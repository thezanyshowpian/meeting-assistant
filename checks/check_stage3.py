#!/usr/bin/env python3
"""
Stage 3 + Stage 4 check harness — grounding the meeting record.

Runs with NO model: grounding is a pure function over a record and a transcript,
so the part that protects 40 points is testable in isolation — the same property
that made Stage 2's gates testable.

The traps here mirror the real failure modes of a generative model asked for
structured facts: fabricated quotes, invented owners, invented deadlines, and
proposals promoted to decisions.

USAGE
    python checks/check_stage3.py
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.ground import find_evidence, ground  # noqa: E402
from app.schemas import ActionItem, Decision, MeetingRecord, Segment, Word  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def record_result(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    print(f"  {status}  {name}" + (f"  — {detail}" if detail else ""), flush=True)


def check(name: str):
    def wrap(fn):
        try:
            record_result("PASS", name, fn() or "")
        except AssertionError as exc:
            record_result("FAIL", name, str(exc))
        except Exception as exc:                                   # noqa: BLE001
            record_result("FAIL", name, f"unexpected {type(exc).__name__}: {exc}")
            traceback.print_exc()
        return fn
    return wrap


def make_segments() -> list[Segment]:
    lines = [
        "Okay let's do it, we're moving the session store from Redis to PostgreSQL",
        "I'll benchmark the PostgreSQL session store latency and have numbers by Friday",
        "Sam can you update the Grafana dashboard",
        "And someone needs to document the rollback procedure",
    ]
    segs = []
    for i, text in enumerate(lines):
        words = [Word(w, i * 10 + j * 0.4, i * 10 + j * 0.4 + 0.4, 0.95)
                 for j, w in enumerate(text.split())]
        segs.append(Segment(id=i, start=i * 10.0, end=i * 10.0 + 9.0, text=text,
                            avg_logprob=-0.2, no_speech_prob=0.0,
                            compression_ratio=1.0, words=words))
    return segs


def main() -> int:
    segments = make_segments()
    print("=" * 70)
    print("STAGE 3/4 CHECK HARNESS — grounding the meeting record")
    print("=" * 70)

    print("\nEVIDENCE LOOKUP")

    @check("a real quote is located and anchored to a timestamp")
    def _():
        seg = find_evidence("we're moving the session store from Redis to PostgreSQL",
                            segments)
        assert seg is not None, "real quote not found"
        assert seg.id == 0, f"anchored to wrong segment {seg.id}"
        return f"anchored to segment {seg.id} at {seg.start:.0f}s"

    @check("a paraphrased quote still grounds")
    def _():
        seg = find_evidence("moving the session store from Redis to PostgreSQL", segments)
        assert seg is not None, "paraphrase rejected (threshold too strict)"
        return "tolerates light paraphrase"

    @check("a FABRICATED quote is not found")
    def _():
        seg = find_evidence("we agreed to migrate everything to MongoDB next quarter",
                            segments)
        assert seg is None, f"fabricated quote matched segment {seg.id if seg else None}"
        return "invented quote correctly unmatched"

    @check("an empty quote is not accepted as evidence")
    def _():
        assert find_evidence("", segments) is None
        assert find_evidence("the a of", segments) is None, "stopwords counted as evidence"
        return "empty / contentless evidence rejected"

    print("\nGROUNDING THE RECORD")

    @check("supported decision is KEPT and timestamped")
    def _():
        rec = MeetingRecord(decisions=[Decision(
            statement="Move the session store from Redis to PostgreSQL",
            evidence="we're moving the session store from Redis to PostgreSQL")])
        res = ground(rec, segments)
        assert len(res.record.decisions) == 1, "dropped a supported decision"
        d = res.record.decisions[0]
        assert d.timestamp is not None, "no timestamp anchored"
        return f"kept, anchored at {d.timestamp:.0f}s"

    @check("decision with FABRICATED evidence is DROPPED")
    def _():
        rec = MeetingRecord(decisions=[Decision(
            statement="Migrate everything to MongoDB",
            evidence="we agreed to migrate everything to MongoDB next quarter")])
        res = ground(rec, segments)
        assert res.record.decisions == [], "kept an unsupported decision!"
        assert res.dropped_decisions, "drop not reported"
        return res.dropped_decisions[0][1]

    @check("decision with NO evidence is DROPPED")
    def _():
        rec = MeetingRecord(decisions=[Decision(statement="Something was decided")])
        res = ground(rec, segments)
        assert res.record.decisions == [], "kept a decision with no evidence"
        return "no-evidence decision removed"

    print("\nINVENTED ATTRIBUTION — the graded response")

    @check("real owner spoken aloud is KEPT")
    def _():
        rec = MeetingRecord(action_items=[ActionItem(
            task="Update the Grafana dashboard", owner="Sam",
            evidence="Sam can you update the Grafana dashboard")])
        res = ground(rec, segments)
        assert res.record.action_items[0].owner == "Sam", "dropped a real owner"
        return "owner 'Sam' preserved"

    @check("INVENTED owner is downgraded, task KEPT")
    def _():
        rec = MeetingRecord(action_items=[ActionItem(
            task="Document the rollback procedure", owner="Priya",
            evidence="someone needs to document the rollback procedure")])
        res = ground(rec, segments)
        assert len(res.record.action_items) == 1, "threw away a real task"
        assert res.record.action_items[0].owner is None, "kept an invented owner!"
        assert res.downgraded, "downgrade not reported"
        return "task kept, invented owner -> unspecified"

    @check("INVENTED deadline is downgraded, task KEPT")
    def _():
        rec = MeetingRecord(action_items=[ActionItem(
            task="Update the Grafana dashboard", owner="Sam", deadline="next Tuesday",
            evidence="Sam can you update the Grafana dashboard")])
        res = ground(rec, segments)
        a = res.record.action_items[0]
        assert a.deadline is None, f"kept invented deadline {a.deadline!r}"
        assert a.owner == "Sam", "downgraded the owner too (should be precise)"
        return "deadline -> unspecified, owner untouched"

    @check("stated deadline is KEPT")
    def _():
        rec = MeetingRecord(action_items=[ActionItem(
            task="Benchmark the PostgreSQL session store latency", deadline="Friday",
            evidence="benchmark the PostgreSQL session store latency and have numbers by Friday")])
        res = ground(rec, segments)
        assert res.record.action_items[0].deadline == "Friday", "dropped a real deadline"
        return "deadline 'Friday' preserved"

    @check("action with fabricated evidence is DROPPED entirely")
    def _():
        rec = MeetingRecord(action_items=[ActionItem(
            task="Deploy to production tonight", owner="Arjun",
            evidence="let's deploy straight to production tonight")])
        res = ground(rec, segments)
        assert res.record.action_items == [], "kept an unsupported action item"
        assert res.dropped_actions, "drop not reported"
        return res.dropped_actions[0][1]

    print("\nREPORTING + EDGE CASES")

    @check("removals and downgrades are surfaced as warnings")
    def _():
        rec = MeetingRecord(
            decisions=[Decision(statement="X", evidence="totally invented quote here")],
            action_items=[ActionItem(task="Update the Grafana dashboard",
                                     owner="Nobody",
                                     evidence="Sam can you update the Grafana dashboard")])
        res = ground(rec, segments)
        assert res.record.warnings, "nothing surfaced to the user"
        assert res.to_dict()["dropped_decisions"], "not serialisable for the UI"
        return f"{len(res.record.warnings)} warning(s) raised"

    @check("empty record is handled")
    def _():
        res = ground(MeetingRecord(), segments)
        assert res.record.decisions == [] and res.record.action_items == []
        assert res.total_removed == 0
        return "no crash, nothing invented"

    print("\nSHORT QUOTES — 'Do that.' (found on the held-out meeting)")
    held = [Segment(i, i * 5.0, i * 5.0 + 4, t, -0.2, 0.0, 1.0, []) for i, t in enumerate([
        "The payment screen still feels slow.",
        "That's the API call. We fetch the full order history before rendering anything.",
        "Can we load it lazily?",
        "Yes. I can change it to load the history after the screen appears.",
        "Do that. Dev, can you have it ready by Thursday?",
    ])]

    @check("short quote + supporting context -> decision KEPT")
    def _():
        rec = MeetingRecord(decisions=[Decision(
            statement="Implement lazy loading of order history on the payment screen",
            evidence="Do that.")])
        res = ground(rec, held)
        assert len(res.record.decisions) == 1, res.dropped_decisions
        assert res.record.decisions[0].segment_id == 4
        return res.context_verified[0][:95]

    @check("short quote, but the claim is NOT in the context -> DROPPED")
    def _():
        rec = MeetingRecord(decisions=[Decision(
            statement="Migrate all services to MongoDB next quarter", evidence="Do that.")])
        assert ground(rec, held).record.decisions == []
        return "verbatim 'Do that.' alone does not support an unrelated claim"

    @check("short quote that is not verbatim in the transcript -> DROPPED")
    def _():
        rec = MeetingRecord(decisions=[Decision(
            statement="Implement lazy loading of order history", evidence="Ship it.")])
        assert ground(rec, held).record.decisions == []
        return "'Ship it.' was never said"

    @check("a quote spanning TWO Whisper segments is still found")
    def _():
        split = [Segment(i, i * 5.0, i * 5.0 + 4, t, -0.2, 0.0, 1.0, []) for i, t in enumerate([
            "Then let's fix it for everyone. From now on, every on call shift gets a",
            "named secondary engineer, who is paged automatically for any high severity incident,",
            "and whose job is communication.",
        ])]
        rec = MeetingRecord(decisions=[Decision(
            statement="Add a secondary engineer to every on-call shift",
            evidence="From now on, every on-call shift gets a named secondary engineer, "
                     "who is paged automatically for any high severity incident")])
        res = ground(rec, split)
        assert len(res.record.decisions) == 1, res.dropped_decisions
        return f"kept, anchored to segment {res.record.decisions[0].segment_id}"

    print("\nDUPLICATE ACTION ITEMS")

    @check("request + acceptance listed twice (same owner) -> MERGED")
    def _():
        rec = MeetingRecord(action_items=[
            ActionItem(task="Update the Grafana dashboard to show new metrics", owner="Sam",
                       evidence="Sam can you update the Grafana dashboard"),
            ActionItem(task="Take ownership of the Grafana dashboard", owner="Sam",
                       deadline="Friday",
                       evidence="Sam can you update the Grafana dashboard")])
        res = ground(rec, segments)
        acts = res.record.action_items
        assert len(acts) == 1, [a.task for a in acts]
        assert acts[0].task.startswith("Update") and acts[0].deadline == "Friday"
        return res.merged[0][:95]

    @check("same owner, different work -> NOT merged")
    def _():
        rec = MeetingRecord(action_items=[
            ActionItem(task="Update the Grafana dashboard", owner="Sam",
                       evidence="Sam can you update the Grafana dashboard"),
            ActionItem(task="Benchmark the PostgreSQL session store latency", owner="Sam",
                       evidence="benchmark the PostgreSQL session store latency")])
        res = ground(rec, segments)
        assert len(res.record.action_items) == 2 and not res.merged
        return "2 kept"

    @check("two UNOWNED overlapping tasks are never merged")
    def _():
        rec = MeetingRecord(action_items=[
            ActionItem(task="Document the rollback procedure",
                       evidence="someone needs to document the rollback procedure"),
            ActionItem(task="Document the rollback procedure for staging",
                       evidence="someone needs to document the rollback procedure")])
        assert len(ground(rec, segments).record.action_items) == 2
        return "no owner, no merge"

    @check("grounding never ADDS anything")
    def _():
        rec = MeetingRecord(
            summary="a summary",
            decisions=[Decision(statement="Move the session store from Redis to PostgreSQL",
                                evidence="we're moving the session store from Redis to PostgreSQL")])
        res = ground(rec, segments)
        assert len(res.record.decisions) <= 1, "grounding invented a decision"
        assert res.record.summary == "a summary", "grounding altered the summary"
        return "output is a subset of the input"

    passed = sum(1 for s, _, _ in RESULTS if s == "PASS")
    failed = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    print("\n" + "=" * 70)
    print(f"RESULT: {passed} passed, {failed} failed")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
