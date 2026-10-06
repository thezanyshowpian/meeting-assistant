#!/usr/bin/env python3
"""
Speaker naming + owner attribution check harness — runs with NO model.

Covers what we wrote: name evidence rules, conflict resolution, the speaker-split
refined transcript, Stage 4's speaker-aware owner checks, and the final
"Speaker 2 -> Arjun (inferred)" attribution. The LLM is not involved: owners are
supplied as the model would emit them, so each test isolates one rule.

USAGE
    python checks/check_naming.py
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.glossary import Glossary  # noqa: E402
from app.pipeline.ground import ground  # noqa: E402
from app.pipeline.naming import (Utterance, attribute_owners,  # noqa: E402
                                 infer_names, speaker_utterances)
from app.pipeline.summarize import system_prompt  # noqa: E402
from app.pipeline.verification import Edit  # noqa: E402
from app.schemas import ActionItem, MeetingRecord, Segment, Word  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []
GLOSSARY = Glossary(["Redis", "PostgreSQL", "Grafana", "Kubernetes", "gRPC"])


def record(status, name, detail=""):
    RESULTS.append((status, name, detail))
    print(f"  {status}  {name}" + (f"  — {detail}" if detail else ""), flush=True)


def check(name):
    def wrap(fn):
        try:
            record("PASS", name, fn() or "")
        except AssertionError as exc:
            record("FAIL", name, str(exc))
        except Exception as exc:                                   # noqa: BLE001
            record("FAIL", name, f"unexpected {type(exc).__name__}: {exc}")
            traceback.print_exc()
        return fn
    return wrap


def meeting(lines):
    """[(speaker, text), ...] -> (segments, utterances), one segment per line."""
    segs, utts, t = [], [], 0.0
    for i, (spk, text) in enumerate(lines):
        segs.append(Segment(i, t, t + 4, text, -0.2, 0.0, 1.0, []))
        utts.append(Utterance(i, spk, t, t + 4, text))
        t += 5
    return segs, utts


# The sample meeting, as diarization labels it. Priya is never addressed by name.
SAMPLE = [
    ("Speaker 1", "Alright, let's start. Main thing today is the session store. Arjun, where are we?"),
    ("Speaker 2", "So right now sessions live in Redis, and we're seeing latency spikes."),
    ("Speaker 1", "Is that a Redis problem or a networking problem?"),
    ("Speaker 2", "Mostly Redis. I think we should move the session store to PostgreSQL."),
    ("Speaker 3", "PostgreSQL will be slower per read, but it's far more predictable."),
    ("Speaker 1", "Okay. Let's do it. We're moving the session store to PostgreSQL."),
    ("Speaker 2", "Agreed. I'll benchmark the PostgreSQL session store latency and have numbers by Friday."),
    ("Speaker 1", "Good. Sam, can you update the Grafana dashboard so we can see the new metrics?"),
    ("Speaker 3", "Yes, I'll take the Grafana dashboard."),
    ("Speaker 1", "And someone needs to document the rollback procedure."),
    ("Speaker 1", "I'll send the notes round after this."),
]


def main() -> int:
    print("=" * 70)
    print("SPEAKER NAMING + OWNER ATTRIBUTION CHECK HARNESS (no model)")
    print("=" * 70)

    print("\nNAME EVIDENCE")

    @check("addressed-then-answered names the answering voice")
    def _():
        _, utts = meeting(SAMPLE)
        n = infer_names(utts, GLOSSARY)
        assert n.name_for("Speaker 2") == "Arjun", n.to_dict()
        assert n.name_for("Speaker 3") == "Sam", n.to_dict()
        return "Speaker 2 -> Arjun, Speaker 3 -> Sam"

    @check("a speaker nobody addresses stays ANONYMOUS (no guess)")
    def _():
        _, utts = meeting(SAMPLE)
        n = infer_names(utts, GLOSSARY)
        assert n.name_for("Speaker 1") is None, n.name_for("Speaker 1")
        return "Speaker 1 (Priya, never named aloud) left unnamed"

    @check("you don't address yourself: the addresser is excluded")
    def _():
        _, utts = meeting([
            ("Speaker 1", "Arjun, can you check the logs?"),
            ("Speaker 1", "Never mind."),           # same voice continues
            ("Speaker 2", "I'm Arjun, by the way."),
        ])
        n = infer_names(utts, GLOSSARY)
        assert "Arjun" in n.excluded.get("Speaker 1", set())
        assert n.name_for("Speaker 1") is None
        assert n.name_for("Speaker 2") == "Arjun"
        return "Speaker 1 excluded; self-introduction names Speaker 2"

    @check("a request mid-turn does not hand the floor (only the last sentence does)")
    def _():
        _, utts = meeting([
            ("Speaker 1", "Sam, can you look at this? Actually, forget it, I'll do it myself."),
            ("Speaker 2", "Sounds fine."),
        ])
        assert infer_names(utts, GLOSSARY).name_for("Speaker 2") is None
        return "no binding"

    @check("discourse words and glossary terms are not names")
    def _():
        _, utts = meeting([
            ("Speaker 1", "Okay, can you start?"), ("Speaker 2", "Sure."),
            ("Speaker 1", "Redis, is that the bottleneck?"), ("Speaker 2", "Yes."),
            ("Speaker 1", "Great, could you share the numbers?"), ("Speaker 2", "Yes."),
        ])
        n = infer_names(utts, GLOSSARY)
        assert not n.bindings, n.to_dict()
        return "'Okay', 'Redis', 'Great' rejected"

    @check("'This is Jenkins' mid-sentence is not a self-introduction")
    def _():
        _, utts = meeting([("Speaker 1", "I think this is Jenkins failing again.")])
        assert not infer_names(utts, GLOSSARY).bindings
        return "no binding"

    print("\nCONFLICTS")

    @check("equal evidence for two names -> left unnamed, conflict reported")
    def _():
        _, utts = meeting([
            ("Speaker 1", "Arjun, where are we?"), ("Speaker 2", "Fine."),
            ("Speaker 1", "Dev, any update?"), ("Speaker 2", "Nothing new."),
        ])
        n = infer_names(utts, GLOSSARY)
        assert n.name_for("Speaker 2") is None and n.conflicts
        return n.conflicts[0]

    @check("one name, two voices -> stronger evidence wins")
    def _():
        _, utts = meeting([
            ("Speaker 1", "Tom, how's the build?"), ("Speaker 2", "Green."),
            ("Speaker 1", "Tom, and the tests?"), ("Speaker 2", "Passing."),
            ("Speaker 1", "Tom, are you there?"), ("Speaker 3", "He stepped out."),
        ])
        n = infer_names(utts, GLOSSARY)
        assert n.name_for("Speaker 2") == "Tom" and n.name_for("Speaker 3") is None
        return "Speaker 2 (2 votes) beats Speaker 3 (1 vote)"

    @check("one name, two voices, TIED -> neither named")
    def _():
        _, utts = meeting([
            ("Speaker 1", "Tom, how's the build?"), ("Speaker 2", "Green."),
            ("Speaker 1", "Tom, are you there?"), ("Speaker 3", "He stepped out."),
        ])
        n = infer_names(utts, GLOSSARY)
        assert not n.bindings, n.to_dict()
        return n.conflicts[-1]

    print("\nSPEAKER-SPLIT REFINED TRANSCRIPT")

    @check("a merged two-speaker segment is split, with Stage 2 edits re-applied")
    def _():
        words = [Word(" Is", 0.0, 0.2, .9, "Speaker 1"), Word(" that", .2, .4, .9, "Speaker 1"),
                 Word(" red", .4, .6, .9, "Speaker 1"), Word(" is?", .6, .8, .9, "Speaker 1"),
                 Word(" Mostly", 1.5, 1.8, .9, "Speaker 2"), Word(" red", 1.8, 2.0, .9, "Speaker 2"),
                 Word(" is.", 2.0, 2.2, .9, "Speaker 2")]
        seg = Segment(0, 0.0, 2.2, "Is that Redis? Mostly Redis.", -0.2, 0, 1, words)
        edit = Edit(segment_id=0, original="red is", replacement="Redis", reason="t",
                    confidence=.9, source="glossary", accepted=True)
        utts = speaker_utterances([seg], [edit])
        assert [u.speaker for u in utts] == ["Speaker 1", "Speaker 2"]
        assert [u.text for u in utts] == ["Is that Redis?", "Mostly Redis."], [u.text for u in utts]
        return " | ".join(f"{u.speaker}: {u.text}" for u in utts)

    @check("Stage 3 prompt switches to label rules only when diarized")
    def _():
        assert "Speaker 2" in system_prompt(True) and "never guess" in system_prompt(True)
        assert "Speaker 2" not in system_prompt(False)
        assert "{owner_rules}" not in system_prompt(False)
        return "plain and diarized prompts differ as intended"

    print("\nSTAGE 4 — SPEAKER-AWARE OWNERS")
    segs, utts = meeting(SAMPLE)
    naming = infer_names(utts, GLOSSARY)

    def run(owner, evidence, task="Benchmark the PostgreSQL session store latency"):
        rec = MeetingRecord(action_items=[ActionItem(task=task, owner=owner, evidence=evidence)])
        res = ground(rec, segs, utterances=utts, naming=naming)
        return res, rec.action_items[0]

    @check("label owner KEPT when that voice spoke the commitment")
    def _():
        _, a = run("Speaker 2", "I'll benchmark the PostgreSQL session store latency")
        assert a.owner == "Speaker 2" and a.owner_source == "speaker", (a.owner, a.owner_source)
        return a.owner_evidence

    @check("label owner DOWNGRADED when a different voice spoke it")
    def _():
        res, a = run("Speaker 3", "I'll benchmark the PostgreSQL session store latency")
        assert a.owner is None and res.downgraded
        return res.downgraded[0][:90]

    @check("guessed name far from the evidence is DOWNGRADED without naming")
    def _():
        # The latent hole in the old check: "Arjun" IS in the transcript (0:00),
        # so a model guessing Arjun owns the 0:30 benchmark used to pass.
        rec = MeetingRecord(action_items=[ActionItem(
            task="Benchmark the PostgreSQL session store latency", owner="Arjun",
            evidence="I'll benchmark the PostgreSQL session store latency")])
        res = ground(rec, segs, utterances=utts, naming=None)
        assert rec.action_items[0].owner is None, "a guessed owner passed grounding"
        return res.downgraded[0][:90]

    @check("same name WITH naming evidence linking the voice -> inferred")
    def _():
        _, a = run("Arjun", "I'll benchmark the PostgreSQL session store latency")
        assert a.owner == "Arjun" and a.owner_source == "inferred", (a.owner, a.owner_source)
        return "accepted, flagged inferred"

    @check("name spoken next to the evidence -> stated")
    def _():
        _, a = run("Sam", "Yes, I'll take the Grafana dashboard", task="Update the Grafana dashboard")
        assert a.owner == "Sam" and a.owner_source == "stated", (a.owner, a.owner_source)
        return a.owner_evidence

    print("\nFINAL ATTRIBUTION")

    @check("grounded 'Speaker 2' becomes 'Arjun (inferred)' with both evidences")
    def _():
        rec = MeetingRecord(action_items=[ActionItem(
            task="Benchmark the PostgreSQL session store latency", owner="Speaker 2",
            evidence="I'll benchmark the PostgreSQL session store latency")])
        ground(rec, segs, utterances=utts, naming=naming)
        attribute_owners(rec, naming)
        a = rec.action_items[0]
        assert a.owner == "Arjun" and a.owner_source == "inferred" and a.owner_label == "Speaker 2"
        assert "addressed by name" in a.owner_evidence and "said it" in a.owner_evidence
        return a.owner_display

    @check("a voice that merely MENTIONS work is not its owner (no first person)")
    def _():
        res, a = run("Speaker 1", "someone needs to document the rollback procedure",
                     task="Document the rollback procedure")
        assert a.owner is None, a.owner
        return res.downgraded[0][:90]

    @check("an unnamed voice keeps its label, flagged voice-only")
    def _():
        rec = MeetingRecord(action_items=[ActionItem(
            task="Send the notes round", owner="Speaker 1",
            evidence="I'll send the notes round after this")])
        ground(rec, segs, utterances=utts, naming=naming)
        attribute_owners(rec, naming)
        a = rec.action_items[0]
        assert a.owner == "Speaker 1" and a.owner_display.endswith("(voice only)")
        return a.owner_display

    print("\nDIARIZATION CROSS-CHECK (the real sample-meeting failure)")
    # As observed: Sam's reply was clustered with Priya's voice (Speaker 1).
    MISSED = [(spk if "take the Grafana" not in txt else "Speaker 1", txt)
              for spk, txt in SAMPLE]
    msegs, mutts = meeting(MISSED)
    mnaming = infer_names(mutts, GLOSSARY)

    @check("a question to Sam 'answered' in the asker's voice is flagged")
    def _():
        assert len(mnaming.suspects) == 1, mnaming.suspects
        m = mnaming.suspects[0]
        assert m.speaker == "Speaker 1" and m.addressee == "Sam"
        return m.describe()[:95]

    @check("...and no name is guessed for anyone from it")
    def _():
        assert mnaming.label_for("Sam") is None and mnaming.name_for("Speaker 1") is None
        return "Sam unnamed; Speaker 1 still anonymous"

    @check("...the flagged voice is never the owner; the NAMED addressee is (stated)")
    def _():
        # As observed on the real sample: Stage 3 gave 'Speaker 1' (Priya's label).
        rec = MeetingRecord(action_items=[ActionItem(
            task="Update the Grafana dashboard", owner="Speaker 1",
            evidence="Yes, I'll take the Grafana dashboard")])
        ground(rec, msegs, utterances=mutts, naming=mnaming)
        a = rec.action_items[0]
        assert a.owner == "Sam" and a.owner_source == "stated", (a.owner, a.owner_source)
        return a.owner_evidence[:95]

    @check("...and if the request was about a DIFFERENT task, the owner is withheld")
    def _():
        rec = MeetingRecord(action_items=[ActionItem(
            task="Fix the checkout flow latency", owner="Speaker 1",
            evidence="Yes, I'll take the Grafana dashboard")])
        res = ground(rec, msegs, utterances=mutts, naming=mnaming)
        assert rec.action_items[0].owner is None, rec.action_items[0].owner
        return res.downgraded[0][:95]

    @check("...while the STATED owner of the same task is unaffected")
    def _():
        rec = MeetingRecord(action_items=[ActionItem(
            task="Update the Grafana dashboard", owner="Sam",
            evidence="Sam, can you update the Grafana dashboard")])
        ground(rec, msegs, utterances=mutts, naming=mnaming)
        a = rec.action_items[0]
        assert a.owner == "Sam" and a.owner_source == "stated"
        return "Sam (stated) kept"

    @check("no false alarm on the correctly diarized meeting, or on self-correction")
    def _():
        assert not naming.suspects, naming.suspects
        _, u2 = meeting([("Speaker 1", "Sam, can you look at this? Actually, forget it, "
                                       "I'll do it myself.")])
        assert not infer_names(u2, GLOSSARY).suspects
        return "0 suspects in both"

    passed = sum(1 for s, _, _ in RESULTS if s == "PASS")
    failed = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    print("\n" + "=" * 70)
    print(f"RESULT: {passed} passed, {failed} failed")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
