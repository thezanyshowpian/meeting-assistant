#!/usr/bin/env python3
"""
Stage 2 check harness — verification gates and glossary.

Runs with NO model: the gates are pure functions over text + Stage 1 metadata,
so the safety-critical half of Stage 2 is testable on its own. That's deliberate
— the gates are what protect the 20 faithfulness points, so they get tested
hardest and earliest.

USAGE
    python checks/check_stage2.py
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.glossary import (Glossary, default_glossary,  # noqa: E402
                                   phonetically_equal)
from app.pipeline.verification import (Edit, EditType, apply_edits,  # noqa: E402
                                       verify_edit, verify_edits)
from app.pipeline.glossary import normalize  # noqa: E402
from app.schemas import Segment, Word  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def normalize_eq(a: str, b: str) -> bool:
    return normalize(a) == normalize(b)


def record(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    print(f"  {status}  {name}" + (f"  — {detail}" if detail else ""), flush=True)


def check(name: str):
    def wrap(fn):
        try:
            record("PASS", name, fn() or "")
        except AssertionError as exc:
            record("FAIL", name, str(exc))
        except Exception as exc:                                  # noqa: BLE001
            record("FAIL", name, f"unexpected {type(exc).__name__}: {exc}")
            traceback.print_exc()
        return fn
    return wrap


def seg(text: str, sid: int = 0, probs: dict[str, float] | None = None) -> Segment:
    """Segment whose words carry confidences (default: LOW, so gates don't block)."""
    probs = probs or {}
    words = [Word(w, i * 0.5, i * 0.5 + 0.5, probs.get(w.lower(), 0.3))
             for i, w in enumerate(text.split())]
    return Segment(id=sid, start=0.0, end=len(words) * 0.5, text=text,
                   avg_logprob=-0.4, no_speech_prob=0.0,
                   compression_ratio=1.0, words=words)


def edit(original: str, replacement: str, sid: int = 0) -> Edit:
    return Edit(segment_id=sid, original=original, replacement=replacement,
                reason="test", confidence=0.9)


def main() -> int:
    g = default_glossary()
    print("=" * 70)
    print("STAGE 2 CHECK HARNESS — glossary + verification gates")
    print("=" * 70)

    print("\nPHONETICS — the core discrimination")

    @check("real ASR errors are phonetically matched (allow)")
    def _():
        for a, b in [("cooper netties", "Kubernetes"), ("post gres", "Postgres"),
                     ("sequel", "SQL")]:
            assert phonetically_equal(a, b), f"{a!r} !~ {b!r}"
        return "kubernetes / postgres / SQL all matched"

    @check("meaning changes are phonetically REJECTED")
    def _():
        for a, b in [("might", "will"), ("deadline", "headline"),
                     ("we will", "we might")]:
            assert not phonetically_equal(a, b), f"{a!r} wrongly matched {b!r}"
        return "'deadline'->'headline' rejected despite 0.87 string similarity"

    print("\nGLOSSARY — deterministic pre-pass is conservative")

    @check("best_match finds mis-heard jargon")
    def _():
        m = g.best_match("cooper netties")
        assert m == "Kubernetes", f"got {m!r}"
        return f"'cooper netties' -> {m!r}"

    @check("best_match ignores already-correct terms")
    def _():
        assert g.best_match("Kubernetes") is None, "fired on a correct term"
        return ""

    @check("best_match ignores ordinary English")
    def _():
        for phrase in ["we think", "the that", "so we will"]:
            assert g.best_match(phrase) is None, f"fired on ordinary speech: {phrase!r}"
        return "no false positives on common speech"

    print("\nTYPED POLICY — insertion / deletion denied as categories")

    @check("insertions are denied")
    def _():
        e = verify_edit(edit("", "Kubernetes"), seg("we deployed it"), g)
        assert not e.accepted
        assert any("insertion" in r for r in e.rejections), e.rejections
        return e.rejections[0]

    @check("deletions are denied")
    def _():
        e = verify_edit(edit("not", ""), seg("we will not ship"), g)
        assert not e.accepted
        assert any("deletion" in r or "negation" in r for r in e.rejections), e.rejections
        return "blocked"

    @check("stutter collapse is the one allowed deletion")
    def _():
        e = Edit(0, "the the", "", reason="stutter")
        from app.pipeline.verification import gate_type
        assert gate_type(e, seg("the the system"), g) is None, "stutter blocked"
        return "'the the' -> '' permitted by gate_type"

    print("\nSUBSTITUTION GATES")

    @check("legitimate jargon correction is ACCEPTED end-to-end")
    def _():
        s = seg("we deployed on cooper netties last week")
        e = verify_edit(edit("cooper netties", "Kubernetes"), s, g)
        assert e.accepted, f"wrongly rejected: {e.rejections}"
        return "'cooper netties' -> 'Kubernetes' passed all gates"

    @check("semantic drift is rejected (might -> will)")
    def _():
        e = verify_edit(edit("might", "will"), seg("we might ship friday"), g)
        assert not e.accepted
        return e.rejections[0][:70]

    @check("replacement outside the glossary is rejected")
    def _():
        e = verify_edit(edit("cooper netties", "Cubernetties"), seg("on cooper netties"), g)
        assert not e.accepted
        assert any("glossary" in r for r in e.rejections), e.rejections
        return "unknown replacement blocked"

    @check("missing anchor is rejected")
    def _():
        e = verify_edit(edit("cooper netties", "Kubernetes"), seg("nothing like it here"), g)
        assert not e.accepted
        assert any("anchor" in r for r in e.rejections), e.rejections
        return "anchor check works"

    @check("number changes are rejected")
    def _():
        e = verify_edit(edit("redis 5", "Redis 6"), seg("we run redis 5 now"), g)
        assert not e.accepted
        assert any("number" in r for r in e.rejections), e.rejections
        return "digit guard works"

    @check("over-long spans are rejected")
    def _():
        e = verify_edit(edit("a b c d e f g", "SQL"), seg("a b c d e f g"), g)
        assert not e.accepted
        assert any("span" in r for r in e.rejections), e.rejections
        return "rewrite-sized edit blocked"

    print("\nLLM ROLE — off-glossary proposals and the context veto")

    @check("deterministic pass may NOT invent off-glossary terms")
    def _():
        e = edit("cooper netties", "Kubernetz")      # not a glossary term
        e.source = "glossary"
        verify_edit(e, seg("we deployed on cooper netties"), g)
        assert not e.accepted
        assert any("glossary" in r for r in e.rejections), e.rejections
        return "deterministic pass stays glossary-bounded"

    @check("LLM MAY propose an off-glossary technical term")
    def _():
        # The coverage gap: a real term nobody put in the glossary.
        s = seg("we store vectors in pine cone now", probs={"pine": 0.3, "cone": 0.3})
        e = edit("pine cone", "Pinecone")
        e.source, e.confidence = "llm", 0.9
        verify_edit(e, s, g)
        assert e.accepted, f"wrongly rejected: {e.rejections}"
        assert e.glossary_backed is False, "should be flagged as off-glossary"
        return "accepted, flagged glossary_backed=False"

    @check("off-glossary proposal rejected when model confidence is low")
    def _():
        s = seg("we store vectors in pine cone now", probs={"pine": 0.3, "cone": 0.3})
        e = edit("pine cone", "Pinecone")
        e.source, e.confidence = "llm", 0.4
        verify_edit(e, s, g)
        assert not e.accepted
        assert any("confidence" in r for r in e.rejections), e.rejections
        return "low-confidence off-glossary edit blocked"

    @check("off-glossary replacement may not be ordinary vocabulary")
    def _():
        s = seg("the readies are not ready", probs={"readies": 0.3})
        e = edit("readies", "ready")
        e.source, e.confidence = "llm", 0.95
        verify_edit(e, s, g)
        assert not e.accepted
        assert any("ordinary" in r or "phonetic" in r for r in e.rejections), e.rejections
        return "blocked from drifting into everyday words"

    @check("context veto catches the deterministic pass's false positive")
    def _():
        # THE failure mode a context-blind phonetic matcher cannot see:
        # "red is" genuinely means "red, is" here, not Redis.
        from app.pipeline.refine import glossary_pass
        s = seg("the light was red is it not", probs={"red": 0.3, "is": 0.3})
        proposals = glossary_pass([s], g)
        assert any(normalize_eq(p.replacement, "Redis") for p in proposals), \
            "expected the phonetic matcher to (wrongly) propose Redis here"
        # Gates alone accept it — only context review can reject it.
        verify_edit(proposals[0], s, g)
        return (f"deterministic pass proposes {proposals[0].original!r}->"
                f"{proposals[0].replacement!r}; accepted_by_gates="
                f"{proposals[0].accepted} -> needs the LLM veto")

    print("\nCONFIDENCE GATE — only unsure spans may be edited")

    @check("high-confidence text cannot be edited")
    def _():
        s = seg("we deployed on cooper netties", probs={"cooper": 0.97, "netties": 0.98})
        e = verify_edit(edit("cooper netties", "Kubernetes"), s, g)
        assert not e.accepted
        assert any("confidence" in r for r in e.rejections), e.rejections
        return e.rejections[0][:70]

    @check("low-confidence text can be edited")
    def _():
        s = seg("we deployed on cooper netties", probs={"cooper": 0.31, "netties": 0.22})
        e = verify_edit(edit("cooper netties", "Kubernetes"), s, g)
        assert e.accepted, f"wrongly rejected: {e.rejections}"
        return "passes when Whisper was unsure"

    print("\nAGGREGATE + APPLICATION")

    @check("edit budget blocks an avalanche of edits")
    def _():
        # budget = max(MIN_EDIT_BUDGET=5, words // EDIT_BUDGET_PER_WORDS=20)
        # 200 words -> 10 allowed. Propose 25; the excess must be rejected.
        s = seg("cooper netties " * 25, sid=0)
        edits = [edit("cooper netties", "Kubernetes") for _ in range(25)]
        verified = verify_edits(edits, {0: s}, g, total_words=200)
        accepted = [e for e in verified if e.accepted]
        assert len(accepted) == 10, f"{len(accepted)} accepted, expected 10"
        assert any("budget" in r for e in verified for r in e.rejections)
        return "10 of 25 allowed for a 200-word transcript"

    @check("short transcripts get the budget floor, not a budget of 1")
    def _():
        # Regression guard: the original formula gave a 30-word transcript a
        # budget of 1, which silently discarded valid jargon corrections.
        s = seg("cooper netties " * 6, sid=0)
        edits = [edit("cooper netties", "Kubernetes") for _ in range(6)]
        verified = verify_edits(edits, {0: s}, g, total_words=30)
        accepted = [e for e in verified if e.accepted]
        assert len(accepted) == 5, f"{len(accepted)} accepted, expected floor of 5"
        return "floor of 5 applied to a 30-word transcript"

    @check("apply_edits leaves untouched text byte-identical")
    def _():
        segments = [seg("we deployed on cooper netties", 0), seg("nothing changes here", 1)]
        e = edit("cooper netties", "Kubernetes", sid=0)
        e.accepted = True
        out = apply_edits(segments, [e])
        assert out[0].text == "we deployed on Kubernetes", out[0].text
        assert out[1].text == segments[1].text, "untouched segment was modified!"
        assert out[0].words == segments[0].words, "word data lost"
        return "edited segment changed, all else identical"

    @check("rejected edits are never applied")
    def _():
        segments = [seg("we might ship friday", 0)]
        e = verify_edit(edit("might", "will"), segments[0], g)
        out = apply_edits(segments, [e])
        assert out[0].text == "we might ship friday", out[0].text
        return "rejected edit had no effect"

    passed = sum(1 for s, _, _ in RESULTS if s == "PASS")
    failed = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    print("\n" + "=" * 70)
    print(f"RESULT: {passed} passed, {failed} failed")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
