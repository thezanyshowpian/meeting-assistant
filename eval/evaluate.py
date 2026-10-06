#!/usr/bin/env python3
"""
Evaluation harness — measures the pipeline against known ground truth.

Judges score us on a recording we've never seen, so we need our own numbers
before submission rather than impressions. The sample meeting was written with
its answers known, which makes four things measurable:

  1. WER of the raw transcript                 (Stage 1 — 20 pts)
  2. Whether refinement HELPED or HURT         (Stage 2 — 20 pts)
     Measured as the change in WER. Refinement that raises WER is damage, and
     without measuring it you would never know.
  3. Decisions: the agreed one found, the parked PROPOSAL correctly absent
                                                (Stage 3 — 25 pts)
  4. Action items: owners/deadlines captured when stated and left UNSPECIFIED
     when not                                   (Stage 3 — 15 pts)

USAGE
    python eval/evaluate.py
    python eval/evaluate.py --no-llm        # deterministic refinement only
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import re  # noqa: E402

from app.pipeline.glossary import Glossary  # noqa: E402
from app.pipeline.orchestrator import run_pipeline  # noqa: E402
from app.pipeline.refine import refine  # noqa: E402
from app.schemas import Segment, Transcript, Word  # noqa: E402
from eval.wer import compute_wer, normalize  # noqa: E402

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "fifth", "first", "second", "third",
}


def numeric_content(text: str) -> list[str]:
    """Digits and number-words, in order. The thing Stage 2 must never alter."""
    return [t for t in normalize(text) if t.isdigit() or t in NUMBER_WORDS
            or re.fullmatch(r"\d+(st|nd|rd|th)", t)]


def inject_errors(transcript: Transcript, corruptions: dict[str, str]) -> tuple[Transcript, list[tuple[str, str]]]:
    """
    Replace correct terms with realistic mis-hearings so the refinement path can
    be measured. Injected words are given LOW confidence, because real ASR errors
    genuinely do come with low confidence — and because the confidence gate
    (correctly) refuses to edit text the recogniser was sure about.
    """
    applied: list[tuple[str, str]] = []
    new_segments: list[Segment] = []
    for seg in transcript.segments:
        text = seg.text
        injected_here: set[str] = set()
        for term, wrong in corruptions.items():
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(wrong, text)
                applied.append((wrong, term))
                injected_here.update(w.lower() for w in wrong.split())

        words = []
        clock = seg.start
        step = (seg.end - seg.start) / max(len(text.split()), 1)
        for tok in text.split():
            bare = re.sub(r"[^\w]", "", tok).lower()
            prob = 0.35 if bare in injected_here else 0.95
            words.append(Word(word=tok, start=clock, end=clock + step, probability=prob))
            clock += step
        new_segments.append(Segment(
            id=seg.id, start=seg.start, end=seg.end, text=text,
            avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob,
            compression_ratio=seg.compression_ratio, words=words,
        ))

    corrupted = Transcript(
        text=" ".join(s.text for s in new_segments).strip(),
        segments=new_segments, language=transcript.language,
        duration=transcript.duration, model=transcript.model,
        decode_method=transcript.decode_method,
    )
    return corrupted, applied

SCRIPT = os.path.join(HERE, "data", "sample_meeting_script.json")
AUDIO = os.path.join(HERE, "data", "sample_meeting.wav")


def contains_idea(text: str, phrase: str, threshold: float = 0.6) -> bool:
    """Loose containment: do most of the key words of `phrase` appear in `text`?"""
    want = [w for w in normalize(phrase) if len(w) > 3]
    if not want:
        return False
    have = set(normalize(text))
    return sum(1 for w in want if w in have) / len(want) >= threshold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--audio", default=AUDIO)
    args = ap.parse_args()

    if not os.path.exists(args.audio):
        print(f"missing {args.audio}\nRun: python scripts/make_sample_meeting.py")
        return 1

    with open(SCRIPT) as fh:
        spec = json.load(fh)
    truth = spec["ground_truth"]
    reference = " ".join(line["text"] for line in spec["lines"])

    glossary = Glossary(truth["domain_terms"])
    print("=" * 72)
    print("EVALUATION — sample meeting with known ground truth")
    print("=" * 72)

    result = run_pipeline(args.audio, glossary=glossary,
                          use_llm_refinement=not args.no_llm,
                          progress=lambda s, m: print(f"  [{s}] {m}", flush=True))
    if not result.ok:
        print(f"\nPipeline failed at {result.failed_stage}: {result.error}")
        return 1

    # Surface warnings LOUDLY. Without this a hard backend failure is caught,
    # degraded, and shows up as "no decisions reported" — indistinguishable from
    # a meeting that genuinely had none. That cost us a full debugging cycle.
    if result.warnings:
        print("\n" + "!" * 72)
        print("PIPELINE WARNINGS")
        for w in result.warnings:
            print(f"  ! {w}")
        print("!" * 72)

    if result.record:
        print(f"\n  stage 3 model: {result.record.model}")

    score: list[tuple[str, bool, str]] = []

    # ---------------------------------------------------- 1 & 2: transcription
    print("\n" + "-" * 72)
    print("TRANSCRIPTION + REFINEMENT")
    print("-" * 72)
    raw = compute_wer(reference, result.raw_text)
    ref = compute_wer(reference, result.refined_text)
    print(f"  raw      {raw}")
    print(f"  refined  {ref}")
    delta = raw.wer - ref.wer
    verdict = ("improved" if delta > 0.0005 else
               "no change" if abs(delta) <= 0.0005 else "MADE IT WORSE")
    print(f"  refinement {verdict} (delta {delta:+.2%})")
    score.append(("refinement did not increase WER", delta >= -0.0005,
                  f"{verdict} ({delta:+.2%})"))

    if result.refinement:
        print(f"\n  applied {len(result.refinement.applied)} / rejected "
              f"{len(result.refinement.rejected)}")
        for e in result.refinement.applied:
            print(f"    + {e.original!r} -> {e.replacement!r}")
        for e in result.refinement.rejected[:8]:
            print(f"    - {e.original!r} -> {e.replacement!r}  "
                  f"[{'; '.join(e.rejections)[:60]}]")

    print(f"\n  raw transcript: {result.raw_text[:200]}…")

    # Preserved content. NOTE the baseline: we compare refined against the RAW
    # TRANSCRIPT, not against the script. Whisper legitimately normalises spoken
    # numbers ("two hundred milliseconds" -> "200 milliseconds"); that is ASR
    # behaviour, not refinement damage. The invariant we actually care about is
    # that STAGE 2 changed nothing numeric.
    neg = truth["must_preserve"]["negation"]
    score.append(("negation preserved through refinement",
                  contains_idea(result.refined_text, neg, 0.7), neg[:50]))

    raw_nums, ref_nums = numeric_content(result.raw_text), numeric_content(result.refined_text)
    score.append(("Stage 2 altered no numbers (refined vs raw)",
                  raw_nums == ref_nums,
                  f"raw={raw_nums[:6]} refined={ref_nums[:6]}" if raw_nums != ref_nums else ""))

    # ------------------------------------------------------------ 3: decisions
    print("\n" + "-" * 72)
    print("DECISIONS")
    print("-" * 72)
    record = result.record
    found = [d.statement for d in record.decisions] if record else []
    for d in found:
        print(f"  reported: {d}")
    if not found:
        print("  (none reported)")

    for expected in truth["decisions"]:
        if expected.get("scored") is False:
            # Recorded for the record but deliberately not scored — e.g. a valid
            # decision discovered after the fixture was written.
            continue
        present = any(contains_idea(f, expected["statement"]) for f in found)
        if expected["must_appear"]:
            score.append((f"decision FOUND: {expected['statement'][:45]}", present, ""))
        else:
            score.append((f"parked proposal correctly ABSENT: "
                          f"{expected['statement'][:35]}", not present,
                          "reported as a decision!" if present else ""))

    # --------------------------------------------------------- 4: action items
    print("\n" + "-" * 72)
    print("ACTION ITEMS")
    print("-" * 72)
    actions = record.action_items if record else []
    for a in actions:
        print(f"  {a.task[:52]:<52} owner={a.owner_display:<12} "
              f"deadline={a.deadline_display}")
    if not actions:
        print("  (none reported)")

    for expected in truth["action_items"]:
        match = next((a for a in actions if contains_idea(a.task, expected["task"])), None)
        label = expected["task"][:40]
        score.append((f"action found: {label}", match is not None, ""))
        if not match:
            continue
        # owner
        if expected["owner"]:
            ok = bool(match.owner) and expected["owner"].lower() in match.owner.lower()
            score.append((f"  owner = {expected['owner']}", ok, f"got {match.owner!r}"))
        else:
            note = " (unattributable without diarization)" if expected.get(
                "requires_diarization") else ""
            score.append((f"  owner correctly UNSPECIFIED{note}", match.owner is None,
                          f"INVENTED {match.owner!r}" if match.owner else ""))
        # deadline
        if expected["deadline"]:
            ok = bool(match.deadline) and expected["deadline"].lower() in match.deadline.lower()
            score.append((f"  deadline = {expected['deadline']}", ok,
                          f"got {match.deadline!r}"))
        else:
            score.append(("  deadline correctly UNSPECIFIED", match.deadline is None,
                          f"INVENTED {match.deadline!r}" if match.deadline else ""))

    # ------------------------------------- Stage 2 under simulated ASR errors
    # The sample audio is synthetic, so Whisper hears the jargon perfectly and
    # Stage 2 has nothing to correct. That means the happy path above proves
    # NOTHING about refinement. Here we inject verified phonetic mis-hearings and
    # measure whether refinement actually recovers them.
    inj = truth.get("injections")
    if inj and result.transcript:
        print("\n" + "-" * 72)
        print("STAGE 2 UNDER SIMULATED ASR ERRORS")
        print("-" * 72)
        neg_ctl = inj["negative_control"]
        # The control is injected through the SAME path as every other
        # corruption — an earlier version only rewrote Transcript.text, which
        # refine() never reads, so the control silently never ran.
        all_corruptions = {**inj["corruptions"], neg_ctl["term"]: neg_ctl["corruption"]}
        corrupted, applied = inject_errors(result.transcript, all_corruptions)

        injected = sorted({w for w, _ in applied})
        print(f"  injected {len(applied)} instance(s) of {len(injected)} corruption(s)")
        control_present = neg_ctl["corruption"] in corrupted.text
        print(f"  negative control {neg_ctl['corruption']!r} actually injected: "
              f"{control_present}")

        recovered = refine(corrupted, glossary=glossary, use_llm=not args.no_llm)
        print(f"\n  refinement applied {len(recovered.applied)}, "
              f"rejected {len(recovered.rejected)}")
        for e in recovered.applied:
            print(f"    + {e.original!r} -> {e.replacement!r}")
        for e in recovered.rejected:
            print(f"    - {e.original!r} -> {e.replacement!r}")
            for reason in e.rejections:
                print(f"        because: {reason}")

        # Score per UNIQUE corruption: absent from the refined text means every
        # instance of it was recovered.
        fixed_count = 0
        for wrong in injected:
            if wrong == neg_ctl["corruption"]:
                continue
            term = next(t for w, t in applied if w == wrong)
            ok = wrong.lower() not in recovered.refined_text.lower()
            fixed_count += ok
            score.append((f"recovered {wrong!r} -> {term}", ok,
                          "" if ok else "still present in refined text"))
        total = len([w for w in injected if w != neg_ctl["corruption"]])
        print(f"\n  recovery rate: {fixed_count}/{total} "
              f"({fixed_count / total:.0%})" if total else "")

        control_survived = neg_ctl["corruption"].lower() in recovered.refined_text.lower()
        score.append((
            f"negative control {neg_ctl['corruption']!r} left UNCORRECTED",
            control_present and control_survived,
            "" if control_present else "CONTROL WAS NEVER INJECTED — test invalid",
        ))

    # ------------------------------------------------------------------ report
    print("\n" + "=" * 72)
    print("SCORECARD")
    print("=" * 72)
    passed = 0
    for name, ok, detail in score:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   — {detail}" if detail else ""))
        passed += bool(ok)
    print("-" * 72)
    print(f"  {passed}/{len(score)} checks passed")
    print(f"  raw WER {raw.wer:.2%} | refined WER {ref.wer:.2%}")
    print("=" * 72)
    return 0 if passed == len(score) else 1


if __name__ == "__main__":
    raise SystemExit(main())
