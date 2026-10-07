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

from app.pipeline.glossary import Glossary, default_glossary  # noqa: E402
from app.pipeline.orchestrator import run_pipeline  # noqa: E402
from app.pipeline.refine import refine  # noqa: E402
from app.schemas import Segment, Transcript, Word  # noqa: E402
from eval.der import compute_der, load_rttm  # noqa: E402
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


def _stem(word: str) -> str:
    """
    Crude suffix stripping, so 'lazily'~'lazy' and 'loading'~'load'. Added after
    the held-out run where the model reported "Implement lazy loading of order
    history…" and exact-word matching scored the decision as MISSING (3/6 words).
    Applied identically to every check; it makes matching fairer, not looser on
    meaning (negation is handled separately by declines()).
    """
    for suffix, repl in (("ily", "y"), ("ing", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: len(word) - len(suffix)] + repl
    return word


def contains_idea(text: str, phrase: str, threshold: float = 0.6) -> bool:
    """Loose containment: do most of the key words of `phrase` appear in `text`?"""
    want = [_stem(w) for w in normalize(phrase) if len(w) > 3]
    if not want:
        return False
    have = {_stem(w) for w in normalize(text)}
    return sum(1 for w in want if w in have) / len(want) >= threshold


_DECLINES = re.compile(r"\b(?:not|no|don't|do not|won't|will not|defer\w*|postpone\w*|"
                       r"park\w*|declin\w*|reject\w*|later|skip\w*|hold off)\b", re.I)


def declines(statement: str) -> bool:
    """
    Does a reported decision state that something will NOT be done?

    A proposal the group declined may legitimately come back as the decision
    "do not redesign onboarding" — that is correct, not the trap. Keyword
    overlap alone can't tell "redesign onboarding" from "do NOT redesign
    onboarding"; this was a scorer bug found on the held-out meeting.
    """
    return bool(_DECLINES.search(statement))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--fixture", choices=["sample", "heldout", "long"], default="sample",
                    help="heldout: the 4-speaker meeting never used for tuning; "
                         "long: the ~30-minute, 6-speaker demo meeting")
    ap.add_argument("--cache", action="store_true",
                    help="reuse a cached Stage 1 transcript (data/.cache) if present")
    ap.add_argument("--speakers", type=int, default=None,
                    help="known number of speakers, as the app's sidebar allows")
    ap.add_argument("--out", help="also write meeting_record.{json,md} and "
                                  "transcripts.md to this directory")
    args = ap.parse_args()
    script = SCRIPT.replace("sample_", f"{args.fixture}_")
    args.audio = args.audio or script.replace("_script.json", ".wav")
    rttm = script.replace("_script.json", "_reference.rttm")

    if not os.path.exists(args.audio):
        print(f"missing {args.audio}\nRun: python scripts/make_sample_meeting.py")
        return 1

    with open(script) as fh:
        spec = json.load(fh)
    truth = spec["ground_truth"]
    reference = " ".join(line["text"] for line in spec["lines"])

    glossary = Glossary(truth["domain_terms"]) if "domain_terms" in truth \
        else default_glossary()
    print("=" * 72)
    print("EVALUATION — sample meeting with known ground truth")
    print("=" * 72)

    transcriber = None
    if args.cache:
        from eval.cache import CachedTranscriber
        transcriber = CachedTranscriber()
    result = run_pipeline(args.audio, glossary=glossary, transcriber=transcriber,
                          num_speakers=args.speakers,
                          use_llm_refinement=not args.no_llm,
                          progress=lambda s, m: print(f"  [{s}] {m}", flush=True))
    if not result.ok:
        print(f"\nPipeline failed at {result.failed_stage}: {result.error}")
        return 1
    print("  stage times: " + ", ".join(f"{k} {v:.0f}s" for k, v in result.stage_times.items()))
    if args.out:
        from app.export import to_json, to_markdown, transcripts_markdown
        os.makedirs(args.out, exist_ok=True)
        for name, text in (("meeting_record.json", to_json(result)),
                           ("meeting_record.md", to_markdown(result)),
                           ("transcripts.md", transcripts_markdown(result))):
            with open(os.path.join(args.out, name), "w") as fh:
                fh.write(text)
        print(f"  wrote outputs to {args.out}")

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
    # What Stage 4 removed or downgraded — a count alone can't distinguish a
    # caught fabrication from a valid item lost to over-strict matching.
    g = result.grounding
    if g and (g.total_removed or g.downgraded or g.context_verified or g.merged):
        print("\n  STAGE 4 ACTIONS")
        for st, why in g.dropped_decisions:
            print(f"    - dropped decision: {st!r} ({why})")
        for st, why in g.dropped_actions:
            print(f"    - dropped action:   {st!r} ({why})")
        for note in g.downgraded:
            print(f"    ~ {note}")
        for note in g.context_verified:
            print(f"    + short quote kept: {note}")
        for note in g.merged:
            print(f"    = merged: {note}")

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
    if "must_preserve" in truth:
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
        hits = [f for f in found if contains_idea(f, expected["statement"])]
        present = bool(hits)
        if not expected["must_appear"]:
            if any(declines(f) for f in hits):
                print(f"  note: declined proposal reported AS declined (correct): "
                      f"{[f for f in hits if declines(f)]}")
            present = any(not declines(f) for f in hits)
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
        print(f"  {a.task[:52]:<52} owner={a.owner_display:<18} "
              f"deadline={a.deadline_display}")
        if a.owner_evidence and a.owner_source != "stated":
            print(f"      owner evidence: {a.owner_evidence[:150]}")
    if not actions:
        print("  (none reported)")

    for expected in truth["action_items"]:
        match = next((a for a in actions if contains_idea(a.task, expected["task"])), None)
        label = expected["task"][:40]
        score.append((f"action found: {label}", match is not None, ""))
        if not match:
            continue
        # owner — a first-person commitment is attributable only with diarization
        expected = dict(expected)
        if result.diarization and expected.get("owner_with_diarization"):
            expected["owner"] = expected["owner_with_diarization"]
            expected["requires_diarization"] = False
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

    # --------------------------------------------------- speaker diarization
    if result.diarization and os.path.exists(rttm):
        print("\n" + "-" * 72)
        print("SPEAKER DIARIZATION")
        print("-" * 72)
        ref_turns = load_rttm(rttm)
        n_ref = len({s for _, _, s in ref_turns})
        d = compute_der(ref_turns, result.diarization.hypothesis())
        print(f"  {d}")
        print(f"  speakers: found {result.diarization.num_speakers}, true {n_ref}  "
              f"| {result.diarization.method}")
        print(f"  label mapping: {d.mapping}")
        for t in result.diarization.turns[:6]:
            print(f"    {t.speaker:<10} [{t.start:5.1f}s] {t.text[:58]}")
        score.append((f"diarization found the true number of speakers ({n_ref})",
                      result.diarization.num_speakers == n_ref,
                      f"found {result.diarization.num_speakers}"))

        # ---- naming: right name on the right voice, and NO name without evidence
        names = truth.get("speaker_names")
        if names and result.naming:
            print("\n  names inferred:")
            for b in result.naming.bindings.values():
                print(f"    {b.speaker} -> {b.name}  (true voice: "
                      f"{d.mapping.get(b.speaker, '?')})  {b.evidence[0].describe()[:90]}")
            for c in result.naming.conflicts:
                print(f"    not named: {c}")
            for person in names["expected"]:
                label = result.naming.label_for(person)
                ok = label is not None and d.mapping.get(label) == person
                score.append((f"name '{person}' bound to {person}'s voice", ok,
                              f"bound to {label} = {d.mapping.get(label)}" if label
                              else "not named"))
            wrong = [f"{b.speaker}->{b.name} (is {d.mapping.get(b.speaker)})"
                     for b in result.naming.bindings.values()
                     if d.mapping.get(b.speaker) != b.name]
            score.append(("no voice was given a WRONG name (precision)", not wrong,
                          ", ".join(wrong)))
            for m in result.naming.suspects:
                print(f"    cross-check: {m.describe()}")
            for person in names["anonymous"]:
                label = next((h for h, r in d.mapping.items() if r == person), None)
                given = result.naming.name_for(label) if label else None
                score.append((f"{person} (never addressed) left ANONYMOUS", given is None,
                              f"named {given!r}!" if given else ""))
    elif os.path.exists(rttm):
        print("\n  (diarization not run — speechbrain not installed, or DIARIZE=off)")

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

    # ------------------------------- Stage 2 on REAL mispronunciations (long)
    # The voice actually says "cough ka" for Kafka. Whether Whisper mishears it
    # is up to Whisper; we score recovery only where it did.
    variants = truth.get("spoken_variants")
    if variants and result.refinement:
        print("\n" + "-" * 72)
        print("STAGE 2 ON SPOKEN MISPRONUNCIATIONS")
        print("-" * 72)
        raw_l, ref_l = result.raw_text.lower(), result.refined_text.lower()
        for meant, spoken in variants.items():
            if spoken.lower() in raw_l:
                ok = spoken.lower() not in ref_l
                print(f"  {spoken!r:<18} misheard by Whisper -> "
                      f"{'recovered as ' + meant if ok else 'NOT recovered'}")
                score.append((f"misheard {spoken!r} recovered -> {meant}", ok, ""))
            else:
                print(f"  {spoken!r:<18} Whisper heard it as intended — nothing for "
                      f"Stage 2 to do (not scored)")

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
