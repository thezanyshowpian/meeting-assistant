#!/usr/bin/env python3
"""
Choose the diarization threshold on one meeting, REPORT it on another.

Tuning a threshold on the same recording you report it on measures memory, not
generalisation. So:

  tune fixture  : data/sample_meeting     (3 speakers)
  test fixture  : data/heldout_meeting    (4 speakers, different voices, many
                                           very short turns) — never used to
                                           choose anything

Embedding is the expensive step and does not depend on the threshold, so each
fixture is embedded ONCE and the threshold sweep reruns only the clustering.

Selection rule: not the single best threshold, but the CENTRE of the plateau of
thresholds within 0.5 percentage points of the best. A value in the middle of a
flat region tolerates shifts in a new recording; a value at a sharp minimum is
usually fitted to noise.

USAGE
    python scripts/make_sample_meeting.py --name heldout_meeting   # once
    python eval/tune_diarization.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from app.pipeline.audio import decode  # noqa: E402
from app.pipeline.diarize import (DEFAULT_THRESHOLD, EcapaEmbedder, label,  # noqa: E402
                                  prepare)
from eval.cache import CachedTranscriber  # noqa: E402
from eval.der import compute_der, load_rttm  # noqa: E402

DATA = os.path.join(HERE, "data")
TUNE, TEST = "sample_meeting", "heldout_meeting"
LONG = "long_meeting"
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(15)]      # 0.30 … 1.00
PLATEAU_PP = 0.005


def load(name, transcriber, embedder):
    wav = os.path.join(DATA, f"{name}.wav")
    rttm = os.path.join(DATA, f"{name}_reference.rttm")
    for path in (wav, rttm):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}\nRun: python scripts/make_sample_meeting.py "
                             f"--name {name}")
    print(f"  {name}: transcribing + embedding…", flush=True)
    transcript = transcriber.transcribe(wav)
    samples, _ = decode(wav)
    prep = prepare(transcript, samples, embedder)
    ref = load_rttm(rttm)
    n_ref = len({s for _, _, s in ref})
    kinds = {k: prep.kind.count(k) for k in ("long", "short", "tiny")}
    print(f"    {len(prep.chunks)} chunks {kinds}, {n_ref} true speakers, "
          f"embedded in {prep.embed_seconds:.1f}s")
    return prep, ref, n_ref


def main() -> int:
    print("=" * 74)
    print("DIARIZATION THRESHOLD — tune on one meeting, report on another")
    print("=" * 74)
    transcriber, embedder = CachedTranscriber(), EcapaEmbedder()
    tune_prep, tune_ref, tune_n = load(TUNE, transcriber, embedder)
    test_prep, test_ref, test_n = load(TEST, transcriber, embedder)

    rows = []
    for th in THRESHOLDS:
        a = label(tune_prep, threshold=th)
        b = label(test_prep, threshold=th)
        rows.append((th, compute_der(tune_ref, a.hypothesis()), a.num_speakers,
                     compute_der(test_ref, b.hypothesis()), b.num_speakers))

    print(f"\n  {'threshold':>9} | {'TUNE DER':>9} {'spk':>4} | {'TEST DER':>9} {'spk':>4}")
    print("  " + "-" * 44)
    for th, da, na, db, nb in rows:
        flag = "  <- current default" if abs(th - DEFAULT_THRESHOLD) < 1e-9 else ""
        print(f"  {th:>9.2f} | {da.der:>8.1%} {na:>4} | {db.der:>8.1%} {nb:>4}{flag}")

    best = min(r[1].der for r in rows)
    plateau = [r for r in rows if r[1].der <= best + PLATEAU_PP]
    chosen = plateau[len(plateau) // 2]
    th, tune_der, tune_spk, test_der, test_spk = chosen
    oracle = min(rows, key=lambda r: r[3].der)

    known_tune = compute_der(tune_ref, label(tune_prep, num_speakers=tune_n).hypothesis())
    known_test = compute_der(test_ref, label(test_prep, num_speakers=test_n).hypothesis())

    print("\n" + "=" * 74)
    print(f"  plateau on TUNE: thresholds {plateau[0][0]:.2f}–{plateau[-1][0]:.2f} "
          f"(within {PLATEAU_PP:.1%} of best {best:.1%})")
    print(f"  CHOSEN threshold : {th:.2f}  (centre of plateau)")
    print(f"    tune  DER {tune_der.der:6.1%}  speakers {tune_spk}/{tune_n}")
    print(f"    TEST  DER {test_der.der:6.1%}  speakers {test_spk}/{test_n}   "
          f"<- the honest number")
    print(f"    test-set oracle: {oracle[3].der:.1%} at {oracle[0]:.2f} "
          f"(gap {test_der.der - oracle[3].der:+.1%} = cost of not peeking)")
    print(f"  known speaker count: tune DER {known_tune.der:.1%} | "
          f"test DER {known_test.der:.1%}")
    print(f"  test breakdown at chosen: {test_der}")

    # The 30-minute, 6-speaker meeting: the case that broke 0.65.
    if os.path.exists(os.path.join(DATA, f"{LONG}.wav")):
        long_prep, long_ref, long_n = load(LONG, transcriber, embedder)
        print(f"\n  {LONG} ({long_n} speakers), same sweep:")
        print(f"  {'threshold':>9} | {'SAMPLE':>7} | {'HELDOUT':>7} | {'LONG':>7} {'spk':>4}")
        for t_, da, na, db, nb in rows:
            r = label(long_prep, threshold=t_)
            dl = compute_der(long_ref, r.hypothesis())
            mark = "  <- default" if abs(t_ - DEFAULT_THRESHOLD) < 1e-9 else ""
            print(f"  {t_:>9.2f} | {da.der:>6.1%} | {db.der:>6.1%} | {dl.der:>6.1%} "
                  f"{r.num_speakers:>4}{mark}")
    if abs(th - DEFAULT_THRESHOLD) > 1e-9:
        print(f"\n  note: the sample-only plateau centre is {th:.2f}; the default is "
              f"{DEFAULT_THRESHOLD:.2f} because the 30-minute meeting showed merges cost "
              f"more than splits (see the comment in app/pipeline/diarize.py). Check "
              f"that the default sits inside the sample plateau above.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
