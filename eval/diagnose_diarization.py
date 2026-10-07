#!/usr/bin/env python3
"""
Why did diarization fail on a recording? Three questions, answered separately:

  1. SEGMENTATION — are the chunks we embed single-speaker? A chunk that spans a
     speaker change produces a blended voice embedding that sits between two
     speakers, and no threshold can cluster it correctly.
  2. SEPARABILITY — on PURE chunks, how far apart are the true speakers in
     ECAPA space, compared with how spread out each speaker is? If two voices
     overlap here, the embedder (not our clustering) is the limit.
  3. CLUSTERING — DER and speaker count across thresholds, and with the true
     speaker count given.

Written after the 30-minute meeting scored DER 39.7% with 5 of 6 speakers, on a
threshold tuned on 1-minute meetings. It diagnoses; it does not choose.

USAGE
    python eval/diagnose_diarization.py --name long_meeting
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from app.pipeline.audio import decode  # noqa: E402
from app.pipeline.diarize import DEFAULT_THRESHOLD, EcapaEmbedder, label, prepare  # noqa: E402
from eval.cache import CachedTranscriber  # noqa: E402
from eval.der import compute_der, load_rttm  # noqa: E402

DATA = os.path.join(HERE, "data")
PURE = 0.9          # a chunk is "pure" if >= 90% of its speech is one true speaker


def ref_share(start, end, ref):
    """Seconds of each reference speaker inside [start, end]."""
    out: dict[str, float] = {}
    for s, e, spk in ref:
        ov = min(end, e) - max(start, s)
        if ov > 0:
            out[spk] = out.get(spk, 0.0) + ov
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="long_meeting")
    args = ap.parse_args()
    wav = os.path.join(DATA, f"{args.name}.wav")
    ref = load_rttm(os.path.join(DATA, f"{args.name}_reference.rttm"))
    speakers = sorted({s for _, _, s in ref})

    print("=" * 74)
    print(f"DIARIZATION DIAGNOSIS — {args.name} ({len(speakers)} true speakers)")
    print("=" * 74)
    transcript = CachedTranscriber().transcribe(wav)
    samples, _ = decode(wav)
    prep = prepare(transcript, samples, EcapaEmbedder())
    print(f"  {len(prep.chunks)} chunks, embedded in {prep.embed_seconds:.0f}s")

    # ---- 1. segmentation
    mixed, mixed_s, total_s, owner = 0, 0.0, 0.0, []
    for c, k in zip(prep.chunks, prep.kind):
        share = ref_share(c.start, c.end, ref)
        tot = sum(share.values())
        top = max(share, key=share.get) if share else None
        owner.append(top if tot and share[top] / tot >= PURE else None)
        if k == "tiny" or not tot:
            continue
        total_s += c.duration
        if share[top] / tot < PURE:
            mixed += 1
            mixed_s += c.duration
    print("\n1. SEGMENTATION (are embedded chunks single-speaker?)")
    print(f"   mixed chunks: {mixed}, covering {mixed_s:.0f}s of {total_s:.0f}s "
          f"({mixed_s / max(total_s, 1):.0%} of embedded speech)")
    longest = sorted(prep.chunks, key=lambda c: -c.duration)[:3]
    print("   longest chunks: " + ", ".join(f"{c.duration:.0f}s" for c in longest))

    # ---- 2. separability on pure chunks
    by_spk: dict[str, list[np.ndarray]] = {s: [] for s in speakers}
    for e, k, o in zip(prep.embeddings, prep.kind, owner):
        if k == "long" and o is not None and e is not None:
            by_spk[o].append(e / np.linalg.norm(e))
    cent = {s: np.mean(v, axis=0) / np.linalg.norm(np.mean(v, axis=0))
            for s, v in by_spk.items() if v}
    print("\n2. SEPARABILITY (pure chunks only; cosine distance)")
    print("   within-speaker spread (mean distance to own centroid):")
    for s, v in by_spk.items():
        if v:
            d = [1 - float(x @ cent[s]) for x in v]
            print(f"     {s:<8} n={len(v):3d}  mean {np.mean(d):.2f}  p90 {np.percentile(d, 90):.2f}")
    print("   between-speaker centroid distance:")
    names = list(cent)
    print("     " + " " * 8 + "".join(f"{n[:7]:>8}" for n in names))
    for a in names:
        print(f"     {a:<8}" + "".join(
            f"{1 - float(cent[a] @ cent[b]):8.2f}" if a != b else f"{'—':>8}" for b in names))

    # ---- 3. clustering
    print("\n3. CLUSTERING")
    print(f"   {'threshold':>9} {'DER':>7} {'speakers':>9}")
    for th in [round(0.40 + 0.05 * i, 2) for i in range(10)]:
        r = label(prep, threshold=th)
        d = compute_der(ref, r.hypothesis())
        mark = "  <- default" if abs(th - DEFAULT_THRESHOLD) < 1e-9 else ""
        print(f"   {th:>9.2f} {d.der:>7.1%} {r.num_speakers:>9}{mark}")
    r = label(prep, num_speakers=len(speakers))
    d = compute_der(ref, r.hypothesis())
    print(f"   known count ({len(speakers)}): {d}")
    print(f"   mapping: {d.mapping}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
