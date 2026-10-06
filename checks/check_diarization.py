#!/usr/bin/env python3
"""
Diarization check harness — runs with NO model.

The embedder is injectable, so these tests use a stand-in: each synthetic
"speaker" is a pure tone, and the stand-in embedding is a log band spectrum.
That isolates what we wrote — pause-splitting, the clustering maths, short-chunk
assignment, labelling — from the pretrained network we didn't. The real ECAPA
model is exercised by eval/tune_diarization.py against recordings with ground
truth.

USAGE
    python checks/check_diarization.py
"""
from __future__ import annotations

import itertools
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from app.pipeline.diarize import (agglomerative, label, prepare,  # noqa: E402
                                  split_on_pauses)
from app.schemas import Segment, Transcript, Word  # noqa: E402
from eval.der import compute_der  # noqa: E402

SR = 16_000
RESULTS: list[tuple[str, str, str]] = []


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


def _bands(wav, power=2):
    spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav)))) ** power
    freqs = np.fft.rfftfreq(len(wav), 1 / SR)
    edges = np.geomspace(100, 4000, 33)
    return np.array([spec[(freqs >= lo) & (freqs < hi)].sum()
                     for lo, hi in zip(edges[:-1], edges[1:])])


class BandEmbedder:
    """
    Stand-in speaker embedding: energy in 32 log-spaced bands. Different tones
    are near-orthogonal (cosine distance ~1.0); the same tone is ~0.0.
    """

    def __call__(self, wav):
        return _bands(wav)


class BlurryEmbedder:
    """
    A deliberately WORSE stand-in: centred log magnitude. Log compression lets
    spectral leakage dominate, so tones an octave apart land only ~0.62 apart.
    Kept because it demonstrates that a threshold only means something relative
    to one embedder's geometry - the first version of these tests used it, and
    two 'speakers' merged at the 0.70 threshold.
    """

    def __call__(self, wav):
        v = np.log1p(_bands(wav, power=1))
        return v - v.mean()


def synth_meeting(turns, merge_segments=()):
    """
    turns: [(speaker, freq_hz, [word, ...]), ...] — one pure tone per speaker.
    merge_segments: turn indices whose words are put into the PREVIOUS turn's
    Whisper segment — reproducing Whisper merging two speakers into one segment.
    Returns samples, transcript, reference turns.
    """
    word_dur, word_gap, turn_gap = 0.35, 0.08, 0.50
    t, audio, ref, segs = 0.20, [], [], []
    rng = np.random.default_rng(0)
    for ti, (spk, freq, words) in enumerate(turns):
        start = t
        ws = []
        for w in words:
            n = int(word_dur * SR)
            tone = 0.3 * np.sin(2 * np.pi * freq * np.arange(n) / SR)
            audio.append((int(t * SR), tone))
            ws.append(Word(w, t, t + word_dur, 0.9))
            t += word_dur + word_gap
        end = ws[-1].end
        ref.append((start, end, spk))
        if ti in merge_segments and segs:
            segs[-1]["words"] += ws
        else:
            segs.append({"words": ws})
        t = end + turn_gap
    samples = (rng.normal(0, 1e-4, int((t + 0.2) * SR))).astype(np.float64)
    for off, tone in audio:
        samples[off:off + len(tone)] += tone
    segments = [Segment(i, s["words"][0].start, s["words"][-1].end,
                        " ".join(w.word for w in s["words"]), -0.2, 0.0, 1.0, s["words"])
                for i, s in enumerate(segs)]
    tr = Transcript(text=" ".join(s.text for s in segments), segments=segments,
                    language="en", duration=t, model="synthetic", decode_method="n/a")
    return samples.astype(np.float32), tr, ref


def blobs(k, per, dim=192, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, dim))
    x = np.concatenate([c + spread * rng.normal(size=(per, dim)) for c in centres])
    return x, np.repeat(np.arange(k), per)


def same_partition(a, b) -> bool:
    groups = lambda lab: {frozenset(np.flatnonzero(lab == l)) for l in set(lab.tolist())}
    return groups(np.asarray(a)) == groups(np.asarray(b))


def brute_average_linkage(x, k):
    """Textbook average linkage, recomputing every cluster distance from scratch."""
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    d = 1 - xn @ xn.T
    clusters = [[i] for i in range(len(x))]
    while len(clusters) > k:
        best = None
        for i, j in itertools.combinations(range(len(clusters)), 2):
            dist = np.mean([d[a, b] for a in clusters[i] for b in clusters[j]])
            if best is None or dist < best[0]:
                best = (dist, i, j)
        _, i, j = best
        clusters[i] += clusters.pop(j)
    lab = np.empty(len(x), int)
    for c, members in enumerate(clusters):
        lab[members] = c
    return lab


def main() -> int:
    print("=" * 70)
    print("DIARIZATION CHECK HARNESS (no model)")
    print("=" * 70)

    print("\nCLUSTERING MATHS")

    @check("recovers 3 well-separated speakers with a threshold")
    def _():
        x, truth = blobs(3, 8)
        lab = agglomerative(x, threshold=0.7)
        assert len(set(lab.tolist())) == 3, f"found {len(set(lab.tolist()))}"
        assert same_partition(lab, truth), "wrong grouping"
        return "3 clusters, correct membership"

    @check("Lance–Williams update matches brute-force average linkage")
    def _():
        x = np.random.default_rng(7).normal(size=(14, 16))
        for k in (6, 4, 2):
            fast = agglomerative(x, threshold=None, n_clusters=k)
            slow = brute_average_linkage(x, k)
            assert same_partition(fast, slow), f"partitions differ at k={k}"
        return "identical partitions at k = 6, 4, 2"

    @check("known speaker count is honoured")
    def _():
        x, _ = blobs(3, 6)
        assert len(set(agglomerative(x, n_clusters=2).tolist())) == 2
        return "forced to 2"

    @check("degenerate inputs: empty, single, identical")
    def _():
        assert len(agglomerative(np.zeros((0, 4)))) == 0
        assert agglomerative(np.ones((1, 4))).tolist() == [0]
        assert len(set(agglomerative(np.ones((5, 4))).tolist())) == 1
        return "no crash"

    print("\nPAUSE SPLITTING")

    @check("splits a Whisper segment that contains two speakers")
    def _():
        _, tr, _ = synth_meeting([("A", 220, ["is", "that", "slow"]),
                                  ("B", 880, ["mostly", "yes"])], merge_segments={1})
        assert len(tr.segments) == 1, "fixture should merge both turns"
        chunks = split_on_pauses(tr)
        assert len(chunks) == 2, f"{len(chunks)} chunks"
        return "1 segment -> 2 chunks at the 0.5s pause"

    @check("does not split at ordinary inter-word gaps")
    def _():
        _, tr, _ = synth_meeting([("A", 220, ["one", "two", "three", "four"])])
        assert len(split_on_pauses(tr)) == 1
        return "0.08s gaps kept together"

    print("\nEND TO END (stand-in embedder)")

    meeting = [("Priya", 220, ["alright", "lets", "start"]),
               ("Arjun", 880, ["sessions", "live", "in", "redis"]),
               ("Priya", 220, ["is", "that", "redis"]),
               ("Arjun", 880, ["mostly", "redis", "yes"]),
               ("Sam", 1760, ["postgres", "is", "slower", "per", "read"]),
               ("Priya", 220, ["great"]),
               ("Sam", 1760, ["yes", "ill", "take", "it"])]

    @check("three speakers, including a merged segment: DER ~0")
    def _():
        samples, tr, ref = synth_meeting(meeting, merge_segments={3})
        res = label(prepare(tr, samples, BandEmbedder()))
        der = compute_der(ref, res.hypothesis(), collar=0)
        assert res.num_speakers == 3, f"found {res.num_speakers} speakers"
        assert der.der < 0.02, f"{der}"
        return f"{res.num_speakers} speakers, {der}"

    @check("speakers are numbered by first appearance")
    def _():
        samples, tr, _ = synth_meeting(meeting)
        res = label(prepare(tr, samples, BandEmbedder()))
        assert res.turns[0].speaker == "Speaker 1"
        assert res.turns[1].speaker == "Speaker 2"
        return " -> ".join(t.speaker for t in res.turns[:4])

    @check("a one-word turn ('great') is assigned via centroid, correctly")
    def _():
        samples, tr, _ = synth_meeting(meeting)
        prep = prepare(tr, samples, BandEmbedder())
        res = label(prep)
        great = next(w for s in tr.segments for w in s.words if w.word == "great")
        first = tr.segments[0].words[0]
        assert "short" in prep.kind, "fixture should contain a short chunk"
        assert great.speaker == first.speaker, f"{great.speaker} vs {first.speaker}"
        return f"'great' -> {great.speaker} (same as Priya's first word)"

    @check("every word receives a speaker label")
    def _():
        samples, tr, _ = synth_meeting(meeting)
        label(prepare(tr, samples, BandEmbedder()))
        missing = [w.word for s in tr.segments for w in s.words if not w.speaker]
        assert not missing, f"unlabelled: {missing}"
        return "all words labelled"

    @check("the threshold is embedder-relative (why we tune on the real model)")
    def _():
        # Same audio, worse embedding geometry. Arjun and Sam (an octave apart)
        # sit ~0.62 apart, so 0.70 merges them and 0.50 separates them. Nothing
        # about the CLUSTERING changed - only what the distances mean.
        samples, tr, _ = synth_meeting(meeting, merge_segments={3})
        prep = prepare(tr, samples, BlurryEmbedder())
        at70 = label(prep, threshold=0.70).num_speakers
        at50 = label(prep, threshold=0.50).num_speakers
        assert at70 == 2, f"expected the merge at 0.70, got {at70}"
        assert at50 == 3, f"expected separation at 0.50, got {at50}"
        return f"blurry embedder: 0.70 -> {at70} speakers, 0.50 -> {at50}"

    @check("known count mode on 3-speaker audio gives exactly 2")
    def _():
        samples, tr, _ = synth_meeting(meeting)
        res = label(prepare(tr, samples, BandEmbedder()), num_speakers=2)
        assert res.num_speakers == 2
        return res.method

    passed = sum(1 for s, _, _ in RESULTS if s == "PASS")
    failed = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    print("\n" + "=" * 70)
    print(f"RESULT: {passed} passed, {failed} failed")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
