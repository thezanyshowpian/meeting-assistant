"""
Diarization Error Rate.

    DER = (missed speech + false-alarm speech + speaker confusion) / total reference speech

All three are measured in TIME, not words. Computed frame by frame (10 ms):

  - missed      : reference says someone is speaking, hypothesis says nobody
  - false alarm : hypothesis says someone is speaking, reference says nobody
  - confusion   : both say someone is speaking, but the wrong person

A diarizer's labels are arbitrary ("cluster 0", "cluster 1") — they don't know
the reference calls someone "Arjun". So before scoring, hypothesis clusters are
mapped one-to-one onto reference speakers so as to MAXIMISE agreement, and only
then is confusion counted. Without this, a perfect diarizer that happened to
number its speakers differently would score 100% confusion.

Collar: by convention, a short window (±0.25 s) around each reference boundary
is not scored, because the exact instant a turn starts is ambiguous even to
human annotators.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

Turn = tuple[float, float, str]   # (start, end, speaker)

STEP = 0.01


def load_rttm(path: str) -> list[Turn]:
    turns: list[Turn] = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start, dur = float(parts[3]), float(parts[4])
                turns.append((start, start + dur, parts[7]))
    return turns


def _frames(turns: list[Turn], n: int, index: dict[str, int]) -> list[int]:
    """Per-frame speaker index, -1 for silence. Overlaps: later turn wins."""
    lab = [-1] * n
    for start, end, spk in turns:
        for f in range(max(0, int(start / STEP)), min(n, int(math.ceil(end / STEP)))):
            lab[f] = index[spk]
    return lab


def _best_mapping(overlap: list[list[int]]) -> dict[int, int]:
    """hyp index -> ref index, one-to-one, maximising total overlap."""
    n_ref, n_hyp = len(overlap), len(overlap[0]) if overlap else 0
    if n_ref == 0 or n_hyp == 0:
        return {}
    best, best_score = {}, -1
    if max(n_ref, n_hyp) <= 7:                       # exact: tiny search space
        refs = list(range(n_ref))
        for perm in itertools.permutations(range(n_hyp), min(n_ref, n_hyp)):
            score = sum(overlap[r][h] for r, h in zip(refs, perm))
            if score > best_score:
                best_score = score
                best = {h: r for r, h in zip(refs, perm)}
        return best
    # greedy fallback for many speakers
    pairs = sorted(((overlap[r][h], r, h) for r in range(n_ref) for h in range(n_hyp)),
                   reverse=True)
    used_r, used_h = set(), set()
    for _, r, h in pairs:
        if r not in used_r and h not in used_h:
            best[h] = r
            used_r.add(r)
            used_h.add(h)
    return best


@dataclass
class DERResult:
    der: float
    missed: float
    false_alarm: float
    confusion: float
    total: float
    mapping: dict[str, str]

    def __str__(self) -> str:
        return (f"DER {self.der:6.2%}  (miss {self.missed:.1f}s, false alarm "
                f"{self.false_alarm:.1f}s, confusion {self.confusion:.1f}s "
                f"/ {self.total:.1f}s speech)")


def compute_der(reference: list[Turn], hypothesis: list[Turn],
                collar: float = 0.25) -> DERResult:
    end = max([e for _, e, _ in reference + hypothesis] or [0.0])
    n = int(math.ceil(end / STEP)) + 1

    ref_spk = sorted({s for _, _, s in reference})
    hyp_spk = sorted({s for _, _, s in hypothesis})
    ref = _frames(reference, n, {s: i for i, s in enumerate(ref_spk)})
    hyp = _frames(hypothesis, n, {s: i for i, s in enumerate(hyp_spk)})

    scored = [True] * n
    if collar > 0:
        c = int(collar / STEP)
        for start, end_, _ in reference:
            for edge in (start, end_):
                e = int(edge / STEP)
                for f in range(max(0, e - c), min(n, e + c + 1)):
                    scored[f] = False

    overlap = [[0] * len(hyp_spk) for _ in ref_spk]
    for f in range(n):
        if scored[f] and ref[f] >= 0 and hyp[f] >= 0:
            overlap[ref[f]][hyp[f]] += 1
    mapping = _best_mapping(overlap)

    miss = fa = conf = total = 0
    for f in range(n):
        if not scored[f]:
            continue
        r, h = ref[f], hyp[f]
        if r >= 0:
            total += 1
            if h < 0:
                miss += 1
            elif mapping.get(h) != r:
                conf += 1
        elif h >= 0:
            fa += 1

    return DERResult(
        der=(miss + fa + conf) / total if total else 0.0,
        missed=miss * STEP, false_alarm=fa * STEP, confusion=conf * STEP,
        total=total * STEP,
        mapping={hyp_spk[h]: ref_spk[r] for h, r in mapping.items()},
    )
