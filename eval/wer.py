"""
Word Error Rate.

    WER = (Substitutions + Deletions + Insertions) / words in the reference

Computed by Levenshtein alignment over word sequences. Text is normalised first
(casefold, strip punctuation) because we are measuring whether the WORDS are
right, not whether the model guessed our comma placement — scoring punctuation
as errors would make the number meaningless.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


def normalize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


@dataclass
class WERResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    def __str__(self) -> str:
        return (f"WER {self.wer:6.2%}  (S={self.substitutions} D={self.deletions} "
                f"I={self.insertions} / N={self.reference_words})")


def compute_wer(reference: str, hypothesis: str) -> WERResult:
    ref, hyp = normalize(reference), normalize(hypothesis)
    n, m = len(ref), len(hyp)

    # d[i][j] = (cost, S, D, I) for ref[:i] vs hyp[:j]
    d = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        d[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
                continue
            sub = d[i - 1][j - 1]
            dele = d[i - 1][j]
            ins = d[i][j - 1]
            best = min(sub[0], dele[0], ins[0]) + 1
            if sub[0] <= dele[0] and sub[0] <= ins[0]:
                d[i][j] = (best, sub[1] + 1, sub[2], sub[3])
            elif dele[0] <= ins[0]:
                d[i][j] = (best, dele[1], dele[2] + 1, dele[3])
            else:
                d[i][j] = (best, ins[1], ins[2], ins[3] + 1)

    cost, s, dl, ins = d[n][m]
    return WERResult(wer=cost / n if n else 0.0, substitutions=s, deletions=dl,
                     insertions=ins, reference_words=n)
