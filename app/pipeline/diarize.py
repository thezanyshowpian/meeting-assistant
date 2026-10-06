"""
Speaker diarization — who spoke when.

Three steps, each deliberately simple enough to derive by hand:

  1. CHUNK   Split the transcript into single-speaker pieces at PAUSES, using the
             word timestamps Stage 1 kept. Whisper segments are not enough on
             their own: on our own fixture Whisper put Priya's question and
             Arjun's answer into ONE segment, so segment-level labels would be
             wrong by construction. Turn changes almost always sit in a pause.

  2. EMBED   Map each chunk of audio to a 192-dimensional ECAPA-TDNN speaker
             embedding (SpeechBrain, trained on VoxCeleb so the same voice lands
             close together and different voices far apart). Compare embeddings
             by COSINE distance: 1 - (a·b)/(|a||b|).

  3. CLUSTER Agglomerative clustering, average linkage. Start with every chunk
             as its own cluster; repeatedly merge the two closest; stop when the
             closest pair is further apart than a threshold. The number of
             speakers is DISCOVERED, not supplied — unless the user knows it, in
             which case we stop at that count instead.

Very short chunks ("Yes.", "Great.") give noisy embeddings, so they don't vote
in clustering; they are assigned afterwards to the nearest speaker centroid.

Output labels are anonymous ("Speaker 1", "Speaker 2"). Turning a label into a
NAME is a separate, evidence-based step (naming.py) — diarization never guesses.

The embedder is injectable, so all of this is testable without the model.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from ..schemas import Transcript
from .audio import SAMPLE_RATE, decode

PAUSE_SPLIT_S = 0.30        # a gap longer than this between words starts a new chunk
MIN_EMBED_S = 0.60          # chunks at least this long vote in clustering
MIN_ASSIGN_S = 0.25         # shorter-but-not-tiny chunks are assigned to a centroid
CONTEXT_PAD_S = 0.10        # audio context either side (inside the pause, so safe)

# Cosine distance, average linkage. Chosen on the tuning fixture and REPORTED on a
# held-out fixture with different voices and a different speaker count — see
# eval/tune_diarization.py. Do not change it by eyeballing one recording.
#
# Measured (ECAPA-TDNN): DER is flat at 8.0% on the tuning meeting for
# thresholds 0.55–0.75; 0.65 is the centre of that plateau. On the HELD-OUT
# 4-speaker meeting it gives 4/4 speakers, DER 4.7%, ZERO speaker confusion, and
# matches the held-out oracle exactly (gap +0.0%). The held-out set's own best
# (0.50) sits at the edge of the tuning plateau and invents a 5th speaker there —
# which is why we take the plateau centre, not either set's minimum.
DEFAULT_THRESHOLD = 0.65

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


# ------------------------------------------------------------------ data
@dataclass
class Chunk:
    start: float
    end: float
    words: list[tuple[int, int]] = field(default_factory=list)   # (segment idx, word idx)
    whole_segment: int | None = None    # used only when a segment has no word timings

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SpeakerTurn:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class Prepared:
    """Everything expensive, computed once: chunks and their embeddings."""
    transcript: Transcript
    chunks: list[Chunk]
    embeddings: list[np.ndarray | None]
    kind: list[str]                     # "long" | "short" | "tiny"
    embed_seconds: float = 0.0


@dataclass
class DiarizationResult:
    turns: list[SpeakerTurn]
    labelled_chunks: list[tuple[float, float, str]]
    num_speakers: int
    method: str
    processing_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def hypothesis(self) -> list[tuple[float, float, str]]:
        """(start, end, speaker) spans covering only where words were spoken — for DER."""
        return self.labelled_chunks

    def to_dict(self) -> dict:
        return {
            "num_speakers": self.num_speakers,
            "method": self.method,
            "processing_seconds": round(self.processing_seconds, 2),
            "turns": [{"speaker": t.speaker, "start": round(t.start, 2),
                       "end": round(t.end, 2), "text": t.text} for t in self.turns],
            "warnings": self.warnings,
        }


# ------------------------------------------------------------------ step 1
def split_on_pauses(transcript: Transcript, gap: float = PAUSE_SPLIT_S) -> list[Chunk]:
    chunks: list[Chunk] = []
    for si, seg in enumerate(transcript.segments):
        if not seg.words:
            chunks.append(Chunk(seg.start, seg.end, whole_segment=si))
            continue
        current = [(si, 0)]
        for wi in range(1, len(seg.words)):
            if seg.words[wi].start - seg.words[wi - 1].end > gap:
                chunks.append(_chunk(transcript, current))
                current = []
            current.append((si, wi))
        chunks.append(_chunk(transcript, current))
    return chunks


def _chunk(transcript: Transcript, refs: list[tuple[int, int]]) -> Chunk:
    first = transcript.segments[refs[0][0]].words[refs[0][1]]
    last = transcript.segments[refs[-1][0]].words[refs[-1][1]]
    return Chunk(first.start, last.end, list(refs))


# ------------------------------------------------------------------ step 2
class EcapaEmbedder:
    """SpeechBrain ECAPA-TDNN. Ungated, CPU. Loaded lazily on first use."""

    def __init__(self, source: str = ECAPA_SOURCE, savedir: str | None = None):
        self.source = source
        self.savedir = savedir or os.path.expanduser(
            "~/.cache/meeting-assistant/spkrec-ecapa-voxceleb")
        self._model = None

    def _load(self):
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier
            self._model = EncoderClassifier.from_hparams(
                source=self.source, savedir=self.savedir, run_opts={"device": "cpu"})
        return self._model

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            emb = self._load().encode_batch(torch.from_numpy(wav).float().unsqueeze(0))
        return emb.squeeze().cpu().numpy()


def speechbrain_available() -> bool:
    try:
        import speechbrain  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:                                              # noqa: BLE001
        return False


def prepare(transcript: Transcript, samples: np.ndarray, embedder) -> Prepared:
    started = time.time()
    chunks = split_on_pauses(transcript)
    pad = int(CONTEXT_PAD_S * SAMPLE_RATE)
    embeddings: list[np.ndarray | None] = []
    kind: list[str] = []
    for c in chunks:
        if c.duration >= MIN_EMBED_S:
            kind.append("long")
        elif c.duration >= MIN_ASSIGN_S:
            kind.append("short")
        else:
            kind.append("tiny")
            embeddings.append(None)
            continue
        lo = max(0, int(c.start * SAMPLE_RATE) - pad)
        hi = min(len(samples), int(c.end * SAMPLE_RATE) + pad)
        embeddings.append(np.asarray(embedder(samples[lo:hi]), dtype=np.float64))
    return Prepared(transcript, chunks, embeddings, kind, time.time() - started)


# ------------------------------------------------------------------ step 3
def _normalise(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def agglomerative(emb: np.ndarray, threshold: float | None = DEFAULT_THRESHOLD,
                  n_clusters: int | None = None) -> np.ndarray:
    """
    Average-linkage agglomerative clustering on cosine distance.

    The distance between two clusters is the MEAN pairwise distance between
    their members. After merging clusters a and b, the distance from any other
    cluster k to the merged one follows from the Lance–Williams update:

        d(k, a∪b) = ( |a|·d(k,a) + |b|·d(k,b) ) / ( |a| + |b| )

    so the full pairwise matrix never needs recomputing. Stops when the closest
    pair exceeds `threshold` — or, if `n_clusters` is given, at that count.
    """
    n = len(emb)
    if n <= 1:
        return np.zeros(n, dtype=int)
    x = _normalise(np.asarray(emb, dtype=np.float64))
    d = 1.0 - x @ x.T
    np.fill_diagonal(d, np.inf)
    size = np.ones(n)
    alive = np.ones(n, dtype=bool)
    members = {i: [i] for i in range(n)}
    k = n
    while k > 1:
        if n_clusters is not None and k <= n_clusters:
            break
        masked = np.where(alive[:, None] & alive[None, :], d, np.inf)
        a, b = np.unravel_index(np.argmin(masked), masked.shape)
        if n_clusters is None and masked[a, b] > threshold:
            break
        merged = (size[a] * d[a] + size[b] * d[b]) / (size[a] + size[b])
        d[a, :] = merged
        d[:, a] = merged
        d[a, a] = np.inf
        d[b, :] = np.inf
        d[:, b] = np.inf
        size[a] += size[b]
        alive[b] = False
        members[a] += members.pop(b)
        k -= 1
    labels = np.empty(n, dtype=int)
    for lab, idx in enumerate(members.values()):
        labels[idx] = lab
    return labels


def label(prep: Prepared, threshold: float | None = DEFAULT_THRESHOLD,
          num_speakers: int | None = None) -> DiarizationResult:
    started = time.time()
    n = len(prep.chunks)
    warnings: list[str] = []
    long_idx = [i for i, k in enumerate(prep.kind) if k == "long"]
    short_idx = [i for i, k in enumerate(prep.kind) if k == "short"]
    if not long_idx and short_idx:
        long_idx, short_idx = short_idx, []
        warnings.append("All speech chunks were short; speaker labels are less reliable.")

    labels = [-1] * n
    if long_idx:
        e = _normalise(np.stack([prep.embeddings[i] for i in long_idx]))
        lab = agglomerative(e, threshold=threshold, n_clusters=num_speakers)
        for i, l in zip(long_idx, lab):
            labels[i] = int(l)
        centroids = {int(l): _normalise(e[lab == l].mean(axis=0)) for l in set(lab.tolist())}
        for i in short_idx:
            v = _normalise(prep.embeddings[i])
            labels[i] = min(centroids, key=lambda c: 1.0 - float(v @ centroids[c]))

    # Tiny chunks inherit from the nearest labelled chunk, preferring the previous.
    for i in range(n):
        if labels[i] < 0:
            prev = next((labels[j] for j in range(i - 1, -1, -1) if labels[j] >= 0), -1)
            nxt = next((labels[j] for j in range(i + 1, n) if labels[j] >= 0), -1)
            labels[i] = prev if prev >= 0 else (nxt if nxt >= 0 else 0)

    # Number speakers by first appearance, so "Speaker 1" is whoever spoke first.
    order: dict[int, str] = {}
    for lab_ in labels:
        if lab_ not in order:
            order[lab_] = f"Speaker {len(order) + 1}"

    segments = prep.transcript.segments
    labelled: list[tuple[float, float, str]] = []
    turns: list[SpeakerTurn] = []
    for chunk, lab_ in zip(prep.chunks, labels):
        name = order[lab_]
        if chunk.whole_segment is not None:
            text = segments[chunk.whole_segment].text.strip()
        else:
            words = [segments[s].words[w] for s, w in chunk.words]
            for w in words:
                w.speaker = name
            text = " ".join(w.word.strip() for w in words)
        labelled.append((chunk.start, chunk.end, name))
        if turns and turns[-1].speaker == name:
            turns[-1].end = chunk.end
            turns[-1].text += " " + text
        else:
            turns.append(SpeakerTurn(name, chunk.start, chunk.end, text))

    mode = (f"known count={num_speakers}" if num_speakers
            else f"threshold={threshold:.2f}")
    return DiarizationResult(
        turns=turns, labelled_chunks=labelled, num_speakers=len(order),
        method=f"ECAPA-TDNN + average-linkage agglomerative ({mode})",
        processing_seconds=prep.embed_seconds + (time.time() - started),
        warnings=warnings,
    )


_DEFAULT_EMBEDDER: EcapaEmbedder | None = None


def default_embedder() -> EcapaEmbedder:
    """One model per process — loading it per request would cost seconds each time."""
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = EcapaEmbedder()
    return _DEFAULT_EMBEDDER


def diarize(transcript: Transcript, audio_path: str, *, embedder=None,
            threshold: float | None = DEFAULT_THRESHOLD,
            num_speakers: int | None = None) -> DiarizationResult:
    samples, _ = decode(audio_path)
    prep = prepare(transcript, samples, embedder or default_embedder())
    return label(prep, threshold=threshold, num_speakers=num_speakers)
