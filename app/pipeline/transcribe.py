"""
Stage 1 — speech to text.

Flow (see docs/DESIGN_DECISIONS.md):
    validate -> decode (ours) -> sanity warnings -> Whisper -> rich Transcript
    -> post-check (empty transcript is HANDLED, never a crash)

Stage 1 does transcription and nothing else. No domain correction (Stage 2), no
speaker attribution, no summarisation. Keeping it narrow is what makes it
independently testable and keeps the "two distinct LLM stages" requirement clean
— Stage 1 isn't an LLM at all.
"""
from __future__ import annotations

import time

from ..errors import TranscriptionError
from ..schemas import Segment, Transcript, Word
from .audio import SAMPLE_RATE, analyze, decode, validate_path

DEFAULT_MODEL = "large-v3-turbo"


class Stage1Transcriber:
    """
    Wraps faster-whisper. The model is loaded lazily on first use so that
    validation/decoding can be exercised (and tested) without pulling 1.5 GB of
    weights into memory.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "en",      # forced: PS is English-only; skips lang-detect
        beam_size: int = 5,        # accuracy over speed — accuracy is what's scored
        vad_filter: bool = True,   # skip silence; Whisper can hallucinate in dead air
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise TranscriptionError(
                    f"faster-whisper not installed: {exc}",
                    user_message="The speech-to-text engine isn't installed. "
                                 "Run: pip install -r requirements.txt",
                    cause=exc,
                ) from exc
            try:
                self._model = WhisperModel(
                    self.model_size, device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as exc:                       # noqa: BLE001
                raise TranscriptionError(
                    f"could not load model {self.model_size}: {exc}",
                    user_message="The speech-to-text model could not be loaded. "
                                 "Check your connection for the first-time "
                                 "model download, then try again.",
                    cause=exc,
                ) from exc
        return self._model

    # ------------------------------------------------------------------ main
    def transcribe(self, path: str) -> Transcript:
        validate_path(path)                      # L1 — hard fail
        samples, decode_method = decode(path)    # L2 — hard fail; decode == validation
        warnings = analyze(samples)              # L3 — soft warnings only
        duration = len(samples) / SAMPLE_RATE

        model = self._ensure_model()
        started = time.time()
        try:
            segment_iter, info = model.transcribe(
                samples,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                word_timestamps=True,
            )
            raw_segments = list(segment_iter)     # generator — force the work
        except Exception as exc:                  # noqa: BLE001
            raise TranscriptionError(
                f"transcription failed: {exc}",
                user_message="Something went wrong while transcribing this "
                             "recording. Please try again.",
                cause=exc,
            ) from exc
        elapsed = time.time() - started

        segments = [
            Segment(
                id=i,
                start=s.start,
                end=s.end,
                text=s.text.strip(),
                avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,
                compression_ratio=s.compression_ratio,
                words=[
                    Word(word=w.word, start=w.start, end=w.end,
                         probability=w.probability)
                    for w in (s.words or [])
                ],
            )
            for i, s in enumerate(raw_segments)
        ]

        text = " ".join(s.text for s in segments).strip()

        transcript = Transcript(
            text=text,
            segments=segments,
            language=getattr(info, "language", self.language),
            duration=duration,
            model=self.model_size,
            decode_method=decode_method,
            warnings=warnings,
            processing_seconds=elapsed,
        )

        # Post-check: an empty result is a legitimate outcome (silence, music,
        # non-speech), not an exception. Say so plainly and carry on.
        if transcript.is_empty:
            transcript.warnings.append(
                "No speech was detected in this recording."
            )
        if transcript.low_confidence_segments:
            transcript.warnings.append(
                f"{len(transcript.low_confidence_segments)} passage(s) were "
                "transcribed with low confidence and may contain errors."
            )
        return transcript


def transcribe_file(path: str, **kwargs) -> Transcript:
    """Convenience one-shot. Prefer reusing a Stage1Transcriber to cache the model."""
    return Stage1Transcriber(**kwargs).transcribe(path)
