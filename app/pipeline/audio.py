"""
Audio validation and decoding — layers B, C and D of the Stage 1 flow.

We decode the audio OURSELVES rather than letting faster-whisper do it. Two
reasons, both learned the hard way:

  1. Error quality. A failure here is ours to catch and phrase for the user,
     instead of a library traceback surfacing in the UI.
  2. Version resilience. faster-whisper's internal decoder calls
     `av.open(..., metadata_errors=...)`, which PyAV >= 19 removed — it crashes
     outright on a current PyAV. Owning the decode makes us immune to that.

Validation is LAYERED and fail-honest. No single check is treated as proof the
input is good; each layer only rules out one class of known-bad. Anything
ambiguous proceeds with a soft warning rather than being discarded on our
judgment — the user's file may still be usable, and that call isn't ours to make.
"""
from __future__ import annotations

import os
import subprocess
from shutil import which

import numpy as np

from ..errors import AudioDecodeError, AudioValidationError

SAMPLE_RATE = 16_000

# Soft-warning thresholds (never hard rejections).
MIN_REASONABLE_DURATION_S = 1.0
NEAR_SILENT_RMS = 1e-4


# --------------------------------------------------------------- layer 1
def validate_path(path: str) -> None:
    """Cheap structural checks. Hard-fails only on the unambiguously broken."""
    if not os.path.exists(path):
        raise AudioValidationError(
            f"file not found: {path}",
            user_message="That file could not be found. Please upload it again.",
        )
    if os.path.isdir(path):
        raise AudioValidationError(
            f"path is a directory: {path}",
            user_message="That's a folder, not an audio file.",
        )
    size = os.path.getsize(path)
    if size == 0:
        raise AudioValidationError(
            f"file is empty: {path}",
            user_message="That file is empty (0 bytes). Please upload a valid "
                         "audio recording.",
        )


# --------------------------------------------------------------- layer 2
def _decode_pyav(path: str) -> np.ndarray:
    """Primary path. PyAV directly, WITHOUT the kwarg that breaks on PyAV>=19."""
    import av

    with av.open(path, mode="r") as container:
        if not container.streams.audio:
            raise RuntimeError("file contains no audio stream")
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=SAMPLE_RATE
        )
        chunks: list[np.ndarray] = []

        def emit(frame) -> None:
            out = resampler.resample(frame)
            if out is None:
                return
            if not isinstance(out, list):      # API varies across PyAV versions
                out = [out]
            for rf in out:
                chunks.append(rf.to_ndarray().reshape(-1))

        for frame in container.decode(stream):
            emit(frame)
        emit(None)                             # flush the resampler

    if not chunks:
        raise RuntimeError("no audio frames decoded")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _decode_ffmpeg(path: str) -> np.ndarray:
    """Fallback. Robust, but requires ffmpeg on PATH (we do not assume it)."""
    if not which("ffmpeg"):
        raise RuntimeError("ffmpeg not on PATH")
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-threads", "0", "-i", path,
         "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
         "-ar", str(SAMPLE_RATE), "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
    )
    if not proc.stdout:
        raise RuntimeError("ffmpeg produced no audio")
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


def _decode_library(path: str) -> np.ndarray:
    """Last resort. Broken on PyAV>=19; kept only as a final fallback."""
    from faster_whisper.audio import decode_audio

    return decode_audio(path, sampling_rate=SAMPLE_RATE)


DECODERS = (
    ("pyav", _decode_pyav),
    ("ffmpeg", _decode_ffmpeg),
    ("faster-whisper", _decode_library),
)


def decode(path: str) -> tuple[np.ndarray, str]:
    """
    Decode to mono float32 @16 kHz. Returns (samples, method_used).

    Tries each decoder in order; a decode failure in all of them is what
    "unsupported / unreadable / corrupt" actually means in practice, so that's
    where we raise. Succeeding here is also our proof the bytes are real audio —
    the decode IS the validation.
    """
    attempts: list[str] = []
    for name, fn in DECODERS:
        try:
            samples = fn(path)
            if samples.size == 0:
                raise RuntimeError("decoded to zero samples")
            return samples, name
        except Exception as exc:                       # noqa: BLE001 - try next
            attempts.append(f"{name}: {type(exc).__name__}: {exc}")

    raise AudioDecodeError(
        "all decoders failed -> " + " | ".join(attempts),
        user_message="That file could not be read as audio. It may be corrupted, "
                     "or in a format we don't support. Try WAV, MP3, M4A or FLAC.",
    )


# --------------------------------------------------------------- layer 3
def analyze(samples: np.ndarray) -> list[str]:
    """
    Post-decode sanity. Returns SOFT WARNINGS only — never raises.

    We flag what looks wrong and let the run proceed. Rejecting here would mean
    trusting our own heuristic over the user's actual file, and a near-silent or
    very short recording can still be legitimate.
    """
    warnings: list[str] = []
    duration = len(samples) / SAMPLE_RATE

    if duration < MIN_REASONABLE_DURATION_S:
        warnings.append(
            f"This recording is very short ({duration:.1f}s). "
            "Results may be empty or incomplete."
        )

    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    if rms < NEAR_SILENT_RMS:
        warnings.append(
            "This recording appears to be silent or almost silent. "
            "Little or no speech may be detected."
        )
    return warnings
