#!/usr/bin/env python3
"""
Stage 1 check harness.

Verifies the BEHAVIOURS we specified, not just that the happy path runs:
  - hard failures raise the right error type with a user-facing message
  - soft-warning cases PROCEED (we never discard a file on our own judgment)
  - an empty transcript is handled gracefully, not crashed on
  - the rich output contract (word timestamps + confidence) is actually populated
  - the output is JSON-serialisable (needed for the machine-readable deliverable)

Fixtures are generated here (silence, too-short, corrupt, non-audio), so the only
external asset needed is one real speech file.

USAGE
    python checks/check_stage1.py --audio jfk.flac      # full battery
    python checks/check_stage1.py --no-model            # skip model-dependent cases
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
import traceback
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from app.errors import AudioDecodeError, AudioValidationError  # noqa: E402
from app.pipeline.audio import analyze, decode, validate_path  # noqa: E402
from app.schemas import Segment, Transcript, Word  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []   # (status, name, detail)


def record(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    mark = {"PASS": "  PASS", "FAIL": "  FAIL", "SKIP": "  SKIP"}[status]
    print(f"{mark}  {name}" + (f"  — {detail}" if detail else ""), flush=True)


def check(name: str):
    """Decorator: run a check, catching anything it throws as a FAIL."""
    def wrap(fn):
        try:
            detail = fn() or ""
            record("PASS", name, detail)
        except AssertionError as exc:
            record("FAIL", name, str(exc))
        except Exception as exc:                            # noqa: BLE001
            record("FAIL", name, f"unexpected {type(exc).__name__}: {exc}")
            traceback.print_exc()
        return fn
    return wrap


# ------------------------------------------------------------------ fixtures
def write_wav(path: str, samples: np.ndarray, rate: int = 16000) -> str:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def build_fixtures(tmp: str) -> dict[str, str]:
    f = {}

    f["empty"] = os.path.join(tmp, "empty.wav")
    open(f["empty"], "wb").close()

    f["missing"] = os.path.join(tmp, "does_not_exist.wav")

    f["dir"] = os.path.join(tmp, "a_folder")
    os.makedirs(f["dir"], exist_ok=True)

    f["text"] = os.path.join(tmp, "not_audio.wav")
    with open(f["text"], "w") as fh:
        fh.write("This is plainly a text file wearing a .wav extension.\n" * 20)

    f["corrupt"] = os.path.join(tmp, "corrupt.wav")
    with open(f["corrupt"], "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 9999) + b"WAVEfmt ")
        fh.write(os.urandom(4096))                 # garbage where audio should be

    f["silence"] = write_wav(os.path.join(tmp, "silence.wav"),
                             np.zeros(16000 * 3, dtype=np.float32))

    f["tiny"] = write_wav(os.path.join(tmp, "tiny.wav"),
                          (np.random.randn(int(16000 * 0.2)) * 0.05).astype(np.float32))

    f["tone"] = write_wav(os.path.join(tmp, "tone.wav"),
                          (0.3 * np.sin(2 * np.pi * 440
                                        * np.arange(16000 * 2) / 16000)).astype(np.float32))
    return f


# ------------------------------------------------------- group A (no model)
def group_a(fx: dict[str, str]) -> None:
    print("\nGROUP A — validation, decoding, schema (no model required)")

    @check("missing file raises AudioValidationError")
    def _():
        try:
            validate_path(fx["missing"])
        except AudioValidationError as e:
            assert e.user_message, "no user_message set"
            return f'user_message: "{e.user_message}"'
        raise AssertionError("did not raise")

    @check("empty file (0 bytes) raises AudioValidationError")
    def _():
        try:
            validate_path(fx["empty"])
        except AudioValidationError as e:
            return f'user_message: "{e.user_message}"'
        raise AssertionError("did not raise")

    @check("directory raises AudioValidationError")
    def _():
        try:
            validate_path(fx["dir"])
        except AudioValidationError:
            return ""
        raise AssertionError("did not raise")

    @check("valid file passes validation")
    def _():
        validate_path(fx["tone"])        # must not raise
        return ""

    @check("text-file-as-.wav raises AudioDecodeError")
    def _():
        try:
            decode(fx["text"])
        except AudioDecodeError as e:
            assert e.user_message
            return f'user_message: "{e.user_message[:60]}…"'
        raise AssertionError("did not raise")

    @check("corrupt audio raises AudioDecodeError")
    def _():
        try:
            decode(fx["corrupt"])
        except AudioDecodeError:
            return ""
        raise AssertionError("did not raise")

    @check("valid audio decodes to float32 @16kHz mono")
    def _():
        samples, method = decode(fx["tone"])
        assert samples.dtype == np.float32, f"dtype {samples.dtype}"
        assert samples.ndim == 1, "not mono"
        dur = len(samples) / 16000
        assert 1.9 < dur < 2.1, f"duration {dur:.2f}s, expected ~2.0"
        return f"via {method}, {dur:.2f}s"

    @check("silence produces a SOFT WARNING and does not raise")
    def _():
        samples, _ = decode(fx["silence"])
        warns = analyze(samples)
        assert any("silent" in w.lower() for w in warns), f"warnings were {warns}"
        return f"{len(warns)} warning(s)"

    @check("too-short clip produces a SOFT WARNING and does not raise")
    def _():
        samples, _ = decode(fx["tiny"])
        warns = analyze(samples)
        assert any("short" in w.lower() for w in warns), f"warnings were {warns}"
        return f"{len(warns)} warning(s)"

    @check("normal audio produces no spurious warnings")
    def _():
        samples, _ = decode(fx["tone"])
        warns = analyze(samples)
        assert warns == [], f"unexpected warnings: {warns}"
        return ""

    @check("Transcript contract: confidence flagging + JSON round-trip")
    def _():
        import json
        t = Transcript(
            text="hello world",
            segments=[
                Segment(0, 0.0, 1.0, "hello world", avg_logprob=-0.1,
                        no_speech_prob=0.0, compression_ratio=1.0,
                        words=[Word("hello", 0.0, 0.5, 0.9),
                               Word("world", 0.5, 1.0, 0.8)]),
                Segment(1, 1.0, 2.0, "mumble", avg_logprob=-2.5,
                        no_speech_prob=0.1, compression_ratio=1.0, words=[]),
            ],
            language="en", duration=2.0, model="test",
            decode_method="pyav", processing_seconds=1.0,
        )
        assert not t.is_empty
        assert [s.id for s in t.low_confidence_segments] == [1], "low-conf flagging wrong"
        assert abs(t.realtime_factor - 0.5) < 1e-9, "RTF wrong"
        parsed = json.loads(t.to_json())
        assert parsed["segments"][0]["words"][0]["probability"] == 0.9
        assert parsed["low_confidence_segment_ids"] == [1]
        return "serialises, flags low-confidence, RTF correct"


# ---------------------------------------------------- group B (needs model)
def group_b(fx: dict[str, str], audio: str | None) -> None:
    print("\nGROUP B — full pipeline (loads the model)")
    from app.pipeline.transcribe import Stage1Transcriber

    transcriber = Stage1Transcriber()

    if not audio:
        record("SKIP", "real speech transcribes correctly", "no --audio given")
    else:
        @check("real speech transcribes with rich output contract")
        def _():
            t = transcriber.transcribe(audio)
            assert not t.is_empty, "transcript came back empty"
            assert t.segments, "no segments"
            words = t.segments[0].words
            assert words, "NO WORD TIMESTAMPS — output contract broken"
            assert words[0].probability is not None, "no word confidence"
            assert t.segments[0].avg_logprob is not None, "no segment confidence"
            import json
            json.loads(t.to_json())          # must be serialisable
            return (f'"{t.text[:55]}…" | {len(t.segments)} seg | '
                    f"RTF={t.realtime_factor:.2f} | decode={t.decode_method}")

    @check("silence is handled gracefully (empty transcript, not a crash)")
    def _():
        t = transcriber.transcribe(fx["silence"])
        assert t.is_empty, f"expected empty transcript, got: {t.text[:60]!r}"
        assert any("no speech" in w.lower() for w in t.warnings), \
            f"missing 'no speech' warning; warnings={t.warnings}"
        return f"{len(t.warnings)} warning(s), no exception"

    @check("corrupt file fails cleanly through the full pipeline")
    def _():
        try:
            transcriber.transcribe(fx["corrupt"])
        except AudioDecodeError as e:
            assert e.user_message
            return "clean AudioDecodeError with user message"
        raise AssertionError("did not raise")


# -------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", help="a real speech file for the full test")
    ap.add_argument("--no-model", action="store_true",
                    help="run only checks that don't need the Whisper model")
    args = ap.parse_args()

    audio = args.audio
    if audio and not os.path.exists(audio):
        print(f"warning: --audio {audio} not found; skipping that check\n")
        audio = None

    print("=" * 66)
    print("STAGE 1 CHECK HARNESS")
    print("=" * 66)

    with tempfile.TemporaryDirectory() as tmp:
        fx = build_fixtures(tmp)
        group_a(fx)
        if args.no_model:
            print("\nGROUP B — skipped (--no-model)")
        else:
            group_b(fx, audio)

    passed = sum(1 for s, _, _ in RESULTS if s == "PASS")
    failed = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    skipped = sum(1 for s, _, _ in RESULTS if s == "SKIP")

    print("\n" + "=" * 66)
    print(f"RESULT: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 66)
    if failed:
        print("\nFailed:")
        for s, name, detail in RESULTS:
            if s == "FAIL":
                print(f"  - {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
