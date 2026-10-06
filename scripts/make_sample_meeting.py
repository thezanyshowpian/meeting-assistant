#!/usr/bin/env python3
"""
Generate the sample meeting recording from the script, using macOS `say`.

Why synthetic rather than a real recording:
  - it is freely shareable (a real meeting raises privacy problems);
  - it is reproducible — anyone can regenerate the exact same audio;
  - and crucially we KNOW the ground truth, so WER and extraction accuracy become
    measurable rather than impressionistic.

Different `say` voices stand in for different speakers. Audio is written as
16 kHz mono WAV and concatenated with the stdlib `wave` module, so this needs
no ffmpeg and no third-party packages.

USAGE
    python scripts/make_sample_meeting.py
    -> data/sample_meeting.wav
    -> data/sample_meeting_reference.txt   (ground-truth transcript)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import wave

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(HERE, "data", "sample_meeting_script.json")
OUT_WAV = os.path.join(HERE, "data", "sample_meeting.wav")
OUT_REF = os.path.join(HERE, "data", "sample_meeting_reference.txt")

RATE = 16_000
GAP_SECONDS = 0.35          # pause between turns, so it sounds like a conversation


def available_voices() -> set[str]:
    try:
        out = subprocess.check_output(["say", "-v", "?"], text=True)
    except Exception:                                              # noqa: BLE001
        return set()
    return {line.split()[0] for line in out.splitlines() if line.strip()}


def synth_line(text: str, voice: str | None, path: str) -> None:
    cmd = ["say"]
    if voice:
        cmd += ["-v", voice]
    cmd += ["-o", path, "--data-format=LEI16@16000", text]
    subprocess.run(cmd, check=True)


def main() -> int:
    if sys.platform != "darwin":
        print("This script needs macOS `say`. Run it on the Mac.", file=sys.stderr)
        return 1

    with open(SCRIPT) as fh:
        spec = json.load(fh)

    voices = available_voices()
    chosen: dict[str, str | None] = {}
    for speaker, preferred in spec["voices"].items():
        chosen[speaker] = preferred if preferred in voices else None
        if chosen[speaker] is None:
            print(f"  note: voice {preferred!r} unavailable for {speaker}; using default")

    frames: list[bytes] = []
    gap = b"\x00\x00" * int(RATE * GAP_SECONDS)

    with tempfile.TemporaryDirectory() as tmp:
        for i, line in enumerate(spec["lines"]):
            wav_path = os.path.join(tmp, f"line_{i:03d}.wav")
            synth_line(line["text"], chosen.get(line["speaker"]), wav_path)
            with wave.open(wav_path, "rb") as w:
                assert w.getnchannels() == 1 and w.getsampwidth() == 2, "unexpected format"
                frames.append(w.readframes(w.getnframes()))
            frames.append(gap)
            print(f"  [{i + 1:2d}/{len(spec['lines'])}] {line['speaker']}: "
                  f"{line['text'][:60]}…")

    os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)
    with wave.open(OUT_WAV, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(b"".join(frames))

    reference = " ".join(line["text"] for line in spec["lines"])
    with open(OUT_REF, "w") as fh:
        fh.write(reference + "\n")

    seconds = sum(len(f) for f in frames) / 2 / RATE
    print(f"\nwrote {OUT_WAV}  ({seconds:.1f}s, {len(spec['lines'])} turns)")
    print(f"wrote {OUT_REF}  ({len(reference.split())} words of ground truth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
