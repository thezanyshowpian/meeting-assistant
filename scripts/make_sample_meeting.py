#!/usr/bin/env python3
"""
Generate the sample meeting recording from the script, using macOS `say`.

Why synthetic rather than a real recording:
  - it is freely shareable (a real meeting raises privacy problems);
  - it is reproducible — anyone can regenerate the exact same audio;
  - we KNOW the ground truth, so WER, extraction accuracy and now diarization
    error (DER) are measurable rather than impressionistic.

Speakers MUST have acoustically distinct voices. The first version fell back to
the system default voice for one speaker, which on many Macs is the same voice as
another speaker — making them indistinguishable to any diarizer and the fixture
useless for testing it. This version refuses to run with duplicate voices.

Writes:
  data/sample_meeting.wav             16 kHz mono WAV
  data/sample_meeting_reference.txt   ground-truth transcript
  data/sample_meeting_reference.rttm  ground-truth speaker turns (standard RTTM)
  data/sample_meeting_turns.json      the same turns, with text, for evaluation

USAGE
    python scripts/make_sample_meeting.py                       # the sample meeting
    python scripts/make_sample_meeting.py --name heldout_meeting  # the held-out one
    python scripts/make_sample_meeting.py --name long_meeting     # ~30-min demo meeting
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

RATE = 16_000
GAP_SECONDS = 0.35            # pause between turns, so it sounds like a conversation
SPEECH_THRESHOLD = 328        # ~1% of int16 full scale: what counts as "speaking"

# Natural English voices, in preference order. Novelty voices (Bells, Bubbles…)
# are deliberately absent.
FALLBACK_VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Tessa", "Rishi",
                   "Fiona", "Alex", "Fred", "Victoria", "Veena", "Tom"]


def available_voices() -> set[str]:
    try:
        out = subprocess.check_output(["say", "-v", "?"], text=True)
    except Exception:                                              # noqa: BLE001
        return set()
    return {line.split()[0] for line in out.splitlines() if line.strip()}


def assign_voices(preferences: dict, available: set[str]) -> dict[str, str]:
    """One DISTINCT voice per speaker, or fail loudly."""
    chosen: dict[str, str] = {}
    for speaker, pref in preferences.items():
        wanted = [pref] if isinstance(pref, str) else list(pref)
        for voice in wanted + FALLBACK_VOICES:
            if voice in available and voice not in chosen.values():
                chosen[speaker] = voice
                break
        else:
            raise SystemExit(
                f"No distinct voice available for {speaker}. Need "
                f"{len(preferences)} different voices; have {sorted(available)}.\n"
                "Install more: System Settings > Accessibility > Spoken Content > "
                "System Voice > Manage Voices."
            )
    return chosen


def synth_line(text: str, voice: str, path: str, rate: int | None = None) -> None:
    cmd = ["say", "-v", voice, "-o", path, "--data-format=LEI16@16000"]
    if rate:
        cmd += ["-r", str(rate)]              # words per minute
    subprocess.run(cmd + [text], check=True)


def speech_bounds(pcm: np.ndarray) -> tuple[int, int]:
    """First and last sample that is actually speech (say pads each clip)."""
    loud = np.flatnonzero(np.abs(pcm.astype(np.int32)) > SPEECH_THRESHOLD)
    if loud.size == 0:
        return 0, len(pcm)
    return int(loud[0]), int(loud[-1]) + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sample_meeting",
                    help="reads data/<name>_script.json, writes data/<name>.wav etc.")
    args = ap.parse_args()

    script = os.path.join(DATA, f"{args.name}_script.json")
    out_wav = os.path.join(DATA, f"{args.name}.wav")
    out_ref = os.path.join(DATA, f"{args.name}_reference.txt")
    out_rttm = os.path.join(DATA, f"{args.name}_reference.rttm")
    out_turns = os.path.join(DATA, f"{args.name}_turns.json")

    if sys.platform != "darwin":
        print("This script needs macOS `say`. Run it on the Mac.", file=sys.stderr)
        return 1

    with open(script) as fh:
        spec = json.load(fh)

    voices = assign_voices(spec["voices"], available_voices())
    for speaker, voice in voices.items():
        print(f"  {speaker:<8} -> {voice}")

    chunks: list[np.ndarray] = []
    turns: list[dict] = []
    cursor = 0                                    # samples written so far
    gap = np.zeros(int(RATE * GAP_SECONDS), dtype=np.int16)

    with tempfile.TemporaryDirectory() as tmp:
        for i, line in enumerate(spec["lines"]):
            path = os.path.join(tmp, f"line_{i:03d}.wav")
            # "spoken" (optional) is what the voice SAYS; "text" is what the speaker
            # MEANT, and is the reference. Used to simulate a mumbled or accented
            # term ("cough ka" for Kafka) that Stage 2 should recover.
            synth_line(line.get("spoken", line["text"]), voices[line["speaker"]], path,
                       spec.get("rate"))
            with wave.open(path, "rb") as w:
                assert w.getnchannels() == 1 and w.getsampwidth() == 2
                assert w.getframerate() == RATE, f"unexpected rate {w.getframerate()}"
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

            lo, hi = speech_bounds(pcm)
            turns.append({
                "speaker": line["speaker"],
                "start": round((cursor + lo) / RATE, 3),
                "end": round((cursor + hi) / RATE, 3),
                "text": line["text"],
            })
            chunks += [pcm, gap]
            cursor += len(pcm) + len(gap)
            print(f"  [{i + 1:3d}/{len(spec['lines'])}] {turns[-1]['start']:6.2f}–"
                  f"{turns[-1]['end']:6.2f}s  {line['speaker']}: {line['text'][:50]}…")

    audio = np.concatenate(chunks)
    with wave.open(out_wav, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(audio.tobytes())

    with open(out_ref, "w") as fh:
        fh.write(" ".join(t["text"] for t in turns) + "\n")

    with open(out_rttm, "w") as fh:
        for t in turns:
            fh.write(f"SPEAKER {args.name} 1 {t['start']:.3f} "
                     f"{t['end'] - t['start']:.3f} <NA> <NA> {t['speaker']} <NA> <NA>\n")

    with open(out_turns, "w") as fh:
        json.dump({"voices": voices, "turns": turns}, fh, indent=2)

    print(f"\nwrote {out_wav}  ({len(audio) / RATE:.1f}s, {len(turns)} turns, "
          f"{len(voices)} distinct voices)")
    print(f"wrote {out_rttm}  (ground-truth speaker turns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
