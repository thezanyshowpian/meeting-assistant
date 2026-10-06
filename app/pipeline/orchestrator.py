"""
Pipeline orchestrator — runs the stages in order and owns failure handling.

Centralising this is what makes "one workflow, in the stated order" a checkable
property rather than an implicit one, and gives a single place where every
failure mode is handled.

Stage 1 failures are FATAL (no audio, no transcript, nothing downstream can run).
Stage 2 and 3 failures DEGRADE: a missing refinement still leaves the raw
transcript, and a failed record still leaves both transcripts. We would rather
return partial, honest output than nothing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..errors import PipelineError
from ..schemas import MeetingRecord, Transcript
import os

from .diarize import DiarizationResult, diarize, speechbrain_available
from .glossary import Glossary, default_glossary
from .ground import GroundingResult, ground
from .refine import RefinementResult, refine
from .summarize import summarize
from .transcribe import Stage1Transcriber


@dataclass
class PipelineResult:
    transcript: Transcript | None = None
    diarization: DiarizationResult | None = None
    refinement: RefinementResult | None = None
    record: MeetingRecord | None = None
    grounding: GroundingResult | None = None
    warnings: list[str] = field(default_factory=list)
    stage_times: dict[str, float] = field(default_factory=dict)
    failed_stage: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.failed_stage is None

    @property
    def raw_text(self) -> str:
        return self.transcript.text if self.transcript else ""

    @property
    def refined_text(self) -> str:
        if self.refinement:
            return self.refinement.refined_text
        return self.raw_text

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "failed_stage": self.failed_stage,
            "error": self.error,
            "warnings": self.warnings,
            "stage_times": {k: round(v, 2) for k, v in self.stage_times.items()},
            "raw_transcript": self.transcript.to_dict() if self.transcript else None,
            "diarization": self.diarization.to_dict() if self.diarization else None,
            "refinement": self.refinement.to_dict() if self.refinement else None,
            "meeting_record": self.record.to_dict() if self.record else None,
            "grounding": self.grounding.to_dict() if self.grounding else None,
        }


def run_pipeline(audio_path: str, *, glossary: Glossary | None = None,
                 transcriber: Stage1Transcriber | None = None,
                 use_llm_refinement: bool = True,
                 num_speakers: int | None = None,
                 progress=None) -> PipelineResult:
    """
    audio -> transcript -> refined transcript -> meeting record.

    `progress` is an optional callable(stage_name, message) for UI status.
    """
    glossary = glossary or default_glossary()
    result = PipelineResult()

    def say(stage: str, message: str) -> None:
        if progress:
            progress(stage, message)

    # ---- Stage 1 — fatal on failure
    say("transcribe", "Transcribing audio…")
    t0 = time.time()
    try:
        transcriber = transcriber or Stage1Transcriber()
        result.transcript = transcriber.transcribe(audio_path)
    except PipelineError as exc:
        result.failed_stage = "transcribe"
        result.error = exc.user_message
        return result
    result.stage_times["transcribe"] = time.time() - t0
    result.warnings.extend(result.transcript.warnings)

    if result.transcript.is_empty:
        # Not an error: a silent or non-speech recording is a legitimate input.
        result.warnings.append(
            "No speech was detected, so no minutes could be produced."
        )
        return result

    # ---- Stage 1.5 — speaker diarization (optional, degrades)
    # DIARIZE=auto (default): run if speechbrain is installed; off: never; on:
    # run, and complain loudly if it can't.
    mode = os.environ.get("DIARIZE", "auto").lower()
    if mode != "off":
        if speechbrain_available():
            say("diarize", "Identifying speakers…")
            t0 = time.time()
            try:
                result.diarization = diarize(result.transcript, audio_path,
                                             num_speakers=num_speakers)
                result.warnings.extend(result.diarization.warnings)
            except Exception as exc:                               # noqa: BLE001
                result.warnings.append(
                    f"Speaker diarization could not run ({exc}); continuing "
                    "without speaker labels.")
            result.stage_times["diarize"] = time.time() - t0
        else:
            result.warnings.append(
                "Speaker diarization skipped: speechbrain is not installed "
                "(pip install -r requirements-diarization.txt to enable).")

    # ---- Stage 2 — degrades on failure
    say("refine", "Correcting domain terminology…")
    t0 = time.time()
    try:
        result.refinement = refine(result.transcript, glossary=glossary,
                                   use_llm=use_llm_refinement)
        result.warnings.extend(result.refinement.warnings)
    except Exception as exc:                                       # noqa: BLE001
        result.warnings.append(
            f"Refinement could not run ({exc}); using the raw transcript instead."
        )
    result.stage_times["refine"] = time.time() - t0

    # ---- Stage 3 — degrades on failure
    say("summarize", "Writing minutes, decisions and action items…")
    t0 = time.time()
    segments = result.refinement.segments if result.refinement else result.transcript.segments
    try:
        result.record = summarize(segments)
        result.warnings.extend(result.record.warnings)
    except Exception as exc:                                       # noqa: BLE001
        result.warnings.append(f"Meeting record could not be produced ({exc}).")
    result.stage_times["summarize"] = time.time() - t0

    # ---- Stage 4 — ground every claim against the transcript.
    # Stage 3 was ASKED for evidence; this is what checks it was telling the truth.
    if result.record:
        say("verify", "Checking every claim against the transcript…")
        t0 = time.time()
        try:
            result.grounding = ground(result.record, segments)
            result.warnings.extend(
                w for w in result.record.warnings if w not in result.warnings
            )
        except Exception as exc:                                   # noqa: BLE001
            result.warnings.append(f"Grounding check could not run ({exc}).")
        result.stage_times["verify"] = time.time() - t0

    say("done", "Complete.")
    return result
