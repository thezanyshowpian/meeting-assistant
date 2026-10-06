"""
Data contracts passed between pipeline stages.

Stage 1 emits a RICH Transcript: text + segments + word-level timestamps +
per-segment confidence. The richness is deliberate:
  - word/segment timestamps let Stage 4 anchor each decision or action item to
    WHERE in the recording it came from ("supported at 12:30");
  - confidence lets Stage 2 target shaky passages rather than rewriting blindly.

Design note: plain dataclasses (no pydantic) — Stage 1's shapes are simple and
stable, and this keeps the dependency surface small. Stage 3 may introduce
pydantic where validating *model-generated* structured output actually earns it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

# Heuristic, not truth: segments below this average log-probability are FLAGGED
# for downstream attention — never silently dropped or "corrected" on this basis.
LOW_CONFIDENCE_LOGPROB = -1.0


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float
    speaker: str | None = None      # set by diarization; anonymous label, never a guessed name


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    words: list[Word] = field(default_factory=list)

    @property
    def is_low_confidence(self) -> bool:
        return self.avg_logprob < LOW_CONFIDENCE_LOGPROB


@dataclass
class Transcript:
    """Stage 1 output. `text` is the raw transcript; everything else is context."""

    text: str
    segments: list[Segment]
    language: str
    duration: float            # seconds of audio
    model: str
    decode_method: str         # which decode path actually worked
    warnings: list[str] = field(default_factory=list)
    processing_seconds: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def low_confidence_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.is_low_confidence]

    @property
    def realtime_factor(self) -> float:
        """proc/audio. <1.0 means faster than realtime."""
        return self.processing_seconds / self.duration if self.duration else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_empty"] = self.is_empty
        d["low_confidence_segment_ids"] = [s.id for s in self.low_confidence_segments]
        d["realtime_factor"] = round(self.realtime_factor, 3)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# --------------------------------------------------------------- Stage 3
# owner/deadline are OPTIONAL at the type level. That is the mechanism by which
# "unspecified, not guessed" is enforced: there is no way to represent a made-up
# owner as a confirmed one, and the exporters render None as "unspecified".
UNSPECIFIED = "unspecified"


@dataclass
class Decision:
    statement: str
    evidence: str = ""          # supporting quote from the transcript
    segment_id: int | None = None
    timestamp: float | None = None


@dataclass
class ActionItem:
    task: str
    owner: str | None = None        # None => unspecified. Never guessed.
    deadline: str | None = None     # None => unspecified. Never guessed.
    evidence: str = ""
    segment_id: int | None = None
    timestamp: float | None = None
    # How the owner is known, set by Stage 4 / naming — never by the model:
    #   "stated"   the name is spoken near the evidence ("Sam, can you…")
    #   "speaker"  a voice committed in the first person; owner is its label
    #   "inferred" that label was bound to a name by evidence-based naming
    owner_source: str | None = None
    owner_evidence: str = ""
    owner_label: str | None = None   # the diarization label behind an inferred name

    @property
    def owner_display(self) -> str:
        if not self.owner:
            return UNSPECIFIED
        if self.owner_source == "inferred":
            return f"{self.owner} (inferred)"
        if self.owner_source == "speaker":
            return f"{self.owner} (voice only)"
        return self.owner

    @property
    def deadline_display(self) -> str:
        return self.deadline or UNSPECIFIED


@dataclass
class MeetingRecord:
    summary: str = ""
    minutes: list[str] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    model: str = ""
    warnings: list[str] = field(default_factory=list)
    processing_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "minutes": self.minutes,
            "decisions": [asdict(d) for d in self.decisions],
            "action_items": [
                {**asdict(a),
                 "owner": a.owner,            # null, not a guess
                 "deadline": a.deadline}
                for a in self.action_items
            ],
            "model": self.model,
            "warnings": self.warnings,
            "processing_seconds": round(self.processing_seconds, 2),
        }
