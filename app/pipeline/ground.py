"""
Stage 4 — grounding the meeting record against the transcript.

Stage 3's prompt DEMANDS an evidence quote for every decision and action item.
Nothing checked that those quotes exist. A language model generates plausible
text, and a plausible-sounding quote is exactly the kind of thing it generates
well — so "the prompt asked for evidence" is not evidence.

This stage checks. The response is GRADED rather than all-or-nothing, because
the two failure modes are different:

  - evidence not found in the transcript  -> DROP the item. It is unsupported.
  - evidence fine but the OWNER never appears in the transcript
                                          -> keep the task, downgrade the owner
                                             to unspecified. The work is real;
                                             the attribution was invented.

Dropping an invented owner but keeping the task preserves true information
instead of throwing it away with the false part.

Nothing is deleted silently: everything removed or downgraded is returned so the
interface can show what was rejected and why — the same pattern as Stage 2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import ActionItem, Decision, MeetingRecord, Segment

# Fraction of the quote's content words that must appear in the transcript.
# Not 1.0: models paraphrase slightly ("we're moving" vs "we are moving") and we
# are checking that the claim is SUPPORTED, not that it was copied perfectly.
EVIDENCE_OVERLAP = 0.6
MIN_EVIDENCE_WORDS = 3

_STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for",
         "is", "are", "was", "were", "be", "it", "that", "this", "we", "i"}


def _tokens(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", (text or "").lower()).split()


def _content(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in _STOP and len(t) > 2]


@dataclass
class GroundingResult:
    record: MeetingRecord
    dropped_decisions: list[tuple[str, str]] = field(default_factory=list)
    dropped_actions: list[tuple[str, str]] = field(default_factory=list)
    downgraded: list[str] = field(default_factory=list)

    @property
    def total_removed(self) -> int:
        return len(self.dropped_decisions) + len(self.dropped_actions)

    def to_dict(self) -> dict:
        return {
            "dropped_decisions": [{"statement": s, "why": w}
                                  for s, w in self.dropped_decisions],
            "dropped_actions": [{"task": s, "why": w} for s, w in self.dropped_actions],
            "downgraded_fields": self.downgraded,
        }


def find_evidence(evidence: str, segments: list[Segment]) -> Segment | None:
    """
    Locate the segment that supports this quote, or None if nothing does.

    Returns the segment so the claim can be anchored to a TIMESTAMP — the payoff
    of keeping word-level timing all the way from Stage 1.
    """
    want = _content(evidence)
    if len(want) < MIN_EVIDENCE_WORDS:
        return None

    best, best_score = None, 0.0
    for seg in segments:
        have = set(_content(seg.text))
        if not have:
            continue
        score = sum(1 for w in want if w in have) / len(want)
        if score > best_score:
            best, best_score = seg, score
    return best if best_score >= EVIDENCE_OVERLAP else None


def _mentioned(value: str | None, transcript: str) -> bool:
    """Is this owner/deadline actually spoken anywhere in the transcript?"""
    if not value:
        return True                      # None is already "unspecified"
    toks = _content(value)
    if not toks:
        return True
    have = set(_tokens(transcript))
    return any(t in have for t in toks)


def ground(record: MeetingRecord, segments: list[Segment]) -> GroundingResult:
    if record is None:
        return GroundingResult(record=MeetingRecord())

    transcript = " ".join(s.text for s in segments)
    result = GroundingResult(record=record)

    kept_decisions: list[Decision] = []
    for d in record.decisions:
        seg = find_evidence(d.evidence, segments)
        if seg is None:
            result.dropped_decisions.append((
                d.statement,
                "no supporting quote found in the transcript"
                if d.evidence else "no evidence quote was provided",
            ))
            continue
        d.segment_id, d.timestamp = seg.id, seg.start
        kept_decisions.append(d)
    record.decisions = kept_decisions

    kept_actions: list[ActionItem] = []
    for a in record.action_items:
        seg = find_evidence(a.evidence, segments)
        if seg is None:
            result.dropped_actions.append((
                a.task,
                "no supporting quote found in the transcript"
                if a.evidence else "no evidence quote was provided",
            ))
            continue
        a.segment_id, a.timestamp = seg.id, seg.start

        # Evidence is sound, but the attribution may still be invented.
        if not _mentioned(a.owner, transcript):
            result.downgraded.append(
                f'owner {a.owner!r} for "{a.task[:40]}" was never named in the '
                f"recording — set to unspecified"
            )
            a.owner = None
        if not _mentioned(a.deadline, transcript):
            result.downgraded.append(
                f'deadline {a.deadline!r} for "{a.task[:40]}" was never stated — '
                f"set to unspecified"
            )
            a.deadline = None
        kept_actions.append(a)
    record.action_items = kept_actions

    if result.total_removed:
        record.warnings.append(
            f"{result.total_removed} generated item(s) could not be supported by "
            f"the transcript and were removed."
        )
    if result.downgraded:
        record.warnings.append(
            f"{len(result.downgraded)} owner/deadline value(s) were not stated in "
            f"the recording and were set to unspecified."
        )
    return result
