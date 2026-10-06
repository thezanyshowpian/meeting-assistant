"""
Stage 2 edit verification — an ALLOWLIST, not a blocklist.

A blocklist ("don't change numbers, don't flip negations") can only deny the bad
patterns its author thought of, so it is bounded by imagination. "We might ship
Friday" -> "We WILL ship Friday" passes every such rule. An allowlist is bounded
by construction: an edit is rejected unless it positively demonstrates it is the
narrow thing Stage 2 is for — correcting a mis-heard domain term.

Every gate returns a rejection reason or None. An edit must pass ALL of them.
Gates are deliberately independent so no single one is load-bearing.

Even together these make bad edits rare and bounded, NOT impossible — which is
why applied edits are surfaced in the UI and the raw transcript is always kept.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from ..schemas import Segment
from .glossary import (Glossary, is_ordinary_speech, normalize,
                       phonetically_equal)

MAX_SPAN_WORDS = 5          # an edit longer than this is a rewrite, not a fix
MAX_LENGTH_RATIO = 2.0      # replacement may not balloon relative to the original
LOW_CONFIDENCE_PROB = 0.60  # "Whisper was unsure here"
MIN_LLM_CONFIDENCE = 0.70   # bar for an LLM edit that is NOT glossary-backed
# Budget exists to catch a rogue model proposing an avalanche of individually
# plausible edits — a failure the per-edit gates cannot see. Originally 1 per 50
# words, which measurement showed was far too tight: a 238-word meeting got a
# budget of 4 and silently discarded VALID jargon corrections. A real meeting is
# dense with domain terms. 1 per 20 words (~5% of the transcript) still flags a
# runaway run while leaving room for genuine corrections; the floor stops short
# transcripts from being over-constrained.
EDIT_BUDGET_PER_WORDS = 20
MIN_EDIT_BUDGET = 5

NEGATIONS = {
    "not", "no", "never", "none", "cannot", "cant", "wont", "dont", "doesnt",
    "didnt", "isnt", "arent", "wasnt", "werent", "shouldnt", "wouldnt",
    "couldnt", "nor", "neither", "without",
}

STUTTER = re.compile(r"^(\w+)(\s+\1)+$", re.IGNORECASE)


class EditType(str, Enum):
    SUBSTITUTION = "substitution"
    INSERTION = "insertion"
    DELETION = "deletion"


@dataclass
class Edit:
    """One proposed correction. `original` must appear verbatim in the segment."""

    segment_id: int
    original: str
    replacement: str
    reason: str = ""
    confidence: float = 0.0        # the model's own stated confidence (advisory only)
    source: str = "glossary"       # "glossary" (deterministic) or "llm"

    # filled in by verification
    accepted: bool = False
    rejections: list[str] = field(default_factory=list)
    evidence: str = ""             # e.g. acoustic re-verification result
    glossary_backed: bool = True   # False => replacement is NOT a known term
    vetoed: bool = False           # rejected by the LLM context review

    @property
    def edit_type(self) -> EditType:
        if not self.original.strip():
            return EditType.INSERTION
        if not self.replacement.strip():
            return EditType.DELETION
        return EditType.SUBSTITUTION

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "original": self.original,
            "replacement": self.replacement,
            "edit_type": self.edit_type.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "source": self.source,
            "glossary_backed": self.glossary_backed,
            "vetoed": self.vetoed,
            "accepted": self.accepted,
            "rejections": self.rejections,
            "evidence": self.evidence,
        }


# ------------------------------------------------------------------- gates
def gate_anchor(edit: Edit, segment: Segment, _g: Glossary) -> str | None:
    """The text being replaced must actually exist where the model says it does."""
    if edit.edit_type is EditType.INSERTION:
        return None                                    # handled by gate_type
    if edit.original not in segment.text:
        if normalize(edit.original) not in normalize(segment.text):
            return f"anchor not found in segment {segment.id}"
    return None


def gate_type(edit: Edit, segment: Segment, _g: Glossary) -> str | None:
    """
    Typed policy. Insertions and deletions are denied as CATEGORIES.

    Adding words never spoken is definitionally not terminology correction, and a
    deletion removes something that *was* said and that we cannot verify should
    go. The single exception is collapsing an obvious stutter.
    """
    if edit.edit_type is EditType.INSERTION:
        return "insertions are not permitted in refinement"
    if edit.edit_type is EditType.DELETION:
        if STUTTER.match(edit.original.strip()):
            return None                                # "the the" -> "the"
        return "deletions are not permitted (except collapsing a stutter)"
    return None


def gate_phonetic(edit: Edit, _s: Segment, _g: Glossary) -> str | None:
    """
    A real ASR error SOUNDS like the truth. Semantic drift does not rhyme.

    This is the gate that rejects "might"->"will" and "deadline"->"headline"
    while allowing "sequel"->"SQL" — all three of which a string-similarity
    check gets wrong.
    """
    if edit.edit_type is EditType.DELETION:
        return None
    if not phonetically_equal(edit.original, edit.replacement):
        return (f"not phonetically similar: {edit.original!r} -> "
                f"{edit.replacement!r} (likely a meaning change, not a mis-hearing)")
    return None


def gate_glossary(edit: Edit, _s: Segment, glossary: Glossary) -> str | None:
    """
    Replacements should come from the known glossary — that bounds the output
    vocabulary so nothing can be invented from nothing.

    BUT measurement showed that bounding it *absolutely* made the LLM stage
    useless: the deterministic pass already scans every low-confidence n-gram
    against the whole glossary, so a glossary-bounded LLM can only ever propose
    a subset of what has already been found. With the LLM disabled, recovery was
    still 100% — it was contributing nothing.

    So a non-glossary replacement is permitted, but ONLY when:
      - it came from the LLM (the deterministic pass has no business inventing),
      - it is not ordinary English (corrections should look like jargon),
      - and the model is confident.
    Every other gate (phonetic, confidence, span, digits, negation) still applies.
    Such edits are marked `glossary_backed=False` and surfaced distinctly, since
    they carry strictly weaker evidence than a glossary-matched correction.
    """
    if edit.edit_type is EditType.DELETION:
        return None
    if glossary.contains(edit.replacement):
        edit.glossary_backed = True
        return None

    if edit.source != "llm":
        return f"replacement {edit.replacement!r} is not a known glossary term"
    if is_ordinary_speech(edit.replacement):
        return (f"replacement {edit.replacement!r} is ordinary vocabulary, "
                "not domain terminology")
    if edit.confidence < MIN_LLM_CONFIDENCE:
        return (f"non-glossary replacement {edit.replacement!r} with low model "
                f"confidence ({edit.confidence:.2f})")
    edit.glossary_backed = False
    return None


def gate_span(edit: Edit, _s: Segment, _g: Glossary) -> str | None:
    """Bound the blast radius of any single edit."""
    words = len(edit.original.split())
    if words > MAX_SPAN_WORDS:
        return f"span too long ({words} words > {MAX_SPAN_WORDS}) — that's a rewrite"
    if edit.original and len(edit.replacement) > MAX_LENGTH_RATIO * len(edit.original):
        return "replacement is disproportionately longer than the original"
    # Symmetric guard. Phonetic codes can collide when extra words are glued on
    # ("stage inn. Okay," matched "staging"), so a span far longer than the term
    # it supposedly mis-heard is a collision, not a correction.
    if edit.replacement and len(edit.original) > MAX_LENGTH_RATIO * len(edit.replacement):
        return ("original span is disproportionately longer than the replacement "
                "— likely a phonetic collision, not a mis-hearing")
    return None


def gate_digits(edit: Edit, _s: Segment, _g: Glossary) -> str | None:
    """Numbers are never a terminology correction. Preserve them exactly."""
    if sorted(re.findall(r"\d+", edit.original)) != sorted(re.findall(r"\d+", edit.replacement)):
        return "edit would change a number"
    return None


def gate_negation(edit: Edit, _s: Segment, _g: Glossary) -> str | None:
    """Flipping a negation inverts meaning — the most damaging possible edit."""
    def negs(text: str) -> set[str]:
        return {w for w in normalize(text).replace("'", "").split() if w in NEGATIONS}

    if negs(edit.original) != negs(edit.replacement):
        return "edit would add or remove a negation"
    return None


def gate_confidence(edit: Edit, segment: Segment, _g: Glossary) -> str | None:
    """
    Only text Whisper was UNSURE about may be edited.

    If a word was transcribed at 0.98 confidence, the LLM has no business
    "correcting" it. Uses the per-word probabilities Stage 1 already stores, and
    structurally prevents the model from touching confidently-heard speech —
    exactly where a meaning-changing edit would do the most damage.
    """
    if not segment.words:
        return None                      # no word data: gate cannot apply, don't block
    target = normalize(edit.original)
    if not target:
        return None
    hits = [w for w in segment.words if normalize(w.word) and normalize(w.word) in target]
    if not hits:
        return None                      # anchor spans words we can't resolve; other gates cover it
    if min(w.probability for w in hits) >= LOW_CONFIDENCE_PROB:
        return (f"transcribed with high confidence "
                f"({min(w.probability for w in hits):.2f}) — not a likely mis-hearing")
    return None


GATES = (
    gate_anchor, gate_type, gate_phonetic, gate_glossary,
    gate_span, gate_digits, gate_negation, gate_confidence,
)


# ------------------------------------------------------------------ driver
def verify_edit(edit: Edit, segment: Segment, glossary: Glossary) -> Edit:
    """Run every gate. The edit is accepted only if all of them pass."""
    edit.rejections = [r for r in (g(edit, segment, glossary) for g in GATES) if r]
    edit.accepted = not edit.rejections
    return edit


def verify_edits(edits: list[Edit], segments: dict[int, Segment],
                 glossary: Glossary, total_words: int) -> list[Edit]:
    """
    Verify a batch, then apply the aggregate budget.

    The budget is a defence against the failure mode individual gates can't see:
    an avalanche of individually-plausible edits. If the model proposes a
    correction every few words, something is wrong with the run as a whole.
    """
    for edit in edits:
        segment = segments.get(edit.segment_id)
        if segment is None:
            edit.accepted = False
            edit.rejections = [f"unknown segment id {edit.segment_id}"]
            continue
        verify_edit(edit, segment, glossary)

    budget = max(MIN_EDIT_BUDGET, total_words // EDIT_BUDGET_PER_WORDS)
    accepted = [e for e in edits if e.accepted]
    if len(accepted) > budget:
        for edit in accepted[budget:]:
            edit.accepted = False
            edit.rejections.append(
                f"edit budget exceeded ({len(accepted)} proposed, {budget} allowed "
                f"for {total_words} words) — suspicious volume of changes"
            )
    return edits


def apply_edits(segments: list[Segment], edits: list[Edit]) -> list[Segment]:
    """
    Apply accepted edits. Everything not explicitly edited is byte-identical by
    construction — this is what makes drift impossible rather than merely
    discouraged.
    """
    by_segment: dict[int, list[Edit]] = {}
    for edit in edits:
        if edit.accepted:
            by_segment.setdefault(edit.segment_id, []).append(edit)

    out: list[Segment] = []
    for seg in segments:
        text = seg.text
        for edit in by_segment.get(seg.id, []):
            if edit.original in text:
                text = text.replace(edit.original, edit.replacement, 1)
        out.append(Segment(
            id=seg.id, start=seg.start, end=seg.end, text=text,
            avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob,
            compression_ratio=seg.compression_ratio, words=seg.words,
        ))
    return out
