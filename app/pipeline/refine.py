"""
Stage 2 — domain-aware transcript refinement.

The model NEVER emits the transcript. It emits only a list of proposed edits,
which we verify and apply ourselves. Everything not explicitly edited is
byte-identical by construction, so meaning drift is impossible rather than
merely discouraged.

Two sources of corrections:
  1. deterministic glossary pre-pass — phonetic matching, no model involved
  2. LLM edit-list pass — for the contextual cases the glossary can't reach

Both go through the same verification gates. Neither is trusted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..llm import LLMClient, LLMError, stage2_client
from ..schemas import Segment, Transcript
from .glossary import Glossary, default_glossary, normalize
from .verification import (Edit, LOW_CONFIDENCE_PROB, apply_edits, verify_edits)

__all__ = ["RefinementResult", "refine", "glossary_pass", "llm_pass",
           "llm_adjudicate"]

# Must match verification.MAX_SPAN_WORDS: the pre-pass can only find spans it
# can form. At 3 it could never see a 4-word mis-hearing like "g r p c" -> gRPC,
# even though the gates would happily have accepted it.
MAX_NGRAM = 5
EDGE_PUNCT = ".,!?;:\"'()[] "


@dataclass
class RefinementResult:
    raw_text: str
    refined_text: str
    segments: list[Segment]
    edits: list[Edit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model: str = ""
    processing_seconds: float = 0.0

    @property
    def applied(self) -> list[Edit]:
        return [e for e in self.edits if e.accepted]

    @property
    def rejected(self) -> list[Edit]:
        return [e for e in self.edits if not e.accepted]

    @property
    def vetoed(self) -> list[Edit]:
        """Deterministic proposals the LLM rejected as wrong in context."""
        return [e for e in self.edits if e.vetoed]

    @property
    def off_glossary(self) -> list[Edit]:
        """Applied edits whose replacement was NOT a known term — weaker evidence."""
        return [e for e in self.applied if not e.glossary_backed]

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "refined_text": self.refined_text,
            "model": self.model,
            "processing_seconds": round(self.processing_seconds, 2),
            "edits_applied": [e.to_dict() for e in self.applied],
            "edits_rejected": [e.to_dict() for e in self.rejected],
            "warnings": self.warnings,
        }


# ------------------------------------------------------- deterministic pass
def _low_confidence_spans(segment: Segment) -> list[str]:
    """
    Candidate phrases to check against the glossary: n-grams containing at least
    one word Whisper was unsure about. Confident speech is left alone.
    """
    if not segment.words:
        return []
    words = segment.words
    out: list[str] = []

    def ends_sentence(word: str) -> bool:
        return word.strip().endswith((".", "?", "!"))

    for n in range(1, MAX_NGRAM + 1):
        for i in range(len(words) - n + 1):
            window = words[i:i + n]
            # Never span a sentence boundary. Found by the negative control:
            # "stage inn" alone does NOT match "staging", but the scanner also
            # formed "stage inn. Okay," — and concatenated, the extra word made
            # the phonetic code collide with "staging", so a valid sentence got
            # mangled. A genuine mis-hearing does not cross a full stop.
            if any(ends_sentence(w.word) for w in window[:-1]):
                continue
            if all(w.probability >= LOW_CONFIDENCE_PROB for w in window):
                continue
            phrase = " ".join(w.word.strip() for w in window).strip()
            if phrase:
                out.append(phrase)
    return out


def glossary_pass(segments: list[Segment], glossary: Glossary) -> list[Edit]:
    """Phonetic glossary matching. No model, fully deterministic and explainable."""
    edits: list[Edit] = []
    seen: set[tuple[int, str]] = set()
    for seg in segments:
        for phrase in _low_confidence_spans(seg):
            # Strip punctuation at the EDGES of the phrase. Otherwise
            # 'g r p c.' -> 'gRPC' deletes the sentence's full stop, and the same
            # phrase with different trailing punctuation ('red is,' vs 'red is.')
            # becomes several edits, only one of which survived deduplication.
            core = phrase.strip(EDGE_PUNCT)
            if not core:
                continue
            term = glossary.best_match(core)
            if not term or normalize(core) == normalize(term):
                continue
            key = (seg.id, normalize(core))
            if key in seen:
                continue
            seen.add(key)
            edits.append(Edit(
                segment_id=seg.id, original=core, replacement=term,
                reason="phonetic match to glossary term (deterministic pass)",
                confidence=1.0, source="glossary",
            ))
    return edits


# ------------------------------------------------- LLM job A: veto in context
ADJUDICATE_PROMPT = """\
You are checking proposed corrections to a meeting transcript. Each proposal \
replaces some words with a technical term because those words SOUND like it. \
The proposals come from a phonetic matcher that cannot read context - your only \
job is to catch the few that are wrong in context.

KEEP is the default. Keep a proposal when the original words are not meaningful \
English in that sentence, or when the technical term obviously fits what the \
meeting is about. Garbled fragments are mis-hearings - keep those corrections.

REJECT only when the original words genuinely make sense as ordinary English in \
that exact sentence, so replacing them would destroy the meaning.

Examples:
  "the light was red is it not" + replace "red is" with "Redis"
     -> keep: false   ("red, is it not" is ordinary English about a colour)
  "right now sessions live in red is" + replace "red is" with "Redis"
     -> keep: true    (sessions living in Redis is the obvious reading)
  "we deployed it on cooper netties" + replace "cooper netties" with "Kubernetes"
     -> keep: true    ("cooper netties" is not English - clearly a mis-hearing)
  "we should roll back the change" + replace "roll back" with "rollback"
     -> keep: true    (same meaning, correct technical spelling)

If you are unsure, KEEP. A wrong rejection throws away a real correction.

Return JSON only: {"verdicts": [{"index": int, "keep": bool, "why": str}]}
Include a verdict for EVERY proposal, by its index.\
"""


def llm_adjudicate(candidates: list[Edit], segments: dict[int, Segment],
                   client: LLMClient) -> list[str]:
    """
    Ask the model to veto context-wrong proposals from the deterministic pass.

    Returns a list the same length as `candidates`: a reason string where the
    edit should be vetoed, or "" to keep it. On any failure we keep everything —
    degrading to the deterministic pass alone, which is the previous behaviour.
    """
    if not candidates:
        return []
    lines = []
    for i, e in enumerate(candidates):
        context = segments[e.segment_id].text if e.segment_id in segments else ""
        lines.append(f'{i}. in "{context}"\n   replace {e.original!r} '
                     f'with {e.replacement!r}?')
    user = "Proposals:\n" + "\n".join(lines) + "\n\nReturn the verdicts as JSON."

    try:
        data = client.chat_json(ADJUDICATE_PROMPT, user)
    except LLMError:
        return [""] * len(candidates)

    verdicts = {}
    for v in (data.get("verdicts") or []):
        if isinstance(v, dict):
            try:
                verdicts[int(v.get("index", -1))] = (
                    bool(v.get("keep", True)), str(v.get("why", ""))[:160])
            except (TypeError, ValueError):
                continue

    out: list[str] = []
    for i in range(len(candidates)):
        keep, why = verdicts.get(i, (True, ""))
        out.append("" if keep else f"vetoed by context review: {why or 'wrong in context'}")
    return out


# ---------------------------------------------------------------- LLM pass
SYSTEM_PROMPT = """\
You correct speech-recognition errors in meeting transcripts. You fix ONLY \
mis-heard domain terminology — product names, technical jargon, acronyms.

You do NOT rewrite, rephrase, summarise, translate, fix grammar, fix punctuation, \
or improve style. The transcript is a record of what people actually said; clumsy \
phrasing is accurate and must be preserved.

Return JSON only: {"edits": [{"segment_id": int, "original": str, \
"replacement": str, "reason": str, "confidence": float}]}

Rules:
- "original" MUST be copied character-for-character from the segment text.
- Prefer a term from the supplied glossary. You MAY also correct a technical \
term that is NOT in the glossary if you are confident it is a mis-hearing of \
real domain jargon - the glossary is incomplete by nature. Set "confidence" \
honestly: below 0.7 for anything you are unsure of, and off-glossary \
corrections below that bar will be discarded.
- Never replace something with an ordinary English word. Corrections are \
technical terminology, not vocabulary changes.
- The replacement must SOUND LIKE the original. If it doesn't sound similar, it \
is a meaning change, not a correction - do not propose it.
- Never change numbers, names, dates, or negations ("not", "won't", "can't").
- Never add or delete words. Substitutions only.
- Only correct text marked as low-confidence.
- If nothing needs correcting, return {"edits": []}. An empty list is a correct \
and common answer. Do not invent work.\
"""

FEW_SHOT = """\
EXAMPLE 1 — a genuine mis-hearing:
segment 0 (low-confidence: "cooper netties"): "we deployed it on cooper netties last week"
glossary: ["Kubernetes", "Docker"]
-> {"edits": [{"segment_id": 0, "original": "cooper netties", \
"replacement": "Kubernetes", "reason": "phonetically matches Kubernetes", "confidence": 0.9}]}

EXAMPLE 2 — clumsy but ACCURATE, leave it alone:
segment 1: "so yeah we uh we think maybe we should probably do that"
glossary: ["Kubernetes"]
-> {"edits": []}

EXAMPLE 3 — tempting but FORBIDDEN (this would change meaning):
segment 2: "we might ship on Friday"
glossary: ["staging"]
-> {"edits": []}

EXAMPLE 4 — sounds similar but is NOT a glossary term, so no edit:
segment 3 (low-confidence: "headline"): "the headline is next Tuesday"
glossary: ["Kubernetes", "Docker"]
-> {"edits": []}\
"""


def _render_segments(segments: list[Segment]) -> str:
    lines = []
    for seg in segments:
        unsure = [w.word.strip() for w in seg.words
                  if w.probability < LOW_CONFIDENCE_PROB] if seg.words else []
        marker = f' (low-confidence: {", ".join(repr(w) for w in unsure[:8])})' if unsure else ""
        lines.append(f'segment {seg.id}{marker}: "{seg.text}"')
    return "\n".join(lines)


def llm_pass(segments: list[Segment], glossary: Glossary,
             client: LLMClient) -> tuple[list[Edit], list[str]]:
    """Ask the model for an edit-list. Failure here degrades, never crashes."""
    if not segments:
        return [], []
    user = (
        f"{FEW_SHOT}\n\nNOW THE REAL TASK.\n\nGlossary: "
        f"{sorted(glossary.terms)}\n\nTranscript:\n{_render_segments(segments)}\n\n"
        "Return the JSON edit list."
    )
    try:
        data = client.chat_json(SYSTEM_PROMPT, user)
    except LLMError as exc:
        # The deterministic pass already ran; losing the LLM pass degrades
        # quality but must not lose the run.
        return [], [f"Refinement model unavailable: {exc.user_message}"]

    raw_edits = data.get("edits") or []
    if not isinstance(raw_edits, list):
        return [], ["Refinement model returned an unexpected shape; ignored."]

    edits: list[Edit] = []
    for item in raw_edits:
        if not isinstance(item, dict):
            continue
        try:
            edits.append(Edit(
                segment_id=int(item.get("segment_id", -1)),
                original=str(item.get("original", "")),
                replacement=str(item.get("replacement", "")),
                reason=str(item.get("reason", ""))[:200],
                confidence=float(item.get("confidence", 0.0) or 0.0),
                source="llm",
            ))
        except (TypeError, ValueError):
            continue
    return edits, []


# ------------------------------------------------------------------- stage
def refine(transcript: Transcript, *, glossary: Glossary | None = None,
           client: LLMClient | None = None, use_llm: bool = True) -> RefinementResult:
    glossary = glossary or default_glossary()
    client = client or stage2_client()
    started = time.time()
    warnings: list[str] = []

    by_id = {s.id: s for s in transcript.segments}

    # 1. deterministic: phonetic glossary matching, no model involved
    edits = glossary_pass(transcript.segments, glossary)

    vetoed: list[Edit] = []
    if use_llm:
        # 2. LLM job A — veto proposals that are wrong IN CONTEXT. The phonetic
        #    matcher is context-blind ("the light was red, is it not" -> Redis),
        #    and this is the only thing that can catch that.
        verdicts = llm_adjudicate(edits, by_id, client)
        keep: list[Edit] = []
        for edit, reason in zip(edits, verdicts):
            if reason:
                edit.vetoed = True
                edit.accepted = False
                edit.rejections.append(reason)
                vetoed.append(edit)
            else:
                keep.append(edit)
        edits = keep

        # 3. LLM job B — propose corrections the glossary cannot cover.
        llm_edits, llm_warnings = llm_pass(transcript.segments, glossary, client)
        warnings.extend(llm_warnings)
        seen = {(e.segment_id, normalize(e.original)) for e in edits}
        edits.extend(e for e in llm_edits
                     if (e.segment_id, normalize(e.original)) not in seen)

    total_words = len(transcript.text.split())
    edits = verify_edits(edits, by_id, glossary, total_words) + vetoed

    refined_segments = apply_edits(transcript.segments, edits)
    refined_text = " ".join(s.text for s in refined_segments).strip()

    return RefinementResult(
        raw_text=transcript.text,
        refined_text=refined_text,
        segments=refined_segments,
        edits=edits,
        warnings=warnings,
        model=client.model,
        processing_seconds=time.time() - started,
    )
