"""
Stage 3 — meeting minutes, decisions and action items.

Runs on the REFINED transcript, using a model distinct from Stage 2's.

The whole design problem here is invention. A language model generates the most
plausible continuation, and "by Friday" is a very plausible continuation of
"we'll get that done" — so an unguarded model will confidently supply deadlines
and owners nobody said. Three defences:

  1. The schema makes owner/deadline OPTIONAL, so "unspecified" is representable
     and is the default rather than something the model must remember to say.
  2. Every decision and action item must carry an `evidence` quote, which makes
     fabrication harder to produce and trivial to check.
  3. Stage 4 (verification) later checks that evidence against the transcript.
"""
from __future__ import annotations

import time

from ..llm import LLMClient, LLMError, local_stage3_client, stage3_client
from ..schemas import ActionItem, Decision, MeetingRecord, Segment

SYSTEM_PROMPT = """\
You write meeting records from transcripts. You report ONLY what the transcript \
says. You never infer, assume, or fill gaps.

Return JSON only:
{"summary": str,
 "minutes": [str],
 "decisions": [{"statement": str, "evidence": str}],
 "action_items": [{"task": str, "owner": str|null, "deadline": str|null, "evidence": str}]}

Hard rules:
- "evidence" MUST be a short quote copied from the transcript. If you cannot \
quote it, do not include the item.
- A DECISION is something the group actually settled. A suggestion, proposal or \
idea under discussion is NOT a decision. When in doubt, leave it out.
- An ACTION ITEM is work someone committed to. "We should maybe look into X" is \
not a commitment.
- "owner" is null unless the transcript names who will do it. Do NOT guess from \
who was speaking.
- "deadline" is null unless the transcript states a time. Do NOT infer "soon", \
"next week", or "by Friday" unless those words were said.
- Empty lists are correct and expected answers when nothing qualifies.\
"""


def _render(segments: list[Segment]) -> str:
    return "\n".join(f"[{s.start:.0f}s] (segment {s.id}) {s.text}" for s in segments)


def summarize(segments: list[Segment], *, client: LLMClient | None = None,
              model_label: str = "") -> MeetingRecord:
    started = time.time()
    text = _render(segments)
    if not text.strip():
        return MeetingRecord(
            summary="", model=model_label or "n/a",
            warnings=["No transcript content to summarise."],
            processing_seconds=0.0,
        )

    explicit_client = client is not None
    if not explicit_client:
        client, model_label = stage3_client(text)

    user = ("Transcript:\n" + text +
            "\n\nProduce the meeting record as JSON. Remember: quote your "
            "evidence, and use null for any owner or deadline that was not stated.")

    warnings: list[str] = []
    data = None
    try:
        data = client.chat_json(SYSTEM_PROMPT, user)
    except LLMError as exc:
        # Fall back to local rather than returning nothing. A remote failure
        # (outage, rate limit, deprecated model id) must not cost the user their
        # entire meeting record — and silently returning an empty record is
        # indistinguishable from a meeting that genuinely had no decisions.
        if not explicit_client and client.backend != "ollama":
            warnings.append(
                f"{model_label} failed ({exc.user_message}) — retried locally."
            )
            client, model_label = local_stage3_client()
            try:
                data = client.chat_json(SYSTEM_PROMPT, user)
            except LLMError as local_exc:
                return MeetingRecord(
                    model=model_label,
                    warnings=warnings + [f"Local fallback also failed: "
                                         f"{local_exc.user_message}"],
                    processing_seconds=time.time() - started,
                )
        else:
            return MeetingRecord(
                model=model_label,
                warnings=[f"Meeting-record model unavailable: {exc.user_message}"],
                processing_seconds=time.time() - started,
            )

    def _clean(value) -> str | None:
        """Normalise the many ways a model says 'nothing here' into None."""
        if value is None:
            return None
        s = str(value).strip()
        if not s or s.lower() in {"null", "none", "n/a", "unspecified", "unknown",
                                  "not specified", "not stated", "tbd", "-"}:
            return None
        return s

    decisions = [
        Decision(statement=str(d.get("statement", "")).strip(),
                 evidence=str(d.get("evidence", "")).strip())
        for d in (data.get("decisions") or []) if isinstance(d, dict)
        and str(d.get("statement", "")).strip()
    ]
    actions = [
        ActionItem(task=str(a.get("task", "")).strip(),
                   owner=_clean(a.get("owner")),
                   deadline=_clean(a.get("deadline")),
                   evidence=str(a.get("evidence", "")).strip())
        for a in (data.get("action_items") or []) if isinstance(a, dict)
        and str(a.get("task", "")).strip()
    ]
    minutes = [str(m).strip() for m in (data.get("minutes") or []) if str(m).strip()]

    return MeetingRecord(
        summary=str(data.get("summary", "")).strip(),
        minutes=minutes,
        decisions=decisions,
        action_items=actions,
        model=model_label or client.model,
        warnings=warnings,
        processing_seconds=time.time() - started,
    )
