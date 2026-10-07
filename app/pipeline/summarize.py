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
- An ACTION ITEM is a concrete piece of work the meeting says must be done. It \
does NOT need an owner: "someone needs to document the rollback procedure" IS an \
action item, with owner null. A tentative idea ("maybe we could look into X") or \
a proposal the group declined or parked is NOT an action item.
{owner_rules}
- "deadline" is null unless the transcript states a time. Do NOT infer "soon", \
"next week", or "by Friday" unless those words were said.
- Empty lists are correct and expected answers when nothing qualifies.\
"""


OWNER_RULES_PLAIN = """\
- "owner" is null unless the transcript names who will do it. Do NOT guess from \
who was speaking."""

# Used only when diarization ran. The model sees ANONYMOUS labels and is told
# never to map them to names: naming is done afterwards by auditable rules
# (naming.py), not by the model's guess.
OWNER_RULES_DIARIZED = """\
- Each line starts with an anonymous speaker label such as "Speaker 2:". Labels \
come from voice analysis. They are NOT names, and you must never guess which \
person's name belongs to a label.
- "owner": if the transcript names who will do it ("Sam, can you update the \
dashboard?"), use that name. If a speaker commits to the work in the first \
person ("I'll benchmark it", "I can change it", "Yes, I'll take it"), the owner \
is that speaker's label exactly as written, e.g. "Speaker 2". Otherwise null.
- For a first-person commitment, "evidence" must be quoted from that speaker's \
own line."""


def _render(segments: list[Segment]) -> str:
    return "\n".join(f"[{s.start:.0f}s] (segment {s.id}) {s.text}" for s in segments)


def system_prompt(diarized: bool) -> str:
    return SYSTEM_PROMPT.replace(
        "{owner_rules}", OWNER_RULES_DIARIZED if diarized else OWNER_RULES_PLAIN)


def summarize(segments: list[Segment], *, utterances=None,
              client: LLMClient | None = None,
              model_label: str = "") -> MeetingRecord:
    """`utterances` (from naming.speaker_utterances) switches on speaker labels."""
    started = time.time()
    diarized = bool(utterances) and any(u.speaker for u in utterances)
    if diarized:
        from .naming import render_utterances
        text = render_utterances(utterances)
    else:
        text = _render(segments)
    prompt = system_prompt(diarized)
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
    # Ollama does not fail on an over-long prompt: it silently drops the start.
    # A record built from the last 40 minutes of a 90-minute meeting would look
    # complete. Say so instead.
    if client.backend == "ollama":
        from ..llm import OLLAMA_NUM_CTX, estimate_tokens
        need = estimate_tokens(text) + 3000          # + room for the JSON answer
        if need > OLLAMA_NUM_CTX:
            warnings.append(
                f"Transcript is ~{need:,} tokens with the answer, over the local "
                f"context of {OLLAMA_NUM_CTX:,}: the start of the meeting may be "
                f"ignored. Raise OLLAMA_NUM_CTX in .env.")
    data = None
    try:
        data = client.chat_json(prompt, user)
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
                data = client.chat_json(prompt, user)
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
