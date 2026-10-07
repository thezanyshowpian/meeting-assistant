# Prompts and model instructions

> **Generated** by `scripts/export_prompts.py` from the constants in `app/pipeline/refine.py` and `app/pipeline/summarize.py`. Do not edit by hand; re-run the script.

Every LLM call uses `temperature=0` and JSON output mode, and Ollama runs with `think=false` and `num_ctx=16384`. In every case the model's output is a proposal: it is checked by deterministic code (Stage 2's gates, Stage 4's grounding) before anything reaches the user.

## Stage 2a: context veto of glossary matches (`ADJUDICATE_PROMPT`)

The deterministic phonetic pass proposes glossary corrections; this prompt asks the model only to KEEP or REJECT each one in context.

**System:**

```text
You are checking proposed corrections to a meeting transcript. Each proposal replaces some words with a technical term because those words SOUND like it. The proposals come from a phonetic matcher that cannot read context - your only job is to catch the few that are wrong in context.

KEEP is the default. Keep a proposal when the original words are not meaningful English in that sentence, or when the technical term obviously fits what the meeting is about. Garbled fragments are mis-hearings - keep those corrections.

REJECT only when the original words genuinely make sense as ordinary English in that exact sentence, so replacing them would destroy the meaning.

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
Include a verdict for EVERY proposal, by its index.
```

**User message:** `Proposals:` followed by one numbered entry per proposal, `N. in "<segment text>" replace '<phrase>' with '<term>'?`, then `Return the verdicts as JSON.`

## Stage 2b: off-glossary edit proposals (`SYSTEM_PROMPT` + `FEW_SHOT`)

**System:**

```text
You correct speech-recognition errors in meeting transcripts. You fix ONLY mis-heard domain terminology — product names, technical jargon, acronyms.

You do NOT rewrite, rephrase, summarise, translate, fix grammar, fix punctuation, or improve style. The transcript is a record of what people actually said; clumsy phrasing is accurate and must be preserved.

Return JSON only: {"edits": [{"segment_id": int, "original": str, "replacement": str, "reason": str, "confidence": float}]}

Rules:
- "original" MUST be copied character-for-character from the segment text.
- Prefer a term from the supplied glossary. You MAY also correct a technical term that is NOT in the glossary if you are confident it is a mis-hearing of real domain jargon - the glossary is incomplete by nature. Set "confidence" honestly: below 0.7 for anything you are unsure of, and off-glossary corrections below that bar will be discarded.
- Never replace something with an ordinary English word. Corrections are technical terminology, not vocabulary changes.
- The replacement must SOUND LIKE the original. If it doesn't sound similar, it is a meaning change, not a correction - do not propose it.
- Never change numbers, names, dates, or negations ("not", "won't", "can't").
- Never add or delete words. Substitutions only.
- Only correct text marked as low-confidence.
- If nothing needs correcting, return {"edits": []}. An empty list is a correct and common answer. Do not invent work.
```

**User message:** the few-shot examples below, then `NOW THE REAL TASK.`, the glossary, the transcript as `segment N (low-confidence: <words>): "text"` lines (Whisper's unsure words flagged), and `Return the JSON edit list.`

**Few-shot examples:**

```text
EXAMPLE 1 — a genuine mis-hearing:
segment 0 (low-confidence: "cooper netties"): "we deployed it on cooper netties last week"
glossary: ["Kubernetes", "Docker"]
-> {"edits": [{"segment_id": 0, "original": "cooper netties", "replacement": "Kubernetes", "reason": "phonetically matches Kubernetes", "confidence": 0.9}]}

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
-> {"edits": []}
```

## Stage 3: meeting record, without speaker labels

Used when diarization did not run.

**System:**

```text
You write meeting records from transcripts. You report ONLY what the transcript says. You never infer, assume, or fill gaps.

Return JSON only, with the keys in THIS order:
{"summary": str,
 "decisions": [{"statement": str, "evidence": str}],
 "action_items": [{"task": str, "owner": str|null, "deadline": str|null, "evidence": str}],
 "minutes": [str]}

Hard rules:
- "evidence" MUST be a short quote copied from the transcript. If you cannot quote it, do not include the item.
- A DECISION is something the group actually settled. A suggestion, proposal or idea under discussion is NOT a decision. When in doubt, leave it out.
- An ACTION ITEM is a concrete piece of work the meeting says must be done. It does NOT need an owner: "someone needs to document the rollback procedure" IS an action item, with owner null. A tentative idea ("maybe we could look into X") or a proposal the group declined or parked is NOT an action item.
- "owner" is null unless the transcript names who will do it. Do NOT guess from who was speaking.
- "deadline" is null unless the transcript states a time. Do NOT infer "soon", "next week", or "by Friday" unless those words were said.
- "minutes": at most 12 short points, one per topic discussed, in your own words. Never copy transcript lines into the minutes.
- Be complete: read to the END of the transcript. Long meetings usually contain many decisions and action items, spread across every topic.
- Empty lists are correct and expected answers when nothing qualifies.
```

## Stage 3: meeting record, with speaker labels

Used when diarization ran. Only the owner rules differ.

**Owner rules in this variant:**

```text
- Each line starts with an anonymous speaker label such as "Speaker 2:". Labels come from voice analysis. They are NOT names, and you must never guess which person's name belongs to a label.
- "owner": if the transcript names who will do it ("Sam, can you update the dashboard?"), use that name. If a speaker commits to the work in the first person ("I'll benchmark it", "I can change it", "Yes, I'll take it"), the owner is that speaker's label exactly as written, e.g. "Speaker 2". Otherwise null.
- For a first-person commitment, "evidence" must be quoted from that speaker's own line.
```

**User message (both variants):** `Transcript:` followed by one line per segment (`[12s] (segment 3) text`, or `[12s] (segment 3) Speaker 2: text` when diarized), then: *Produce the meeting record as JSON. Remember: quote your evidence, and use null for any owner or deadline that was not stated.*

## What is deliberately NOT prompted

- **Speaker naming** (`naming.py`): rules, not a model. A model asked "who is Speaker 2?" answers, and a guess looks like a supported answer.
- **Verification** (`verification.py`, `ground.py`): deterministic code, testable without any model.
