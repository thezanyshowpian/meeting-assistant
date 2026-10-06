# Pipeline — Per-Stage Contracts

Inputs, outputs and guarantees for each stage. Rationale is in
`DESIGN_DECISIONS.md`; models in `MODELS.md`; the full narrative in
`technical_description.md`. The orchestrator (`app/pipeline/orchestrator.py`)
runs the stages in this order and owns the failure policy.

| Stage | Module | On failure |
| :-- | :-- | :-- |
| 1 Transcribe | `transcribe.py`, `audio.py` | **fatal**: nothing downstream is meaningful |
| 1.5 Diarize | `diarize.py` | degrade: no speaker labels, warning |
| 2 Refine | `refine.py`, `verification.py`, `glossary.py` | degrade: raw transcript used, warning |
| 1.6 Name | `naming.py` | degrade: labels stay anonymous, warning |
| 3 Document | `summarize.py`, `llm.py` | Groq → local fallback; then degrade, warning |
| 4 Verify | `ground.py`, `naming.attribute_owners` | degrade: unverified record, warning |

---

## Stage 1 — Transcribe

- **In:** audio file path.
- **Out:** `Transcript`: text, segments, words (with timestamps and confidence),
  warnings.
- **Guarantees:**
  - Missing, empty or undecodable files raise errors with user-facing messages.
  - Short or silent audio produces warnings, never a refusal.
  - No speech means an empty transcript plus a warning, not a crash.

## Stage 1.5 — Diarize (optional: `DIARIZE=auto|on|off`)

- **In:** transcript and audio. **Out:** `words[].speaker = "Speaker N"`, speaker
  turns, the method used.
- **Guarantees:**
  - Every word gets a label.
  - Labels are anonymous and numbered by first appearance.
  - A known speaker count is honoured.

## Stage 2 — Refine

- **In:** transcript and glossary. **Out:** refined segments, plus every edit
  (applied, rejected with reasons, or vetoed).
- **Guarantees:**
  - Text not explicitly edited is byte-identical.
  - No change to digits or negation.
  - No insertions.
  - Only low-confidence spans may change.
  - At most max(5, words ÷ 20) edits.

## Stage 1.6 — Name (runs when diarization ran)

- **In:** refined segments, accepted edits, glossary.
- **Out:** `Utterance[]` (refined text per speaker per segment), and
  `NamingResult`: bindings with evidence, conflicts, and suspected diarization
  misses.
- **Guarantees:**
  - A name only comes with recorded evidence.
  - A tie means no name.
  - Nobody is named after someone they addressed.
  - One name goes to at most one voice.

## Stage 3 — Document

- **In:** refined segments, or speaker-labelled utterances.
- **Out:** `MeetingRecord`: summary, minutes, decisions, and action items, each
  with an evidence quote.
- **Guarantees:**
  - Owner and deadline are `null` unless stated.
  - A label owner is used only for a first-person commitment.
  - The model is never asked to map labels to names.

## Stage 4 — Verify

- **In:** the record, segments, utterances and naming.
- **Out:** `GroundingResult`: a pruned, timestamped record, plus the dropped and
  downgraded lists.
- **Guarantees:**
  - Unsupported items are dropped.
  - Unsupported owners and deadlines become `unspecified`.
  - Every owner carries `owner_source`: `stated`, `speaker` or `inferred`.
  - Every owner carries `owner_evidence`: the quote, timestamp, and naming
    evidence behind it.

## Outputs

`export.py` produces Markdown and JSON from the same `PipelineResult`.

- They include the raw, refined and speaker-labelled transcripts.
- They include the record, with owner provenance.
- They include the inferred names with evidence, and every warning.
