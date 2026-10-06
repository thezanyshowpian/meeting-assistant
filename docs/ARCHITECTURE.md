# Architecture

> System-level design and data flow. Stage responsibilities are fixed; the
> concrete components inside each stage are TBD (see `docs/MODELS.md`).

## High-level data flow
```
audio file
  └─[1] transcribe ──────────► raw transcript
          └─[2] refine ───────► refined transcript
                  └─[3] summarize ─► minutes / decisions / action items
                          └─[4] verify ─► final verified record ─► export + UI
```

## Component responsibilities

### [1] Transcription
_Role fixed: audio → raw transcript. Model: TBD._

### [2] Refinement (LLM A)
_Role fixed: correct domain-specific recognition errors, preserve meaning
(names, numbers, negation, commitments). Must be a distinct stage. Model: TBD._

### [3] Summarization (LLM B)
_Role fixed: refined transcript → summary, minutes, decisions, action items.
Must be a distinct stage from [2]. Model: TBD._

### [4] Verification (differentiator)
_Role fixed: check each generated decision/action item against the transcript;
drop or downgrade unsupported claims; enforce `unspecified` for missing
owner/deadline. Approach: TBD._

## Orchestration
_TBD — how stages are sequenced, where errors are caught, sync vs async,
long-recording handling (chunking strategy)._

## Boundaries & interfaces
_TBD — what data object passes between each stage (see `docs/DATA_CONTRACTS.md`)._
