# CLAUDE.md — Agent Orientation

> Read this first. This file is the single source of truth for *fixed* facts and
> points to the other docs for everything still being decided. Do not invent
> decisions that are marked TBD — ask or check the relevant doc.

---

## 1. What this project is

An AI-powered **meeting assistant**: it takes a recorded English-language meeting
and turns it into (a) a raw transcript, (b) a domain-refined transcript, and
(c) a structured written record — meeting minutes, key decisions, and action
items — through a coordinated multi-model pipeline exposed via an interactive
interface.

This is a submission for **Inter IIT Tech Meet 15.0 — Bootcamp Phase 2, ML PS**
(IIT Guwahati Tech Board). It doubles as a selection round, so code clarity and
defensible design choices matter as much as a working demo.

- **Submission deadline:** 7 October 2026
- **Team size:** 1–3

## 2. Hard constraints (non-negotiable)

These come straight from the problem statement and the project owner. Never
violate them.

- **Everything must be free** — APIs, model weights, model sources, tools,
  hosting, every dependency. No paid services, no paid tiers, no trial credits
  that expire.
- **English-language audio** input only.
- **Two LLM roles MUST be distinct processing stages** — transcript refinement
  and meeting-documentation generation are separate models/calls, not one.
- **No hardcoded or prewritten outputs.** Every output must be produced by the
  pipeline running on the actual input. This disqualifies the submission if
  violated.
- **Never invent facts.** Where a task owner or deadline was not stated in the
  recording, output it as `unspecified` — never guessed. A proposal is not a
  decision; an unstated assignment is not a confirmed task.
- **Both output formats must agree.** The human-readable and machine-readable
  records must convey the same decisions and tasks.

## 3. The pipeline (fixed order)

```
audio → [1] transcribe → raw transcript
            → [2] refine (domain-aware) → refined transcript
                → [3] summarize → minutes / decisions / action items
                    → [4] verify (check claims against transcript) → final record
```

Stage 4 (verification) is our chosen differentiator, not a PS requirement — see
`docs/ARCHITECTURE.md` and `docs/PIPELINE.md`.

## 4. Required outputs (every run)

| Output | Notes |
| :-- | :-- |
| Raw transcript | STT result, before refinement |
| Refined transcript | after domain-aware correction; shown separately for comparison |
| Meeting minutes | concise summary + organized discussion points |
| Key decisions | structured list; empty list if none |
| Action items | structured list; owner/deadline only if stated, else `unspecified`; empty list if none |

All must be viewable in the UI and downloadable in both human-readable and
machine-readable form.

## 5. Evaluation rubric (100 pts) — what we're optimizing for

| Criterion | Pts |
| :-- | :-- |
| Speech transcription | 20 |
| Transcript refinement | 20 |
| Minutes & decisions | 25 |
| Action items | 15 |
| End-to-end application | 15 |
| Submission quality | 5 |

~80 of 100 points are about **faithfulness** (accuracy + not inventing). Design
every stage to protect those points first.

## 6. Decisions still open (do NOT assume)

| Decision | Where it will be recorded | Status |
| :-- | :-- | :-- |
| STT model | `docs/MODELS.md` | TBD |
| Refinement LLM | `docs/MODELS.md` | TBD |
| Summarization LLM | `docs/MODELS.md` | TBD |
| How models are sourced (self-host vs free API) | `docs/MODELS.md` | TBD |
| UI framework | `docs/UI_SPEC.md` | TBD |
| Orchestration approach | `docs/ARCHITECTURE.md` | TBD |
| Implementation language | this file §7 | Working assumption: Python (confirm) |
| Output schema details | `docs/DATA_CONTRACTS.md` | TBD |
| Prompt designs | `docs/PROMPTS.md` | TBD |

## 7. Conventions

- **Language / runtime:** TBD (working assumption: Python 3.x).
- **Style / formatting / linting:** TBD.
- **Testing:** TBD.
- **Secrets:** any keys via environment variables; never commit secrets. See
  `.env.example`.

## 8. Doc map

- `docs/PLANNING.md` — phases, timeline, task breakdown
- `docs/ARCHITECTURE.md` — system architecture & data flow
- `docs/PIPELINE.md` — per-stage specification
- `docs/MODELS.md` — model choices, roles, sourcing (also feeds the deliverable write-up)
- `docs/DATA_CONTRACTS.md` — schemas for transcripts, decisions, action items
- `docs/PROMPTS.md` — prompt / instruction designs for the LLM stages
- `docs/EVALUATION.md` — self-evaluation methodology (WER + quality)
- `docs/UI_SPEC.md` — interface requirements & framework
- `docs/ERROR_HANDLING.md` — edge cases & failure behavior
- `docs/DESIGN_DECISIONS.md` — running log of decisions and their rationale
- `docs/technical_description.md` — required deliverable write-up
