# AI Meeting Assistant

Turns a recorded meeting into an accurate transcript and a usable written record:
raw transcript → domain-corrected transcript → who spoke → minutes, key decisions
and action items, through a coordinated multi-model pipeline with an interactive
interface.

Built for **Inter IIT Tech Meet 15.0 — Bootcamp Phase 2 (ML PS)**, IIT Guwahati
Tech Board.

**Everything is free, and the whole pipeline runs offline with no API keys.** An
optional free Groq key speeds up Stage 3 for short meetings.

---

## What it does

```
audio ──► [1] TRANSCRIBE ─────────► raw transcript (words + timestamps + confidence)
               Whisper, local          │
                                       ├──► [1.5] DIARIZE ──► "Speaker 1/2/3" per word
                                       │         ECAPA-TDNN + our own clustering   (optional)
                                       │
                                       ├──► [2] REFINE ─────► corrected transcript
                                       │        glossary + LLM edit-list, 9 verification gates
                                       │
                                       ├──► [1.6] NAME ─────► "Speaker 2 is Arjun" — only with evidence
                                       │         rules, not a model                (optional)
                                       │
                                       └──► [3] DOCUMENT ───► summary · minutes · decisions · actions
                                                second, distinct LLM
                                                    │
                                                    └──► [4] VERIFY ──► every claim checked against
                                                             the transcript (and the voices)
```

The design principle throughout: **a model's output is a proposal, never a fact.**

- Stage 2 may not rewrite the transcript. It proposes edits, which we verify and apply.
- Stage 3 never decides who a voice belongs to. It sees anonymous labels only.
- Stage 4 discards any decision or action item whose supporting quote cannot be
  found in the transcript, downgrades any owner the recording does not support,
  and merges a task the model listed twice.
- Where the recording does not state an owner or a deadline, the output says
  `unspecified`. It is never guessed.

---

## Requirements

- **Python 3.10+**
- **[Ollama](https://ollama.com)** runs the local language models
- ~15 GB disk (Whisper ~1.5 GB, two Ollama models ~14 GB; +~1 GB for the optional speaker stage)
- No GPU required. Developed and measured on an Apple M4 / 24 GB, CPU only.

---

## Setup

```bash
# 1. dependencies
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. local models
ollama serve                       # leave running in its own terminal
ollama pull qwen3:8b               # Stage 2  (~5 GB)
ollama pull qwen3:14b              # Stage 3  (~9 GB)

# 3. OPTIONAL — speaker diarization + naming (adds torch; ~1 GB)
pip install -r requirements-diarization.txt

# 4. OPTIONAL — config (the app works without this)
cp .env.example .env               # e.g. a free Groq key for faster Stage 3
```

Whisper (~1.5 GB) and the speaker model (~80 MB) download automatically on first use.

> **Restricted networks:** model weights come from Hugging Face. If your network
> blocks it, pre-download on an unrestricted machine and copy
> `~/.cache/huggingface` (and `~/.cache/meeting-assistant`) across. We hit this
> exact block during development.

---

## Running

**Interactive interface** (the main deliverable):

```bash
python -m streamlit run app/ui/streamlit_app.py
```

(`python -m` forces Streamlit to use your virtualenv's interpreter. Plain
`streamlit run` can pick up a different Python and fail with `ModuleNotFoundError`.)

Upload a recording and press **Process recording**. The tabs show:

- **Transcripts:** raw and refined, side by side.
- **Meeting record:** summary, minutes, decisions, and action items. Each owner
  shows *how* it is known.
- **Speakers:** the speaker-labelled transcript, each inferred name with its
  evidence, and any speaker label the conversation contradicts.
- **Corrections:** every edit, applied *and rejected*, with the reason.
- **Downloads:** both formats.

**Command line**, end to end, writing the output files:

```bash
python checks/check_e2e.py --audio data/sample_meeting.wav --out data/sample_outputs
```

---

## Models

| Stage | Model | Where | Role |
| :-- | :-- | :-- | :-- |
| 1 Transcribe | **Whisper large-v3-turbo** (faster-whisper) | local | audio → transcript with word timestamps and confidence |
| 1.5 Diarize | **ECAPA-TDNN** (SpeechBrain, VoxCeleb) | local | one 192-d voice embedding per speech chunk; clustering is our own code |
| 2 Refine | **qwen3:8b** (Ollama) | local | vetoes context-wrong glossary matches; proposes corrections the glossary lacks |
| 3 Document | **qwen3:14b** (Ollama) or **openai/gpt-oss-120b** (Groq, free) | local / API | refined, speaker-labelled transcript → summary, minutes, decisions, actions |

Stages 2 and 3 use **distinct models**, as the problem statement requires.
Stages 1.6 (naming) and 4 (verification) are deliberately **not** models. They
are deterministic rules, so their behaviour can be tested exhaustively.

**Stage 3 routing.** Groq's free tier allows **6,000 tokens per minute**, and a
30-minute meeting is ~6,800 tokens in one call. Chunking can't fix this (you'd
wait a minute between chunks). So Stage 3 routes **by transcript size**:

- Groq below ~4,500 estimated tokens (~20 minutes of audio).
- Local above that.
- Local again if the remote call fails for any reason.
- With no API key, everything is local.

---

## Who said what, and who owns what

Diarization tells you *which stretches of audio are the same voice*. It cannot
tell you anyone's name. We bind a name to a voice only where the conversation
itself provides evidence:

- **Addressed, then answered.** "*Arjun*, where are we?" followed immediately by a
  different voice means that voice is Arjun.
- **Self-introduction.** "Hi, I'm Arjun."
- **Nobody addresses themselves.** Whoever says "Arjun, …" is *not* Arjun.

Ties and contradictions leave a speaker **unnamed**. In the sample meeting nobody
ever says Priya's name, so her voice correctly stays "Speaker 1".

Every action-item owner records how it is known:

| Shown as | Meaning |
| :-- | :-- |
| `Sam` | **stated**: the name is spoken next to the task ("Sam, can you update…") |
| `Speaker 3 (voice only)` | that voice committed in the first person ("I'll take it"), but no name is known |
| `Arjun (inferred)` | that voice committed, **and** was linked to the name by evidence, which is shown |
| `unspecified` | nothing in the recording supports an owner |

**The words also check the voices.** If a voice asks Sam a question and the
"Yes, I'll take it" that follows carries the *same* voice label, diarization
almost certainly missed a speaker change. We found exactly this on our own
sample. The line is flagged, and its voice label is never used as an owner. If
the flagged reply accepts a request about the same task, the owner is the person
the request named ("*Sam*, can you update the Grafana dashboard?"), tagged
`stated` and citing that request.

---

## Outputs

| Output | Format |
| :-- | :-- |
| Raw transcript, refined transcript, speaker-labelled transcript | Markdown |
| Summary, minutes, decisions, action items (with owner provenance) | **Markdown** (human) + **JSON** (machine) |

Both formats are generated from the same object, so they cannot disagree.
Unstated owners and deadlines are `unspecified` in both. Examples from the
sample meeting are in `data/sample_outputs/`.

---

## Verifying it works

```bash
python checks/check_stage1.py --audio data/sample_meeting.wav   # 14 checks
python checks/check_stage2.py                                   # 29 checks, no model
python checks/check_stage3.py                                   # 21 checks, no model
python checks/check_diarization.py                              # 12 checks, no model
python checks/check_naming.py                                   # 25 checks, no model
python checks/check_groq.py                                     # API, if configured

python eval/evaluate.py                     # sample meeting, full measured evaluation
python eval/evaluate.py --fixture heldout   # 4-speaker meeting never used for tuning
python eval/stage3_trials.py                # Stage 3 repeated trials per backend
python eval/tune_diarization.py             # threshold sweep, DER per setting
```

The evaluation reports:

- **Transcription:** word error rate, and whether refinement helped or hurt.
- **Diarization:** diarization error rate (DER).
- **Naming:** whether names went to the right voices.
- **Invention:** whether the system invented any decision, owner or deadline.

Regenerate the test recordings (macOS only, uses `say`):

```bash
python scripts/make_sample_meeting.py
python scripts/make_sample_meeting.py --name heldout
```

---

## Measured results

Two synthetic meetings with known ground truth, including the speaker timeline.
The second was used **only** for final reporting, never for tuning.

| | Sample (3 speakers, 238 words) | Held-out (4 speakers, 134 words) |
| :-- | :-- | :-- |
| Transcript WER | **2.52%** | **3.73%** |
| Refinement effect on WER | no degradation | no degradation |
| Jargon recovery under simulated ASR errors | **5/5**, negative control held | — |
| Speakers found | 3/3 | 4/4 |
| Diarization error rate | 7.97% | 4.71% (zero speaker confusion) |
| Names bound to the right voice | Arjun ✓; Sam not named (see limitations) | Tom ✓ Dev ✓ Lena ✓ |
| **Wrong names given** | **0** | **0** |
| Never-addressed speaker left anonymous | Priya ✓ | Meera ✓ |
| Owners from first-person commitments | "I'll benchmark" → **Arjun (inferred)** | Tom, Dev, Lena (all inferred, all correct) |
| Full evaluation (latest run) | **24/25**: the miss is Sam's voice not being named | **18/19**: the miss is Lena's "next week" deadline |

**Stage 3 repeated trials** (same transcript, model the only variable; plain prompt):

- Groq gpt-oss-120b passed **45/45** checks over 5 runs.
- Local qwen3:14b passed **9/9** over 1 run.
- Groq's wording varies between runs; its substance did not.

Stage 1 runs faster than real time (RTF ≈ 0.45 on CPU).

---

## Known limitations

- **Similar voices can be merged.** In the sample, Sam's one short reply was
  clustered with Priya (two similar synthetic female voices). The cross-check
  flags it, and naming refuses to guess, so Sam's *voice* stays unnamed. His task
  is still attributed to him, because Priya's request named him out loud. A lost
  name, not a wrong one.
- **Naming needs evidence in the words.** A speaker who is never addressed by
  name and never introduces themselves stays anonymous, by design.
- **Very short approvals.** "Do that." carries no content of its own. Stage 4
  now accepts such a quote only if it appears verbatim **and** the decision's
  content is in the preceding context. Before this fix our own verifier dropped
  the held-out lazy-loading decision, which the model had reported correctly.
  The prompt was **not** tuned on the held-out meeting.
- **A deadline stated in a request can be lost.** "Lena, could you run another
  usability test *next week*?" → "Sure, I'll book five participants." The model
  sometimes builds the task from the reply and leaves out the deadline (2 of 3
  held-out runs). This is not invented, just missed.
- **Stage 3 is not deterministic on the hosted model.** Wording varies between
  runs and, occasionally, a borderline item does. That's why the backends are
  compared with repeated trials, not single runs.
- **Synthetic audio.** Both test meetings are text-to-speech: clean, no overlap,
  no crosstalk. Real meetings will have higher WER and DER. The architecture
  (verification, grounding, conservative naming) is what is designed to hold up.
- **Long recordings.** Stage 3 sends the transcript in one call. `OLLAMA_NUM_CTX`
  is 16,384 (~2 hours of speech); beyond that, chunking would be needed.
- **Acoustic re-verification** of Stage 2 edits is designed but not built. See
  `docs/DESIGN_DECISIONS.md` for the circularity pitfall in the naive version and
  the decoy-control design that fixes it.

---

## Repository layout

```
app/
├── llm.py              model clients + Stage 3 size routing
├── schemas.py          typed contracts (owner/deadline optional; owner provenance)
├── export.py           Markdown + JSON from one object
├── errors.py           errors carrying user-facing messages
├── ui/streamlit_app.py the interface
└── pipeline/
    ├── audio.py        validation + decoding
    ├── transcribe.py   Stage 1
    ├── diarize.py      Stage 1.5 — pause-split chunks, ECAPA, average-linkage clustering
    ├── glossary.py     phonetic matching
    ├── verification.py Stage 2's gates
    ├── refine.py       Stage 2
    ├── naming.py       Stage 1.6 — speaker-split transcript, name evidence, cross-check
    ├── summarize.py    Stage 3
    ├── ground.py       Stage 4 — evidence, owners, voices
    └── orchestrator.py runs the stages in order, owns failure policy
checks/                 behavioural test harnesses
eval/                   WER, DER, ground-truth evaluation, trials, tuning
scripts/                test-meeting generator, prompt export
data/                   two test meetings, scripts, ground truth, sample outputs
docs/                   technical description, design decisions, models, pipeline, prompts
```

**Docs:**

- `docs/technical_description.md`: the required submission artifact.
- `docs/DESIGN_DECISIONS.md`: every significant decision, alternative, and
  mistake, with measurements.
- `docs/PROMPTS.md`: every prompt sent to a model, generated from the code by
  `scripts/export_prompts.py`, so it can't drift.
- `docs/MODELS.md`: model sourcing.
- `docs/PIPELINE.md`: per-stage contracts.
