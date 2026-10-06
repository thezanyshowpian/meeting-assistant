# AI Meeting Assistant

Turns a recorded meeting into an accurate transcript and a usable written record:
raw transcript → domain-corrected transcript → minutes, key decisions and action
items — through a coordinated multi-model pipeline with an interactive interface.

Built for **Inter IIT Tech Meet 15.0 — Bootcamp Phase 2 (ML PS)**, IIT Guwahati
Tech Board.

**Runs fully offline with no API keys.** Everything used is free.

---

## What it does

```
audio ──► [1] TRANSCRIBE ──► raw transcript
               (Whisper, local)
                   │
                   ├──► [2] REFINE ──────► corrected transcript
                   │        (glossary + LLM edit-list, verified)
                   │
                   └──► [3] DOCUMENT ────► summary · minutes · decisions · actions
                            (second, distinct LLM)
                                │
                                └──► [4] VERIFY ──► every claim checked
                                         against the transcript
```

The design principle throughout: **a model's output is treated as a proposal,
never as fact.** Stage 2 may not rewrite the transcript, only propose edits we
verify and apply. Stage 4 discards any decision or action item whose supporting
quote cannot be found in the transcript. Where the recording does not state an
owner or a deadline, the output says `unspecified` — it is never guessed.

---

## Requirements

- **Python 3.10+**
- **[Ollama](https://ollama.com)** — runs the local language models
- ~15 GB disk (Whisper ~1.5 GB, two Ollama models ~14 GB)
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

# 3. optional config (the app works without this)
cp .env.example .env
```

Whisper's weights (~1.5 GB) download automatically on the first run.

> **Restricted networks:** model weights come from Hugging Face. If your network
> blocks it, pre-download the weights on an unrestricted machine and copy
> `~/.cache/huggingface` across. We hit this exact block during development.

---

## Running

**Interactive interface** (the main deliverable):

```bash
streamlit run app/ui/streamlit_app.py
```

> If you see `ModuleNotFoundError`, Streamlit is running from a different Python
> than your virtualenv. Use `python -m streamlit run app/ui/streamlit_app.py`,
> which forces the active interpreter.

Upload a recording, press **Process recording**, and you get raw and refined
transcripts side by side, the meeting record, every correction applied *and
rejected*, and downloads in both formats.

**Command line**, end to end:

```bash
python checks/check_e2e.py --audio data/sample_meeting.wav
```

---

## Models

| Stage | Model | Where it runs | Role |
| :-- | :-- | :-- | :-- |
| 1 Transcribe | **Whisper large-v3-turbo** (faster-whisper) | local | audio → raw transcript with word-level timestamps and confidence |
| 2 Refine | **qwen3:8b** (Ollama) | local | vetoes context-wrong glossary matches; proposes corrections for terms the glossary lacks |
| 3 Document | **qwen3:14b** (Ollama) or **openai/gpt-oss-120b** (Groq) | local / API | refined transcript → summary, minutes, decisions, action items |

Stages 2 and 3 use **distinct models**, as the problem statement requires.

**Stage 3 routing.** Groq's free tier allows **6,000 tokens per minute**, and a
30-minute meeting is ~6,800 tokens in a single call — so the API cannot serve
long recordings, and chunking cannot rescue it (you would have to wait a minute
between chunks). Stage 3 therefore routes **by transcript size**: Groq below
~4,500 estimated tokens (~20 minutes of audio), local above it, and local again
if the remote call fails for any reason. **With no API key set, everything runs
locally** — which is why no key is needed to run this project.

---

## Outputs

Every run produces, viewable in the UI and downloadable:

| Output | Format |
| :-- | :-- |
| Raw transcript | text |
| Refined transcript | text |
| Meeting minutes, decisions, action items | **Markdown** (human) + **JSON** (machine) |

Both formats are generated from the same object, so they cannot disagree.
Unstated owners and deadlines render as `unspecified` in both.

---

## Verifying it works

```bash
python checks/check_stage1.py --audio data/sample_meeting.wav   # 14 checks
python checks/check_stage2.py                                   # 25 checks
python checks/check_stage3.py                                   # 15 checks
python checks/check_groq.py                                     # API, if configured
python eval/evaluate.py                                         # full measured evaluation
```

`eval/evaluate.py` runs the pipeline against a meeting whose correct answers are
known, and reports Word Error Rate, whether refinement helped or hurt, and
whether the system invented anything.

Regenerate the sample recording (macOS only, uses `say`):

```bash
python scripts/make_sample_meeting.py
```

---

## Measured results

On the included sample meeting (86s, 223 words, known ground truth):

| Metric | Result |
| :-- | :-- |
| Raw transcript WER | **1.68%** |
| Refinement effect on WER | no degradation |
| Jargon recovery under simulated ASR errors | **5/5 (100%)** |
| Parked proposal reported as a decision | **no** |
| Unstated owners/deadlines invented | **none (3/3 correct)** |
| Stage 1 speed | RTF ≈ 0.45 (faster than real time) |

---

## Known limitations

**No speaker diarization.** The transcript has no speaker labels, so a
first-person commitment ("*I'll* benchmark that") cannot be attributed and comes
back as `unspecified`. An owner named aloud ("*Sam*, can you…") is captured
correctly. We chose to report unspecified rather than guess from context.

**Long recordings.** Stage 3 sends the transcript in one call. `OLLAMA_NUM_CTX`
is set to 16,384 (~2 hours of speech); beyond that, chunking would be needed.

**Acoustic re-verification** is designed but not implemented — see
`docs/DESIGN_DECISIONS.md`, including the circularity pitfall we identified in
the naive version.

---

## Repository layout

```
app/
├── llm.py              model clients + Stage 3 size routing
├── schemas.py          typed contracts (owner/deadline optional by design)
├── export.py           Markdown + JSON from one object
├── errors.py           errors carrying user-facing messages
└── pipeline/
    ├── audio.py        validation + decoding
    ├── transcribe.py   Stage 1
    ├── glossary.py     phonetic matching
    ├── verification.py Stage 2's gates
    ├── refine.py       Stage 2
    ├── summarize.py    Stage 3
    ├── ground.py       Stage 4
    └── orchestrator.py runs 1→4, owns failure policy
checks/                 behavioural test harnesses
eval/                   WER + ground-truth evaluation
data/                   sample meeting, script, ground truth
docs/                   design decisions, technical description
```
