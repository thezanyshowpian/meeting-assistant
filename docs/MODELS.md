# Models

> Records the concrete model choices, each model's role, and how it's sourced.
> This doc feeds directly into the required technical description deliverable.
> Free-only constraint applies to every entry.

## Stage 1 — Speech-to-text  [LOCKED]
- **Model:** Whisper **large-v3-turbo** (OpenAI open weights).
- **Runtime:** **faster-whisper** (CTranslate2) — CPU, single portable backend. mlx-whisper rejected (see DESIGN_DECISIONS).
- **Source:** self-hosted locally (no hosted API). Weights pulled from the model hub on first run, then cached (~1.5 GB).
- **Role:** audio → raw transcript (rich: text + segments + word-level timestamps + per-segment confidence).
- **Config:** `language="en"` forced, `beam_size=5`, `vad_filter=True`, `word_timestamps=True`.
- **Audio decoding:** **we decode ourselves via PyAV-direct** and pass a float32 numpy array @16 kHz. faster-whisper's internal decoder is BROKEN on PyAV ≥19 (`av.open()` dropped `metadata_errors`), and ffmpeg is not assumed present. Fallback order: PyAV-direct → ffmpeg subprocess → library internal.
- **Measured:** load 2.6s (cached); RTF ≈0.45 on dense speech (Apple M4). Transcript verbatim-correct on the test clip; `avg_logprob −0.085`.
- **Why:** free, robust without fine-tuning, runs anywhere (any judge's machine), no network/rate-limit dependency at demo time.
- **Free?:** yes (local weights + open-source runtime).
- **Dependency pinning (important):** pin `faster-whisper` and `av` in `requirements.txt`. An unpinned install produced a confusing crash; judges hitting that would cost us "setup instructions" points.

## Stage 2 — Refinement LLM
- **Model:** _TBD_
- **Source:** _TBD_
- **Why:** _TBD_
- **Free?:** _must be yes_

## Stage 3 — Summarization LLM
- **Model:** _TBD_
- **Source:** _TBD_
- **Why:** _TBD_
- **Free?:** _must be yes_

## Stage 4 — Verification (if a model is used)
- **Model:** _TBD_
- **Source:** _TBD_
- **Why:** _TBD_

## Data flow between models
_TBD — what each model receives and emits. (Stage 1 emits the rich Transcript.)_

## Candidates considered but not used
- **STT:** wav2vec2 (needs domain fine-tuning; no built-in punctuation), NVIDIA NeMo/Canary/Parakeet (wants CUDA, not available on the Mac), Vosk/Kaldi (weaker on terminology/numbers), hosted free-tier Whisper APIs e.g. Groq (network/rate-limit/free-tier risk on demo day). Keep for the interview + write-up.
