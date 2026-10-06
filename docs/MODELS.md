# Models

Every model: its role, how it is sourced, why it was chosen, and what was
rejected. Free-only constraint applies to every entry.

| Stage | Model | Source | Size | Free? |
| :-- | :-- | :-- | :-- | :-- |
| 1 | Whisper large-v3-turbo | open weights via Hugging Face, run by faster-whisper | ~1.5 GB | yes (MIT) |
| 1.5 | ECAPA-TDNN `speechbrain/spkrec-ecapa-voxceleb` | Hugging Face, **ungated** | ~80 MB | yes (Apache-2.0) |
| 2 | qwen3:8b | Ollama, local | ~5 GB | yes (Apache-2.0) |
| 3 (local) | qwen3:14b | Ollama, local | ~9 GB | yes (Apache-2.0) |
| 3 (API) | openai/gpt-oss-120b | Groq free tier | — | yes (free tier, 6k tokens/min) |

Not models, by design: **Double Metaphone** (phonetic codes for the glossary
pass), the **clustering** in Stage 1.5, **speaker naming** (1.6) and
**grounding** (4). These are our own deterministic code.

---

## Stage 1 — Whisper large-v3-turbo

- **Runtime:** faster-whisper (CTranslate2), on CPU, `compute_type="auto"`.
  - mlx-whisper was rejected: it's Apple-only, and a judge's machine may not be a
    Mac.
  - The spike that compared them first measured the model *download* instead of
    the speed. We caught that and re-ran it.
- **Config:** `language="en"`, `beam_size=5`, `vad_filter=True`,
  `word_timestamps=True`.
- **Decoding:** PyAV directly, then ffmpeg, then the library's own decoder. The
  library's decoder is broken on PyAV ≥19.
- **Measured:** RTF ≈ 0.45 on an Apple M4; WER 2.52% / 3.73% on the two test
  meetings.
- **Rejected:**
  - wav2vec2: needs fine-tuning, no punctuation.
  - NeMo / Canary / Parakeet: need CUDA.
  - Vosk / Kaldi: weaker on jargon and numbers.
  - Hosted Whisper APIs: network and rate limits on demo day.

## Stage 1.5 — ECAPA-TDNN (speaker embeddings)

- **Role:** one 192-dimensional voice embedding per pause-split chunk. Turning
  embeddings into speakers (clustering, threshold, short-chunk assignment) is
  our code.
- **Why ECAPA:**
  - Strong on VoxCeleb.
  - CPU-friendly.
  - Ungated: it downloads without an account or token, so anyone can run the
    project.
- **Rejected for now — pyannote.audio 3.x:**
  - It is the strongest end-to-end open diarizer.
  - But its models are **gated**: each user must accept terms on Hugging Face and
    supply a token. That is free, but it breaks "clone and run".
  - Planned as an optional backend when `HF_TOKEN` is set. Not built.
- **Install:** `requirements-diarization.txt` (adds torch). It is optional, so
  the core install stays light.

## Stage 2 — qwen3:8b

- **Role:** context **veto** of glossary matches, plus **off-glossary**
  proposals. It returns an edit-list, never a transcript.
- **Why local:**
  - Stage 2 runs on every meeting.
  - It must be deterministic (`temperature=0`, `think=false`).
  - A small model suffices, because every proposal passes 9 deterministic gates.
- **Why distinct from Stage 3:** the problem statement requires it. It also
  separates "fix words" from "understand the meeting".

## Stage 3 — qwen3:14b (local) / openai/gpt-oss-120b (Groq)

- **Routing by size:**
  - Groq's free tier is 6,000 tokens per minute.
  - Under ~4,500 estimated tokens (~20 minutes of audio), Groq is used: ~4 s
    instead of ~66 s locally.
  - Above that, local is used. A failure on Groq also falls back to local.
  - With no key, everything is local.
- **Groq model IDs rotate.** `llama-3.3-70b-versatile`, widely documented, was
  not available on our account. `checks/check_groq.py` lists what yours has.
- **Two integration traps, both fixed:**
  - Cloudflare rejects urllib's default User-Agent (HTTP 403, error 1010).
  - Ollama's default context window is small and silently truncates long
    transcripts. `num_ctx` is now 16,384.
- **Measured (repeated trials, plain prompt):** Groq 45/45 over 5 runs; local
  9/9.
