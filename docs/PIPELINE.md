# Pipeline — Per-Stage Specification

> One section per stage. Inputs/outputs and the fixed role are stated; the "how"
> is TBD until decided. See `docs/DESIGN_DECISIONS.md` for rationale.

## Stage 1 — Transcribe  [LOCKED]
- **Input:** uploaded audio file path
- **Output:** rich Transcript object — full text + segments `[{start,end,text,confidence}]` + word-level timestamps
- **Model:** Whisper **large-v3-turbo**, self-hosted, via **faster-whisper** (CPU, portable). Verified on Apple M4/24 GB: load 2.6s, RTF ≈0.45.
- **Config:** `language="en"`, `beam_size=5`, `vad_filter=True`, `word_timestamps=True`.
- **Decoding:** PyAV-direct → ffmpeg subprocess → library internal (first that works). We pass the model a float32 numpy array @16 kHz; we do NOT use faster-whisper's internal decoder (broken on PyAV ≥19).
- **Internal flow:**
  1. **Validate (layered, fail-honest):**
     - L1 exists + non-zero size → hard error on fail
     - L2 ffmpeg decode probe → hard error on fail (corrupt/unreadable/unsupported); yields the decoded 16 kHz mono samples
     - L3 post-decode sanity (duration floor, VAD energy) → **soft warning, proceed** if too short / near-silent
  2. **Decode:** owned by us (from L2), not delegated to the library
  3. **Clean:** VAD skips silence before the model
  4. **Transcribe:** run Whisper per config
  5. **Assemble:** build the rich Transcript object; flag (don't hide) low-confidence patches
  6. **Post-check:** empty / near-empty transcript → "no speech detected", handled gracefully
- **Does NOT do:** diarization (who-spoke), punctuation/casing fixes beyond Whisper's output, translation. (Diarization is a good "how I'd extend this" interview answer, not a build item.)
- **Deferred:** long-recording chunking → edge-hardening step.

## Stage 2 — Refine (domain-aware)
- **Input:** rich Transcript (+ domain glossary). Can use per-segment confidence to target shaky patches.
- **Output:** refined transcript
- **Model:** _TBD_
- **Must preserve:** names, numbers, negation, commitments, intended meaning
- **Glossary mechanism:** _TBD_
- **Prompt:** _see `docs/PROMPTS.md`_

## Stage 3 — Summarize
- **Input:** refined transcript
- **Output:** summary, minutes, key decisions, action items
- **Model:** _TBD (distinct from Stage 2)_
- **Rules:** proposal ≠ decision; unstated assignment ≠ task; owner/deadline only if stated
- **Prompt:** _see `docs/PROMPTS.md`_

## Stage 4 — Verify
- **Input:** generated decisions/tasks + transcript (with timestamps for anchoring)
- **Output:** verified record with unsupported claims removed/downgraded; each kept claim anchored to a transcript timestamp
- **Approach:** _TBD_
- **Prompt:** _see `docs/PROMPTS.md`_
