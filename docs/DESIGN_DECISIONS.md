# Design Decisions (running log)

> One entry per significant decision, newest first. Captures *why*, not just
> *what* — this is interview ammunition and keeps agents from re-litigating
> settled choices.

<!--
Template for each entry:

## [YYYY-MM-DD] <decision title>
- **Decision:** what we chose
- **Alternatives considered:** what else, and why not
- **Rationale:** why this, tied to constraints / rubric
- **Status:** proposed / accepted / superseded
-->

## [2026-10-05] Stage 2 verification — LOCKED: allowlist gates, not a blocklist

**The problem with the first draft.** The original rules (reject unanchored edits, guard numbers/negations, reject pure punctuation changes) are a **blocklist** — they deny *enumerated* bad patterns and therefore can never be complete. "We might ship Friday" → "We **will** ship Friday" passes every one of them. A blocklist is bounded by the author's imagination; an allowlist is bounded by construction. We switch to permitting only known-safe edits.

**Typed edit policy** (the three edit operations get different policies, not one shared rule):
- **Insertion → DENY by default.** Adding words never spoken is definitionally not terminology correction. Forbid the category rather than police each case.
- **Deletion → DENY by default,** with one narrow exception: collapsing an obvious stutter ("the the" → "the"). Any other deletion removes something that *was* said and we cannot verify it should go.
- **Substitution → the only broadly permitted type,** and it must still clear every gate below.

**Gates on surviving substitutions:**
1. **Phonetic similarity required** (Double Metaphone / similar). A real ASR error *sounds like* the truth ("cooper netties" → "Kubernetes"); semantic drift does not ("might" → "will" fails instantly). Semantic drift doesn't rhyme — this single rule kills most of it.
2. **Replacement must come from the glossary** (or a close variant). Bounds the output vocabulary so the model structurally cannot invent new content. Matches the PS framing ("correct transcription errors in domain-specific terms").
3. **Span + budget limits.** Each edit capped at a few words; total edits capped relative to transcript length (~1 per 50 words). An avalanche of individually-plausible edits is detectable in aggregate.
4. **Confidence gating.** Edits may ONLY be anchored to spans Whisper was *unsure* about, using the per-word probabilities Stage 1 already stores. If a word was transcribed at 0.98 confidence, the LLM has no business "correcting" it. The model cannot touch confidently-heard text — where meaning-changing edits would do the most damage.
5. **Acoustic re-verification.** For each proposed substitution we know its exact time span (Stage 1 word timestamps). Re-run Whisper on *just that slice*, biased with the glossary term via `initial_prompt`, and accept only if the audio actually supports the correction. This replaces "an LLM suggested it and the text looked plausible" with **"the audio itself supports this."** Direct payoff from the Stage 1 decision to keep rich timestamps — one choice enabling another.

**Residual honesty (not optional).** Layered gates make bad edits rare and bounded, NOT impossible. So: every applied edit is surfaced in the UI with its reason and evidence, and the raw transcript is always preserved beside the refined one, so a human can see and undo any change. We reduce error; we never claim to have eliminated it.

**Edit-list schema (approved):** `{segment_id, original, replacement, edit_type, reason, confidence}`. `segment_id` is included so the same wrong phrase occurring twice can be disambiguated and anchored precisely.

**Glossary:** ships with a sample AND is user-uploadable through the UI (both).

**Risk flagged:** acoustic re-verification is the most ambitious piece and the deadline is 2026-10-07. Build it behind a flag so the stage degrades gracefully to gates 1–4 if it proves too slow or noisy.

## [2026-10-05] LLM sourcing — LOCKED: hybrid (local Stage 2, free API Stage 3)

- **Decision:** Stage 2 (refinement) runs a **local** model via Ollama. Stage 3 (minutes/decisions/actions) uses a **strong free-tier API** model. This also satisfies the PS requirement that the two LLM roles be distinct models.
- **Rationale:** Stage 2 wants determinism, auditability and offline reliability, and its output is a *short* edit-list — exactly the workload a local model handles well. Stage 3 is the hardest reasoning task and carries 40 of the 100 points, so it gets the strongest free model available.
- **Hardware reality (corrects the published "24GB VRAM" guidance):** the M4 Air has 24 GB *unified* memory and is *fanless*. macOS+apps ~6–8 GB, Whisper ~2–3 GB when loaded → ~12–14 GB usable. So Qwen3 30B (~18 GB @ Q4) is NOT viable alongside Stage 1; **14B @ Q4 (~9 GB) or 12B (~8 GB) is the ceiling**, and sustained inference will thermally throttle on this chassis.
- **Architecture synergy:** the edit-list decision is what makes local Stage 2 practical — we need only a short JSON list of corrections, not thousands of generated transcript tokens. Generation length is where local models are slow; prefill of the long transcript input is far cheaper per token.
- **Free-tier landscape (checked 2026-10-05):** Groq (30 req/min, 1,000/day, Llama 3.3 70B, fast), Cerebras (30/min, ~1M tok/day, Llama 3.3 70B), Google AI Studio (5–15/min, up to 1,500/day). OpenRouter free is 20/min but only **50/day** — too tight for development. Documented risk: free tiers can tighten limits or go down without warning, so Stage 3 needs a graceful-failure path.
- **Known cost accepted:** judges running the app themselves will need their own free API key for Stage 3 — must be documented clearly in the README.
- **Open:** which local model (Qwen3 14B vs 8B vs Gemma 3 12B), which free API provider, glossary format, edit-verification rules, chunking.

## [2026-10-05] Stage 2 architecture — LOCKED: edit-list + glossary pre-pass

- **Decision:** Stage 2 does NOT emit a rewritten transcript. Pipeline is:
  1. **Deterministic glossary pre-pass** — phonetic (Double Metaphone) + edit-distance matching against a domain glossary catches the known-jargon ASR errors with zero hallucination risk.
  2. **LLM edit-list pass** — the model outputs only structured corrections `{original_text, replacement, reason/evidence}` for the contextual cases the glossary can't reach.
  3. **We apply the edits programmatically** with exact string matching + verification (reject any edit whose anchor text isn't present; guard numbers, names, negations).
- **Rationale (the mechanism, not a preference):** generation is autoregressive — if the model emits the transcript, every one of thousands of tokens is re-rolled, so meaning drift can appear anywhere and prompting can only *reduce* the rate, never bound it. With an edit-list, everything not on the list is byte-identical **by construction**: drift becomes impossible rather than discouraged. Failure degrades to a *missed* correction (benign) instead of a *corrupted* transcript (catastrophic and invisible).
- **Secondary benefits:** short output (fast, no context pressure); fully auditable — we can show exactly what changed and why, which is both UI material and rubric evidence; machine-verifiable before applying.
- **Alternatives rejected:** full rewrite (simplest, but unbounded invisible drift — weakest on the 20 faithfulness points); pure deterministic glossary (no hallucination risk but context-blind, and the PS *requires* an LLM in this stage).
- **Also decided:** **temperature = 0 / greedy decoding.** Correction is not a creative task; this also makes runs reproducible, so a judge re-running the demo gets identical output.
- **Prompting note:** few-shot examples will include *negative* cases (correct output == input, including clumsy-but-accurate text the model would be tempted to polish). Instruction-tuned models are trained by RLHF to "improve" text; showing restraint conditions better than describing it.
- **Status:** accepted. Open: model choice + sourcing, quantization, chunking, glossary format, edit-verification rules.

## [2026-10-05] Stage 1 — IMPLEMENTED and VERIFIED (14/14 checks pass)

- **Code:** `app/errors.py`, `app/schemas.py`, `app/pipeline/audio.py`, `app/pipeline/transcribe.py`, `checks/check_stage1.py`, pinned `requirements.txt`.
- **Verified on the real machine (Apple M4):** 14/14 checks pass — hard failures (missing / empty / directory / non-audio / corrupt) raise the right error types with user-facing messages; soft-warning cases (silence, too-short) proceed rather than being discarded; silence yields an empty transcript plus a "no speech detected" warning instead of a crash; real speech returns word timestamps AND confidence populated; output is JSON-serialisable. Live run: `RTF=0.48`, `decode=pyav`.
- **Confirmed in practice:** on the Mac (PyAV 19) the decode path used is `pyav` — the faster-whisper internal decoder would have failed, exactly as predicted. Owning the decode is doing real work, not ceremony.
- **Implementation notes worth defending in the interview:**
  - `analyze()` *cannot* raise — it returns warnings only, so the soft-warning stance is enforced by the function's signature rather than by discipline at each call site.
  - Decode order is PyAV → ffmpeg → library (reversed from the spike: our own path is primary, the known-broken library path is last resort).
  - The model loads **lazily**, so validation/decoding/schemas are testable without 1.5 GB of weights — and the UI can reject a bad upload instantly without ever loading the model.
  - Plain dataclasses, not pydantic: Stage 1's shapes are simple and stable. Pydantic may be introduced at Stage 3 where validating *model-generated* structured output actually earns the dependency.
- **Status:** Stage 1 complete. Deferred to edge-hardening: long-recording chunking.

## [2026-10-04] Stage 1 runtime — LOCKED: faster-whisper only

- **Decision:** faster-whisper as the single STT runtime. mlx-whisper rejected.
- **Measured (3.7 min audio, large-v3-turbo, Apple M4 / 24 GB), total wall-clock:**

  | runtime | total | RTF | extrapolated 30-min meeting |
  | :-- | --: | --: | --: |
  | faster-whisper (CPU) | 98.7s (96.1 proc + 2.6 load) | 0.45 | ~13.5 min |
  | mlx-whisper (M4 GPU) | 79.7s (load+transcribe) | 0.36 | ~10.9 min |

- **Rationale:** mlx is only ~1.24× faster (~19% less wall-clock) — real but not a step change; it saves <3 min on a 30-min meeting. Against that it is **macOS-only** (would not run at all on a judge's Linux/Windows machine), adds a second dependency, a second 1.6 GB weight download in a different format, and more code paths to test with 3 days left. Portability is scored ("works on new audio", judges may run it themselves); 19% is not worth losing it.
- **Also noted:** this benchmark is pessimistic for both — tiled audio is 100% dense speech with no silence, while real meetings have pauses that our VAD skips, so effective RTF should be better in practice. And a realistic 3–5 min demo recording processes in ~1.5–2.5 min either way, which is acceptable. This further weakens the case for mlx.
- **Rejected alternative kept for the write-up/interview:** pluggable runtime (auto-select mlx on Apple Silicon, fall back to faster-whisper). Good design, genuinely better speed+portability, but not worth the complexity under deadline.
- **Status:** accepted. **Stage 1 is now fully decided.**

## [2026-10-04] Stage 1 spike — RESULTS on the real machine (macOS)

**Machine:** Apple M4, 24 GB RAM, macOS 27, arm64, Python 3.13.9.
Versions: faster_whisper 1.2.1, av 19.0.1, ctranslate2 4.8.2, mlx_whisper 0.4.3, numpy 2.5.3. **ffmpeg NOT installed.**

**Result (large-v3-turbo, 11s clip, faster-whisper, CPU):**
- `[load] 2.5s` (weights cached) · `[transcribe] audio=11.0s proc=5.8s RTF=0.53 (1.9x realtime)`
- Transcript verbatim-correct incl. punctuation. `avg_logprob=-0.085`, `no_speech_prob=0.000`.
- **Output contract confirmed:** word-level timestamps + per-segment confidence present and populated.

**DECIDED — decode path (hard requirement, evidence-backed):**
- faster-whisper's INTERNAL decoder is **broken on PyAV 19.0.1** (`av.open()` no longer accepts `metadata_errors`; PyAV removed it between 17.x and 19.x). `ffmpeg` is not installed and we will NOT require judges to `brew install` anything.
- **We decode ourselves via PyAV-direct** (11s decoded in 0.28s — negligible) with ffmpeg-subprocess and library paths as fallbacks. This was already our design choice for error-message quality; the spike proves it is also required for library-version resilience.

**DECIDED — model size:** keep **large-v3-turbo**. The 8 GB RAM contingency (fall back to `medium`) is dropped — 24 GB makes memory a non-issue and load is 2.5s.

**NOT yet settled:**
- RTF=0.53 is **overhead-dominated** (fixed VAD/encoder warm-up costs amortized over only 11s). Not representative of long-form. Added `--repeat` to `stage1_spike_v2.py` to measure realistic long-form RTF.
- **Runtime (faster-whisper vs mlx-whisper)** pending the long-form + mlx comparison. Note faster-whisper is CPU-only on Mac, leaving the M4 GPU unused; mlx uses it but is macOS-only (portability tradeoff still stands).

## [2026-10-04] Stage 1 spike — environment findings (sandbox)

- **Where the bridge shell runs:** NOT macOS directly — a constrained Linux VM on the Mac (aarch64, 4 CPU, 3.8 GB RAM, no GPU). Not the demo environment.
- **Model weights unreachable in sandboxes:** `huggingface.co` is denied by egress policy in BOTH the VM and the cloud workspace (403 at CONNECT; only package registries like pypi are allowlisted). GitHub raw IS reachable. faster-whisper 1.2.1 installed fine on aarch64 CPU (needed `pip install --use-deprecated=legacy-resolver` to dodge a pip resolver AssertionError), but could not download model weights → spike must run on native macOS (unrestricted net + Apple GPU).
- **Reinforces runtime lean:** mlx-whisper is Mac-only; faster-whisper is the portable path that works even in restricted environments. Leaning faster-whisper as primary; mlx as a Mac-only speed option to compare in the real spike.
- **README implication:** locked-down environments can't fetch weights at runtime — the submission should pre-download weights or document the model source clearly so judges aren't blocked.
- **Real spike:** `spikes/stage1_whisper_spike.py` (run on macOS; compares faster-whisper vs mlx-whisper, reports load time + RTF + validates word-timestamps/confidence output). Pending user's numbers → then lock model size + runtime.

## [2026-10-04] Stage 1 (Speech-to-text) — model family, sourcing, runtime, flow

- **Decision — model family:** Whisper (OpenAI, open weights).
- **Alternatives considered:** wav2vec2 (CTC; needs domain fine-tuning to compete, no built-in punctuation), NVIDIA NeMo/Canary/Parakeet (excellent accuracy but wants CUDA — we're on an Apple-Silicon MacBook Air, high delivery risk), Vosk/Kaldi (fully offline but noticeably weaker on terminology/numbers — would bleed the 20 accuracy points). Whisper wins: free, robust without fine-tuning, trivially available, runs on Mac.
- **Decision — sourcing:** Self-host locally (NOT a hosted free API).
- **Rationale:** live judged demo + interview both reward local control — no network dependency, no rate limits, no free-tier expiry, identical behavior every run. Hosted APIs (e.g. Groq free Whisper) are faster/zero-setup but tie demo-day reliability to external services and a free tier that may change, and are a weaker interview story. **Gated on the spike** (must confirm it runs acceptably on this Mac before committing).
- **Decision — model size:** large-v3-turbo as the starting point, to be confirmed/adjusted against real timings in the spike; fall back to medium/small.en if too heavy on the Air.
- **Decision — runtime:** mlx-whisper (Apple-Silicon-native GPU via MLX, likely fastest on this Mac) or faster-whisper (CTranslate2, CPU, most battle-tested) — settle empirically in the spike. Reversible, low-stakes (identical output, only speed differs).
- **Decision — config:** force language = English; beam search (accuracy over speed — accuracy is the graded thing); VAD on (reduces silence-hallucination); keep timestamps.

### Stage 1 internal flow (locked)
Layered, fail-honest validation — rule out known-bad loudly, carry uncertainty
forward, never trust a single check as conclusive:

- **[B] Validate (layered):**
  - L1 — exists + non-zero size → hard error on fail.
  - L2 — decode probe via ffmpeg → hard error on fail (catches corrupt / unreadable / unsupported codec). On success we already hold the decoded 16 kHz mono samples, so decode *is* the validation.
  - L3 — post-decode sanity (duration floor, VAD energy): too-short / near-silent → **SOFT WARNING, proceed anyway** (do not discard on our judgment — the file may still be usable; flag it to the user instead).
- **[C] Decode:** owned by us (done in L2), not delegated to the library.
- **[D] Clean:** VAD skips silence before the model.
- **[E] Transcribe:** Whisper per the config above; word-level timestamps + per-segment confidence on.
- **[F] Assemble:** rich Transcript object — full text + segments `[{start,end,text,confidence}]` + word-level timestamps. Low-confidence patches are flagged, not hidden.
- **[G] Post-check:** empty / near-empty transcript → "no speech detected" (handled gracefully, never a crash).

- **Output contract:** Rich — text + segments + word-level timestamps + per-segment
  confidence. Chosen over minimal/text-only because timestamps unlock
  timestamp-anchored verification in Stage 4 ("decision supported at 12:30") and
  confidence lets Stage 2 target shaky patches.
- **Guiding principle (carries to Stage 4):** don't let any single check or model
  output masquerade as ground truth — rule out known-bad, carry uncertainty
  forward, fail honestly. Validation and anti-hallucination are two instances of
  one philosophy.
- **Deferred:** long-recording chunking strategy → decided in the edge-hardening step.
- **Status:** accepted (model size + runtime pending spike confirmation).
