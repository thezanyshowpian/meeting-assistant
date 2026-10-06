# Technical Description

Required submission artifact: the models used, each model's role, and how data
moves between stages.

---

## 1. The models

| # | Model | Runtime | Role |
| :-- | :-- | :-- | :-- |
| 1 | **Whisper large-v3-turbo** | faster-whisper (CTranslate2), local CPU | Speech-to-text |
| 2 | **qwen3:8b** | Ollama, local | Transcript refinement |
| 3 | **qwen3:14b** *or* **openai/gpt-oss-120b** | Ollama local / Groq API | Meeting documentation |

Stages 2 and 3 are distinct models executed as separate processing stages, as
required. All components are free; the application runs end to end with no API
keys and no network.

---

## 2. Stage 1 — Speech-to-text

**Model:** Whisper large-v3-turbo, self-hosted via faster-whisper.
**Input:** an uploaded audio file. **Output:** a `Transcript` object.

Config: `language="en"` (forced — skips language detection and prevents
mis-detection), `beam_size=5` (accuracy over speed), `vad_filter=True` (Whisper
can hallucinate text during long silences), `word_timestamps=True`.

**Why Whisper, and why self-hosted.** Whisper is free, robust without
fine-tuning, and runs on CPU. Alternatives were rejected on concrete grounds:
NVIDIA NeMo wants CUDA we don't have; wav2vec2 needs domain fine-tuning to
compete and emits no punctuation; Vosk/Kaldi are measurably weaker on
terminology and numbers. Self-hosting removes any network or rate-limit
dependency at demo time, and makes behaviour identical on every run.

**Audio decoding is ours, not the library's.** faster-whisper's internal decoder
calls `av.open(..., metadata_errors=...)`, which PyAV ≥19 removed — it crashes
outright on current PyAV. We decode via PyAV directly (falling back to an
ffmpeg subprocess, then the library) and hand the model a float32 array at
16 kHz. This was originally chosen for error-message quality; it turned out to
be required for version resilience.

**Output is deliberately rich:** full text, segments, **word-level timestamps**
and **per-word confidence**. Both extras are load-bearing downstream —
confidence gates Stage 2, timestamps anchor Stage 4.

Measured: model load 2.6 s; RTF ≈ 0.45; WER **1.68%** on the sample meeting.

---

## 3. Stage 2 — Domain-aware refinement

**Input:** the `Transcript` from Stage 1, plus a domain glossary.
**Output:** a refined transcript and a list of every edit, applied or rejected.

**The model never emits the transcript.** Generation is autoregressive, so a
model asked to return a corrected transcript re-rolls every token — thousands of
independent opportunities for meaning to drift, and prompting can only reduce
that rate, never bound it. Instead the model returns a structured **edit-list**
(`segment_id`, `original`, `replacement`, `reason`, `confidence`) which *we*
apply with exact string matching. Everything not explicitly edited is
byte-identical **by construction**: drift becomes impossible rather than
discouraged, and the failure mode degrades to a *missed* correction (benign)
instead of a *corrupted* transcript (catastrophic and invisible).

Corrections come from two sources, both verified by the same gates:

**(a) Deterministic glossary pass — no model.** ASR errors on jargon are
overwhelmingly phonetic, so glossary terms are indexed by **Double Metaphone**
and matched against every low-confidence n-gram. Measurement showed phonetics is
not merely *a* way to do this but the correct one:

| proposed edit | phonetic | string similarity | correct verdict |
| :-- | :-- | --: | :-- |
| "cooper netties" → Kubernetes | match | 0.75 | accept ✓ |
| "sequel" → SQL | match | 0.67 | accept ✓ |
| "deadline" → headline | **no match** | **0.87** | reject ✓ |
| "might" → will | no match | 0.48 | reject ✓ |

A fuzzy string matcher accepts "deadline → headline" (0.87) and rejects
"sequel → SQL" (0.67) — both backwards. Phonetics gets both right.

**(b) LLM pass — two jobs the deterministic pass cannot do.** We originally
bounded every replacement to the glossary; measurement then showed the LLM was
contributing *nothing* (disabling it left recovery at 100%), because the
deterministic pass already scans the whole glossary exhaustively. So the LLM was
given the two jobs that are genuinely its own:

- **Context veto.** The phonetic matcher is context-blind: in *"the light was
  red, is it not"* it proposes "red is" → **Redis**, and every gate accepts it.
  Only a model reading the sentence can reject that.
- **Off-glossary proposals.** Correcting technical terms nobody put in the
  glossary — permitted only above a confidence bar, never into ordinary
  vocabulary, and flagged `glossary_backed=false` since the evidence is weaker.

**Verification gates (an allowlist, not a blocklist).** A blocklist denies only
the bad patterns its author imagined; "we might ship Friday" → "we **will** ship
Friday" passes every obvious rule. So edits are rejected unless they positively
demonstrate they are the narrow thing this stage is for:

1. **Anchor** — `original` must exist in the named segment
2. **Typed policy** — insertions denied as a category; deletions denied except stutter collapse
3. **Phonetic similarity** — semantic drift does not rhyme
4. **Glossary bound** — or the off-glossary exception above
5. **Span limit** — ≤5 words; longer is a rewrite
6. **Digit guard** — numbers may never change
7. **Negation guard** — "not"/"won't"/"can't" may never be added or removed
8. **Confidence gate** — only spans Whisper was *unsure* about may be edited at all
9. **Edit budget** — an avalanche of individually plausible edits is caught in aggregate

Decoding is greedy (`temperature=0`): correction is not a creative task, and
determinism means a judge re-running the demo sees identical output.

---

## 4. Stage 3 — Meeting documentation

**Input:** the refined transcript. **Output:** summary, minutes, decisions,
action items.

Three defences against invention, because a language model produces plausible
continuations and "by Friday" is a very plausible continuation of "we'll get
that done":

1. **The schema makes `owner` and `deadline` optional**, so `unspecified` is the
   default state rather than something the model must remember to say. There is
   no way to represent a guess as a confirmed fact.
2. **Every decision and action item must carry an evidence quote.**
3. **Stage 4 checks those quotes** (below).

**Backend routing by size.** Groq's free tier allows **6,000 tokens per minute**;
a 30-minute meeting is ~6,800 tokens in one Stage 3 call. Chunking cannot fix
this — you would have to wait a minute between chunks. So Stage 3 routes on
estimated transcript size: Groq (`openai/gpt-oss-120b`) under ~4,500 tokens,
local `qwen3:14b` above it, and local again on any remote failure. With no key
configured, everything runs locally.

Measured difference on the sample (n=1, reported as such): gpt-oss-120b found an
additional valid decision our ground truth had missed, but dropped one unassigned
action item that qwen3:14b kept. They fail in opposite directions.

---

## 5. Stage 4 — Grounding

**Input:** the generated record plus the transcript. **Output:** the record with
unsupported content removed, each surviving claim anchored to a timestamp.

Stage 3 was *asked* for evidence; this is what checks it was telling the truth.
The response is **graded**, because the two failure modes differ:

- **Evidence not found in the transcript → the item is dropped.** It is unsupported.
- **Evidence sound but the owner never appears in the transcript → keep the task,
  downgrade the owner to `unspecified`.** The work is real; the attribution was
  invented. Discarding the whole item would throw away true information along
  with the false part.

Surviving claims are anchored to the segment and timestamp that supports them —
the payoff of carrying word-level timing from Stage 1. Nothing is removed
silently: every drop and downgrade is surfaced in the interface and serialised
into the JSON output.

---

## 6. How data moves between stages

```
audio file
   │  validate (layered) → decode (PyAV) → VAD
   ▼
Transcript {text, segments[{start,end,text,confidence,words[]}], warnings}
   │  glossary pass ─┐
   │  LLM veto ──────┤→ Edit[] → 9 gates → apply
   ▼                 ┘
RefinementResult {refined_text, segments, edits[applied|rejected|vetoed]}
   │  size estimate → route (Groq | local) → fallback on failure
   ▼
MeetingRecord {summary, minutes[], decisions[{statement,evidence}],
               action_items[{task, owner?, deadline?, evidence}]}
   │  evidence lookup against segments
   ▼
GroundingResult {record (pruned, timestamped), dropped[], downgraded[]}
   │
   ▼
Markdown (human-readable) + JSON (machine-readable), from the same object
```

**Failure policy.** Stage 1 failing is fatal — without a transcript nothing
downstream is meaningful. Stages 2, 3 and 4 **degrade**: a failed refinement
still leaves the raw transcript, a failed record still leaves both transcripts.
Every degradation raises a visible warning, because a system that degrades
silently is indistinguishable from one that succeeded on an empty meeting — a
failure mode we hit during development and had to design against.

---

## 7. Verification summary

| Harness | Checks | Covers |
| :-- | --: | :-- |
| `check_stage1.py` | 14 | validation layers, decode fallbacks, soft warnings, output contract |
| `check_stage2.py` | 25 | all 9 gates, phonetics, off-glossary policy, context veto, edit application |
| `check_stage3.py` | 15 | evidence lookup, fabricated quotes, invented owners/deadlines, graded response |
| `eval/evaluate.py` | 20 | WER, refinement effect, planted traps, error injection, negative control |

Harnesses for Stages 2–4 run **without any model**, because the logic protecting
faithfulness is deterministic — which is itself a consequence of the edit-list
and grounding designs.
