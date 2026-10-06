# Technical Description

Required submission artifact: the models used, each model's role, and how data
moves between stages.

---

## 1. The models

| # | Model | Runtime | Role |
| :-- | :-- | :-- | :-- |
| 1 | **Whisper large-v3-turbo** | faster-whisper (CTranslate2), local CPU | Speech-to-text |
| 1.5 | **ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`) | SpeechBrain, local CPU | Speaker embeddings (optional stage) |
| 2 | **qwen3:8b** | Ollama, local | Transcript refinement |
| 3 | **qwen3:14b** *or* **openai/gpt-oss-120b** | Ollama local / Groq free API | Meeting documentation |

Stages 2 and 3 are distinct models, executed as separate processing stages, as
the problem statement requires.

Two further stages are **deliberately not models**: speaker naming (1.6) and
verification (4). They decide what counts as *true*, so they are deterministic
rules, testable without any model.

All components are free. The application runs end to end with no API keys and no
network.

---

## 2. Stage 1 — Speech-to-text

**Model:** Whisper large-v3-turbo, self-hosted via faster-whisper.
**Input:** an uploaded audio file. **Output:** a `Transcript`: full text,
segments, and **word-level timestamps** and **per-word confidence**.

**Configuration:**

| Setting | Why |
| :-- | :-- |
| `language="en"` | forced: skips language detection, so it can't mis-detect |
| `beam_size=5` | accuracy over speed |
| `vad_filter=True` | Whisper hallucinates text in long silences |
| `word_timestamps=True` | needed downstream |

**Why Whisper, and why self-hosted.** Whisper is free, robust without
fine-tuning, and runs on CPU. The alternatives were rejected on concrete grounds:

- **NVIDIA NeMo** needs CUDA, which we don't have.
- **wav2vec2** needs domain fine-tuning to compete, and emits no punctuation.
- **Vosk/Kaldi** are weaker on terminology and numbers.

Self-hosting removes any network or rate-limit dependency at demo time.

**Audio decoding is ours.** faster-whisper's internal decoder passes an argument
that PyAV ≥19 removed, so it crashes on current PyAV. We decode with PyAV
directly (falling back to ffmpeg, then the library). The model gets float32 at
16 kHz.

**Layered validation:**

- Missing, empty or undecodable files are **hard errors**.
- Very short or near-silent audio gets a **soft warning**, and processing
  continues. Our heuristic could be wrong, so we don't refuse the file.

**Every extra in the rich output is load-bearing:**

- Confidence decides what Stage 2 may edit.
- Timestamps drive diarization and anchor Stage 4.

Measured: RTF ≈ 0.45 (faster than real time). WER 2.52% on the sample meeting,
3.73% on the held-out meeting.

---

## 3. Stage 1.5 — Speaker diarization (optional)

**Input:** the transcript and the audio. **Output:** an anonymous label
("Speaker 1", "Speaker 2", …) on every word, numbered by order of first
appearance.

1. **Pause-split chunks.** Whisper segments are cut wherever the gap between
   words exceeds 0.3 s. Segment-level labels would be wrong by construction: on
   our own sample, Whisper put Priya's question and Arjun's answer in **one**
   segment.
2. **Embed.** Each chunk (with 0.1 s padding) becomes a 192-dimensional
   ECAPA-TDNN voice embedding. Chunks under 0.6 s are too short to embed
   reliably, so they don't vote.
3. **Cluster — our own code.**
   - Average-linkage agglomerative clustering on cosine distance.
   - Implemented with the Lance–Williams update. A test checks it is identical
     to brute-force average linkage.
   - Stops at distance threshold **0.65**, or at a known speaker count if given.
4. **Assign the short chunks.**
   - Chunks of 0.25–0.6 s go to the nearest cluster centroid.
   - Shorter fragments inherit the label of a neighbouring chunk.

**Choosing the threshold.**

- The error rate is flat (8.0% DER) for thresholds from 0.55 to 0.75 on the
  tuning meeting. We take the **centre of that plateau**.
- We then report it on a **held-out** meeting never used for tuning: 4 speakers,
  different voices, many one-word turns.
- Result there: 4/4 speakers, DER 4.7%, zero confusion.
- The held-out meeting's own best threshold (0.50) invents a fifth speaker on the
  tuning meeting. That is why we take the plateau centre, not either minimum.

**Why a threshold is not portable.** With a deliberately blurrier stand-in
embedder, the same audio gives 2 speakers at 0.70 and 3 at 0.50. A threshold
only means something relative to one embedder's geometry. That's why we tuned
on the real model and reported on held-out data. This is encoded as a test.

**Optional by design.** Torch is large, so the core install stays light.

- Without `requirements-diarization.txt`, the pipeline runs and says how to
  enable diarization.
- `DIARIZE=auto|on|off` controls it.

---

## 4. Stage 2 — Domain-aware refinement

**Input:** the `Transcript`, plus a domain glossary. **Output:** a refined
transcript and a list of every edit, applied or rejected.

**The model never emits the transcript.**

- Generation is autoregressive. A model asked to return a corrected transcript
  re-rolls every token: thousands of chances for meaning to drift, which
  prompting can reduce but never bound.
- Instead the model returns a structured **edit-list** (`segment_id`, `original`,
  `replacement`, `reason`, `confidence`).
- *We* apply each edit to every occurrence in its segment, on word boundaries.
- Everything not explicitly edited is byte-identical **by construction**. The
  worst case becomes a *missed* correction (benign), not a *corrupted*
  transcript (invisible).

**(a) Deterministic glossary pass, no model.**

- ASR errors on jargon are overwhelmingly phonetic, so glossary terms are indexed
  by **Double Metaphone** (a phonetic code).
- They are matched against every low-confidence n-gram of up to 5 words.
- N-grams never cross a sentence boundary.

Phonetics turned out to be the correct tool, not just one option:

| proposed edit | phonetic | string similarity | correct verdict |
| :-- | :-- | --: | :-- |
| "cooper netties" → Kubernetes | match | 0.75 | accept ✓ |
| "sequel" → SQL | match | 0.67 | accept ✓ |
| "deadline" → headline | **no match** | **0.87** | reject ✓ |
| "might" → will | no match | 0.48 | reject ✓ |

A fuzzy string matcher gets both of the first two pairs backwards; phonetics gets
both right.

**(b) LLM pass, doing the two jobs rules cannot.** An earlier version bounded
every replacement to the glossary. Measurement then showed the LLM contributed
*nothing*: disabling it left recovery at 100%. So it now has two jobs of its own:

- **Context veto.** In *"the light was red, is it not"* the matcher proposes
  "red is" → **Redis**. Only a reader of the sentence can reject that.
- **Off-glossary proposals.** It may correct technical terms nobody listed. These
  are allowed only at confidence ≥ 0.7, never into ordinary vocabulary, and are
  flagged as weaker evidence.

**Verification gates: an allowlist, not a blocklist.** A blocklist misses
"we might ship" → "we **will** ship". Edits are rejected unless they demonstrate
they are a narrow terminology fix:

1. **Anchor:** the original text exists in the segment.
2. **No-op:** a change of case or punctuation only is rejected.
3. **Typed policy:**
   - insertions are denied;
   - deletions are denied, except collapsing a stutter.
4. **Phonetic similarity.**
5. **Glossary bound**, or the off-glossary exception above.
6. **Span limit:** at most 5 words, with a symmetric length-ratio check.
7. **Digit guard.**
8. **Negation guard.**
9. **Confidence gate:** only spans Whisper was unsure of (confidence < 0.60) may
   be edited.

On top of the gates, an **edit budget** of at most max(5, words ÷ 20) per
transcript catches an avalanche of individually plausible edits.

Decoding is greedy (`temperature=0`). Correction is not a creative task, and a
re-run of the demo should give the same output.

---

## 5. Stage 1.6 — Evidence-based speaker naming (optional)

**Input:** the refined transcript split by speaker. **Output:** label → name
bindings, each with its evidence, plus any label the words contradict.

**Speaker-split refined text.**

- Diarization labels raw *words*, but refinement edits segment *text*.
- A single-speaker segment uses its refined text directly.
- For a segment containing two voices, each voice's piece is rebuilt from the raw
  words, and that segment's accepted Stage 2 edits are re-applied to it.

**Why rules and not the LLM.** Asked "who is Speaker 2?", a model will answer,
and a guess looks exactly like a supported answer. The evidence patterns are
detectable deterministically:

| Evidence | Example | Weight |
| :-- | :-- | --: |
| addressed, then answered | Speaker 1: "*Arjun*, where are we?" → Speaker 2 answers next | 1 |
| self-introduction | "Hi, I'm Arjun" / "This is Arjun." | 2 |
| self-address exclusion | whoever says "Arjun, …" is not Arjun | veto |

**Details that matter:**

- Only a request that *ends* the speaker's turn hands the floor to the person
  named.
- Words like "Okay," / "Great," and glossary terms like "Redis," are never names.
- "This is Jenkins" mid-sentence is not an introduction.

**Resolution is conservative:**

- A tie between two names leaves the speaker unnamed.
- If two voices claim one name, the stronger evidence wins. On a tie, neither is
  named.
- Unnamed is a correct output. Nobody ever says Priya's name, so she stays
  "Speaker 1".

**Cross-check: the words audit the voices.** Suppose a request addressed to
someone else ("Sam, can you…?") is followed directly by a reply opener ("Yes, …")
carrying the **same** voice label. Then diarization almost certainly missed a
speaker change. The label is marked untrustworthy and a warning is raised.
Nothing downstream is built on it.

This is not hypothetical. On our sample, Sam's "Yes, I'll take the Grafana
dashboard" was clustered with Priya's voice. Before the check, that became a wrong
owner ("Speaker 1", which is Priya).

After the check, the label is never used as an owner. Withholding alone lost a
correct owner, though: Priya had said "*Sam*, can you update the Grafana
dashboard?" out loud. So when the flagged reply accepts a request about the same
task (at least half the task's content words appear in the request), the owner
becomes the person the request named, tagged `stated`, citing the request.
Otherwise the owner is withheld.

---

## 6. Stage 3 — Meeting documentation

**Input:** the refined transcript, speaker-labelled when diarization ran.
**Output:** summary, minutes, decisions, action items.

Defences against invention:

1. **The schema makes `owner` and `deadline` optional.** `unspecified` is the
   default state, not something the model must remember to say.
2. **Every decision and action item must carry an evidence quote.**
3. **Precise definitions.**
   - A *decision* is something settled; a proposal is not.
   - An *action item* is work the meeting says must be done. It does not need an
     owner, so "someone needs to document the rollback" counts.
   - A tentative or declined idea is not an action item.
4. **Anonymous labels only.**
   - With diarization, a first-person commitment ("I'll benchmark it") gets the
     speaker's label as its owner, e.g. "Speaker 2".
   - The model is told never to map labels to names. Naming happens afterwards,
     by Stage 1.6's rules.
5. **Stage 4 checks all of it.**

**Backend routing by size.**

- Groq's free tier is 6,000 tokens per minute, and a 30-minute meeting is ~6,800
  tokens in one call.
- So Stage 3 uses Groq (`openai/gpt-oss-120b`) under ~4,500 estimated tokens,
  local `qwen3:14b` above, and local again on any remote failure.

**Measured with repeated trials, not single runs.**

- Hosted inference varies even at temperature 0, and we twice changed our minds
  on single runs before measuring rates.
- An early "Groq misses owner-less tasks" result turned out to be **our prompt**:
  it defined an action item as a commitment. After the fix, Groq passed 45/45
  checks over 5 runs, and local passed 9/9.
- Groq's wording varies between runs; its substance did not.

---

## 7. Stage 4 — Grounding

**Input:** the record, the transcript, the speaker-split utterances and the
naming result. **Output:** the record with unsupported content removed, each
surviving claim anchored to a timestamp, and each owner tagged with how it is
known.

**Evidence lookup.** A quote is supported if ≥ 60% of its content words appear in
one segment. This tolerates paraphrase while rejecting fabrication. Surviving
claims get that segment's timestamp.

**Short quotes.** "Do that." approves a proposal, and it can be the only quote
that captures the decision. But it has no content words, so overlap lookup can't
place it. Our verifier used to drop such items. On the held-out meeting that made
a decision the model had *correctly* reported disappear: a recall bug in this
stage, not in the model. A quote with fewer than 3 content words is now accepted
only if both hold:

- it appears **verbatim** in a segment, which pins the moment;
- at least half of the claim's own content words appear in that segment or the
  3 before it, so the context backs the content.

"Do that." cannot support an unrelated claim, and a short quote that was never
said is still dropped. Both are tested.

**Duplicates.** The model sometimes lists a request and its acceptance as two
tasks ("Update the Grafana dashboard" / "Take ownership of the Grafana
dashboard"). Two action items with the **same named owner** and strongly
overlapping work (at least 2 shared word stems, and ≥ 50% of the smaller task)
are merged. The longer wording is kept, plus any deadline from either. Two
*unowned* tasks are never merged, since too little ties them together. Every merge
is listed.

**Graded response:**

- **Quote not found:** the item is **dropped**. It is unsupported.
- **Quote found, but the owner or deadline is unsupported:** the task is
  **kept**, and that field is downgraded to `unspecified`. Discarding the whole
  item would lose true information along with the false.

**Owner rules:**

| Owner given | Accepted when | Tagged |
| :-- | :-- | :-- |
| a name | spoken in the evidence segment or the two before it | `stated` |
| a name | that name's voice (per Stage 1.6) spoke the evidence | `inferred` |
| a speaker label | that voice spoke the evidence, **in the first person**, and the line is not flagged by the cross-check | `speaker` |
| a speaker label on a **flagged** reply | the reply accepts a request about the same task that named someone | `stated` (that name) |
| anything else | — | downgraded to `unspecified` |

Then, if a `speaker` owner's voice has a name binding, it is shown as that name,
tagged `inferred`. Both pieces of evidence are kept: who said it, and why that
voice is that person.

**Closing a latent hole.** The original owner check accepted a name spoken
*anywhere* in the meeting. "Arjun, where are we?" at 0:04 therefore made a
*guessed* "Arjun" pass for a task at 0:50. The proximity rule closes that hole,
and it is regression-tested.

**Nothing is removed silently.** Every drop, downgrade, merge, short-quote
acceptance and cross-check flag appears in the evaluation output and the JSON.
Warnings also appear in the interface.

---

## 8. How data moves between stages

```
audio file
   │  validate (layered) → decode (PyAV) → VAD
   ▼
Transcript {text, segments[{id,start,end,text,confidence, words[{word,start,end,p}]}]}
   │  pause-split → ECAPA embeddings → agglomerative clustering   (1.5, optional)
   ▼
words[].speaker = "Speaker N"
   │  glossary pass + LLM veto/proposals → Edit[] → 9 gates + budget → apply
   ▼
RefinementResult {refined segments, edits[applied | rejected | vetoed]}
   │  split refined text by speaker; name evidence; cross-check   (1.6, optional)
   ▼
Utterance[] {segment_id, speaker, start, text}  +  NamingResult {bindings, conflicts, suspects}
   │  render "[t] (segment k) Speaker N: text" → size estimate → Groq | local
   ▼
MeetingRecord {summary, minutes[], decisions[{statement, evidence}],
               action_items[{task, owner?, deadline?, evidence}]}
   │  evidence lookup · owner rules · voice check · name attribution
   ▼
GroundingResult {record (pruned, timestamped, owner_source + owner_evidence),
                 dropped[], downgraded[]}
   │
   ▼
Markdown (human-readable) + JSON (machine-readable), from the same object
```

**Failure policy.**

- Stage 1 failing is fatal: without a transcript nothing downstream means
  anything.
- Every other stage **degrades** with a visible warning:
  - no diarization means no speaker labels;
  - no refinement means the raw transcript is used;
  - no record still leaves both transcripts.
- A silent degradation looks exactly like success on an empty meeting. We hit
  this during development (a Cloudflare 403 swallowed into "no decisions").

---

## 9. Verification summary

| Harness | Checks | Model needed | Covers |
| :-- | --: | :-- | :-- |
| `check_stage1.py` | 14 | Whisper | validation layers, decode fallbacks, soft warnings, output contract |
| `check_stage2.py` | 29 | none | all gates, phonetics, off-glossary policy, veto, edit application, regressions |
| `check_stage3.py` | 21 | none | evidence lookup, fabricated quotes, invented owners/deadlines, graded response, short quotes, duplicate merging |
| `check_diarization.py` | 12 | none | clustering maths vs brute force, pause splitting, short chunks, threshold relativity |
| `check_naming.py` | 25 | none | name evidence, conflicts, speaker split, voice-aware owners, the cross-check |
| `eval/evaluate.py` | 25 / 19 | all | WER, DER, names, traps, error injection, negative control (sample / held-out) |
| `eval/stage3_trials.py` | 9 × N | Stage 3 | backend comparison as rates over repeated runs |

The harnesses that guard faithfulness need **no model**. That follows from the
design: what decides truth (gates, grounding, naming) is deterministic code, not
model output.
