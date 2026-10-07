"""
Stage 4 — grounding the meeting record against the transcript.

Stage 3's prompt DEMANDS an evidence quote for every decision and action item.
Nothing checked that those quotes exist. A language model generates plausible
text, and a plausible-sounding quote is exactly the kind of thing it generates
well — so "the prompt asked for evidence" is not evidence.

This stage checks. The response is GRADED rather than all-or-nothing, because
the two failure modes are different:

  - evidence not found in the transcript  -> DROP the item. It is unsupported.
  - evidence fine but the OWNER never appears in the transcript
                                          -> keep the task, downgrade the owner
                                             to unspecified. The work is real;
                                             the attribution was invented.

Dropping an invented owner but keeping the task preserves true information
instead of throwing it away with the false part.

Nothing is deleted silently: everything removed or downgraded is returned so the
interface can show what was rejected and why — the same pattern as Stage 2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import ActionItem, Decision, MeetingRecord, Segment

# Fraction of the quote's content words that must appear in the transcript.
# Not 1.0: models paraphrase slightly ("we're moving" vs "we are moving") and we
# are checking that the claim is SUPPORTED, not that it was copied perfectly.
EVIDENCE_OVERLAP = 0.6
MIN_EVIDENCE_WORDS = 3

_STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for",
         "is", "are", "was", "were", "be", "it", "that", "this", "we", "i"}


def _tokens(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", (text or "").lower()).split()


def _content(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in _STOP and len(t) > 2]


@dataclass
class GroundingResult:
    record: MeetingRecord
    dropped_decisions: list[tuple[str, str]] = field(default_factory=list)
    dropped_actions: list[tuple[str, str]] = field(default_factory=list)
    downgraded: list[str] = field(default_factory=list)
    context_verified: list[str] = field(default_factory=list)   # short quotes
    merged: list[str] = field(default_factory=list)             # duplicate actions

    @property
    def total_removed(self) -> int:
        return len(self.dropped_decisions) + len(self.dropped_actions)

    def to_dict(self) -> dict:
        return {
            "dropped_decisions": [{"statement": s, "why": w}
                                  for s, w in self.dropped_decisions],
            "dropped_actions": [{"task": s, "why": w} for s, w in self.dropped_actions],
            "downgraded_fields": self.downgraded,
            "verified_by_context": self.context_verified,
            "merged_duplicates": self.merged,
        }


def find_evidence(evidence: str, segments: list[Segment]) -> Segment | None:
    """
    Locate the segment that supports this quote, or None if nothing does.

    Returns the segment so the claim can be anchored to a TIMESTAMP — the payoff
    of keeping word-level timing all the way from Stage 1.
    """
    want = _content(evidence)
    if len(want) < MIN_EVIDENCE_WORDS:
        return None

    # Whisper segments are acoustic chunks, not sentences: a long quote often
    # runs across two of them, and each half alone falls under the bar. (Found
    # on the 30-minute meeting, where two correct items were dropped this way.)
    # So each segment is scored alone AND joined with the next one; a pair is
    # anchored to whichever of its two segments holds more of the quote.
    best, best_score = None, 0.0
    for i, seg in enumerate(segments):
        have = set(_content(seg.text))
        windows = [(seg, have)]
        if i + 1 < len(segments):
            windows.append((None, have | set(_content(segments[i + 1].text))))
        for anchor, words in windows:
            if not words:
                continue
            score = sum(1 for w in want if w in words) / len(want)
            if anchor is None:
                nxt = set(_content(segments[i + 1].text))
                anchor = seg if sum(w in have for w in want) >= sum(w in nxt for w in want) \
                    else segments[i + 1]
            if score > best_score + 1e-9:
                best, best_score = anchor, score
    return best if best_score >= EVIDENCE_OVERLAP else None


# ---- short quotes ------------------------------------------------------
# "Do that." approves a proposal, and is the ONLY quote that captures the
# decision — but it has no content words, so overlap lookup can't place it and
# the item used to be dropped (found on the held-out meeting: a recall bug in
# this stage, not in the model). A short quote is accepted only if
#   (1) it appears VERBATIM in a segment — it pins the moment, and
#   (2) the claim's own content words appear in that segment or the few before
#       it — the context backs the content.
CONTEXT_WINDOW = 3          # segments before the quote
CONTEXT_OVERLAP = 0.5


def _stem(word: str) -> str:
    for suffix, repl in (("ily", "y"), ("ing", ""), ("ed", ""), ("ly", ""),
                         ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: len(word) - len(suffix)] + repl
    return word


def _stems(text: str) -> set[str]:
    return {_stem(t) for t in _content(text)}


def find_short_evidence(evidence: str, claim: str,
                        segments: list[Segment]) -> tuple[Segment, float] | None:
    quote = _tokens(evidence)
    want = _stems(claim)
    if not quote or len(_content(evidence)) >= MIN_EVIDENCE_WORDS or not want:
        return None
    n = len(quote)
    best: tuple[Segment, float] | None = None
    for idx, seg in enumerate(segments):
        toks = _tokens(seg.text)
        if not any(toks[i:i + n] == quote for i in range(len(toks) - n + 1)):
            continue
        window = " ".join(s.text for s in segments[max(0, idx - CONTEXT_WINDOW):idx + 1])
        support = len(want & _stems(window)) / len(want)
        if support >= CONTEXT_OVERLAP and (best is None or support > best[1]):
            best = (seg, support)
    return best


def _locate(item_text: str, evidence: str, segments, result: "GroundingResult"):
    seg = find_evidence(evidence, segments)
    if seg is not None:
        return seg
    short = find_short_evidence(evidence, item_text, segments)
    if short:
        seg, support = short
        result.context_verified.append(
            f'"{item_text[:50]}": short quote {evidence!r} found verbatim at '
            f"{_clock(seg.start)}; {support:.0%} of its content is in the preceding context")
        return seg
    return None


# ---- duplicate action items --------------------------------------------
# Models sometimes list a request AND its acceptance as two tasks ("Update the
# Grafana dashboard" / "Take ownership of the Grafana dashboard", both Sam).
MERGE_OVERLAP = 0.5


def _merge_duplicates(actions: list[ActionItem], result: "GroundingResult") -> list[ActionItem]:
    kept: list[ActionItem] = []
    for a in actions:
        twin = None
        if a.owner:                     # never merge two UNOWNED tasks: too risky
            for k in kept:
                if (k.owner or "").lower() != a.owner.lower():
                    continue
                x, y = _stems(a.task), _stems(k.task)
                shared = x & y
                if len(shared) >= 2 and len(shared) / min(len(x), len(y)) >= MERGE_OVERLAP:
                    twin = k
                    break
        if twin is None:
            kept.append(a)
            continue
        keep, drop = (twin, a) if len(twin.task) >= len(a.task) else (a, twin)
        keep.deadline = keep.deadline or drop.deadline
        if keep is a:
            kept[kept.index(twin)] = a
        result.merged.append(f'"{drop.task}" merged into "{keep.task}" '
                             f"(same owner {keep.owner}, same work)")
    return kept


def _mentioned(value: str | None, transcript: str) -> bool:
    """Is this owner/deadline actually spoken anywhere in the transcript?"""
    if not value:
        return True                      # None is already "unspecified"
    toks = _content(value)
    if not toks:
        return True
    have = set(_tokens(transcript))
    return any(t in have for t in toks)


_SPEAKER = re.compile(r"\s*(Speaker \d+)\b")
# A voice owns a task only if it COMMITTED, i.e. spoke in the first person.
# "Someone needs to document the rollback" said by Speaker 1 is not Speaker 1's task.
_FIRST_PERSON = re.compile(r"\b(?:I|I'll|I'm|I've|I'd|me|my|mine)\b")
NAME_WINDOW = 2   # a stated owner must be named in the evidence segment or the 2 before


def _clock(seconds: float | None) -> str:
    m, s = divmod(int(seconds or 0), 60)
    return f"{m}:{s:02d}"


def evidence_speaker(evidence: str, seg: Segment, utterances) -> str | None:
    """Which voice spoke this quote? (best overlap among the segment's utterances)"""
    u = evidence_utterance(evidence, seg, utterances)
    return u.speaker if u else None


def evidence_utterance(evidence: str, seg: Segment, utterances):
    if not utterances:
        return None
    want = _content(evidence)
    best, best_score = None, -1.0
    for u in utterances:
        if u.segment_id != seg.id:
            continue
        have = set(_content(u.text))
        score = sum(1 for w in want if w in have) / max(len(want), 1)
        if score > best_score:
            best, best_score = u, score
    return best


def _named_near(name: str, segments: list[Segment], seg: Segment) -> bool:
    """
    Is this name spoken in the evidence segment or just before it?

    The original check accepted a name spoken ANYWHERE in the meeting. That has a
    hole: "Arjun, where are we?" at 0:04 makes "Arjun" appear in the transcript,
    so a model that GUESSED Arjun owns a task at 0:50 passed. Proximity is what
    links a name to a task ("Sam, can you update…" -> "Yes, I'll take it").
    """
    idx = next((i for i, s in enumerate(segments) if s.id == seg.id), None)
    if idx is None:
        return _mentioned(name, " ".join(s.text for s in segments))
    window = " ".join(s.text for s in segments[max(0, idx - NAME_WINDOW):idx + 1])
    toks = _content(name)
    return bool(toks) and any(t in set(_tokens(window)) for t in toks)


def _named_request(a: ActionItem, utt, utterances, naming):
    """
    "Hannah, could you get the release notes ready by Monday?" names who the work
    is for. On the 30-minute meeting the model gave the REQUESTER's voice label
    as owner; Stage 4 correctly rejected that but then had nothing left. If the
    evidence is a request addressed by name, about this same task, the person
    named is the stated owner — UNLESS the reply came from a voice identified as
    someone else ("Leo, can you check the restore?" answered by Grace, who takes
    it). Returns (name, request sentence) or None.
    """
    from .naming import _REQUEST, _SENTENCE, _vocative
    if utt is None or not utterances:
        return None
    want = _stems(a.task)
    for sentence in _SENTENCE.split(utt.text.strip()):
        name = _vocative(sentence, None)
        if not name or not _REQUEST.search(sentence):
            continue
        if not want or len(want & _stems(sentence)) / len(want) < 0.5:
            continue
        idx = next((i for i, u in enumerate(utterances) if u is utt), None)
        reply = next((u for u in utterances[idx + 1:] if u.speaker != utt.speaker),
                     None) if idx is not None else None
        # Redirected? The replying voice has evidence for a DIFFERENT name —
        # bound or merely contested (a tie leaves it unbound but still says
        # "this voice may be someone else"). Then the request names the wrong
        # person, and we assign nobody rather than guess.
        if naming and reply:
            bound = naming.name_for(reply.speaker)
            if bound:
                if bound.lower() != name.lower():
                    return None              # a voice known to be someone else
            elif {n.lower() for n in naming.candidates.get(reply.speaker, set())} \
                    - {name.lower()}:
                return None                  # unnamed, but possibly someone else
        return name, sentence
    return None


def _check_owner(a: ActionItem, seg: Segment, segments, utterances, naming,
                 result: "GroundingResult") -> None:
    if not a.owner:
        return
    utt = evidence_utterance(a.evidence, seg, utterances)
    who = utt.speaker if utt else None
    suspect = naming.is_suspect(utt) if naming else None
    m = _SPEAKER.match(a.owner)
    if m:
        # A voice label. True only if THAT voice spoke the evidence.
        a.owner = m.group(1)
        if suspect:
            # The voice is untrustworthy, but the REQUEST was addressed by name.
            # If the flagged reply accepts a request about this same task, the
            # owner is stated out loud: "Sam, can you update the dashboard?"
            want = set(_content(a.task))
            asked = set(_content(suspect.question))
            if want and len(want & asked) / len(want) >= 0.5:
                a.owner, a.owner_source = suspect.addressee, "stated"
                a.owner_evidence = (
                    f'asked by name at {_clock(suspect.asked_at)} ("{suspect.question[:80]}"); '
                    f"the accepting reply's voice label was unreliable, so it was not used")
                return
            result.downgraded.append(
                f'owner {a.owner!r} for "{a.task[:40]}" withheld — {suspect.describe()} '
                f"— set to unspecified")
            a.owner = None
            return
        if who != a.owner or not _FIRST_PERSON.search(a.evidence):
            asked = _named_request(a, utt, utterances, naming)
            if asked:
                a.owner, a.owner_source = asked[0], "stated"
                a.owner_evidence = (f'asked by name at {_clock(seg.start)} '
                                    f'("{asked[1][:80]}")')
                return
        if who == a.owner and not _FIRST_PERSON.search(a.evidence):
            result.downgraded.append(
                f'owner {a.owner!r} for "{a.task[:40]}" — {a.owner} said it, but '
                f"not as a first-person commitment — set to unspecified")
            a.owner = None
            return
        if who == a.owner:
            a.owner_source = "speaker"
            a.owner_evidence = f'{a.owner} said it at {_clock(seg.start)}: "{a.evidence[:80]}"'
            return
        result.downgraded.append(
            f'owner {a.owner!r} for "{a.task[:40]}" — the evidence was spoken by '
            f"{who or 'an unidentified voice'}, not {a.owner} — set to unspecified")
        a.owner = None
        return

    if _named_near(a.owner, segments, seg):
        a.owner_source = "stated"
        a.owner_evidence = f"named in the conversation at {_clock(seg.start)}"
        return
    label = naming.label_for(a.owner) if naming else None
    if label and who == label and not suspect:
        # The model used a name, but only the naming evidence supports it.
        a.owner_label, a.owner_source = label, "inferred"
        a.owner_evidence = (f'{label} said it at {_clock(seg.start)}; '
                            f"{naming.bindings[label].describe()}")
        return
    result.downgraded.append(
        f'owner {a.owner!r} for "{a.task[:40]}" is not named near the evidence '
        f"and no speaker evidence links them to it — set to unspecified")
    a.owner = None


def ground(record: MeetingRecord, segments: list[Segment], *, utterances=None,
           naming=None) -> GroundingResult:
    """`utterances`/`naming` (optional) enable speaker-label and inferred owners."""
    if record is None:
        return GroundingResult(record=MeetingRecord())

    transcript = " ".join(s.text for s in segments)
    result = GroundingResult(record=record)

    kept_decisions: list[Decision] = []
    for d in record.decisions:
        seg = _locate(d.statement, d.evidence, segments, result)
        if seg is None:
            result.dropped_decisions.append((
                d.statement,
                f"no supporting quote found in the transcript: {d.evidence[:90]!r}"
                if d.evidence else "no evidence quote was provided",
            ))
            continue
        d.segment_id, d.timestamp = seg.id, seg.start
        kept_decisions.append(d)
    record.decisions = kept_decisions

    kept_actions: list[ActionItem] = []
    for a in record.action_items:
        seg = _locate(a.task, a.evidence, segments, result)
        if seg is None:
            result.dropped_actions.append((
                a.task,
                f"no supporting quote found in the transcript: {a.evidence[:90]!r}"
                if a.evidence else "no evidence quote was provided",
            ))
            continue
        a.segment_id, a.timestamp = seg.id, seg.start

        # Evidence is sound, but the attribution may still be invented.
        _check_owner(a, seg, segments, utterances, naming, result)
        if not _mentioned(a.deadline, transcript):
            result.downgraded.append(
                f'deadline {a.deadline!r} for "{a.task[:40]}" was never stated — '
                f"set to unspecified"
            )
            a.deadline = None
        kept_actions.append(a)
    record.action_items = _merge_duplicates(kept_actions, result)

    if result.total_removed:
        record.warnings.append(
            f"{result.total_removed} generated item(s) could not be supported by "
            f"the transcript and were removed."
        )
    if result.downgraded:
        record.warnings.append(
            f"{len(result.downgraded)} owner/deadline value(s) were not stated in "
            f"the recording and were set to unspecified."
        )
    return result
