"""
Stage 1.6 — evidence-based speaker naming.

Diarization answers "which stretches of audio are the same voice?" and returns
ANONYMOUS labels ("Speaker 2"). It cannot know anyone's name. This module binds
a label to a name ONLY where the conversation itself provides evidence, and it
keeps that evidence so every inferred name can be checked by a person.

Why rules and not the LLM: a language model asked "who is Speaker 2?" will
answer — that is what it does — and a plausible guess looks exactly like a
supported one. Two observable patterns carry real evidence, and both can be
detected deterministically:

  1. ADDRESSED, THEN ANSWERED.  Speaker 1: "Arjun, where are we?"
                                Speaker 2: "So right now sessions live in Redis…"
     A request addressed to a name, answered immediately by a different voice, is
     evidence that the answering voice is that name. (weight 1)

  2. SELF-INTRODUCTION.         "Hi, I'm Arjun", "this is Arjun".   (weight 2)

And one pattern carries NEGATIVE evidence, which is cheap and catches errors:

  3. You don't address yourself. If Speaker 1 says "Arjun, …", Speaker 1 is not
     Arjun — whatever else the evidence says.

Resolution is conservative. A tie between two names leaves the speaker unnamed;
one name claimed by two speakers goes to the one with more evidence, or to
neither on a tie. An unnamed speaker is a correct output: in our sample meeting
nobody ever says Priya's name, so "Speaker 1" is the honest answer.

Known failure mode (documented, not hidden): the evidence is only as good as the
diarization under it. If the answering words are attributed to the wrong voice,
the name goes to the wrong voice too. The evidence string makes that visible.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..schemas import Segment
from .glossary import Glossary, is_ordinary_speech

SPEAKER_RE = re.compile(r"Speaker \d+")

# Capitalised words that open a sentence with a comma but are NOT names.
# ("Okay, can you…" must not make anyone called "Okay".)
_DISCOURSE = {
    "okay", "ok", "alright", "all", "right", "so", "yes", "yeah", "yep", "no",
    "nope", "well", "now", "good", "great", "sure", "fine", "agreed",
    "understood", "thanks", "thank", "sorry", "hey", "hi", "hello", "morning",
    "afternoon", "evening", "everyone", "everybody", "guys", "team", "folks",
    "look", "listen", "honestly", "actually", "also", "and", "but", "then",
    "first", "second", "third", "finally", "next", "cool", "perfect", "nice",
    "wait", "hmm", "um", "uh", "oh", "anyway", "basically", "again", "today",
    "tomorrow", "yesterday", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "please", "anyone", "someone", "somebody",
    "maybe", "otherwise", "however", "meanwhile", "lastly", "overall",
    "interesting", "exactly", "correct", "true", "absolutely", "definitely",
    "understood", "noted", "done", "there", "here", "this", "that", "it",
}

_SENTENCE = re.compile(r"(?<=[.?!])\s+")
_VOC_START = re.compile(r"^([A-Z][a-z]+),\s+\S")
_VOC_END = re.compile(r",\s+([A-Z][a-z]+)\s*[?.!]$")
_REQUEST = re.compile(r"\?\s*$|\b(?:can|could|would|will) you\b|\bplease\b", re.I)
# "I'm Arjun" anywhere; "This is Arjun." only as a whole sentence — otherwise
# "this is Jenkins" (a tool) would name a person.
_INTRO = re.compile(r"\b(?:I'm|I am|my name is)\s+([A-Z][a-z]+)\b"
                    r"|^(?:(?:Hi|Hello|Hey),?\s+)?[Tt]his is\s+([A-Z][a-z]+)\s*[.,!]?$")

# A reply opener. "Sam, can you…?" followed by "Yes, I'll take it" in the SAME
# voice is almost certainly a speaker change that diarization missed.
_REPLY_START = re.compile(r"^(?:yes|yeah|yep|sure|okay|ok|no|nope|absolutely|"
                          r"will do|on it|sounds good|got it)\b", re.I)

W_INTRO = 2
W_ADDRESSED = 1


# --------------------------------------------------------------- utterances
@dataclass
class Utterance:
    """One speaker's words inside one Whisper segment (segment ids are kept, so
    downstream grounding can still anchor to a segment and a timestamp)."""
    segment_id: int
    speaker: str | None
    start: float
    end: float
    text: str


def _apply(text: str, edits) -> str:
    for e in edits:
        pattern = re.compile(r"(?<!\w)" + re.escape(e.original) + r"(?!\w)")
        text = pattern.sub(lambda _m, r=e.replacement: r, text)
    return text


def speaker_utterances(segments: list[Segment], edits=()) -> list[Utterance]:
    """
    The REFINED transcript, split by speaker.

    Diarization labels raw WORDS; refinement edits segment TEXT. When a segment
    has one speaker we use its refined text directly. When Whisper merged two
    speakers into one segment, we rebuild each speaker's piece from the raw words
    and re-apply that segment's accepted Stage 2 edits to each piece. (An edit
    whose phrase straddles the speaker change is not re-applied — rare, and the
    raw words are still correct speech.)
    """
    by_seg: dict[int, list] = defaultdict(list)
    for e in edits:
        if getattr(e, "accepted", False):
            by_seg[e.segment_id].append(e)

    out: list[Utterance] = []
    for seg in segments:
        speakers = {w.speaker for w in seg.words if w.speaker}
        if len(speakers) <= 1:
            out.append(Utterance(seg.id, next(iter(speakers), None),
                                 seg.start, seg.end, seg.text.strip()))
            continue
        runs: list[list] = []
        for w in seg.words:
            spk = w.speaker or (runs[-1][0] if runs else None)
            if runs and runs[-1][0] == spk:
                runs[-1][1].append(w)
            else:
                runs.append([spk, [w]])
        for spk, ws in runs:
            text = " ".join(w.word.strip() for w in ws)
            out.append(Utterance(seg.id, spk, ws[0].start, ws[-1].end,
                                 _apply(text, by_seg.get(seg.id, []))))
    return out


def render_utterances(utterances: list[Utterance]) -> str:
    return "\n".join(
        f"[{u.start:.0f}s] (segment {u.segment_id}) {u.speaker or 'Unknown'}: {u.text}"
        for u in utterances)


# ------------------------------------------------------------------ naming
@dataclass
class Evidence:
    kind: str                 # "addressed-then-answered" | "self-introduction"
    at: float
    quote: str
    reply: str = ""
    reply_at: float | None = None
    weight: int = 1

    def describe(self) -> str:
        if self.kind == "self-introduction":
            return f'introduced themself at {_clock(self.at)}: "{self.quote}"'
        return (f'addressed by name at {_clock(self.at)} ("{self.quote}") and '
                f'answered next at {_clock(self.reply_at or self.at)} '
                f'("{self.reply[:60]}")')


@dataclass
class NameBinding:
    speaker: str
    name: str
    evidence: list[Evidence]
    support: int
    contested_by: dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        text = f"{self.speaker} identified as {self.name}: " + "; ".join(
            e.describe() for e in self.evidence)
        if self.contested_by:
            text += f" (weaker evidence for {', '.join(self.contested_by)})"
        return text


@dataclass
class SuspectedMiss:
    """A question to someone else, 'answered' in the asker's own voice."""
    speaker: str
    addressee: str
    asked_at: float
    question: str
    reply: str
    segment_id: int
    reply_at: float

    def describe(self) -> str:
        return (f"{self.speaker} asked {self.addressee} a question at "
                f"{_clock(self.asked_at)} and the reply (\"{self.reply[:50]}\", "
                f"{_clock(self.reply_at)}) was attributed to the same voice — "
                f"likely a speaker change diarization missed")


@dataclass
class NamingResult:
    bindings: dict[str, NameBinding] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    excluded: dict[str, set[str]] = field(default_factory=dict)
    suspects: list[SuspectedMiss] = field(default_factory=list)
    # every name each voice has ANY evidence for, bound or not
    candidates: dict[str, set[str]] = field(default_factory=dict)

    def is_suspect(self, utterance) -> SuspectedMiss | None:
        """Is this utterance's voice label contradicted by the conversation?"""
        if utterance is None:
            return None
        return next((m for m in self.suspects
                     if m.segment_id == utterance.segment_id
                     and m.speaker == utterance.speaker
                     and utterance.start <= m.reply_at <= utterance.end + 0.01), None)

    def name_for(self, speaker: str | None) -> str | None:
        b = self.bindings.get(speaker or "")
        return b.name if b else None

    def label_for(self, name: str | None) -> str | None:
        key = (name or "").strip().lower()
        return next((b.speaker for b in self.bindings.values()
                     if b.name.lower() == key), None)

    def display(self, speaker: str | None) -> str:
        name = self.name_for(speaker)
        return f"{speaker} ({name}?)" if name else (speaker or "Unknown")

    def to_dict(self) -> dict:
        return {
            "names": {k: {"name": b.name, "support": b.support,
                          "evidence": [e.describe() for e in b.evidence],
                          "contested_by": b.contested_by}
                      for k, b in self.bindings.items()},
            "conflicts": self.conflicts,
            "suspected_diarization_errors": [m.describe() for m in self.suspects],
            "method": "rule-based: addressed-then-answered (w=1), "
                      "self-introduction (w=2), self-address exclusion",
        }


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _is_name(tok: str, glossary: Glossary | None) -> bool:
    return (tok.lower() not in _DISCOURSE and not is_ordinary_speech(tok)
            and not (glossary and glossary.contains(tok)))


def _vocative(sentence: str, glossary: Glossary | None) -> str | None:
    for rx in (_VOC_START, _VOC_END):
        m = rx.search(sentence)
        if m and _is_name(m.group(1), glossary):
            return m.group(1)
    return None


@dataclass
class _Turn:
    speaker: str | None
    start: float
    text: str


def _turns(utterances: list[Utterance]) -> list[_Turn]:
    turns: list[_Turn] = []
    for u in utterances:
        if turns and turns[-1].speaker == u.speaker:
            turns[-1].text += " " + u.text
        else:
            turns.append(_Turn(u.speaker, u.start, u.text))
    return turns


def infer_names(utterances: list[Utterance],
                glossary: Glossary | None = None) -> NamingResult:
    turns = _turns(utterances)
    votes: dict[str, dict[str, list[Evidence]]] = defaultdict(lambda: defaultdict(list))
    excluded: dict[str, set[str]] = defaultdict(set)

    for i, turn in enumerate(turns):
        if not turn.speaker:
            continue
        sentences = [s for s in _SENTENCE.split(turn.text.strip()) if s]
        for s in sentences:
            for m in _INTRO.finditer(s):
                found = m.group(1) or m.group(2)
                if _is_name(found, glossary):
                    votes[turn.speaker][found].append(
                        Evidence("self-introduction", turn.start, s, weight=W_INTRO))
        for j, s in enumerate(sentences):
            name = _vocative(s, glossary)
            if not name:
                continue
            excluded[turn.speaker].add(name)          # nobody addresses themself
            # Only a request that ENDS the turn hands the floor to the addressee.
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            if (j == len(sentences) - 1 and _REQUEST.search(s) and nxt
                    and nxt.speaker and nxt.speaker != turn.speaker):
                votes[nxt.speaker][name].append(Evidence(
                    "addressed-then-answered", turn.start, s,
                    reply=nxt.text, reply_at=nxt.start, weight=W_ADDRESSED))

    result = NamingResult(excluded={k: set(v) for k, v in excluded.items()},
                          suspects=_suspected_misses(utterances, glossary),
                          candidates={k: set(v) for k, v in votes.items()})
    candidates: dict[str, NameBinding] = {}
    for speaker, by_name in votes.items():
        scored = {}
        for name, evs in by_name.items():
            if name in excluded[speaker]:
                result.conflicts.append(
                    f"{speaker} would be '{name}', but {speaker} addressed "
                    f"'{name}' themself — rejected")
                continue
            scored[name] = sum(e.weight for e in evs)
        if not scored:
            continue
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            result.conflicts.append(
                f"{speaker}: equal evidence for {ranked[0][0]} and {ranked[1][0]} "
                f"— left unnamed")
            continue
        name, support = ranked[0]
        candidates[speaker] = NameBinding(speaker, name, by_name[name], support,
                                          dict(ranked[1:]))

    # One name, one voice.
    by_name: dict[str, list[NameBinding]] = defaultdict(list)
    for b in candidates.values():
        by_name[b.name].append(b)
    for name, bs in by_name.items():
        bs.sort(key=lambda b: -b.support)
        if len(bs) > 1 and bs[0].support == bs[1].support:
            result.conflicts.append(
                f"'{name}' fits {' and '.join(b.speaker for b in bs)} equally "
                f"— none named")
            continue
        result.bindings[bs[0].speaker] = bs[0]
        for loser in bs[1:]:
            result.conflicts.append(
                f"'{name}' also fit {loser.speaker} (weaker evidence) — kept for "
                f"{bs[0].speaker} only")
    return result


def _suspected_misses(utterances: list[Utterance],
                      glossary: Glossary | None) -> list[SuspectedMiss]:
    """
    Language cross-checking the voices. A request addressed by name to someone
    else, followed directly by a reply opener ("Yes, …") carrying the SAME voice
    label, means diarization most likely missed the turn change. (Found on our
    own sample: Sam's "Yes, I'll take the Grafana dashboard" was clustered with
    Priya — two similar voices — and became a wrong voice-based owner.)

    We do not try to repair the label. We mark it untrustworthy, so nothing
    downstream (owners, names) is built on it, and say so.
    """
    flat: list[tuple[int, str]] = []        # (utterance index, sentence)
    for i, u in enumerate(utterances):
        flat += [(i, s) for s in _SENTENCE.split(u.text.strip()) if s]
    misses: list[SuspectedMiss] = []
    for k in range(len(flat) - 1):
        (i, q), (j, r) = flat[k], flat[k + 1]
        asker, replier = utterances[i], utterances[j]
        name = _vocative(q, glossary)
        if (name and _REQUEST.search(q) and asker.speaker
                and replier.speaker == asker.speaker and _REPLY_START.match(r)):
            misses.append(SuspectedMiss(asker.speaker, name, asker.start, q, r,
                                        replier.segment_id, replier.start
                                        if i != j else asker.start))
    return misses


# ------------------------------------------------------- owner attribution
def attribute_owners(record, naming: NamingResult | None) -> list[str]:
    """
    After Stage 4: an action item whose owner was GROUNDED as a speaker label
    ("Speaker 2" said "I'll benchmark it") is shown under that speaker's name if,
    and only if, the name binding has evidence. The result is flagged as
    inferred and carries both pieces of evidence.
    """
    notes: list[str] = []
    if record is None or naming is None:
        return notes
    for a in record.action_items:
        if a.owner_source != "speaker":
            continue
        b = naming.bindings.get(a.owner or "")
        if not b:
            continue
        a.owner_label = a.owner
        a.owner_evidence = f"{a.owner_evidence}; {b.describe()}"
        a.owner, a.owner_source = b.name, "inferred"
        notes.append(f'"{a.task[:40]}": {a.owner_label} -> {b.name} (inferred)')
    return notes
