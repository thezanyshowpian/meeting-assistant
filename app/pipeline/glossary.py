"""
Domain glossary + the deterministic (non-LLM) correction pre-pass.

ASR errors on jargon are overwhelmingly PHONETIC substitutions: the model heard
the sound correctly and picked the wrong words. "cooper netties" -> "kubernetes".

So we index glossary terms by their Double Metaphone code and look up candidates
by sound, not spelling. Measured on real cases, this separates genuine ASR errors
from meaning-changing drift far better than string similarity does:

    cooper netties -> kubernetes   phonetic MATCH    jaro 0.75   (want: allow)
    sequel         -> SQL          phonetic MATCH    jaro 0.67   (want: allow)
    deadline       -> headline     phonetic MISMATCH jaro 0.87   (want: REJECT)
    might          -> will         phonetic MISMATCH jaro 0.48   (want: REJECT)

Note rows 3 and 4: string similarity would have *accepted* "deadline"->"headline"
(0.87) and *rejected* "sequel"->"SQL" (0.67). Both wrong. Phonetics gets both right.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from metaphone import doublemetaphone

# Words we never treat as mis-heard jargon, however they sound. Without this the
# deterministic pass fires on ordinary speech.
_COMMON_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "of", "to", "in",
    "on", "at", "for", "with", "by", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "have", "has", "had", "will",
    "would", "can", "could", "should", "may", "might", "must", "shall", "we",
    "you", "they", "he", "she", "it", "i", "this", "that", "these", "those",
    "there", "here", "what", "when", "where", "who", "how", "why", "all", "any",
    "some", "no", "not", "yes", "okay", "ok", "well", "just", "like", "think",
}


def normalize(text: str) -> str:
    """Casefold, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def phonetic_key(text: str) -> tuple[str, str]:
    """Double Metaphone of the phrase with spaces removed (so 'post gres'=='postgres')."""
    return doublemetaphone(normalize(text).replace(" ", ""))


def is_ordinary_speech(text: str) -> bool:
    """
    True if every word is ordinary English. Used to stop a non-glossary
    replacement from drifting into everyday vocabulary — domain corrections
    should look like jargon, not like common words.
    """
    words = normalize(text).split()
    return bool(words) and all(w in _COMMON_WORDS for w in words)


def phonetically_equal(a: str, b: str) -> bool:
    """True if two phrases plausibly sound the same."""
    pa, pb = phonetic_key(a), phonetic_key(b)
    if not pa[0] and not pb[0]:
        return False
    candidates = [
        (pa[0], pb[0]),
        (pa[0], pb[1]),
        (pa[1], pb[0]),
        (pa[1], pb[1]),
    ]
    return any(x and y and x == y for x, y in candidates)


@dataclass
class GlossaryMatch:
    source_text: str      # what was in the transcript
    term: str             # the canonical glossary term
    start_word: int       # word index in the segment
    end_word: int


class Glossary:
    """Canonical domain terms, indexed by sound."""

    def __init__(self, terms: list[str]):
        self.terms = [t for t in dict.fromkeys(t.strip() for t in terms) if t.strip()]
        self._normalized = {normalize(t) for t in self.terms}
        self._by_sound: dict[str, list[str]] = {}
        for term in self.terms:
            for code in phonetic_key(term):
                if code:
                    self._by_sound.setdefault(code, []).append(term)

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_json(cls, path: str) -> "Glossary":
        with open(path) as fh:
            data = json.load(fh)
        terms = data["terms"] if isinstance(data, dict) else data
        return cls(terms)

    @classmethod
    def empty(cls) -> "Glossary":
        return cls([])

    def __len__(self) -> int:
        return len(self.terms)

    # ---------------------------------------------------------------- lookup
    def contains(self, text: str) -> bool:
        """Is this already a correct glossary term?"""
        return normalize(text) in self._normalized

    def candidates(self, phrase: str) -> list[str]:
        """Glossary terms that SOUND like this phrase."""
        out: list[str] = []
        for code in phonetic_key(phrase):
            if code:
                out.extend(self._by_sound.get(code, []))
        return list(dict.fromkeys(out))

    def best_match(self, phrase: str) -> str | None:
        """
        The glossary term this phrase was most likely a mis-hearing of, or None.

        Returns None for phrases that are already correct, for ordinary English
        words, and for anything with no phonetic match — the deterministic pass
        must be conservative, since it applies without any model in the loop.
        """
        norm = normalize(phrase)
        if not norm or norm in self._normalized:
            return None                       # already correct — nothing to do
        if all(w in _COMMON_WORDS for w in norm.split()):
            return None                       # ordinary speech, not jargon
        matches = self.candidates(phrase)
        return matches[0] if matches else None


# Shipped sample. Users can upload their own through the UI; this exists so the
# app is useful out of the box and so judges see the mechanism working.
SAMPLE_TERMS = [
    "Kubernetes", "PostgreSQL", "Postgres", "SQL", "NoSQL", "Redis", "Kafka",
    "nginx", "Docker", "Terraform", "Ansible", "Grafana", "Prometheus",
    "OAuth", "JWT", "gRPC", "GraphQL", "REST API", "webhook", "CI/CD",
    "Jenkins", "GitHub Actions", "staging", "rollback", "latency", "throughput",
    "sprint", "standup", "retrospective", "backlog", "epic", "Jira", "Figma",
    "onboarding", "churn", "MRR", "ARR", "KPI", "OKR", "runway", "burn rate",
    "PyTorch", "TensorFlow", "Hugging Face", "inference", "fine-tuning",
    "embedding", "quantization", "GPU", "CUDA", "Whisper", "transformer",
]


def default_glossary() -> Glossary:
    return Glossary(SAMPLE_TERMS)
