"""
Cache Stage 1 transcripts for evaluation runs.

Whisper is the slow stage (~9 minutes for the 30-minute meeting) and is
deterministic for a given file, so re-running it to test a change in Stage 1.5,
2, 3 or 4 wastes time. The cache is keyed by the audio file's size and mtime, so
regenerating the audio invalidates it. Evaluation only — the app never uses it.
"""
from __future__ import annotations

import os
import pickle

from app.pipeline.transcribe import Stage1Transcriber

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "data", ".cache")


class CachedTranscriber:
    def __init__(self):
        self._inner = None

    def transcribe(self, path: str):
        st = os.stat(path)
        key = f"{os.path.basename(path)}.{st.st_size}.{int(st.st_mtime)}.pkl"
        cached = os.path.join(CACHE_DIR, key)
        if os.path.exists(cached):
            print(f"  (stage 1 from cache: {key})", flush=True)
            with open(cached, "rb") as fh:
                return pickle.load(fh)
        self._inner = self._inner or Stage1Transcriber()
        transcript = self._inner.transcribe(path)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cached, "wb") as fh:
            pickle.dump(transcript, fh)
        return transcript
