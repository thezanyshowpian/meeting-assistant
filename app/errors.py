"""
Pipeline error types.

Every error carries a `user_message`: a plain-sentence explanation safe to show
in the UI. The PS scores "handle unsupported, empty, or unreadable files with a
clear error message", so the user-facing wording is part of the contract, not an
afterthought — we never surface a raw library traceback.
"""
from __future__ import annotations


class PipelineError(Exception):
    """Base for all pipeline errors."""

    def __init__(self, message: str, *, user_message: str | None = None,
                 cause: Exception | None = None):
        super().__init__(message)
        self.user_message = user_message or message
        self.cause = cause


class AudioValidationError(PipelineError):
    """Layer 1 — the file itself is unusable (missing, empty, not a file)."""


class AudioDecodeError(PipelineError):
    """Layer 2 — the bytes exist but are not decodable audio."""


class TranscriptionError(PipelineError):
    """The speech-to-text model failed."""
