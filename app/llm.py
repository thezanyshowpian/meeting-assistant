"""
Minimal LLM client — stdlib only, no extra dependencies.

Two backends behind one interface:
  - "ollama"  : local models (Stage 2). No key, no network, no rate limit.
  - "openai"  : any OpenAI-compatible endpoint (Groq / Cerebras) for Stage 3.

Stage 3 FALLS BACK to Ollama when no API key is present. That matters for the
submission: judges can run the whole application with zero API keys, and it still
satisfies the "two distinct language models" requirement because Stage 2 and
Stage 3 use different models either way.

temperature=0 everywhere: correction and extraction are not creative tasks, and
deterministic runs mean a judge re-running the demo sees identical output.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def load_env(path: str | None = None) -> None:
    """
    Minimal .env loader (no dependency). Real environment variables always win,
    so an export on the command line overrides the file.
    """
    env_file = Path(path or Path(__file__).resolve().parent.parent / ".env")
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_TIMEOUT = 300

# Ollama defaults its context window LOW (often 2048-4096) no matter what the
# model supports. Left unset, a 30-minute transcript is silently truncated and
# Stage 3 summarises only the opening minutes while reporting a complete record.
# This is set explicitly for that reason.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))

# Cloudflare fronts several of these APIs and 403s the default urllib agent.
USER_AGENT = "meeting-assistant/1.0 (+https://github.com/)"


class LLMError(Exception):
    def __init__(self, message: str, *, user_message: str | None = None):
        super().__init__(message)
        self.user_message = user_message or message


class LLMClient:
    def __init__(self, backend: str, model: str, *, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0):
        self.backend = backend
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature

    # ------------------------------------------------------------------ api
    def chat_json(self, system: str, user: str, *, timeout: int = DEFAULT_TIMEOUT) -> dict:
        """Ask for a JSON object back. Raises LLMError on failure or bad JSON."""
        raw = self._chat(system, user, timeout=timeout)
        return _parse_json(raw)

    # ------------------------------------------------------------- backends
    def _chat(self, system: str, user: str, *, timeout: int) -> str:
        if self.backend == "ollama":
            return self._ollama(system, user, timeout)
        return self._openai_compatible(system, user, timeout)

    def _ollama(self, system: str, user: str, timeout: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": "json",                      # constrain output to JSON
            "options": {"temperature": self.temperature, "num_ctx": OLLAMA_NUM_CTX},
            "think": False,                        # qwen3 is a reasoning model; off for speed
        }
        data = self._post(f"{OLLAMA_URL}/api/chat", payload, {}, timeout,
                          hint="Is `ollama serve` running?")
        return data.get("message", {}).get("content", "")

    def _openai_compatible(self, system: str, user: str, timeout: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = self._post(f"{self.base_url}/chat/completions", payload, headers,
                          timeout, hint="Check your API key and rate limits.")
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _post(url: str, payload: dict, headers: dict, timeout: int, *, hint: str) -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     # REQUIRED. urllib defaults to "Python-urllib/3.x", which
                     # Cloudflare's browser-integrity check rejects with a 403
                     # "error code: 1010" BEFORE the request reaches the API.
                     # Without this, every Groq call silently fails and Stage 3
                     # returns an empty record that looks like "no decisions".
                     "User-Agent": USER_AGENT,
                     **headers},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()[:300]
            raise LLMError(
                f"HTTP {exc.code} from {url}: {body}",
                user_message=f"The language model service returned an error "
                             f"({exc.code}). {hint}",
            ) from exc
        except Exception as exc:                                   # noqa: BLE001
            raise LLMError(
                f"request to {url} failed: {exc}",
                user_message=f"Could not reach the language model service. {hint}",
            ) from exc


def _parse_json(raw: str) -> dict:
    """
    Models sometimes wrap JSON in prose or fences even in JSON mode. Recover what
    we can rather than failing the whole run on a formatting quirk.
    """
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        raw = raw[4:] if raw.lower().startswith("json") else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
    raise LLMError(
        f"model did not return valid JSON: {raw[:200]!r}",
        user_message="The language model returned a malformed response.",
    )


# --------------------------------------------------------------- factories
def stage2_client(model: str | None = None) -> LLMClient:
    """
    Stage 2 is always local: deterministic, offline, no rate limits.
    Override with STAGE2_MODEL to compare models, e.g. STAGE2_MODEL=qwen3:14b
    """
    return LLMClient("ollama", model or os.environ.get("STAGE2_MODEL", "qwen3:8b"))


# Groq's free tier caps at 6,000 TOKENS PER MINUTE. A Stage 3 call costs the
# transcript plus ~1,200 tokens of system prompt and output, so the API is only
# viable for shorter recordings — and chunking cannot rescue it, because you
# would have to wait a minute between chunks. We route by size, with ~25%
# headroom because the estimate is approximate and exceeding the cap does not
# degrade gracefully, it fails outright.
GROQ_TPM_LIMIT = 6_000
STAGE3_PROMPT_OVERHEAD = 1_200
GROQ_TOKEN_THRESHOLD = 4_500          # ≈20 minutes of audio

GROQ_BASE = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_LOCAL_STAGE3 = "qwen3:14b"


def estimate_tokens(text: str) -> int:
    """Rough token count: English averages ~1.33 tokens per word."""
    return int(len(text.split()) * 1.33) + STAGE3_PROMPT_OVERHEAD


def local_stage3_client(model: str | None = None) -> tuple[LLMClient, str]:
    name = model or os.environ.get("STAGE3_MODEL_LOCAL", DEFAULT_LOCAL_STAGE3)
    return LLMClient("ollama", name), f"Ollama / {name} (local)"


def groq_client(model: str | None = None) -> tuple[LLMClient, str] | None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    name = model or os.environ.get("STAGE3_MODEL", DEFAULT_GROQ_MODEL)
    return (LLMClient("openai", name, base_url=GROQ_BASE, api_key=key),
            f"Groq / {name}")


def stage3_client(transcript: str = "", model: str | None = None) -> tuple[LLMClient, str]:
    """
    Pick the Stage 3 backend by transcript SIZE.

    STAGE3_BACKEND: auto (default) | local | groq
      auto  - Groq for short recordings (better quality), local for long ones
              (no throughput cap). This is the only routing that is correct for
              both cases.
      local - force local
      groq  - force Groq regardless of size (will fail on long transcripts)

    Callers should fall back to local if the returned client errors — see
    summarize(). Routing by size prevents the common failure; the fallback
    catches the rest (outages, rate limits, deprecated model IDs).
    """
    mode = os.environ.get("STAGE3_BACKEND", "auto").lower()
    if mode == "local":
        return local_stage3_client(model)

    tokens = estimate_tokens(transcript)
    if mode == "groq":
        return groq_client(model) or local_stage3_client(model)

    remote = groq_client(model)
    if remote and tokens <= GROQ_TOKEN_THRESHOLD:
        client, label = remote
        return client, f"{label} — {tokens} est. tokens, under the {GROQ_TOKEN_THRESHOLD} limit"

    client, label = local_stage3_client(model)
    if remote:
        label += (f" — {tokens} est. tokens exceeds the {GROQ_TOKEN_THRESHOLD} "
                  f"Groq free-tier budget, so this ran locally")
    return client, label
