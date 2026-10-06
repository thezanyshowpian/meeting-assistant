#!/usr/bin/env python3
"""
Groq connectivity smoke test.

Exists because a failed Groq call currently LOOKS like "the model found no
decisions" — the pipeline catches the error, degrades, and the empty record is
indistinguishable from a legitimately empty meeting. This prints the raw truth.

Checks, in order:
  1. is a key present?
  2. which model IDs does the account actually have? (IDs get deprecated)
  3. is our configured model in that list?
  4. does a real request succeed, and what comes back?

USAGE
    python checks/check_groq.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm import USER_AGENT, load_env  # noqa: E402

load_env()
BASE = "https://api.groq.com/openai/v1"


def request(path: str, key: str, payload: dict | None = None) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode() if payload else None,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 # Without a real User-Agent, Cloudflare returns
                 # "403 error code: 1010" before Groq sees the request.
                 "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()
    except Exception as exc:                                       # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def main() -> int:
    print("=" * 70)
    print("GROQ SMOKE TEST")
    print("=" * 70)

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("  FAIL  no GROQ_API_KEY found (checked environment and .env)")
        return 1
    print(f"  key present: {key[:8]}…{key[-4:]}")

    configured = os.environ.get("STAGE3_MODEL", "llama-3.3-70b-versatile")
    print(f"  configured model: {configured}")

    print("\n--- available models ---")
    status, body = request("/models", key)
    if status != 200:
        print(f"  FAIL  HTTP {status}: {body[:400]}")
        return 1
    try:
        ids = sorted(m["id"] for m in json.loads(body).get("data", []))
    except Exception as exc:                                       # noqa: BLE001
        print(f"  FAIL  could not parse model list: {exc}\n{body[:300]}")
        return 1
    for mid in ids:
        mark = "  <-- configured" if mid == configured else ""
        print(f"    {mid}{mark}")

    if configured not in ids:
        print(f"\n  *** {configured!r} IS NOT AVAILABLE ***")
        print("  This is almost certainly why Stage 3 returned nothing: the "
              "request 400s, the pipeline degrades, and the empty record looks "
              "like 'no decisions found'.")
        candidates = [m for m in ids if "llama" in m and "70b" in m] or ids[:5]
        print(f"  Try one of: {candidates}")
        return 1

    print("\n--- live request (JSON mode, same shape Stage 3 uses) ---")
    status, body = request("/chat/completions", key, {
        "model": configured,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": 'Return {"ok": true, "decisions": []}'},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    })
    if status != 200:
        print(f"  FAIL  HTTP {status}: {body[:500]}")
        return 1
    try:
        content = json.loads(body)["choices"][0]["message"]["content"]
        print(f"  OK  response: {content.strip()[:200]}")
        json.loads(content)
        print("  OK  response is valid JSON")
    except Exception as exc:                                       # noqa: BLE001
        print(f"  FAIL  unexpected response shape: {exc}\n{body[:400]}")
        return 1

    print("\n" + "=" * 70)
    print("RESULT: Groq reachable, model valid, JSON mode works")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
