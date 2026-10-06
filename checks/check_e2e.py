#!/usr/bin/env python3
"""
End-to-end pipeline check — audio in, meeting record out.

Verifies the walking skeleton actually walks: all three stages run in order, the
outputs exist and agree, and both export formats are produced.

USAGE
    python checks/check_e2e.py --audio spikes/jfk.flac
    python checks/check_e2e.py --audio spikes/jfk.flac --no-llm   # skip the LLM pass
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.export import to_json, to_markdown  # noqa: E402
from app.pipeline.orchestrator import run_pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("END-TO-END PIPELINE CHECK")
    print("=" * 70)

    result = run_pipeline(
        args.audio, use_llm_refinement=not args.no_llm,
        progress=lambda stage, msg: print(f"  [{stage}] {msg}", flush=True),
    )

    failures: list[str] = []

    print("\n--- stage timings ---")
    for stage, seconds in result.stage_times.items():
        print(f"  {stage:<12} {seconds:6.1f}s")

    if result.warnings:
        print("\n--- warnings ---")
        for w in result.warnings:
            print(f"  ! {w}")

    if not result.ok:
        print(f"\nFAILED at {result.failed_stage}: {result.error}")
        return 1

    print("\n--- transcripts ---")
    print(f"  raw     ({len(result.raw_text):5d} chars): {result.raw_text[:90]}…")
    print(f"  refined ({len(result.refined_text):5d} chars): {result.refined_text[:90]}…")
    if not result.raw_text:
        failures.append("raw transcript is empty")

    ref = result.refinement
    if ref:
        print(f"\n--- refinement ({ref.model}) ---")
        print(f"  applied : {len(ref.applied)}")
        print(f"  rejected: {len(ref.rejected)}")
        for e in ref.applied[:5]:
            print(f"    + {e.original!r} -> {e.replacement!r}  ({e.reason[:50]})")
        for e in ref.rejected[:5]:
            print(f"    - {e.original!r} -> {e.replacement!r}  "
                  f"[{'; '.join(e.rejections)[:70]}]")
    else:
        failures.append("refinement did not run")

    rec = result.record
    if rec:
        print(f"\n--- meeting record ({rec.model}) ---")
        print(f"  summary     : {rec.summary[:80]!r}")
        print(f"  minutes     : {len(rec.minutes)}")
        print(f"  decisions   : {len(rec.decisions)}")
        print(f"  action items: {len(rec.action_items)}")
        for a in rec.action_items:
            print(f"    * {a.task[:50]!r} owner={a.owner_display} "
                  f"deadline={a.deadline_display}")
        # The point of the optional-by-default schema: unstated stays unstated.
        for a in rec.action_items:
            if a.owner is not None and a.owner.strip().lower() in {"unspecified", "unknown", "n/a"}:
                failures.append(f"owner placeholder leaked as a value: {a.owner!r}")
    else:
        failures.append("meeting record did not run")

    print("\n--- exports ---")
    try:
        payload = to_json(result)
        json.loads(payload)
        print(f"  JSON     : {len(payload)} chars, parses")
    except Exception as exc:                                       # noqa: BLE001
        failures.append(f"JSON export failed: {exc}")
    try:
        md = to_markdown(result)
        assert "# Meeting Record" in md
        print(f"  Markdown : {len(md)} chars")
    except Exception as exc:                                       # noqa: BLE001
        failures.append(f"Markdown export failed: {exc}")

    print("\n" + "=" * 70)
    if failures:
        print(f"RESULT: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: end-to-end pipeline OK")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
