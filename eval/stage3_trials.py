#!/usr/bin/env python3
"""
Stage 3 repeated trials — decide the backend on rates, not anecdotes.

Why this exists: we changed our mind about Groq vs local twice on single runs.
After two Groq runs dropped an unassigned action item we called it systematic;
a third run found it and we called it variance; a fourth dropped it again. Each
conclusion came from one data point. Hosted inference is non-deterministic even
at temperature=0, so the only honest comparison is a RATE over repeated trials.

Method: transcribe ONCE, then run Stage 3 (summarise) + Stage 4 (grounding)
N times per backend on the IDENTICAL transcript. The model is the only variable.
Local is expected to be deterministic at temperature=0 — fewer trials suffice,
and identical outputs across them are themselves a result worth confirming.

USAGE
    python eval/stage3_trials.py                  # 5 Groq, 3 local
    python eval/stage3_trials.py --n-groq 10 --n-local 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from app.pipeline.ground import ground  # noqa: E402
from app.pipeline.summarize import summarize  # noqa: E402
from app.pipeline.transcribe import Stage1Transcriber  # noqa: E402
from eval.evaluate import AUDIO, SCRIPT, contains_idea, declines  # noqa: E402


def score(record, truth) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    found = [d.statement for d in record.decisions]
    for d in truth["decisions"]:
        if d.get("scored") is False:
            continue
        present = any(contains_idea(f, d["statement"]) for f in found)
        if d["must_appear"]:
            checks[f"decision found: {d['statement'][:34]}"] = present
        else:
            # Reported AS declined ("do not redesign…") is correct, not the trap.
            present = any(contains_idea(f, d["statement"]) and not declines(f)
                          for f in found)
            checks[f"proposal NOT a decision: {d['statement'][:26]}"] = not present
            # A declined proposal must not sneak back in as an action item
            # either - the risk of loosening the action-item definition.
            as_task = any(contains_idea(a.task, d["statement"])
                          for a in record.action_items)
            checks[f"proposal NOT an action: {d['statement'][:28]}"] = not as_task

    for a in truth["action_items"]:
        label = a["task"][:34]
        m = next((x for x in record.action_items if contains_idea(x.task, a["task"])), None)
        checks[f"action found: {label}"] = m is not None
        if m is None:
            checks[f"  fields correct: {label}"] = False
            continue
        owner_ok = (m.owner is None) if a["owner"] is None else \
            bool(m.owner) and a["owner"].lower() in m.owner.lower()
        dl_ok = (m.deadline is None) if a["deadline"] is None else \
            bool(m.deadline) and a["deadline"].lower() in m.deadline.lower()
        checks[f"  fields correct: {label}"] = owner_ok and dl_ok
    return checks


GROQ_GAP_S = 15   # space Groq calls: a 429 measures the rate limit, not the model


def run_trials(backend: str, n: int, segments, truth) -> tuple[dict, list]:
    os.environ["STAGE3_BACKEND"] = backend
    tally: dict[str, int] = {}
    runs = []
    for i in range(n):
        if backend == "groq" and i > 0:
            time.sleep(GROQ_GAP_S)
        t0 = time.time()
        record = summarize(segments)
        result = ground(record, segments)
        checks = score(result.record, truth)
        for k, ok in checks.items():
            tally[k] = tally.get(k, 0) + int(ok)
        runs.append({
            "model": record.model,
            "seconds": round(time.time() - t0, 1),
            "passed": sum(checks.values()),
            "of": len(checks),
            "warnings": result.record.warnings,
            "actions": [a.task for a in result.record.action_items],
        })
        print(f"  [{backend} {i + 1}/{n}] {runs[-1]['passed']}/{runs[-1]['of']} "
              f"in {runs[-1]['seconds']}s  ({record.model.split(' — ')[0]})", flush=True)
        for w in result.record.warnings:
            print(f"      ! {w}")
        # Show WHAT grounding removed. A drop is either Stage 4 catching a real
        # fabrication, or a recall bug: a valid item whose quote was paraphrased
        # too far to match. The count alone can't tell those apart.
        for statement, why in result.dropped_decisions:
            print(f"      - dropped decision: {statement!r} ({why})")
        for task, why in result.dropped_actions:
            print(f"      - dropped action:   {task!r} ({why})")
        for note in result.downgraded:
            print(f"      ~ {note}")
        for note in result.context_verified + result.merged:
            print(f"      + {note}")
    return tally, runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-groq", type=int, default=5)
    ap.add_argument("--n-local", type=int, default=3)
    args = ap.parse_args()

    with open(SCRIPT) as fh:
        truth = json.load(fh)["ground_truth"]

    print("=" * 74)
    print("STAGE 3 TRIALS — same transcript, repeated, per backend")
    print("=" * 74)
    print("  transcribing once…", flush=True)
    segments = Stage1Transcriber().transcribe(AUDIO).segments

    results = {}
    for backend, n in (("groq", args.n_groq), ("local", args.n_local)):
        if n <= 0:
            continue
        if backend == "groq" and not os.environ.get("GROQ_API_KEY"):
            from app.llm import load_env
            load_env()
            if not os.environ.get("GROQ_API_KEY"):
                print("  (skipping groq: no GROQ_API_KEY)")
                continue
        print(f"\n--- {backend}: {n} trial(s) ---")
        results[backend] = (n, *run_trials(backend, n, segments, truth))

    keys = list(next(iter(results.values()))[1].keys()) if results else []
    print("\n" + "=" * 74)
    print(f"{'check':<52}" + "".join(f"{b:>11}" for b in results))
    print("-" * 74)
    for k in keys:
        row = f"{k:<52}"
        for b, (n, tally, _) in results.items():
            row += f"{tally.get(k, 0):>7}/{n:<3}"
        print(row)
    print("-" * 74)
    for b, (n, tally, runs) in results.items():
        total = sum(tally.values())
        possible = n * len(keys)
        distinct = len({json.dumps(r["actions"]) for r in runs})
        print(f"  {b:<6} {total}/{possible} checks ({total / possible:.0%})  |  "
              f"{distinct} distinct action-item set(s) across {n} run(s)")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
