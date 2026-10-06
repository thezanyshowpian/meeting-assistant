#!/usr/bin/env python3
"""
Write docs/PROMPTS.md from the prompt constants the pipeline actually uses.

Generated, not hand-copied, so the documented prompts can never drift from the
code. Re-run after changing any prompt:

    python scripts/export_prompts.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.pipeline import refine, summarize  # noqa: E402


def block(text: str) -> str:
    return "```text\n" + text.strip() + "\n```\n"


def main() -> int:
    out = [
        "# Prompts and model instructions",
        "",
        "> **Generated** by `scripts/export_prompts.py` from the constants in "
        "`app/pipeline/refine.py` and `app/pipeline/summarize.py`. Do not edit by "
        "hand; re-run the script.",
        "",
        "Every LLM call uses `temperature=0` and JSON output mode, and Ollama runs "
        "with `think=false` and `num_ctx=16384`. In every case the model's output "
        "is a proposal: it is checked by deterministic code (Stage 2's gates, "
        "Stage 4's grounding) before anything reaches the user.",
        "",
        "## Stage 2a: context veto of glossary matches (`ADJUDICATE_PROMPT`)",
        "",
        "The deterministic phonetic pass proposes glossary corrections; this prompt "
        "asks the model only to KEEP or REJECT each one in context.",
        "",
        "**System:**",
        "",
        block(refine.ADJUDICATE_PROMPT),
        "**User message:** `Proposals:` followed by one numbered entry per proposal, "
        "`N. in \"<segment text>\" replace '<phrase>' with '<term>'?`, then "
        "`Return the verdicts as JSON.`",
        "",
        "## Stage 2b: off-glossary edit proposals (`SYSTEM_PROMPT` + `FEW_SHOT`)",
        "",
        "**System:**",
        "",
        block(refine.SYSTEM_PROMPT),
        "**User message:** the few-shot examples below, then `NOW THE REAL TASK.`, "
        "the glossary, the transcript as `segment N (low-confidence: <words>): \"text\"` "
        "lines (Whisper's unsure words flagged), and "
        "`Return the JSON edit list.`",
        "",
        "**Few-shot examples:**",
        "",
        block(refine.FEW_SHOT),
        "## Stage 3: meeting record, without speaker labels",
        "",
        "Used when diarization did not run.",
        "",
        "**System:**",
        "",
        block(summarize.system_prompt(False)),
        "## Stage 3: meeting record, with speaker labels",
        "",
        "Used when diarization ran. Only the owner rules differ.",
        "",
        "**Owner rules in this variant:**",
        "",
        block(summarize.OWNER_RULES_DIARIZED),
        "**User message (both variants):** `Transcript:` followed by one line per "
        "segment (`[12s] (segment 3) text`, or `[12s] (segment 3) Speaker 2: text` "
        "when diarized), then: *Produce the meeting record as JSON. Remember: quote "
        "your evidence, and use null for any owner or deadline that was not "
        "stated.*",
        "",
        "## What is deliberately NOT prompted",
        "",
        "- **Speaker naming** (`naming.py`): rules, not a model. A model asked "
        "\"who is Speaker 2?\" answers, and a guess looks like a supported answer.",
        "- **Verification** (`verification.py`, `ground.py`): deterministic code, "
        "testable without any model.",
        "",
    ]
    path = os.path.join(ROOT, "docs", "PROMPTS.md")
    with open(path, "w") as fh:
        fh.write("\n".join(out))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
