"""
Output formats.

The PS requires a human-readable AND a machine-readable record that "convey the
same decisions and tasks". Both are generated from the SAME PipelineResult
object, so they cannot disagree — agreement is structural, not a thing we have
to remember to maintain.

Missing owners and deadlines render as "unspecified" in both.
"""
from __future__ import annotations

import json

from .schemas import UNSPECIFIED


def to_json(result) -> str:
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


def to_markdown(result) -> str:
    r = result.record
    lines: list[str] = ["# Meeting Record", ""]

    if result.warnings:
        lines.append("> **Notes**")
        lines += [f"> - {w}" for w in result.warnings]
        lines.append("")

    if not r:
        lines.append("_No meeting record was produced._")
        return "\n".join(lines)

    if r.summary:
        lines += ["## Summary", "", r.summary, ""]

    lines += ["## Minutes", ""]
    lines += [f"- {m}" for m in r.minutes] if r.minutes else ["_No minutes recorded._"]
    lines.append("")

    lines += ["## Key Decisions", ""]
    if r.decisions:
        for d in r.decisions:
            lines.append(f"- **{d.statement}**")
            if d.evidence:
                lines.append(f'  - evidence: "{d.evidence}"')
    else:
        lines.append("_No decisions were reached._")
    lines.append("")

    lines += ["## Action Items", ""]
    if r.action_items:
        lines += ["| Task | Owner | Deadline |", "| :-- | :-- | :-- |"]
        for a in r.action_items:
            lines.append(f"| {a.task} | {a.owner_display} | "
                         f"{a.deadline or UNSPECIFIED} |")
        flagged = [a for a in r.action_items if a.owner_source in ("inferred", "speaker")]
        if flagged:
            lines += ["", "**How owners marked _(inferred)_ / _(voice only)_ are known:**", ""]
            lines += [f"- *{a.task}*: {a.owner_evidence}" for a in flagged]
    else:
        lines.append("_No action items were assigned._")
    lines.append("")

    naming = getattr(result, "naming", None)
    if naming and (naming.bindings or naming.conflicts):
        lines += ["## Speaker Names (inferred from the conversation)", ""]
        for b in naming.bindings.values():
            lines.append(f"- {b.describe()}")
        for c in naming.conflicts:
            lines.append(f"- _not named:_ {c}")
        lines.append("")

    if result.refinement and result.refinement.applied:
        lines += ["## Terminology Corrections Applied", "",
                  "| Original | Corrected | Reason |", "| :-- | :-- | :-- |"]
        for e in result.refinement.applied:
            lines.append(f"| {e.original} | {e.replacement} | {e.reason} |")
        lines.append("")

    return "\n".join(lines)


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def transcripts_markdown(result) -> str:
    out = (f"# Raw Transcript\n\n{result.raw_text}\n\n"
           f"# Refined Transcript\n\n{result.refined_text}\n")
    if getattr(result, "diarization", None):
        d = result.diarization
        naming = getattr(result, "naming", None)
        out += (f"\n# Speaker-labelled Transcript\n\n"
                f"_{d.num_speakers} speakers detected ({d.method}). A name in "
                f"brackets with '?' is inferred from evidence in the conversation; "
                f"otherwise labels are anonymous._\n\n")
        utts = getattr(result, "utterances", None)
        if utts:
            turns: list[list] = []
            for u in utts:
                if turns and turns[-1][0] == u.speaker:
                    turns[-1][2] += " " + u.text
                else:
                    turns.append([u.speaker, u.start, u.text])
        else:
            turns = [[t.speaker, t.start, t.text] for t in d.turns]
        show = naming.display if naming else (lambda s: s)
        out += "\n\n".join(f"**{show(spk)}** [{_clock(t0)}]: {text}"
                           for spk, t0, text in turns) + "\n"
    return out
