#!/usr/bin/env python3
"""
Build data/long_meeting_script.json (and the demo glossary) from script.txt.

The ~30-minute demo meeting. Every behaviour the pipeline claims is planted on
purpose and recorded in the answer key below, so evaluate.py can score it:

  naming      Ravi, Hannah, Leo, Grace are addressed by name and answer; Omar
              introduces himself; the chair is NEVER named (must stay anonymous).
  conflict    the chair asks "Leo, can you check the restore?" and GRACE answers
              and takes it — the answering voice gets a vote for "Leo". Leo's own
              voice has more evidence, so the name must still land on Leo.
  owners      stated ("Grace, can you schedule…"), first-person + inferred name
              ("I'll set up budget alerts"), voice-only (the anonymous chair
              commits), and owner-less ("Someone needs to audit…").
  deadlines   stated, absent, and stated only in the REQUEST (the known weak spot).
  decisions   seven real ones, including a negation ("not going to build") and a
              bare "Do that." approval; three traps: parked, declined, tentative.
  Stage 2     six terms are MISPRONOUNCED by the voice ({{Meant|spoken}}), so
              Whisper may genuinely mishear them and refinement has real work.
  routing     ~30 min is ~8,000 tokens: too big for Groq's free tier, so Stage 3
              must route to the local model.

USAGE
    python scripts/long_meeting/build.py
    python scripts/make_sample_meeting.py --name long_meeting     # macOS, ~5 min
    python eval/evaluate.py --fixture long
"""
from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MARK = re.compile(r"\{\{([^|}]+)\|([^}]+)\}\}")

VOICES = {                       # distinct and dissimilar: Tessa is left out on
    "Diana": ["Samantha"],       # purpose — it was confused with Samantha in the
    "Ravi": ["Rishi"],           # sample meeting
    "Hannah": ["Karen"],
    "Leo": ["Daniel"],
    "Grace": ["Moira"],
    "Omar": ["Fred", "Alex", "Tom"],
}

GLOSSARY = ["Kafka", "Kubernetes", "Redis", "PostgreSQL", "Postgres", "OAuth",
            "Okta", "Grafana", "Datadog", "Terraform", "Sentry", "TestFlight",
            "Flutter", "webhook", "idempotency", "PagerDuty", "Confluence",
            "staging", "rollback"]

GROUND_TRUTH = {
    "decisions": [
        {"statement": "Add idempotency keys to the payment webhook", "must_appear": True},
        {"statement": "Use Okta instead of building our own token service", "must_appear": True,
         "why": "Also carries the negation: 'We are not going to build our own token service.'"},
        {"statement": "Rotate OAuth tokens every 90 days", "must_appear": True},
        {"statement": "Delay the mobile release", "must_appear": True,
         "why": "Moved from the 20th to the 27th."},
        {"statement": "Move debug logs to the cold storage tier", "must_appear": True},
        {"statement": "Add a secondary engineer to every on-call shift", "must_appear": True},
        {"statement": "Move the on-call handbook into Confluence", "must_appear": True,
         "why": "Approved only by a bare 'Do that.' — exercises Stage 4's short-quote rule."},
        {"statement": "Rewrite the mobile app in Flutter", "must_appear": False,
         "why": "TRAP: proposed by Grace, parked by the chair ('not this quarter')."},
        {"statement": "Hire a contractor for the backlog", "must_appear": False,
         "why": "TRAP: proposed by Ravi, declined ('we don't have budget')."},
        {"statement": "Self-host Grafana", "must_appear": False,
         "why": "TRAP: Leo says it is 'just an idea, not a proposal'."},
    ],
    "action_items": [
        {"task": "Write the incident postmortem", "owner": "Ravi", "deadline": "Friday",
         "why": "Stated: 'Ravi, can you also write up the full incident postmortem by Friday?'"},
        {"task": "Write the idempotency patch for the payment webhook", "owner": None,
         "owner_with_diarization": "Ravi", "deadline": "Wednesday",
         "why": "First person ('I'll write the idempotency patch myself'), far from Ravi's name."},
        {"task": "Send the penetration test report", "owner": None,
         "owner_with_diarization": "Omar", "deadline": "Thursday",
         "why": "First person; Omar is named only by his own self-introduction."},
        {"task": "Fix the login crash", "owner": "Hannah", "deadline": "week",
         "why": "Hannah was just addressed by name; 'this week'."},
        {"task": "Prepare the release notes", "owner": "Hannah", "deadline": "Monday",
         "why": "Deadline is stated only in the REQUEST ('by next Monday'); the reply omits it."},
        {"task": "Set up budget alerts in Terraform", "owner": None,
         "owner_with_diarization": "Leo", "deadline": None,
         "why": "First person, no deadline: the deadline must stay unspecified."},
        {"task": "Talk to finance about the Datadog contract", "owner": None,
         "owner_with_diarization": "Speaker 1", "deadline": None,
         "why": "The chair commits in the first person but is never named: '(voice only)'."},
        {"task": "Audit the unused staging clusters", "owner": None, "deadline": None,
         "why": "TRAP: 'Someone needs to…' — no owner, no deadline."},
        {"task": "Run a full backup restore test", "owner": None,
         "owner_with_diarization": "Grace", "deadline": "Thursday",
         "why": "TRAP: the chair asks LEO; Grace answers and takes it. Without voices the "
                "owner is unknowable; 'Leo' would be wrong."},
        {"task": "Schedule interviews for the two backend roles", "owner": "Grace",
         "deadline": "month",
         "why": "Stated: 'Grace, can you schedule … before the end of the month?'"},
        {"task": "Move the on-call handbook into Confluence", "owner": None,
         "owner_with_diarization": "Grace", "deadline": None,
         "why": "Grace offers ('I can move…'), the chair approves with 'Do that.'"},
    ],
    "speaker_names": {
        "expected": ["Ravi", "Hannah", "Leo", "Grace", "Omar"],
        "anonymous": ["Diana"],
        "why": "The chair (Diana in this key) is never addressed by name. Omar is named "
               "only by self-introduction. Grace's voice also gets one vote for 'Leo'.",
    },
    "must_preserve": {
        "negation": "We are not going to build our own token service",
        "why": "Stage 2 must not flip it.",
    },
    "domain_terms": GLOSSARY,
}


def main() -> int:
    lines = []
    variants: dict[str, str] = {}
    with open(os.path.join(HERE, "script.txt")) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            speaker, text = raw.split(": ", 1)
            assert speaker in VOICES, f"unknown speaker {speaker!r}"
            meant = MARK.sub(lambda m: m.group(1), text)
            spoken = MARK.sub(lambda m: m.group(2), text)
            for m in MARK.finditer(text):
                variants[m.group(1)] = m.group(2)
            line = {"speaker": speaker, "text": meant}
            if spoken != meant:
                line["spoken"] = spoken
            lines.append(line)

    truth = dict(GROUND_TRUTH)
    truth["spoken_variants"] = variants
    spec = {
        "title": "Weekly engineering sync — payments incident, security, release, costs",
        "note": "Synthetic ~30-minute meeting for the demo and for evaluation. "
                "Built by scripts/long_meeting/build.py from script.txt.",
        "voices": VOICES,
        "rate": 160,
        "lines": lines,
        "ground_truth": truth,
    }
    out = os.path.join(ROOT, "data", "long_meeting_script.json")
    with open(out, "w") as fh:
        json.dump(spec, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(ROOT, "data", "long_meeting_glossary.json"), "w") as fh:
        json.dump({"terms": GLOSSARY}, fh, indent=2)

    words = sum(len(l["text"].split()) for l in lines)
    minutes = words / spec["rate"] + len(lines) * 0.35 / 60
    print(f"wrote {out}: {len(lines)} turns, {words} words, "
          f"~{minutes:.0f} min at {spec['rate']} wpm, {len(variants)} spoken variants")
    print("wrote data/long_meeting_glossary.json (upload it in the app's sidebar)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
