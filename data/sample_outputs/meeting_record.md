# Meeting Record

> **Notes**
> - Speaker labels: Speaker 1 asked Sam a question at 1:06 and the reply ("Yes, I'll take the Grafana dashboard.", 1:12) was attributed to the same voice — likely a speaker change diarization missed; voice-based owners from that line are withheld.

## Summary

The team decided to move the session store from Redis to PostgreSQL, benchmark its latency, update the Grafana dashboard, document the rollback procedure, and postpone the OAuth migration and GRPC switch for this sprint.

## Minutes

- Speaker 1 opened the meeting focusing on the session store.
- Speaker 2 reported sessions in Redis have latency spikes (~200 ms at the 95th percentile) affecting the checkout flow.
- Speaker 1 asked if the issue was Redis or networking.
- Speaker 2 said it was mostly Redis, with connection‑pool thrash, and suggested moving the session store to PostgreSQL.
- Speaker 3 noted PostgreSQL would be slower per read but more predictable and already in staging.
- Speaker 1 agreed to move the session store to PostgreSQL.
- Speaker 1 declared the decision to move the session store from Redis to PostgreSQL.
- Speaker 3 suggested switching internal calls from REST API to gRPC.
- Speaker 1 decided to park the gRPC suggestion for now.
- Speaker 2 committed to benchmark the PostgreSQL session store latency and provide numbers by Friday.
- Speaker 1 asked Sam to update the Grafana dashboard for the new metrics.
- Speaker 1 volunteered to take the Grafana dashboard.
- Speaker 1 noted that someone needs to document the rollback procedure before shipping.
- Speaker 2 announced the OAuth migration will not be done this sprint due to a vendor block.
- Speaker 1 concluded the meeting.

## Key Decisions

- **Move the session store from Redis to PostgreSQL**
  - evidence: "We're moving the session store from Redis to PostgreSQL. That's decided."
- **Defer the REST‑to‑gRPC switch for now**
  - evidence: "Let's park that one."
- **Do not do the OAuth migration this sprint**
  - evidence: "We are not going to do the OAuth migration this sprint."

## Action Items

| Task | Owner | Deadline |
| :-- | :-- | :-- |
| Benchmark PostgreSQL session store latency and provide numbers by Friday | Arjun (inferred) | Friday |
| Take responsibility for the Grafana dashboard | Sam | unspecified |
| Document rollback procedure before shipping | unspecified | unspecified |

**How owners marked _(inferred)_ / _(voice only)_ are known:**

- *Benchmark PostgreSQL session store latency and provide numbers by Friday*: Speaker 2 said it at 0:59: "I'll benchmark the PostgreSQL session store latency and have numbers by Friday."; Speaker 2 identified as Arjun: addressed by name at 0:00 ("Arjun, where are we?") and answered next at 0:05 ("So right now sessions live in Redis, and we're seeing latenc")

## Speaker Names (inferred from the conversation)

- Speaker 2 identified as Arjun: addressed by name at 0:00 ("Arjun, where are we?") and answered next at 0:05 ("So right now sessions live in Redis, and we're seeing latenc")
