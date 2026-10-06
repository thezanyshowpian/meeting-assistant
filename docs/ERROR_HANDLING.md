# Error Handling & Edge Cases

> Explicitly scored and commonly skipped under time pressure. Behavior TBD but
> every listed case must be handled with a clear user-facing message.

## Input validation
- Unsupported format — _TBD_
- Empty file — _TBD_
- Corrupt / unreadable file — _TBD_
- Non-audio file — _TBD_

## Content edge cases
- Silence / no speech — _TBD_
- Very long recording — _TBD (chunking)_
- No decisions reached — _return empty list_
- No action items assigned — _return empty list_

## Stage failures
- STT fails / times out — _TBD_
- LLM returns malformed structured output — _TBD (retry/validate)_
- LLM stage unavailable (rate limit) — _TBD_

## Principle
Fail loudly and clearly in the interface; never silently produce partial or
invented output.
