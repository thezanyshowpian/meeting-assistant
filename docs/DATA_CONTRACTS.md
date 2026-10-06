# Data Contracts / Schemas

> The typed objects that pass between stages and define the machine-readable
> output. Enforcing optionality here (e.g. nullable owner/deadline) is how we
> guarantee "unspecified, not guessed." Field lists are TBD.

## Transcript
_TBD — fields (text, segments?, timestamps?, speaker?)._

## RefinedTranscript
_TBD._

## Decision
_TBD — fields (statement, supporting evidence / transcript reference?)._

## ActionItem
_TBD — fields (task description, owner: optional, deadline: optional, evidence?).
Owner and deadline MUST be optional/nullable and default to unspecified._

## MeetingRecord (top-level)
_TBD — summary, minutes, decisions[], action_items[]._

## Serialization
_TBD — JSON (machine-readable) and Markdown (human-readable) both built from the
same object so they cannot disagree._
