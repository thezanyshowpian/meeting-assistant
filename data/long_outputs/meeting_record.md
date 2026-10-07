# Meeting Record

> **Notes**
> - 1 generated item(s) could not be supported by the transcript and were removed.

## Summary

The meeting covered a range of topics including a recent payments incident, a security review, the mobile release, cloud costs, backups, and hiring and process improvements. Key decisions were made on implementing idempotency keys for the payment webhook, using Okta for the Partner API, delaying the mobile release, and improving on-call procedures. Action items included writing a postmortem, configuring Okta integration, and testing the backup restore.

## Minutes

- Discussed the recent payments incident and its impact, including duplicate charges and customer support issues.
- Decided to implement idempotency keys for the payment webhook to prevent duplicate charges.
- Agreed to use Okta for the Partner API tokens instead of building a custom token service.
- Delayed the mobile release to the 27th to ensure a safe and well-tested release.
- Addressed cloud costs by moving debug logs older than three days to cold storage.
- Improved on-call procedures by introducing a secondary engineer for high severity incidents.
- Planned to run a full restore test for backups to determine recovery time objectives.
- Discussed hiring pipeline and on-call handbook improvements.
- Agreed to move the on-call handbook to Confluence and add a checklist for incident response.
- Reviewed the Partner API timeline and confirmed that the Okta integration would not delay the project.
- Noted the need for a support tool to look up payment events for future reference.
- Agreed to set up budget alerts for cloud spend to prevent unexpected increases.

## Key Decisions

- **Add idempotency keys to the payment webhook handler.**
  - evidence: "We're adding idempotency keys to the payment webhook. Stored in PostgreSQL with the payment record. That's decided."
- **Use Okta for the Partner API tokens instead of building our own token service.**
  - evidence: "We are not going to build our own token service. We'll use Okta for the partner API tokens instead. That's decided."
- **Rotate OAuth tokens every 90 days automatically.**
  - evidence: "We'll rotate OAuth tokens every 90 days. Automated. That's decided too."
- **Delay the mobile release to the 27th.**
  - evidence: "Let's move the release. We're delaying the mobile release to the 27th."
- **Move debug logs older than three days to cold storage.**
  - evidence: "We're moving debug logs older than three days to the cold storage tier. That's decided."
- **Every on-call shift gets a named secondary engineer for high severity incidents.**
  - evidence: "From now on, every on-call shift gets a named secondary engineer, who is paged automatically for any high severity incident."

## Action Items

| Task | Owner | Deadline |
| :-- | :-- | :-- |
| Write up the full incident postmortem by Friday. | Ravi | Friday |
| Implement the idempotency key patch and have it in review by Wednesday. | Ravi (inferred) | Wednesday |
| Configure Okta integration for the Partner API. | Omar (inferred) | unspecified |
| Run a full restore test on Thursday and report the results. | Grace (inferred) | Thursday |
| Get the release notes ready by next Monday. | Hannah | Next Monday |

**How owners marked _(inferred)_ / _(voice only)_ are known:**

- *Implement the idempotency key patch and have it in review by Wednesday.*: Speaker 3 said it at 5:04: "I'll write the idempotency patch myself. And I should have it in review by Wedne"; Speaker 3 identified as Ravi: addressed by name at 0:32 ("Ravi, can you walk us through what happened from the beginning?") and answered next at 0:50 ("Sure. So the first alert fired at 9.42 in the morning. Check"); addressed by name at 5:14 ("Ravi, can you also write up the full incident postmortem by Friday, so we can share it with the wider engineering group?") and answered next at 5:21 ("Yes. I'll have the postmortem in confluence by Friday. I'll ")
- *Configure Okta integration for the Partner API.*: Speaker 2 said it at 25:09: "I'm happy to pair with whoever does the gateway configuration."; Speaker 2 identified as Omar: introduced themself at 0:20: "Hi everyone, I'm Omar."
- *Run a full restore test on Thursday and report the results.*: Speaker 6 said it at 19:52: "I'll run a full restore test on Thursday, into an isolated environment, and I'll"; Speaker 6 identified as Grace: addressed by name at 20:54 ("Grace, how is the hiring pipeline looking for the back-end roles?") and answered next at 21:00 ("Better than last month. We have two open back-end positions."); addressed by name at 21:44 ("Grace, can you schedule the interviews for the two back-end roles before the end of the month?") and answered next at 21:51 ("Yes, I'll schedule them. I'll start with the two candidates ") (weaker evidence for Leo)

## Speaker Names (inferred from the conversation)

- Speaker 2 identified as Omar: introduced themself at 0:20: "Hi everyone, I'm Omar."
- Speaker 3 identified as Ravi: addressed by name at 0:32 ("Ravi, can you walk us through what happened from the beginning?") and answered next at 0:50 ("Sure. So the first alert fired at 9.42 in the morning. Check"); addressed by name at 5:14 ("Ravi, can you also write up the full incident postmortem by Friday, so we can share it with the wider engineering group?") and answered next at 5:21 ("Yes. I'll have the postmortem in confluence by Friday. I'll ")
- Speaker 4 identified as Hannah: addressed by name at 8:31 ("Hannah, where does that leave the release?") and answered next at 8:42 ("So, the current plan was to ship version 4 .2 on the 20th. T"); addressed by name at 9:58 ("Hannah, does the 27th work for you and the team?") and answered next at 10:08 ("Yes, the 27th works. I'll fix the session check myself, and "); addressed by name at 11:19 ("Hannah, could you get the release notes ready by next Monday, so marketing has time to prepare the announcement?") and answered next at 11:27 ("Sure, I'll get those done. I'll make sure they mention the s")
- Speaker 6 identified as Grace: addressed by name at 20:54 ("Grace, how is the hiring pipeline looking for the back-end roles?") and answered next at 21:00 ("Better than last month. We have two open back-end positions."); addressed by name at 21:44 ("Grace, can you schedule the interviews for the two back-end roles before the end of the month?") and answered next at 21:51 ("Yes, I'll schedule them. I'll start with the two candidates ") (weaker evidence for Leo)
