# Raw Transcript

Alright, let's start. Main thing today is the session store. Arjun, where are we? So right now sessions live in Redis, and we're seeing latency spikes on the Kubernetes cluster, about 200 milliseconds at the 95th percentile. It's only on three nodes, but it's enough to hurt the checkout flow. Is that a Redis problem or a networking problem? Mostly Redis. The connection pool thrash is under load. I think we should move the session store to PostgreSQL. PostgreSQL will be slower per read, but it's far more predictable, and we already run it in staging. Okay, let's do it. We're moving the session store from Redis to PostgreSQL. That's decided. While we're at it, maybe we should switch the internal calls from REST API to GRPC. It might help the latency too. Let's park that one. Interesting, but not this sprint, and I don't want to change two things at once. Agreed. I'll benchmark the PostgreSQL session store latency and have numbers by Friday. Good. Sam, can you update the Grafana dashboard so we can actually see the new metrics? Yes, I'll take the Grafana dashboard. And someone needs to document the rollback procedure before we ship anything. One more thing. We are not going to do the OAuth migration this sprint. It's blocked on the vendor. Understood. That's everything. Thanks.

# Refined Transcript

Alright, let's start. Main thing today is the session store. Arjun, where are we? So right now sessions live in Redis, and we're seeing latency spikes on the Kubernetes cluster, about 200 milliseconds at the 95th percentile. It's only on three nodes, but it's enough to hurt the checkout flow. Is that a Redis problem or a networking problem? Mostly Redis. The connection pool thrash is under load. I think we should move the session store to PostgreSQL. PostgreSQL will be slower per read, but it's far more predictable, and we already run it in staging. Okay, let's do it. We're moving the session store from Redis to PostgreSQL. That's decided. While we're at it, maybe we should switch the internal calls from REST API to GRPC. It might help the latency too. Let's park that one. Interesting, but not this sprint, and I don't want to change two things at once. Agreed. I'll benchmark the PostgreSQL session store latency and have numbers by Friday. Good. Sam, can you update the Grafana dashboard so we can actually see the new metrics? Yes, I'll take the Grafana dashboard. And someone needs to document the rollback procedure before we ship anything. One more thing. We are not going to do the OAuth migration this sprint. It's blocked on the vendor. Understood. That's everything. Thanks.

# Speaker-labelled Transcript

_3 speakers detected (ECAPA-TDNN + average-linkage agglomerative (threshold=0.65)). A name in brackets with '?' is inferred from evidence in the conversation; otherwise labels are anonymous._

**Speaker 1** [0:00]: Alright, let's start. Main thing today is the session store. Arjun, where are we?

**Speaker 2 (Arjun?)** [0:05]: So right now sessions live in Redis, and we're seeing latency spikes on the Kubernetes cluster, about 200 milliseconds at the 95th percentile. It's only on three nodes, but it's enough to hurt the checkout flow.

**Speaker 1** [0:19]: Is that a Redis problem or a networking problem?

**Speaker 2 (Arjun?)** [0:22]: Mostly Redis. The connection pool thrash is under load. I think we should move the session store to PostgreSQL.

**Speaker 3** [0:30]: PostgreSQL will be slower per read, but it's far more predictable, and we already run it in staging.

**Speaker 1** [0:37]: Okay, let's do it. We're moving the session store from Redis to PostgreSQL. That's decided.

**Speaker 3** [0:44]: While we're at it, maybe we should switch the internal calls from REST API to GRPC. It might help the latency too.

**Speaker 1** [0:52]: Let's park that one. Interesting, but not this sprint, and I don't want to change two things at once.

**Speaker 2 (Arjun?)** [1:00]: Agreed. I'll benchmark the PostgreSQL session store latency and have numbers by Friday.

**Speaker 1** [1:06]: Good. Sam, can you update the Grafana dashboard so we can actually see the new metrics? Yes, I'll take the Grafana dashboard. And someone needs to document the rollback procedure before we ship anything.

**Speaker 2 (Arjun?)** [1:19]: One more thing. We are not going to do the OAuth migration this sprint. It's blocked on the vendor.

**Speaker 1** [1:26]: Understood. That's everything. Thanks.
