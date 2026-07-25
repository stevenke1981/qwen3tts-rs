# P03 Talker KV Cache

Talker caches use `[batch, num_kv_heads, sequence, head_dim]` K/V tensors.
`TalkerModel::forward` validates cache count and existing K/V shape before any
layer runs, failing closed without partial updates. Prefill creates one entry
per layer; an incremental call appends exactly one sequence position.

The pinned qwentts.cpp oracle (revision `82cd05b9f3a175612dc89fd6943e610fab096ef5`)
verifies the non-ICL prompt layout at sequence length 11. The real gate compares
all 28 layers' K/V prefix and appended metrics, final hidden/logits, and the
independent `next-emb-step0` / `talker-hidden-step1` tensors. Metrics are emitted
only when `P03_METRICS_OUT` is explicitly set.
