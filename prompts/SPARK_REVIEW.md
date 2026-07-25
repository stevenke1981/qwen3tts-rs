# GPT-5.3 Codex Spark — Independent Review

Review task `<TASK_ID>` without implementing unrelated improvements.

Inspect:

- assignment and task card
- complete diff
- worker report
- tests and evidence
- state reset and error paths
- concurrency/cancellation implications
- numerical and performance thresholds

Act adversarially. Specifically look for:

- silent test skips
- O(n²) hidden behind correct output
- full dequantization hidden behind packed files
- hard-coded model/token metadata
- nondeterministic or incorrect sampling order
- state shared across sessions
- per-frame allocations or graph rebuilds
- threshold weakening
- CPU fallback disguised as backend support

Return:

- Verdict: ACCEPT / REJECT / NEEDS_EVIDENCE
- Blocking findings
- Non-blocking findings
- Commands independently rerun
- Evidence inspected
- Exact repair task if rejected
