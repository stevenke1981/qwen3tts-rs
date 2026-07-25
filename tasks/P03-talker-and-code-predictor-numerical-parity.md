# P03 — Talker and Code Predictor numerical parity

## Objective

驗證並修正 Talker 與 Code Predictor 的數值路徑。

## Dependency

Previous phase gate must pass.

## Tasks

- [ ] **P03-T01** — Instrument embeddings, norms, RoPE and layer outputs
- [ ] **P03-T02** — Verify Talker prefill and single-step KV cache
- [ ] **P03-T03** — Verify Code Predictor frame-local prefill and 14 decode steps
- [ ] **P03-T04** — Eliminate avoidable host transfers in acoustic prediction
- [ ] **P03-T05** — Gate stage cosine and logit ranking thresholds

## Sol Delegation Rules

- Delegate one task at a time.
- Before delegation, write exact allowed files and required tests.
- Require a worker report and independent review.
- Do not mark a task complete when real-weight tests were skipped.
- Store evidence under `artifacts/alignment/P03/<task-id>/`.

## Phase Gate

- Required stage cosine thresholds pass.
- KV cache incremental path matches full prefill.
- Predictor resets per frame and uses its own cache.
- Host synchronization count is measured.

## Required Deliverables

- code and tests
- worker report
- review report
- commands and captured results
- numerical/performance artifacts where applicable
- updated `STATUS.md`
