# P08 — Native quantized runtime

## Objective

建立真正使用 packed weights 計算的量化 Runtime。

## Dependency

Previous phase gate must pass.

## Tasks

- [ ] **P08-T01** — Define quantized tensor types and protected tensor policy
- [ ] **P08-T02** — Implement direct Q8 linear and embedding operations
- [ ] **P08-T03** — Implement Q4_K_M-class block layout and kernels
- [ ] **P08-T04** — Add backend-resident packed weight loader
- [ ] **P08-T05** — Integrate quantized Talker and Code Predictor
- [ ] **P08-T06** — Gate memory, token and quality metrics

## Sol Delegation Rules

- Delegate one task at a time.
- Before delegation, write exact allowed files and required tests.
- Require a worker report and independent review.
- Do not mark a task complete when real-weight tests were skipped.
- Store evidence under `artifacts/alignment/P08/<task-id>/`.

## Phase Gate

- Q8/Q4 runtime does not hold a full F32 backbone.
- memory and quality thresholds pass.
- protected RVQ/speaker tensors remain specified precision.
- quantized backends have direct kernels.

## Required Deliverables

- code and tests
- worker report
- review report
- commands and captured results
- numerical/performance artifacts where applicable
- updated `STATUS.md`
