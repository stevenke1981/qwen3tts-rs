# P02 — Sampling parity

## Objective

使固定 seed 的取樣行為可與參考實作逐 Token 比較。

## Dependency

Previous phase gate must pass.

## Tasks

- [x] **P02-T01** — Implement Philox RNG with known-vector tests
- [x] **P02-T02** — Implement repetition penalty with exact operation ordering (GATE_PASSED)
- [ ] **P02-T03** — Separate Talker and Code Predictor sampling configs (READY, assignment pending)
- [ ] **P02-T04** — Match token suppression and EOS handling
- [ ] **P02-T05** — Gate deterministic token sequence parity

## Sol Delegation Rules

- Delegate one task at a time.
- Before delegation, write exact allowed files and required tests.
- Require a worker report and independent review.
- Do not mark a task complete when real-weight tests were skipped.
- Store evidence under `artifacts/alignment/P02/<task-id>/`.

## Phase Gate

- Philox known vectors pass.
- Repetition penalty chain order is exact.
- Talker and predictor have separate sampling settings.
- Fixed-seed corpus produces exact codes.

## Required Deliverables

- code and tests
- worker report
- review report
- commands and captured results
- numerical/performance artifacts where applicable
- updated `STATUS.md`
