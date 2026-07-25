# P12 — C ABI and continuous batching

## Objective

交付 C ABI 與多 Session 有界批次執行。

## Dependency

Previous phase gate must pass.

## Tasks

- [ ] **P12-T01** — Define stable opaque-handle C ABI
- [ ] **P12-T02** — Implement callback ownership and cancellation
- [ ] **P12-T03** — Implement per-session scheduler state
- [ ] **P12-T04** — Implement bounded multi-lane Talker/Predictor batching
- [ ] **P12-T05** — Implement per-slot codec streams and isolation
- [ ] **P12-T06** — Gate C lifecycle, 8-session isolation and throughput

## Sol Delegation Rules

- Delegate one task at a time.
- Before delegation, write exact allowed files and required tests.
- Require a worker report and independent review.
- Do not mark a task complete when real-weight tests were skipped.
- Store evidence under `artifacts/alignment/P12/<task-id>/`.

## Phase Gate

- C lifecycle has no leaks/panics across boundary.
- 8 concurrent sessions remain isolated.
- batching improves throughput without changing deterministic single-session output.

## Required Deliverables

- code and tests
- worker report
- review report
- commands and captured results
- numerical/performance artifacts where applicable
- updated `STATUS.md`
