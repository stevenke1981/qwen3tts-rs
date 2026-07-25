# P06 — End-to-end generation streaming

## Objective

將 Talker Token 事件與 Codec Stream 接成真正端到端串流。

## Dependency

Previous phase gate must pass.

## Tasks

- [ ] **P06-T01** — Expose frame events from Talker generation
- [ ] **P06-T02** — Connect generated codes directly to codec stream session
- [ ] **P06-T03** — Add audio callback, backpressure and cancellation
- [ ] **P06-T04** — Update GUI to consume streaming events
- [ ] **P06-T05** — Gate TTFA-before-completion and long-form output

## Sol Delegation Rules

- Delegate one task at a time.
- Before delegation, write exact allowed files and required tests.
- Require a worker report and independent review.
- Do not mark a task complete when real-weight tests were skipped.
- Store evidence under `artifacts/alignment/P06/<task-id>/`.

## Phase Gate

- First PCM callback occurs before token generation finishes.
- Backpressure/cancellation are safe.
- GUI and library consume the same runtime event API.

## Required Deliverables

- code and tests
- worker report
- review report
- commands and captured results
- numerical/performance artifacts where applicable
- updated `STATUS.md`
