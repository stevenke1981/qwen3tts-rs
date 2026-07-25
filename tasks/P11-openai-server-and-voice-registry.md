# P11 — OpenAI server and voice registry

## Objective

交付 OpenAI 相容 Server 與安全 Voice Registry。

## Dependency

Previous phase gate must pass.

## Tasks

- [ ] **P11-T01** — Implement /v1/audio/speech non-streaming endpoint
- [ ] **P11-T02** — Implement chunked/streaming audio response
- [ ] **P11-T03** — Implement safe cloned-voice registry
- [ ] **P11-T04** — Implement request limits, cancellation and error mapping
- [ ] **P11-T05** — Gate API and concurrency tests

## Sol Delegation Rules

- Delegate one task at a time.
- Before delegation, write exact allowed files and required tests.
- Require a worker report and independent review.
- Do not mark a task complete when real-weight tests were skipped.
- Store evidence under `artifacts/alignment/P11/<task-id>/`.

## Phase Gate

- API contract and streaming tests pass.
- request cancellation releases session state.
- registry paths and metadata are sanitized and versioned.

## Required Deliverables

- code and tests
- worker report
- review report
- commands and captured results
- numerical/performance artifacts where applicable
- updated `STATUS.md`
