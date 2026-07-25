# P10 — CLI and library product surface

## Objective

交付穩定 Rust API 與 qwen-tts CLI。

## Dependency

Previous phase gate must pass.

## Tasks

- [ ] **P10-T01** — Implement stable high-level Rust synthesis/session API
- [ ] **P10-T02** — Implement qwen-tts CLI with streaming and WAV output
- [ ] **P10-T03** — Add model discovery/download and cache policy
- [ ] **P10-T04** — Add structured logs, JSON metrics and exit codes
- [ ] **P10-T05** — Gate compatibility corpus and cancellation

## Sol Delegation Rules

- Delegate one task at a time.
- Before delegation, write exact allowed files and required tests.
- Require a worker report and independent review.
- Do not mark a task complete when real-weight tests were skipped.
- Store evidence under `artifacts/alignment/P10/<task-id>/`.

## Phase Gate

- library session API supports streaming, cancellation and reuse.
- CLI supports model modes and structured metrics.
- product tests pass across required variants.

## Required Deliverables

- code and tests
- worker report
- review report
- commands and captured results
- numerical/performance artifacts where applicable
- updated `STATUS.md`
