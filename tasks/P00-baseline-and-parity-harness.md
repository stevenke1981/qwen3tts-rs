# P00 — Baseline and parity harness

## Objective

建立可信的對齊基線與不會靜默跳過的測試基礎。

## Dependency

None

## Tasks

- [x] **P00-T01** — Pin target/reference/upstream revisions and write baseline delta report
- [x] **P00-T02** — Create fixture manifest and fail-closed fixture resolver
- [x] **P00-T03** — Add stage dump format and dump hooks behind a feature flag
- [x] **P00-T04** — Build reference command adapters for Python and qwentts.cpp
- [x] **P00-T05** — Create CPU F32 smoke corpus and baseline report

## Sol Delegation Rules

- Delegate one task at a time.
- Before delegation, write exact allowed files and required tests.
- Require a worker report and independent review.
- Do not mark a task complete when real-weight tests were skipped.
- Store evidence under `artifacts/alignment/P00/<task-id>/`.

## Phase Gate

- Reference adapters can generate prompt/token/tensor/audio evidence.
- Missing required fixtures produce a failed release gate.
- Every artifact includes revision and SHA-256.
- Existing CPU build and unit tests remain green.

## Required Deliverables

- code and tests
- worker report
- review report
- commands and captured results
- numerical/performance artifacts where applicable
- updated `STATUS.md`
