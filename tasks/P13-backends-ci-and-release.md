# P13 — Backends, CI and release

## Objective

完成所有後端、CI、授權與 Release Audit。

## Dependency

Previous phase gate must pass.

## Tasks

- [ ] **P13-T01** — CPU/CUDA/Metal full matrix and performance report
- [ ] **P13-T02** — Vulkan/ROCm strategy and implementation gate
- [ ] **P13-T03** — Replace permissive CI with fail-closed parity workflow
- [ ] **P13-T04** — Generate SBOM, notices, model manifest and reproducible build notes
- [ ] **P13-T05** — Run final audit and produce alignment release report

## Sol Delegation Rules

- Delegate one task at a time.
- Before delegation, write exact allowed files and required tests.
- Require a worker report and independent review.
- Do not mark a task complete when real-weight tests were skipped.
- Store evidence under `artifacts/alignment/P13/<task-id>/`.

## Phase Gate

- fail-closed full parity CI passes.
- release report includes exact revisions, metrics and known limitations.
- no P0/P1 requirement remains open.

## Required Deliverables

- code and tests
- worker report
- review report
- commands and captured results
- numerical/performance artifacts where applicable
- updated `STATUS.md`
