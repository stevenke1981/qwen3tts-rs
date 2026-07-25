# P09 — Model format and conversion

## Objective

建立完整模型 metadata、轉換、GGUF 相容與量化工具鏈。

## Dependency

Previous phase gate must pass.

## Tasks

- [ ] **P09-T01** — Define metadata-complete Rust model container strategy
- [ ] **P09-T02** — Implement GGUF reader compatibility or lossless converter
- [ ] **P09-T03** — Implement official checkpoint conversion with provenance
- [ ] **P09-T04** — Implement quantization command and protected tensor rules
- [ ] **P09-T05** — Gate five talker variants plus shared tokenizer

## Sol Delegation Rules

- Delegate one task at a time.
- Before delegation, write exact allowed files and required tests.
- Require a worker report and independent review.
- Do not mark a task complete when real-weight tests were skipped.
- Store evidence under `artifacts/alignment/P09/<task-id>/`.

## Phase Gate

- Five official variants and tokenizer convert/load.
- conversion is reproducible and records provenance.
- metadata round-trip has no lost fields.

## Required Deliverables

- code and tests
- worker report
- review report
- commands and captured results
- numerical/performance artifacts where applicable
- updated `STATUS.md`
