# P07 — Tokenizer encoder and qwen-codec

## Objective

完成音訊 Tokenizer Encoder 與 qwen-codec 工具。

## Dependency

Previous phase gate must pass.

## Tasks

- [ ] **P07-T01** — Complete 24 kHz audio preprocessing/resampling contract
- [ ] **P07-T02** — Verify SEANet and encoder transformer
- [ ] **P07-T03** — Implement RVQ encode argmin path
- [ ] **P07-T04** — Define versioned RVQ code file format
- [ ] **P07-T05** — Implement qwen-codec encode/decode/stream CLI
- [ ] **P07-T06** — Gate round-trip and reference code parity

## Sol Delegation Rules

- Delegate one task at a time.
- Before delegation, write exact allowed files and required tests.
- Require a worker report and independent review.
- Do not mark a task complete when real-weight tests were skipped.
- Store evidence under `artifacts/alignment/P07/<task-id>/`.

## Phase Gate

- WAV→codes matches reference.
- codes→WAV matches decoder reference.
- versioned code files round-trip.
- CLI handles resampling and invalid inputs.

## Required Deliverables

- code and tests
- worker report
- review report
- commands and captured results
- numerical/performance artifacts where applicable
- updated `STATUS.md`
