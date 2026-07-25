# P04 — Tokenizer decoder offline parity

## Objective

完成離線 Tokenizer Decoder 的可靠數值對齊。

## Dependency

Previous phase gate must pass.

## Tasks

- [ ] **P04-T01** — Verify RVQ split projections and codebook policy
- [ ] **P04-T02** — Verify decoder transformer and sliding-window semantics
- [ ] **P04-T03** — Verify ConvNeXt upsample and DAC blocks
- [ ] **P04-T04** — Match offline waveform on short/medium/long corpora
- [ ] **P04-T05** — Create buffered chunk decode with left-context trimming

## Sol Delegation Rules

- Delegate one task at a time.
- Before delegation, write exact allowed files and required tests.
- Require a worker report and independent review.
- Do not mark a task complete when real-weight tests were skipped.
- Store evidence under `artifacts/alignment/P04/<task-id>/`.

## Phase Gate

- Offline waveforms pass short/long corpora.
- Buffered chunk decode handles left context without seams.
- Decoder stage dumps meet thresholds.

## Required Deliverables

- code and tests
- worker report
- review report
- commands and captured results
- numerical/performance artifacts where applicable
- updated `STATUS.md`
