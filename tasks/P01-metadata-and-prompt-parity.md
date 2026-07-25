# P01 — Metadata and prompt parity

## Objective

讓模型配置、Token、Prompt 與 RoPE 完全由正確 metadata 驅動。

## Dependency

Previous phase gate must pass.

## Tasks

- [x] **P01-T01** — Replace model-name inference with metadata/config parsing
- [x] **P01-T02** — Load all special token, language, speaker and dialect tables from metadata
- [x] **P01-T03** — Correct M-RoPE semantics and add exact position/rotation tests
- [x] **P01-T04** — Match prompt assembly for Base, CustomVoice and VoiceDesign
- [x] **P01-T05** — Gate exact prompt IDs across the model matrix

## Phase gate

- [x] **P01-GATE** — `P01-T05` passed and validated

## Sol Delegation Rules

- Delegate one task at a time.
- Before delegation, write exact allowed files and required tests.
- Require a worker report and independent review.
- Do not mark a task complete when real-weight tests were skipped.
- Store evidence under `artifacts/alignment/P01/<task-id>/`.

## Phase Gate

- Exact prompt IDs for all five variants.
- No model behavior depends only on filename matching.
- mrope_interleaved and position semantics match model metadata/reference.

## Required Deliverables

- code and tests
- worker report
- review report
- commands and captured results
- numerical/performance artifacts where applicable
- updated `STATUS.md`
