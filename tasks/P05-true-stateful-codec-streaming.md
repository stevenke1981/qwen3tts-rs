# P05 — True stateful codec streaming

## Objective

以持久狀態取代累積歷史重算，建立真正單幀 Codec Streaming。

## Dependency

Previous phase gate must pass.

## Tasks

- [ ] **P05-T01** — Define CodecStreamState and state ownership
- [ ] **P05-T02** — Implement persistent causal-convolution contexts
- [ ] **P05-T03** — Implement transposed-convolution overlap state
- [ ] **P05-T04** — Implement transformer KV ring and absolute RoPE position
- [ ] **P05-T05** — Implement one-frame graph/buffer reuse
- [ ] **P05-T06** — Implement reset, ICL prime and optional state snapshots
- [ ] **P05-T07** — Gate offline-equivalent output and bounded complexity

## Sol Delegation Rules

- Delegate one task at a time.
- Before delegation, write exact allowed files and required tests.
- Require a worker report and independent review.
- Do not mark a task complete when real-weight tests were skipped.
- Store evidence under `artifacts/alignment/P05/<task-id>/`.

## Phase Gate

- No O(n²) history recomputation.
- Stream equals offline for 1–1200 frames.
- State reset, ICL prime and session isolation pass.
- Per-frame latency and memory are bounded.

## Required Deliverables

- code and tests
- worker report
- review report
- commands and captured results
- numerical/performance artifacts where applicable
- updated `STATUS.md`
