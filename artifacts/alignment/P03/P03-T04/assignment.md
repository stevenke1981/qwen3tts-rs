# P03-T04 Assignment — Acoustic Prediction Host-Transfer Elimination

> External DeepSeek V4 Flash must also read
> `deepseek-v4-flash-handoff.md`, which records the current dirty-tree
> implementation and blocking defects discovered by CBM inspection.

## Outcome

Measure host synchronization in Talker and Code Predictor acoustic prediction,
then eliminate every avoidable device-to-host transfer without changing token,
logit, cache, observer, or sampling semantics. The greedy path must transfer at
most one scalar codebook-0 token per completed frame and no Code Predictor
logit/code vectors. The sampled path may retain the full-logit transfer required
by the CPU Philox sampler, but must not perform duplicate code-vector transfers.

## Authoritative Context

- P03-T03 gate and official oracle: `artifacts/alignment/P03/P03-T03/`
- Production paths:
  - `TalkerForConditionalGeneration::{generate,generate_sampled}`
  - `CodePredictor::{generate,generate_sampled}`
  - `sampling::{greedy_select,Sampler::sample_with_mode}`
- Preserve P02 exact qwentts/PyTorch sampling order and P03 stage names.
- Batch size remains the documented production `batch=1` contract.

## Allowed Files

- `src/alignment_stage_dump.rs` for a default-noop transfer telemetry callback
- `src/talker/sampling.rs`
- `src/talker/code_predictor.rs`
- `src/talker/talker.rs`
- tests directly covering acoustic transfer counts and parity
- `docs/alignment/p03-acoustic-host-transfers.md`
- `artifacts/alignment/P03/P03-T04/*`
- `STATUS.md`, `TODOS.md`, and the P03 task index only after Gate acceptance

Do not change probability math, Philox draws, suppression/EOS policy, prompt
assembly, model architecture, public thresholds, unrelated codec/vocoder code,
Git state, or remote state.

## Required Work

1. Record a baseline transfer inventory for greedy and sampled one-frame paths,
   distinguishing scalar, full-logit, and full-code transfers and element count.
2. Greedy Code Predictor must retain the argmax token as an on-device Tensor for
   the next private embedding and construct the final 15 codes on-device. It
   must not call `to_scalar`, `to_vec*`, or rebuild each token from a host value.
3. Talker must assemble `[c0,c1..c15]` frames and the final output on-device; it
   must not copy the 15 predictor codes to a host Vec.
4. Greedy codebook-0 selection must avoid downloading the full vocabulary.
   Preserve reserved-suffix suppression, optional EOS, and deterministic
   lowest-index tie behavior. At most the selected scalar may cross to host.
5. Sampled generation may download each logits vector exactly once because the
   pinned CPU sampler requires it. Remove all later code-vector downloads and
   duplicate cache/container copies.
6. Add default-noop, capture-off-safe telemetry that reports synchronization
   event kind and transferred element count without allocating or formatting in
   the hot path. Prove instrumentation does not add a transfer.
7. Add mutation/parity tests covering suppression/EOS/ties, greedy and sampled
   output equality, exact P02 deterministic sequence, P03 real frame oracle, and
   the expected one-frame transfer budget.
8. Add a benchmark or deterministic complexity artifact showing before/after
   event and element counts. Record remaining unavoidable transfers explicitly.

## Required Commands

```powershell
$env:PATH='C:\Users\steven\.cargo\bin;' + $env:PATH
$env:QWEN3_TTS_REAL_MODEL_DIR='C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc'

cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo check --no-default-features --features cpu,stage-dump
cargo test --test acoustic_host_transfer_test --no-default-features
cargo test --test deterministic_token_sequence_test
cargo test --test suppression_eos_test
cargo test --test code_predictor_frame_test --no-default-features
cargo test --test code_predictor_frame_real_test -- --ignored --nocapture
cargo test --test stage_instrumentation_test --features stage-dump
cargo test --lib sampling
cargo test --lib talker
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
git diff --check -- src/alignment_stage_dump.rs src/talker/sampling.rs src/talker/code_predictor.rs src/talker/talker.rs tests/acoustic_host_transfer_test.rs docs/alignment/p03-acoustic-host-transfers.md artifacts/alignment/P03/P03-T04
```

## Evidence and Completion

Write `commands.txt`, `test-results.txt`, `worker-report.md`, a deterministic
before/after transfer-count artifact, and draft `gate.json` under
`artifacts/alignment/P03/P03-T04/`. Do not write `review.md`, update task/status
indexes, commit, push, delete, or modify files outside the allowed list.
