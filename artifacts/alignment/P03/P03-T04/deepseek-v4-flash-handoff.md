# P03-T04 DeepSeek V4 Flash Handoff

## Mission

Complete `P03-T04` from the current dirty working tree. Do not restart the
implementation from `HEAD`: an incomplete external-agent attempt already
exists in the files listed below.

The authoritative acceptance contract remains `assignment.md`. This handoff
records the current implementation state and the defects that must be fixed
before independent review.

## Repository State

- Branch: `alignment/full-qwentts-parity`
- Current HEAD when this handoff was written: `e70ea9d`
- Remote branch currently ends at: `9146f5b`
- The local branch contains three unpushed commits unrelated to P03-T04.
- P03-T04 currently has uncommitted changes in:
  - `src/alignment_stage_dump.rs`
  - `src/talker/sampling.rs`
  - `src/talker/code_predictor.rs`
  - `src/talker/talker.rs`
  - `tests/acoustic_host_transfer_test.rs` (untracked)
- CBM/OpenCode runtime files are also dirty. They are not task source files.

Do not commit, push, reset, clean, delete, stage, or modify Git history. Do not
stage `.codebase-memory/**`, `.opencode/**`, or any file outside the task card.

## CBM-Verified Call Paths

The active production paths are:

```text
TalkerForConditionalGeneration::generate_with_observer
  -> greedy_select_on_device
  -> CodePredictor::generate_with_observer
  -> embedding_lookup / forward_layers / linear

TalkerForConditionalGeneration::generate_sampled_with_observer
  -> Sampler::sample_with_mode
  -> CodePredictor::generate_sampled_with_observer
  -> Sampler::sample
```

The current greedy Code Predictor keeps argmax tensors on device, and Talker
assembles frames with `Tensor::cat`. Those are useful partial changes and should
be preserved unless a tested correction requires otherwise.

## Blocking Defects in the Current Attempt

1. **Telemetry is disconnected from production.**
   `TransferObserver::on_transfer` is not called at actual `to_scalar`,
   `to_vec*`, or host-to-device reconstruction boundaries.

2. **The transfer test fabricates evidence.**
   `tests/acoustic_host_transfer_test.rs` explicitly infers transfers from
   stage callbacks. A stage callback is not proof that a transfer occurred.
   The test must consume events emitted by the production boundary itself.

3. **Final Tensor return is incorrectly counted as a host transfer.**
   Returning an on-device `Tensor` or calling an observer with `&Tensor` is not
   device-to-host synchronization. Remove synthetic `FinalOutput` transfer
   events unless the production code actually materializes values on the host.

4. **`StageDumpWriter` violates the hot-path contract.**
   Its current `TransferObserver` implementation calls `format!`, allocates a
   CPU tensor, uses `unwrap()`, ignores the result, and records a fake tensor
   stage. Replace it with structured counter/event handling that is default
   noop and performs no allocation or formatting when capture is disabled.
   Library code must not add `unwrap()` or swallow telemetry errors.

5. **Duplicate cache-container copies remain.**
   Both Code Predictor generation paths still return
   `kv_caches.to_vec()` even though the mutable cache slice was already updated.
   Remove the redundant return/copy and update directly related callers/tests.

6. **Greedy masking needs semantic hardening.**
   Prove exact parity with the existing host selector for:
   - no suppression;
   - reserved-suffix suppression;
   - optional EOS inside the suppressed suffix;
   - lowest-index ties;
   - invalid `suppress_from` / allowed-token indices;
   - NaN and positive/negative infinity behavior;
   - the documented `batch=1` shape contract.

   Do not rely on `f32::MIN` arithmetic if it can allow a suppressed `+inf`,
   alter non-finite behavior, or create an invalid winner.

7. **Current tests are insufficient despite passing.**
   The focused test currently reports 14/14 PASS, but its header admits that
   production telemetry is not integrated. Passing inferred-event assertions
   is not Gate evidence.

8. **The full formatting gate currently fails.**
   `cargo fmt --all -- --check` reports formatting drift in P03-T04 files and
   in pre-existing files from the three local commits. Format only task-owned
   files; do not mechanically rewrite unrelated files.

9. **Required evidence is missing.**
   `worker-report.md`, `commands.txt`, `test-results.txt`, a deterministic
   before/after transfer artifact, and draft `gate.json` have not been written.

## Required Implementation Rules

- Instrument the actual synchronization boundary, not a neighboring stage.
- Event data must be structured (`direction`, `kind`, `elements`) and use
  copyable fixed-size values.
- Capture-off must be a branch plus default-noop call at most: no `Vec` growth,
  `String`, `format!`, tensor construction, tensor download, or logging.
- Capture-on test collection may allocate outside the production hot path.
- Greedy completed-frame budget:
  - codebook-0: at most one device-to-host scalar for EOS/control flow;
  - Code Predictor: zero device-to-host logits or code vectors;
  - frame assembly: zero host round trips.
- Sampled path:
  - exactly one full-logit device-to-host transfer for each CPU Philox draw;
  - record any required scalar host-to-device reconstruction honestly;
  - no full code-vector download/re-upload;
  - no duplicate cache-container copy.
- Preserve exact P02 Philox draw order, P03 observer stage names, cache
  transaction behavior, EOS policy, and output tokens.
- Do not change thresholds, model architecture, unrelated codec/vocoder code,
  public sampling math, or fixture policy.

## Test Requirements

Replace inference-based transfer assertions with runtime telemetry assertions.
At minimum add:

1. capture-off observer proves zero telemetry-side transfers/allocating work;
2. greedy CP records zero host transfers;
3. greedy Talker records only the permitted codebook-0 scalar per completed
   frame, with terminal EOS draw handled explicitly;
4. sampled Talker and CP record exactly one full-logit download per Philox draw
   and no code-vector round trip;
5. suppression/EOS/tie/non-finite/error parity against the pinned host selector;
6. output equality for greedy and sampled paths;
7. P02 deterministic token-sequence regression;
8. P03 tiny and real Code Predictor frame regressions;
9. cache mutation/reset/observer-error behavior remains unchanged.

Do not use source-text scans, comments, stage-callback inference, ignored tests,
or silent fixture skips as proof of a transfer budget.

## Deliverables

Only after all required commands have actually run, write:

- `worker-report.md`
- `commands.txt`
- `test-results.txt`
- `before-after-transfer.json`
- draft `gate.json` with status no stronger than `IN_PROGRESS`
- `docs/alignment/p03-acoustic-host-transfers.md`

Report exact commands, exit codes, transfer counts/elements, skipped commands,
fixture paths/hashes, and remaining risks. Do not create `review.md`; the Sol
gate owner will perform independent review and final acceptance.
