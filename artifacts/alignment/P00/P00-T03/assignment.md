# GPT-5.3 Codex Spark — Implementation Assignment

## Task

- Task ID: `P00-T03`
- Goal: add a versioned stage-dump writer and explicit Talker, Code Predictor and offline Codec
  observation hooks without global mutable state or default-build file I/O.
- Phase contract: `tasks/P00-baseline-and-parity-harness.md`

## Allowed production files

- `Cargo.toml`
- `src/lib.rs`
- `src/alignment_stage_dump.rs` (new)
- `src/talker/talker.rs`
- `src/talker/code_predictor.rs`
- `src/decoder_12hz.rs`

## Allowed test/tool files

- `tests/stage_dump_test.rs` (new)
- `schemas/stage-dump.schema.json`
- `tools/compare_stage_dumps.py`

No other file may be modified without stopping and reporting the need.

## Read First

- `AGENTS.md`
- `tasks/P00-baseline-and-parity-harness.md`
- `ACCEPTANCE_CRITERIA.md`
- `schemas/stage-dump.schema.json`
- `tools/compare_stage_dumps.py`
- the complete existing implementations and tests for the three allowed Rust model files

## Required Architecture

1. Add a non-default Cargo feature named `stage-dump`.
2. Use an explicitly owned observer/writer passed through method calls. Do not use global,
   thread-local, environment-selected or singleton mutable recorders.
3. Preserve existing public methods and output behavior. Existing methods should route through a
   generic/no-op observer path so the default build can optimize observation away.
4. File-producing writer APIs must only be available with `feature = "stage-dump"`.
5. Stage output format:
   - manifest schema version 1;
   - source, revision, model, case ID and optional seed;
   - one unique stage name per manifest;
   - dtype, shape, relative file, layout and SHA-256;
   - little-endian contiguous F32 binary stage data.
6. Reject unsafe or ambiguous stage names (`..`, separators, empty names) and duplicate names.
7. Never silently overwrite an existing stage file or manifest.
8. Errors must include the failing stage and path context.

## Required Initial Hook Points

- Talker: codebook-0 logits for every autoregressive frame and final generated code matrix.
- Code Predictor: logits for each generated sub-codebook step and final sub-code matrix.
- Offline 12 Hz Codec: input code matrix and final PCM tensor.

P03 will add deeper per-layer hooks; do not broaden this task into full layer instrumentation.

## Required Tests

- writer produces schema-compatible manifest and exact F32 binary bytes;
- recorded SHA-256 matches the actual stage file;
- duplicate/unsafe stage names fail without overwrite;
- normal no-observer generation/decode behavior remains unchanged;
- capturing observer receives the required hook names in deterministic order;
- compare tool passes identical dumps and fails a changed/missing stage;
- default build works without the feature and performs no file I/O.

Synthetic small tensors/configs are allowed for hook plumbing tests. Do not claim numerical model
parity from them.

## Required Commands

```text
cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo test --lib
cargo test --features stage-dump --test stage_dump_test
python tools/compare_stage_dumps.py <IDENTICAL_REFERENCE> <IDENTICAL_CANDIDATE> --min-cosine 0.999
python tools/compare_stage_dumps.py <REFERENCE> <CHANGED_OR_MISSING> --min-cosine 0.999
```

The final compare command must exit non-zero. Tests may generate temporary manifests and invoke the
tool so no repository fixture binaries are committed.

## Acceptance

- Existing Rust API behavior and 82 library tests remain green.
- Feature-off build compiles without stage writer/file I/O.
- Feature-on tests prove format, hash, overwrite safety and deterministic hook ordering.
- Dumping is explicitly session-owned; no cross-session/shared mutable recorder exists.
- Diff stays within the nine allowed files.

## Constraints

- Do not add Python, PyTorch, ONNX or network dependencies.
- Do not weaken numerical thresholds.
- Do not add model weights or large binary fixtures.
- Do not modify status, TODO, task index or evidence Gate files.
- Do not commit or push.

## Final Report

Return summary, exact files, design choices, commands/exit codes, hook names, evidence and risks.
