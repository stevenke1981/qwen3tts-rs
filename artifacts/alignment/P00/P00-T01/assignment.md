# P00-T01 Assignment

Task ID: `P00-T01`

Goal: pin target, qwentts.cpp and official Qwen3-TTS revisions; capture the local toolchain and
hardware environment; determine whether a source-baseline delta is required; establish a
reproducible CPU baseline before production changes.

Allowed files:

- `SOURCE_BASELINE.md`
- `STATUS.md`
- `TODOS.md`
- `tasks/task-index.yaml`
- `docs/alignment/current-implementation-audit.md`
- `artifacts/alignment/P00/P00-T01/**`

Required verification:

- target/reference/upstream revisions are exact hashes;
- environment contains Rust, Cargo, OS, CPU, GPU and CUDA/Metal capability;
- `cargo fmt --all -- --check`;
- `cargo check --all-targets`;
- `cargo test --lib`;
- static alignment scan produces an artifact;
- no real-weight parity is claimed from skipped fixtures.

Baseline delta rule: create `docs/alignment/baseline-delta-20260725.md` only if the target or
qwentts.cpp reference differs from `SOURCE_BASELINE.md`.
