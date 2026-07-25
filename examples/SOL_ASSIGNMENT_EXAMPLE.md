# Example Sol → Spark Assignment

## Task

- Task ID: `P02-T01`
- Goal: Implement a Philox4x32-compatible counter-based RNG and known-vector tests.
- Phase contract: `tasks/P02-sampling-parity.md`

## Allowed production files

- `src/talker/philox.rs`
- `src/talker/sampling.rs`
- `src/talker/mod.rs`

## Allowed tests

- `tests/philox_vectors.rs`
- existing sampling unit tests

## Required behavior

- Implement deterministic Philox stream with explicit seed, subsequence and offset.
- No OS RNG.
- Do not change sampling order in this task.
- Preserve current public sampling API through an adapter.

## Required tests

- official/reference known vectors
- same seed produces identical sequence
- offset/subsequence independence
- boundary and wrapping behavior

## Commands

```text
cargo fmt --all -- --check
cargo test philox
cargo clippy --all-targets -- -D warnings
```

## Acceptance

All vectors exact; no existing sampling test changes expected values unless documented by Sol.
