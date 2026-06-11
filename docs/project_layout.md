# Project Layout

The repository root is kept for build metadata, project contracts, and primary
entrypoints:

- `Cargo.toml`, `Cargo.lock`: Rust package metadata.
- `AGENTS.md`: agent rules for this repository.
- `plan.md`, `spec.md`: implementation plan and technical contract.
- `build_cuda.cmd`: Windows CUDA build helper.

Primary code and validation areas:

- `src/`: Rust library implementation.
- `examples/`: CLI examples and standalone utilities.
- `tests/`: Rust integration and alignment tests.
- `tests/fixtures/`: small checked-in test fixtures required by tests.
- `benches/`: benchmarks.
- `tools/`: conversion, fixture export, and validation tools.
- `docs/`: user-facing notes, release usage, and engineering records.

Generated or historical validation material:

- `artifacts/`: checked-in non-source artifacts kept for traceability.
- `scripts/local/`: ad-hoc local run wrappers.
- `tools/prototypes/`: temporary upstream/Rust comparison probes.
- `.codebase-memory/`: persisted codebase-memory knowledge graph artifact.

Large local outputs such as new WAV files, unpacked releases, model weights, and
one-off root-level run products are intentionally ignored by `.gitignore`.
