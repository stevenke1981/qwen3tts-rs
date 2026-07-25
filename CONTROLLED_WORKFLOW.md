# Controlled Workflow

## State Files

- `TODOS.md`: planned and completed work.
- `STATUS.md`: current phase, current task, test evidence and blockers.
- `docs/alignment/decisions/ADR-*.md`: architecture decisions.
- `artifacts/alignment/`: generated evidence; large binaries remain gitignored.
- `tasks/task-index.yaml`: machine-readable task dependencies.

## Per-Task Procedure

1. Sol selects the first READY task whose dependencies passed.
2. Sol writes `artifacts/alignment/<phase>/<task>/assignment.md`.
3. Spark implements only the assignment.
4. Spark writes `worker-report.md`.
5. Sol or independent Spark reviews the diff and writes `review.md`.
6. Sol runs the task gate and stores `gate.json`.
7. Only Sol changes the task status.
8. On failure, keep evidence, revert only the task change if necessary, and create a repair task.

## Boundary Rules

- qwentts.cpp is a behavioral reference, not a build dependency.
- qwen3tts-rs public APIs must remain idiomatic Rust.
- Maintain a clean separation between:
  model semantics, stateful runtime, backend kernels, product surfaces and test tooling.
- Do not solve codec streaming by increasing history windows.
- Do not solve quantization by storing packed weights but expanding them permanently to F32.
- Do not solve deterministic parity by forcing greedy decoding.
- Do not label a buffered full decode as streaming.

## Rollback

Every task commit must be independently revertible. Before a high-risk task:

```powershell
git status --short
git switch -c alignment/<phase>-<task>
cargo test --lib
```

If the task fails its gate, preserve evidence and either fix on the same branch or revert the
single task commit. Do not reset unrelated user work.
