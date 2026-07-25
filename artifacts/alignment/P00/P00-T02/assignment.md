# GPT-5.3 Codex Spark — Implementation Assignment

## Task

- Task ID: `P00-T02`
- Goal: create a fixture manifest contract and fail-closed resolver that never turns missing,
  malformed or hash-mismatched required fixtures into a successful parity result.
- Phase contract: `tasks/P00-baseline-and-parity-harness.md`

## Allowed production files

- `config/fixtures.json`
- `tools/fixture_manifest.py`

## Allowed test/tool files

- `tools/test_fixture_manifest.py`

No other file may be modified without stopping and reporting the need.

## Read first

- `AGENTS.md`
- `tasks/P00-baseline-and-parity-harness.md`
- `TEST_PLAN.md` Fixture Policy
- `config/fixtures.json`
- `tools/fixture_manifest.py`
- `schemas/stage-dump.schema.json` only as a style reference

## Required behavior

1. Define and validate manifest version 1.
2. Each fixture entry must contain:
   `id`, `path`, `source`, `revision`, `sha256`, `license`, `generated_by`, `command`,
   and `required`.
3. Resolve relative fixture paths relative to the manifest file, not process CWD.
4. Reject duplicate IDs, missing fields, unsupported versions, malformed SHA-256, missing files
   and hash mismatches.
5. `--require-all` must fail when the manifest contains no required fixtures.
6. Failures must print a stable machine-readable prefix:
   `FIXTURE_MISSING`, `FIXTURE_INVALID`, or `FIXTURE_HASH_MISMATCH`.
7. Success must print a concise verified-count summary.
8. Do not download weights, add model binaries or invent fixture hashes.

## Required tests

Use `unittest` and temporary directories/files. Cover:

- valid manifest and matching hash passes;
- relative path resolves from manifest directory;
- empty required manifest fails under `--require-all`;
- missing required file fails with `FIXTURE_MISSING`;
- malformed/missing metadata fails with `FIXTURE_INVALID`;
- duplicate fixture ID fails;
- hash mismatch fails with `FIXTURE_HASH_MISMATCH`;
- optional missing fixture behavior is explicit and tested.

## Required commands

```text
python -m unittest tools.test_fixture_manifest -v
python tools/fixture_manifest.py config/fixtures.json --require-all
python tools/check_alignment_static.py . --report artifacts/alignment/P00/P00-T02/static-gap-report.json
```

The second command is expected to exit non-zero with `FIXTURE_MISSING` while
`config/fixtures.json` intentionally contains no real local model entries. Record this expected
fail-closed probe separately; do not change the command into a success.

## Acceptance

- Unit test command exits 0 with all required cases passing.
- Empty/missing/hash-invalid required fixtures produce non-zero exit and the exact stable prefix.
- No fixture-dependent parity path can interpret resolver failure as success.
- Diff stays within the three allowed files and remains independently revertible.

## Constraints

- Do not weaken existing thresholds.
- Do not silently skip missing fixtures.
- Do not redesign unrelated modules.
- Do not change task status.
- Do not add network access or model download behavior.
- Do not modify `STATUS.md`, `TODOS.md`, `tasks/task-index.yaml` or Gate evidence.
- Preserve public CLI compatibility where possible; document intentional exit/output changes.

## Final report

Return:

1. Summary
2. Files changed
3. Design choices
4. Commands run
5. Exact results and exit codes
6. Evidence paths
7. Risks or blockers
