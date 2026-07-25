# GPT-5.3 Codex Spark — Implementation Assignment

## Task

- Task ID: `P00-T05`
- Goal: implement a fail-closed CPU F32 smoke-corpus report builder that consumes
  P00-T04 reference-run manifests and emits a revisioned, hashed baseline report.
- Phase contract: `tasks/P00-baseline-and-parity-harness.md`

## Allowed files

- `tools/cpu_f32_smoke.py` (new)
- `tools/test_cpu_f32_smoke.py` (new)
- `schemas/cpu-f32-baseline.schema.json` (new)
- `docs/alignment/cpu-f32-smoke.md` (new)

No other file may be modified. Real model execution, fixture-manifest population,
task evidence, and status files remain the Sol integrator's responsibility.

## Input contract

- Accept an explicit corpus JSON and one or more explicit P00-T04
  `reference-run.json` paths.
- Corpus cases have a stable non-empty `id`, literal UTF-8 `text` or a safe
  relative `text_file`, and explicit language.
- Each reference manifest must be version 1, have `exit_status == 0`, a pinned
  non-empty revision, a unique `(adapter, case_id)` identity, and artifacts with
  safe relative paths, exact byte lengths, and matching SHA-256.
- Never infer revisions, discover arbitrary output directories, download models,
  invoke Python/qwentts/Rust inference, or silently skip missing evidence.

## Output contract

- Refuse to overwrite the output report.
- Emit version-1 deterministic JSON with:
  - corpus SHA-256 and every selected case's normalized text SHA-256;
  - target/reference/official pinned revisions supplied explicitly on the CLI;
  - CPU/F32 backend declaration;
  - sorted runs with adapter, case, model, seed, source revision, source manifest
    path/hash, and sorted artifact metadata;
  - coverage summary for prompt/token/tensor/audio kinds;
  - status `PASS` only when all requested adapters/cases and all required artifact
    kinds are present and verified.
- All report paths are relative to an explicit repository root and must remain
  within it. Reject traversal, absolute paths, symlink escapes, duplicates,
  malformed hashes, size/hash mismatch, non-zero child status, and empty required
  artifacts.
- Stable error prefixes: `BASELINE_CONFIG`, `BASELINE_INPUT`,
  `BASELINE_COVERAGE`, `BASELINE_OUTPUT`.
- Support `--dry-run`: validate all inputs and print the deterministic report JSON
  without creating the output file.

## CLI

At minimum:

```text
python tools/cpu_f32_smoke.py \
  --repo-root <path> \
  --corpus <relative-json> \
  --target-revision <sha> \
  --reference-revision <sha> \
  --official-revision <sha> \
  --require-adapter official-python \
  --require-adapter qwentts-cpp \
  --require-case zh-short \
  --require-kind prompt \
  --require-kind token \
  --require-kind tensor \
  --require-kind audio \
  --run <relative-reference-run.json> \
  --output <relative-report.json>
```

## Tests

Use temporary directories and tiny binary fixtures. Cover:

- deterministic report and ordering;
- exact manifest/artifact hashes and sizes;
- full prompt/token/tensor/audio coverage across multiple successful runs;
- missing adapter/case/kind, non-zero run, unpinned revision, duplicate identity,
  duplicate artifact, unsafe/absolute/symlink escape, missing/empty/tampered
  artifact, malformed corpus, and overwrite refusal;
- literal text and safe relative `text_file`;
- `--dry-run` creates no output.

## Required commands

```text
python -m unittest tools.test_cpu_f32_smoke -v
python tools/cpu_f32_smoke.py --help
python -m json.tool schemas/cpu-f32-baseline.schema.json
cargo fmt --all -- --check
cargo check --no-default-features --features cpu
git diff --check -- tools/cpu_f32_smoke.py tools/test_cpu_f32_smoke.py schemas/cpu-f32-baseline.schema.json docs/alignment/cpu-f32-smoke.md
```

## Acceptance

- Synthetic tests prove fail-closed provenance, coverage, and path handling.
- Report construction is deterministic and performs no inference.
- Real P00-T04 runs can be consumed without rewriting their manifests.
- Diff stays inside the four allowed files.
