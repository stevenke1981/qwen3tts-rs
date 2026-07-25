# CPU F32 Smoke Baseline Report (`tools/cpu_f32_smoke.py`)

## Purpose

`cpu_f32_smoke.py` builds a deterministic CPU/F32 baseline report from explicit
P00-T04 `reference-run.json` manifests. The tool is fail-closed:

- no inference is executed;
- all inputs are validated before writing output;
- malformed or non-reproducible data raises a prefixed error;
- no required path is overwritten.

## CLI contract

```text
python tools/cpu_f32_smoke.py \
  --repo-root <path> \
  --corpus <relative-json> \
  --target-revision <sha> \
  --reference-revision <sha> \
  --official-revision <sha> \
  [--require-adapter official-python] \
  [--require-adapter qwentts-cpp] \
  [--require-case <case-id> ...] \
  [--require-kind prompt|token|tensor|audio ...] \
  --run <relative-reference-run.json> ... \
  --output <relative-report.json> \
  [--dry-run]
```

All paths are interpreted as repository-relative paths.

## Validation and failure classes

- `BASELINE_CONFIG`:
  - malformed command arguments (missing revisions, unknown adapter/kind, etc.).
- `BASELINE_INPUT`:
  - malformed corpus JSON;
  - malformed reference run;
  - artifact hash/size mismatch;
  - non-zero `exit_status`;
  - duplicate `(adapter, case_id)` identity;
  - path traversal, absolute path, or symlink escape.
- `BASELINE_COVERAGE`:
  - required adapter/case/kind coverage incomplete.
- `BASELINE_OUTPUT`:
  - report output path would overwrite an existing file.

## Report output

The report is emitted as pretty JSON with sorted, deterministic keys and fields:

```json
{
  "schema_version": 1,
  "backend": "CPU/F32",
  "target_revision": "...",
  "reference_revision": "...",
  "official_revision": "...",
  "corpus": {
    "path": "<relative corpus path>",
    "sha256": "....",
    "cases": [
      {
        "id": "zh-short",
        "language": "zh",
        "text_sha256": "...."
      }
    ]
  },
  "runs": [
    {
      "adapter": "official-python|qwentts.cpp",
      "case": "zh-short",
      "model": "Qwen3TTS",
      "seed": 0,
      "source_revision": "...",
      "source_manifest_path": "artifacts/reference-run.json",
      "source_manifest_sha256": "...",
      "artifacts": [
        {"kind":"prompt","relative_path":"...","byte_length":123,"sha256":"..."},
        {"kind":"token","relative_path":"...","byte_length":123,"sha256":"..."},
        {"kind":"tensor","relative_path":"...","byte_length":123,"sha256":"...","shape":[]...},
        {"kind":"audio","relative_path":"...","byte_length":123,"sha256":"..."}
      ]
    }
  ],
  "coverage": {
    "prompt": {"required": 1, "present": 1, "missing": 0, "status": "PASS"},
    "token": {"required": 1, "present": 1, "missing": 0, "status": "PASS"},
    "tensor": {"required": 0, "present": 0, "missing": 0, "status": "PASS"},
    "audio": {"required": 1, "present": 1, "missing": 0, "status": "PASS"}
  },
  "required": {
    "adapters": ["official-python"],
    "cases": ["zh-short"],
    "kinds": ["prompt", "token", "audio"]
  },
  "status": "PASS"
}
```

`status` is `PASS` only when:

- every required `(adapter, case)` pair exists in `--run`;
- for each required case, the union of required kinds across all selected adapters is present.

Otherwise `FAIL` and, where applicable, the tool exits with `BASELINE_COVERAGE`.

## `--dry-run`

`--dry-run` performs the full validation path and prints the deterministic report JSON.
No report file is written.

## Notes

- Reference artifact `stdout` is canonicalized to `token` in the report.
- `--require-adapter qwentts-cpp` is canonicalized to `qwentts.cpp` for internal matching.
- The script is intentionally strict and rejects symlink-based escapes, absolute/relative traversal paths, duplicate artifact paths, and mismatched hash/size evidence.
- `stderr` artifacts are preserved in each run record when present, and they are not part of coverage accounting. Coverage keys remain `prompt`, `token`, `tensor`, `audio` only.
