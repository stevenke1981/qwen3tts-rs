# P03-T05 Assignment — Stage Cosine and Logit Ranking Gate

## Outcome

Create a fail-closed real-weight Gate across the existing Talker and Code
Predictor stage dumps. Every required aligned stage must meet the pinned
numerical thresholds, and logits must preserve token-selection ranking. Produce
machine-readable metrics suitable for the P03 Phase Gate.

## External Implementer

- Assigned model: user-designated external agent (DeepSeek V4 Flash compatible).
- The implementer may edit only the allowed files and may not accept the Gate,
  update project status, commit, or push.

## Authoritative Context

- Dependency: `P03-T04 GATE_PASSED`.
- Target baseline: `b08178964504d5a214565ffc4ff5ed592eb8f7ec`.
- qwentts.cpp: `82cd05b9f3a175612dc89fd6943e610fab096ef5`.
- Official Qwen3-TTS: `022e286b98fbec7e1e916cb940cdf532cd9f488e`.
- Stage schema: `artifacts/alignment/P03/P03-T01/`.
- Talker real metrics: `artifacts/alignment/P03/P03-T02/`.
- Code Predictor real metrics: `artifacts/alignment/P03/P03-T03/`.
- Host-transfer Gate: `artifacts/alignment/P03/P03-T04/`.
- Comparator: `tools/compare_stage_dumps.py`.
- Full official reference output:
  `artifacts/alignment/P03/P03-T05/runs/official-python-full/manifest.json`.
- Full Candle candidate output:
  `artifacts/alignment/P03/P03-T05/runs/candle-full/manifest.json`.
- qwentts.cpp anchor reference:
  `artifacts/alignment/P00/P00-T05/runs/qwentts-zh-short/tensors/manifest.json`.
- qwentts.cpp-to-Candle name mapping:
  `artifacts/alignment/P03/P03-T05/qwentts-stage-map.json`.
- Pinned model snapshot:
  `C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc`.

## Allowed Files

- `tools/compare_stage_dumps.py`
- `tools/export_p03_stage_reference.py`
- `tests/stage_dump_test.rs`
- `tests/stage_instrumentation_real_test.rs`
- `tests/stage_threshold_gate_test.rs`
- `docs/alignment/p03-stage-threshold-gate.md`
- `artifacts/alignment/P03/P03-T05/*`, except `review.md` and `gate.json`

All production model code, thresholds outside this task card, prior Gate
evidence, `STATUS.md`, `TODOS.md`, `tasks/task-index.yaml`, Git history, and
remote state are read-only.

## Required Work

1. Inventory every required stage family from P03-T01 and map reference/Rust
   names without silently dropping unmatched stages.
2. Extend the comparator to emit deterministic JSON containing shape, dtype,
   cosine, max absolute error, and logit ranking metrics.
3. Fail closed on missing files, duplicate stages, schema/layout mismatch,
   NaN/Inf metrics, stale revision, or an empty comparison set.
4. For logit stages, report exact top-1 agreement, top-5 set overlap, and the
   score margin around any differing rank. Preserve deterministic lowest-index
   tie behavior.
5. Use real pinned fixtures. Synthetic tests may test the comparator but cannot
   satisfy the Gate.
6. Add mutation tests proving each metric and missing-stage check can fail.
7. Write `stage-metrics.json`, `commands.txt`, `test-results.txt`, and
   `worker-report.md`.
8. Add `P03_STAGE_DUMP_DIR` support to the real Candle test so its complete
   723-stage manifest is written to the fixed candidate path above. The default
   test behavior may continue using a temporary directory.
9. Add `tools/export_p03_stage_reference.py` to produce the matching full
   723-stage official-Python manifest at the fixed reference path. It must pin
   official source revision, model snapshot revision, case, seed, stage
   names/layouts, and file hashes.
10. Extend the comparator with `--max-abs`, logit top-1/top-5 metrics,
    `--mapping`, and fail-closed provenance/stage-count checks.
11. Run two real comparisons:
    - full official-Python 723-stage reference versus Candle 723-stage candidate;
    - pinned qwentts.cpp anchor manifest versus the mapped Candle stage subset.

The existing P00 qwentts.cpp manifest contains only 16 anchor stages and cannot
substitute for the full 723-stage reference. If the full official reference,
full Candle candidate, or mapping cannot be generated and paired exactly, report
F8 `BLOCKED`; do not fall back to partial or synthetic evidence.

## Acceptance Thresholds

- Every required tensor stage: cosine `>= 0.999`.
- Every required tensor stage: maximum absolute error `<= 0.001`.
- Every required logit stage: exact top-1 token agreement.
- Every required logit stage: top-5 set overlap `>= 0.8`.
- Zero missing, duplicate, non-finite, shape, dtype, layout, or provenance
  failures.
- Full comparison stage count is exactly 723 on both sides.
- Every qwentts.cpp anchor declared in `qwentts-stage-map.json` is compared; no
  unmapped required anchor is silently dropped.
- No ignored or skipped real test may be counted as passing.
- Existing P02 deterministic tokens and P03-T02/T03 real oracles remain green.

These thresholds are pinned by Sol. The external implementer must not weaken
them; a near-tie or backend difference is evidence to report, not permission to
change the Gate.

## Required Commands

```powershell
$env:PATH='C:\Users\steven\.cargo\bin;' + $env:PATH
$env:QWEN3_TTS_REAL_MODEL_DIR='C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc'
$env:QWEN3_TTS_OFFICIAL_SOURCE='C:\Users\steven\Qwen3-TTS'
$env:QWEN3_TTS_OFFICIAL_PYTHON='C:\Users\steven\Qwen3-TTS\.venv\Scripts\python.exe'
$env:P03_STAGE_REFERENCE_DIR='artifacts\alignment\P03\P03-T05\runs\official-python-full'
$env:P03_STAGE_DUMP_DIR='artifacts\alignment\P03\P03-T05\runs\candle-full'

# Report repository-wide formatting status without changing unrelated files.
cargo fmt --all -- --check
rustfmt --edition 2021 --check tests/stage_dump_test.rs tests/stage_instrumentation_real_test.rs tests/stage_threshold_gate_test.rs
cargo check --no-default-features --features cpu,stage-dump
cargo test --test stage_dump_test --features stage-dump
cargo test --test stage_instrumentation_test --features stage-dump
cargo test --test stage_instrumentation_real_test --features stage-dump -- --ignored --nocapture
cargo test --test stage_threshold_gate_test --features stage-dump -- --nocapture
cargo test --test talker_kv_cache_real_test -- --ignored --nocapture
cargo test --test code_predictor_frame_real_test -- --ignored --nocapture
cargo test --test deterministic_token_sequence_real_test -- --ignored --nocapture
& "$env:QWEN3_TTS_OFFICIAL_PYTHON" tools/export_p03_stage_reference.py --official-source "$env:QWEN3_TTS_OFFICIAL_SOURCE" --official-revision 022e286b98fbec7e1e916cb940cdf532cd9f488e --model "$env:QWEN3_TTS_REAL_MODEL_DIR" --model-revision 5d83992436eae1d760afd27aff78a71d676296fc --case-id p03-t01-real --seed 42 --output-dir "$env:P03_STAGE_REFERENCE_DIR" --expected-stages 723
python tools/compare_stage_dumps.py "$env:P03_STAGE_REFERENCE_DIR\manifest.json" "$env:P03_STAGE_DUMP_DIR\manifest.json" --min-cosine 0.999 --max-abs 0.001 --logit-top1-exact --min-logit-top5-overlap 0.8 --expected-stages 723 --report artifacts/alignment/P03/P03-T05/stage-metrics.json
python tools/compare_stage_dumps.py artifacts/alignment/P00/P00-T05/runs/qwentts-zh-short/tensors/manifest.json "$env:P03_STAGE_DUMP_DIR\manifest.json" --mapping artifacts/alignment/P03/P03-T05/qwentts-stage-map.json --min-cosine 0.999 --max-abs 0.001 --logit-top1-exact --min-logit-top5-overlap 0.8 --require-all-reference-stages --report artifacts/alignment/P03/P03-T05/qwentts-anchor-metrics.json
git diff --check -- tools/compare_stage_dumps.py tools/export_p03_stage_reference.py tests/stage_dump_test.rs tests/stage_instrumentation_real_test.rs tests/stage_threshold_gate_test.rs docs/alignment/p03-stage-threshold-gate.md artifacts/alignment/P03/P03-T05
```

## External Deliverables

- `worker-report.md`
- `commands.txt`
- `test-results.txt`
- `stage-metrics.json`
- `qwentts-anchor-metrics.json`
- `qwentts-stage-map.json`
- Fixed-path full reference and candidate manifests with provenance/hashes
- Comparator/test/doc changes within allowed files

Do not create `review.md` or `gate.json`. Sol will inspect the actual diff,
rerun every real command, independently review the metrics, and decide both
P03-T05 and the P03 Phase Gate.
