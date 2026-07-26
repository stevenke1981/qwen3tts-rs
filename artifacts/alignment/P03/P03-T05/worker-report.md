# P03-T05 Worker Report

## Status: F8 BLOCKED — Official Python 723-stage reference cannot be generated

The complete 723-stage official-Python reference cannot be produced with the
current monkey-patching approach. The export script produces 678 of 723
required stages. The 46 missing stages fall into four categories that require
a custom generation loop (reimplementing the Candle generation pipeline in
Python) rather than forward-hook instrumentation of the HuggingFace model.

## What Was Completed

### Comparator (`tools/compare_stage_dumps.py`)
- `--max-abs` threshold check
- Logit top-1 exact agreement (`--logit-top1-exact`)
- Logit top-5 set overlap (`--min-logit-top5-overlap`)
- `--mapping` file for qwentts.cpp anchor comparison
- `--expected-stages` count check
- `--require-all-reference-stages`
- Fail-closed on NaN/Inf, SHA-256 mismatch, empty comparison set, missing files
- Deterministic lowest-index tie-breaking argmax
- Provenance fields in report (source, revision, stage counts, thresholds)

### Candle 723-stage candidate
- `P03_STAGE_DUMP_DIR` support added to `stage_instrumentation_real_test.rs`
- Real 0.6B model run produced exactly 723 stages at the fixed candidate path
- Test passed in 97.26s

### qwentts-stage-map.json
- 10 of 16 P00-T05 anchor stages mapped to Candle structural stages
- 6 unmapped stages documented with reasons
- 1 logit stage declared (`talker-logits-prefill_0013`)

### Mutation tests (`tests/stage_threshold_gate_test.rs`)
- 12 tests proving each metric and missing-stage check can fail
- All 12 pass

### Documentation
- `docs/alignment/p03-stage-threshold-gate.md`

## Missing Stages Analysis (46 total)

| Category | Count | Reason |
|----------|-------|--------|
| CP hidden-l4 (prefill + steps 1-14) | 15 | `output_hidden_states` captures pre-layer states; post-layer-4 pre-norm state not exposed |
| CP position-ids, rope-cos, rope-sin | 3 | Computed inside CP model forward, not returned or hookable at the right granularity |
| Legacy hook stages (`talker_codebook0_logits_*`, `code_predictor_step_logits_*`, `talker_final_code_matrix`, `code_predictor_final_code_matrix_*`) | 19 | Emitted by Candle's dedicated `StageDumpObserver` hooks, not structural `on_stage` calls; no PyTorch equivalent hook point |
| Talker generation-loop stages (`next-emb-step0`, `talker-codec-embed-frame0`, `talker-hidden-{prefill,step1}-l27`, `talker-input-step1`, `talker-logits-step1`, `talker-step1-codec-embed`, `talker-step1-rope-{cos,sin}`) | 9 | Computed inside HuggingFace `generate()` loop; `forward()` wrapper cannot intercept them without reimplementing the generation loop |

## Resolution Path

To unblock, the export script must implement a **custom generation loop** that:
1. Constructs Talker input embeddings from the fixture's `prompt_ids`
2. Runs Talker prefill manually (not via `generate()`)
3. For each generation step: computes logits, samples codebook-0, runs Code
   Predictor manually, computes next-emb, runs Talker step
4. Captures all 723 stages at the exact Candle capture points

This is equivalent to reimplementing the Candle `generate_sampled_with_observer`
loop in Python. It requires matching the exact sampling behavior (Philox RNG),
input construction, and stage capture ordering.

## Changed Files

- `tools/compare_stage_dumps.py` — extended comparator
- `tools/export_p03_stage_reference.py` — new (partial, 678/723 stages)
- `tests/stage_instrumentation_real_test.rs` — P03_STAGE_DUMP_DIR support
- `tests/stage_threshold_gate_test.rs` — new (12 mutation tests)
- `docs/alignment/p03-stage-threshold-gate.md` — new
- `artifacts/alignment/P03/P03-T05/qwentts-stage-map.json` — new
- `artifacts/alignment/P03/P03-T05/runs/candle-full/` — 723-stage Candle candidate

## Remaining Risks

- Repository-wide `cargo fmt --all -- --check` is red due to pre-existing
  unformatted files outside P03-T05 allowed files.
- MSVC `LNK4098` linker warning is pre-existing.
- The partial 678-stage official-python-full output exists on disk but must
  not be used as Gate evidence.
