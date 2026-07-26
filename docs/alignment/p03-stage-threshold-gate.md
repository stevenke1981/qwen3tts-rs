# P03-T05 Stage Cosine and Logit Ranking Gate

## Purpose

Fail-closed real-weight Gate across the existing Talker and Code Predictor
stage dumps. Every required aligned stage must meet pinned numerical
thresholds, and logits must preserve token-selection ranking.

## Thresholds (pinned by Sol)

| Metric | Threshold |
|--------|-----------|
| Cosine similarity (all tensor stages) | >= 0.999 |
| Maximum absolute error (all tensor stages) | <= 0.001 |
| Logit top-1 token agreement | exact |
| Logit top-5 set overlap | >= 0.8 |
| Missing / duplicate / non-finite / shape / dtype / layout / provenance failures | zero |
| Full comparison stage count | exactly 723 on both sides |

## Comparisons

1. **Full 723-stage**: official-Python reference vs Candle candidate.
2. **qwentts.cpp anchor**: pinned P00-T05 manifest (16 stages, 10 mapped)
   vs the mapped Candle stage subset.

## Comparator Usage

```powershell
# Full 723-stage comparison
python tools/compare_stage_dumps.py `
    "$env:P03_STAGE_REFERENCE_DIR\manifest.json" `
    "$env:P03_STAGE_DUMP_DIR\manifest.json" `
    --min-cosine 0.999 --max-abs 0.001 `
    --logit-top1-exact --min-logit-top5-overlap 0.8 `
    --expected-stages 723 `
    --report artifacts/alignment/P03/P03-T05/stage-metrics.json

# qwentts.cpp anchor comparison
python tools/compare_stage_dumps.py `
    artifacts/alignment/P00/P00-T05/runs/qwentts-zh-short/tensors/manifest.json `
    "$env:P03_STAGE_DUMP_DIR\manifest.json" `
    --mapping artifacts/alignment/P03/P03-T05/qwentts-stage-map.json `
    --min-cosine 0.999 --max-abs 0.001 `
    --logit-top1-exact --min-logit-top5-overlap 0.8 `
    --require-all-reference-stages `
    --report artifacts/alignment/P03/P03-T05/qwentts-anchor-metrics.json
```

## Fail-Closed Checks

The comparator exits non-zero on:

- Missing reference or candidate manifest file
- Missing stage binary file
- Duplicate stage names in either manifest
- Shape / dtype / layout mismatch
- NaN or Inf values in any stage
- SHA-256 mismatch between manifest and binary
- Stage count != `--expected-stages`
- Empty comparison set
- Mapped reference stage missing from candidate (with `--mapping`)
- Cosine below `--min-cosine`
- Max absolute error above `--max-abs`
- Logit top-1 mismatch (with `--logit-top1-exact`)
- Logit top-5 overlap below `--min-logit-top5-overlap`

## Logit Ranking Metrics

For logit stages (names starting with `talker-logits-` or declared in
`qwentts-stage-map.json` `logit_stages`):

- `top1_ref` / `top1_cand`: deterministic lowest-index argmax
- `top1_match`: boolean exact agreement
- `top5_overlap`: |top5_ref ∩ top5_cand| / 5
- `score_margin_at_diff`: |ref[ref_top1] - cand[cand_top1]| when top-1 differs

## qwentts.cpp Anchor Mapping

`qwentts-stage-map.json` maps 10 of 16 P00-T05 anchor stages to Candle
structural stages. Six stages are unmapped (no structural Candle equivalent):
`codes-full_0000`, `codes-step0_0001`, `output-audio_0003`,
`prompt-ids_0004`, `trailing-text-hidden_0014`, `tts-pad-embed_0015`.

## Mutation Tests

`tests/stage_threshold_gate_test.rs` proves each metric and missing-stage
check can fail using synthetic stage dumps (no real model required).

## Candle Candidate Generation

Set `P03_STAGE_DUMP_DIR` before running the real instrumentation test to
write the 723-stage manifest to the fixed candidate path:

```powershell
$env:P03_STAGE_DUMP_DIR='artifacts\alignment\P03\P03-T05\runs\candle-full'
cargo test --test stage_instrumentation_real_test --features stage-dump -- --ignored --nocapture
```

## Official Python Reference Generation

```powershell
& "$env:QWEN3_TTS_OFFICIAL_PYTHON" tools/export_p03_stage_reference.py `
    --official-source "$env:QWEN3_TTS_OFFICIAL_SOURCE" `
    --official-revision 022e286b98fbec7e1e916cb940cdf532cd9f488e `
    --model "$env:QWEN3_TTS_REAL_MODEL_DIR" `
    --model-revision 5d83992436eae1d760afd27aff78a71d676296fc `
    --case-id p03-t01-real --seed 42 `
    --output-dir "$env:P03_STAGE_REFERENCE_DIR" `
    --expected-stages 723
```
