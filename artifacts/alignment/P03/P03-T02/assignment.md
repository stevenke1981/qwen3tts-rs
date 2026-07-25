# P03-T02 Assignment — Talker Prefill and Single-Step KV Cache Parity

## Outcome

Prove and, where necessary, correct the production Candle Talker prefill plus
single-token incremental path so it is numerically equivalent to a full
recompute of the same sequence and matches the pinned qwentts/official
semantics. Verify every layer's K/V content, cache growth, M-RoPE position, and
last-token hidden/logit output. This is a real-weight gate, not a synthetic-only
task.

## Authoritative Context

- Official Qwen revision:
  `022e286b98fbec7e1e916cb940cdf532cd9f488e`
- qwentts.cpp revision:
  `82cd05b9f3a175612dc89fd6943e610fab096ef5`
- Pinned model snapshot:
  `C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc`
- P03-T01 stage schema and real gate:
  `src/alignment_stage_dump.rs`,
  `tests/stage_instrumentation_real_test.rs`
- Production paths:
  `src/talker/{talker,model,decoder_layer,talker_attention,primitives,input_builder}.rs`
- Existing qwentts baseline:
  `artifacts/alignment/P00/P00-T05/runs/qwentts-zh-short/tensors/manifest.json`

## Allowed Files

- `src/talker/talker.rs`
- `src/talker/model.rs`
- `src/talker/decoder_layer.rs`
- `src/talker/talker_attention.rs`
- `src/talker/primitives.rs`
- `src/alignment_stage_dump.rs` only if an additional cache-stage hook is
  strictly required
- `tests/talker_kv_cache_test.rs`
- `tests/talker_kv_cache_real_test.rs`
- `fixtures/alignment/p03_talker_kv_cache_real.json`
- `tools/generate_talker_kv_cache_fixture.py`
- `docs/alignment/p03-talker-kv-cache.md`
- `config/fixtures.json` only to register a real fixture

Do not modify sampling semantics, Code Predictor behavior, task/status indexes,
P03-T01 evidence, Git state, or remote state.

## Required Work

1. Trace the production path:
   `InputBuilder -> Talker prefill -> per-layer KV cache -> cached M-RoPE
   position -> first incremental Talker forward`.
2. Define an inspectable cache contract without introducing host copies on the
   normal generation path:
   - one cache entry per configured Talker layer;
   - K/V layout exactly `[batch, num_kv_heads, sequence, head_dim]`;
   - prefill length equals prompt sequence length;
   - one incremental call grows sequence length by exactly one;
   - cached prefix K/V remains bit-identical after append;
   - appended K/V equals the final position from a full recompute within the
     required numerical threshold.
3. Build a fast tiny-model production test with at least two Talker layers.
   Compare:
   - full forward over `prefill + next_input`;
   - prefill followed by one cached forward over `next_input`;
   - every layer's K/V prefix and appended position;
   - final normalized hidden state and codec logits.
4. Cover M-RoPE position semantics using production
   `compute_position_ids`, `rope_delta`, and
   `cached_positions_from_delta`. Include a non-zero delta case and mutation
   tests for an off-by-one cache position.
5. Test cache error contracts fail closed: wrong cache count, incompatible
   batch/head/head-dim/sequence shapes, or malformed K/V pairs must return a
   domain error and must not partially update earlier layers. Do not panic.
6. Add an ignored real-weight test that:
   - requires the exact pinned snapshot path and pinned official/qwentts
     revisions;
   - uses production `InputBuilder`;
   - compares full recompute versus prefill+one-step for all 28 layers;
   - asserts exact cache layouts/growth/prefix preservation;
   - asserts per-layer appended K/V cosine >= 0.999 and final hidden/logit
     cosine >= 0.999;
   - records max absolute difference and cosine values in a deterministic JSON
     artifact/fixture;
   - fails with `FIXTURE_MISSING` if any required real input is absent.
7. If the existing implementation diverges, fix the root cause in the allowed
   production files. Do not lower thresholds, skip layers, compare a path to
   itself through shared erroneous state, or hide errors with broad tolerances.
8. Preserve P03-T01 capture-off guarantees and qwentts canonical stage names.

## Required Commands

## Additional Allowed Files (oracle-approved root-cause fix)

- `src/talker/input_builder.rs`
- `tests/prompt_assembly_test.rs`
- `artifacts/alignment/P03/P03-T02/metrics.json`

```powershell
$env:PATH='C:\Users\steven\.cargo\bin;' + $env:PATH
$env:QWEN3_TTS_REAL_MODEL_DIR='C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc'

cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo check --no-default-features --features cpu,stage-dump
cargo test --test talker_kv_cache_test
cargo test --test talker_kv_cache_real_test -- --ignored --nocapture
cargo test --test stage_dump_test --features stage-dump
cargo test --test stage_instrumentation_test --features stage-dump
cargo test --test stage_instrumentation_real_test --features stage-dump -- --ignored --nocapture
cargo test --lib talker
cargo test --test deterministic_token_sequence_test
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
git diff --check -- src/talker/talker.rs src/talker/model.rs src/talker/decoder_layer.rs src/talker/talker_attention.rs src/talker/primitives.rs src/alignment_stage_dump.rs tests/talker_kv_cache_test.rs tests/talker_kv_cache_real_test.rs fixtures/alignment/p03_talker_kv_cache_real.json tools/generate_talker_kv_cache_fixture.py docs/alignment/p03-talker-kv-cache.md config/fixtures.json
```

## Completion Report

Report the discovered root cause or proof of correctness, exact cache layout
and growth contract, tiny and real per-layer thresholds/results, M-RoPE
position evidence, every command result, capture-path regression status, and
remaining risks. Do not commit or push.
