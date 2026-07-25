# P03-T01 Assignment — Stage Instrumentation

## Outcome

Instrument the production Candle Talker and Code Predictor numerical path so
P03-T02 through P03-T05 can compare embeddings, normalization outputs, RoPE
tensors and every decoder-layer output against pinned reference stages.
Instrumentation must be deterministic, uniquely named, shape/layout described,
feature-safe, and effectively no-op when capture is disabled.

## Authoritative Context

- Official Qwen revision:
  `022e286b98fbec7e1e916cb940cdf532cd9f488e`
- qwentts.cpp revision:
  `82cd05b9f3a175612dc89fd6943e610fab096ef5`
- Existing qwentts stage manifest:
  `artifacts/alignment/P00/P00-T05/runs/qwentts-zh-short/tensors/manifest.json`
- Existing observer/writer:
  `src/alignment_stage_dump.rs`
- Existing production paths:
  `src/talker/{talker,model,decoder_layer,talker_attention,code_predictor,input_builder,primitives}.rs`
- Pinned real model:
  `C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc`

## Allowed Files

- `src/alignment_stage_dump.rs`
- `src/lib.rs`
- `src/talker/talker.rs`
- `src/talker/model.rs`
- `src/talker/decoder_layer.rs`
- `src/talker/talker_attention.rs`
- `src/talker/code_predictor.rs`
- `src/talker/input_builder.rs`
- `src/talker/primitives.rs`
- `tests/stage_dump_test.rs`
- `tests/stage_instrumentation_test.rs`
- `tests/stage_instrumentation_real_test.rs`
- `tools/compare_stage_dumps.py`
- `docs/alignment/p03-stage-instrumentation.md`

Do not modify task/status indexes, P02 files, unrelated production modules,
gate closeout evidence, Git state, or remote state.

## Required Design

1. Extend `StageDumpObserver` with a stable structured or generic stage hook
   that defaults to no-op and is implemented by `StageDumpWriter`.
2. Preserve current public forward APIs. Add observer-aware variants and let
   existing methods delegate through `NoopStageDumpObserver`.
3. Capture, with stable unique names and layouts:
   - Talker input/prompt and codec embeddings;
   - Talker position IDs and RoPE cos/sin;
   - for every Talker layer: input norm, attention output, post-attention norm,
     MLP output and final layer output;
   - Talker final norm output;
   - Code Predictor projected/input embeddings and per-step codec embeddings;
   - Code Predictor position IDs and RoPE cos/sin;
   - for every Code Predictor layer: input norm, attention output,
     post-attention norm, MLP output and final layer output;
   - Code Predictor final norm and existing step logits/codes.
4. Stage names must encode component, phase/frame/step, layer and substage so
   prefill, Talker incremental steps and 15 Code Predictor calls never collide.
   Ordering must be deterministic.
5. Document exact mappings to existing qwentts names, including:
   `talker-input-embed`, `talker-hidden-prefill-l{n}`,
   `talker-hidden-prefill-final`, `talker-logits-prefill`,
   `next-emb-step0`, and `talker-hidden-step1`.
6. Normal production with `wants_capture() == false` must not copy tensors to
   host, format dynamic stage names, write files, or allocate capture buffers.
   Capture-only formatting/allocation must sit behind the capture branch.
7. Writer must reject duplicate names and invalid component/phase/layer/step
   identifiers. Manifest dtype, shape, layout, byte order and SHA remain
   fail-closed.
8. Add a fast synthetic production-path test that proves all expected
   categories and ordering for Talker plus Code Predictor.
9. Add an ignored fail-closed real-weight test that loads the exact pinned
   snapshot, runs production `InputBuilder -> generate_sampled_with_observer`
   for one complete frame plus terminal c0, commits a manifest, and asserts:
   - the exact model/snapshot metadata;
   - every configured Talker layer and Code Predictor layer is present;
   - no duplicate stage names;
   - required stage families, shapes/layouts and hashes;
   - normal Noop execution still produces the same token matrix.
10. Actually run the pinned real-weight test. Missing model/config must produce
    `FIXTURE_MISSING`, never a silent return.

## Required Commands

```powershell
$env:PATH='C:\Users\steven\.cargo\bin;' + $env:PATH
$env:QWEN3_TTS_REAL_MODEL_DIR='C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc'

cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo check --no-default-features --features cpu,stage-dump
cargo test --test stage_dump_test --features stage-dump
cargo test --test stage_instrumentation_test --features stage-dump
cargo test --test stage_instrumentation_real_test --features stage-dump -- --ignored --nocapture
cargo test --lib talker
cargo test --test deterministic_token_sequence_test
cargo test --test deterministic_token_sequence_real_test -- --ignored --nocapture
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
git diff --check -- src/alignment_stage_dump.rs src/lib.rs src/talker/talker.rs src/talker/model.rs src/talker/decoder_layer.rs src/talker/talker_attention.rs src/talker/code_predictor.rs src/talker/input_builder.rs src/talker/primitives.rs tests/stage_dump_test.rs tests/stage_instrumentation_test.rs tests/stage_instrumentation_real_test.rs tools/compare_stage_dumps.py docs/alignment/p03-stage-instrumentation.md
```

## Completion Report

Report the observer/API design, exact stage-name families and counts, normal
path overhead protections, synthetic and real manifest counts, real snapshot
evidence, every command result and remaining risks. Do not commit or push.
