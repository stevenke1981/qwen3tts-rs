# GPT-5.3 Codex Spark — P02-T03 Implementation Assignment

## Task

- Task ID: `P02-T03`
- Goal: represent, load and route Talker and Code Predictor sampling settings as
  distinct configurations matching official `generation_config.json`.
- Phase contract: `tasks/P02-sampling-parity.md`
- Dependency: `P02-T02` must be `GATE_PASSED`.

## Allowed production files

- `src/talker/sampling.rs`
- `src/talker/talker.rs`
- `src/talker/code_predictor.rs`
- `src/text_frontend/candle_backend.rs`
- `src/text_frontend/model_catalog.rs`

## Allowed test, fixture, tool and documentation files

- `tests/sampling_config_test.rs`
- `tools/generate_sampling_config_matrix.py`
- `fixtures/alignment/p02_sampling_config_matrix.json`
- `docs/alignment/sampling-config.md`
- `config/fixtures.json`

No other file may be modified. Stop and report if another file is required.

## Read first

- `AGENTS.md`
- `tasks/P02-sampling-parity.md`
- `prompts/SPARK_IMPLEMENT.md`
- `src/talker/sampling.rs`
- `src/talker/talker.rs`
- `src/talker/code_predictor.rs`
- `src/text_frontend/candle_backend.rs`
- installed official wrapper
  `C:\Users\steven\Qwen3-TTS\.venv\Lib\site-packages\qwen_tts\inference\qwen3_tts_model.py`
- real snapshot `generation_config.json`
  `C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc\generation_config.json`
- `E:\qwentts.cpp-reference\src\pipeline-tts.cpp`
- `E:\qwentts.cpp-reference\src\code-predictor-forward.h`

## Required behavior

1. Define distinct typed settings for:
   - Talker: `do_sample`, `temperature`, `top_k`, `top_p`,
     `repetition_penalty`;
   - Code Predictor/subtalker: `subtalker_dosample`,
     `subtalker_temperature`, `subtalker_top_k`, `subtalker_top_p`,
     with repetition penalty fixed to `1.0` and empty history.
2. Parse official sibling `generation_config.json` when Candle weights are loaded.
   Use the installed wrapper's hard defaults only for absent optional keys:
   Talker `true/0.9/50/1.0/1.05`, Code Predictor `true/0.9/50/1.0`.
   A missing file may use these documented official defaults; a present malformed
   file or invalid field must fail closed with a clear config error.
3. Validate finite positive temperatures for sampled paths, `top_p` in `(0, 1]`,
   and valid repetition penalty. `top_k=0` means disabled. When `do_sample=false`,
   the sampler must use greedy argmax, bypass repetition penalty and consume no Philox.
4. Route the two configurations independently through production:
   - Talker c0 calls use Talker settings and c0 history;
   - all 15 Code Predictor calls use Code Predictor settings and empty history;
   - retain one shared Philox stream so the call-order subsequence remains c0 then
     codebooks 1..15.
5. Preserve the current public `SynthesisOptions` surface in this task. Its common
   `temperature/top_k/top_p` values override Talker settings only. Code Predictor
   keeps the parsed subtalker settings. A Talker temperature `<=0` is the existing
   explicit greedy override for Talker only; it must not silently overwrite subtalker
   settings.
6. Do not infer settings from model names or filenames. Store parsed defaults in
   `CandleLLM` and make the resolved pair testable without loading weights.
7. Create an independent matrix fixture for all five public models, pinned to immutable
   revisions, containing literal generation-config hashes and resolved Talker/subtalker
   values. Download only `generation_config.json`, never weights.
8. Add tests for different Talker/subtalker values, sampled/greedy cross-products,
   malformed/missing config, validation boundaries, independent routing, shared Philox
   draw count/order and fixture SHA/provenance.
9. Keep this task scoped to config separation. Do not redesign suppression/EOS handling
   (P02-T04) or deterministic real-token corpus gating (P02-T05).

## Required commands

```powershell
$env:PATH='C:\Users\steven\.cargo\bin;' + $env:PATH
$env:QWEN3_TTS_REAL_MODEL_DIR='C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc'
C:\Users\steven\Qwen3-TTS\.venv\Scripts\python.exe tools/generate_sampling_config_matrix.py --check fixtures/alignment/p02_sampling_config_matrix.json
cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo test --lib talker::sampling
cargo test --test sampling_config_test
cargo test --test repetition_penalty_test
cargo test --test philox_rng_test
cargo test --lib text_frontend
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
git diff --check -- src/talker/sampling.rs src/talker/talker.rs src/talker/code_predictor.rs src/text_frontend/candle_backend.rs src/text_frontend/model_catalog.rs tests/sampling_config_test.rs tools/generate_sampling_config_matrix.py fixtures/alignment/p02_sampling_config_matrix.json docs/alignment/sampling-config.md config/fixtures.json
```

## Acceptance

- Five immutable model revisions/config hashes and exact resolved config pairs recorded.
- Talker and Code Predictor settings can differ and reach only their intended sampler calls.
- Mixed greedy/stochastic modes have exact Philox consumption tests.
- P02-T01/T02 regression gates remain green.
- Fixture SHA-256 matches `config/fixtures.json`.
- Independent reviewer returns `ACCEPT`.

## Final report

Return summary, exact files changed, fixture provenance, commands/results and risks.
Do not change task status and do not commit or push.
