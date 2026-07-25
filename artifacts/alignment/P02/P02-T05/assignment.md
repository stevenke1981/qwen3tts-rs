# P02-T05 Assignment — Deterministic Token Sequence Parity

## Outcome

Gate exact fixed-seed 16-codebook token sequences without conflating sampling
parity with the P03 numerical-parity work. The committed corpus must cover the
independent qwentts.cpp Philox sampling contract, and one real 0.6B Base
production run must reproduce the official patched-PyTorch oracle exactly.

## Authoritative References

- Official Qwen3-TTS revision:
  `022e286b98fbec7e1e916cb940cdf532cd9f488e`
- qwentts.cpp revision:
  `82cd05b9f3a175612dc89fd6943e610fab096ef5`
- qwentts.cpp sampling order:
  `E:\qwentts.cpp-reference\src\sampling.h`
- qwentts.cpp Philox implementation:
  `E:\qwentts.cpp-reference\src\philox.h`
- Patched PyTorch multinomial oracle:
  `E:\qwentts.cpp-reference\tests\cossim_common.py`
- Official installed model implementation:
  `C:\Users\steven\Qwen3-TTS\.venv\Lib\site-packages\qwen_tts\core\models\modeling_qwen3_tts.py`
- Real 0.6B Base snapshot:
  `C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc`

## Allowed Files

Production changes are allowed only when required to expose or correct the
existing production sampling route:

- `src/talker/sampling.rs`
- `src/talker/talker.rs`
- `src/talker/code_predictor.rs`
- `src/talker/input_builder.rs`
- `src/talker/config.rs`
- `src/text_frontend/candle_backend.rs`

Test, oracle, fixture and documentation files:

- `tests/deterministic_token_sequence_test.rs`
- `tests/deterministic_token_sequence_real_test.rs`
- `tools/generate_deterministic_token_sequences.py`
- `tools/export_talker_fixtures.py`
- `fixtures/alignment/p02_deterministic_token_sequences.json`
- `fixtures/alignment/p02_deterministic_token_sequences_real.json`
- `docs/alignment/deterministic-token-sequences.md`
- `config/fixtures.json`

Do not modify task indexes, `STATUS.md`, `TODOS.md`, gate/evidence closeout
files, Git state, or unrelated files.

## Required Semantics and Evidence

1. Build an independent, offline-checkable qwentts.cpp sampling corpus:
   - exact operation order: repetition penalty, temperature, top-k, top-p,
     softmax, Philox multinomial;
   - Talker dynamic reserved-suffix suppression and two-token EOS minimum;
   - Code Predictor receives no Talker suppression;
   - one shared Philox subsequence across c0 then c1..c15;
   - exact c0 history accumulation and empty Code Predictor history;
   - sampled/sampled, sampled/greedy, greedy/sampled and greedy/greedy routes;
   - multiple nontrivial seeds, top-k/top-p boundaries, negative/positive
     repetition-penalty logits, and at least one EOS termination case;
   - literal expected `[frame][16]` token matrices and exact draw counts.
2. The generator must have a deterministic offline `--check` mode that does
   not need a Hugging Face cache, model load, network, or qwentts checkout.
   It must fail closed on changed revisions, metadata, case IDs, duplicates,
   missing routes, changed expected matrices, or a stale manifest SHA.
3. Add a real-weight official oracle fixture generated with the pinned local
   0.6B Base model and the qwentts patched-PyTorch Philox multinomial:
   - use the real official prompt/token construction for a short Chinese Base
     prompt;
   - record prompt IDs, language, seed, exact Talker/subtalker options, maximum
     frame count, expected Philox draws and exact 16-codebook matrix;
   - use at least two generated frames and the normal sampled/sampled route;
   - pin snapshot revision and fixture SHA.
4. Add a fail-closed ignored Rust integration test that loads the exact real
   snapshot, builds inputs through the Rust production `InputBuilder`, calls
   the production `Talker::generate_sampled` route, and asserts the entire
   literal matrix and draw contract. Missing model/snapshot/fixture must fail
   with `FIXTURE_MISSING`; never silently return or skip.
5. Add a normal, fast Rust integration test that replays the independent
   committed corpus through the production `Sampler` and sequence-routing
   contract. It must assert exact token matrices, histories, per-call modes,
   Philox subsequences, EOS truncation and rerun determinism.
6. Tests must prove that changing a seed changes at least one sampled token and
   rerunning the same seed produces byte-identical tokens.
7. Fixture and docs must state clearly that P02 gates sampling decisions using
   authoritative/captured logits; P03 separately gates Rust-vs-reference model
   logits and hidden states.

## Required Commands

```powershell
$env:PATH='C:\Users\steven\.cargo\bin;' + $env:PATH
$env:QWEN3_TTS_REAL_MODEL_DIR='C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc'
$env:QWEN3_TTS_MODEL_SAFETENSORS="$env:QWEN3_TTS_REAL_MODEL_DIR\model.safetensors"

C:\Users\steven\Qwen3-TTS\.venv\Scripts\python.exe tools/generate_deterministic_token_sequences.py --check
cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo test --test deterministic_token_sequence_test
cargo test --lib talker::sampling
cargo test --lib talker::talker
cargo test --lib talker::code_predictor
cargo test --test suppression_eos_test
cargo test --test sampling_config_test
cargo test --test repetition_penalty_test
cargo test --test philox_rng_test
cargo test --test deterministic_token_sequence_real_test -- --ignored --nocapture
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
git diff --check -- src/talker/sampling.rs src/talker/talker.rs src/talker/code_predictor.rs src/talker/input_builder.rs src/talker/config.rs src/text_frontend/candle_backend.rs tests/deterministic_token_sequence_test.rs tests/deterministic_token_sequence_real_test.rs tools/generate_deterministic_token_sequences.py tools/export_talker_fixtures.py fixtures/alignment/p02_deterministic_token_sequences.json fixtures/alignment/p02_deterministic_token_sequences_real.json docs/alignment/deterministic-token-sequences.md config/fixtures.json
```

## Completion Report

Report modified files, oracle-generation method, exact fixture hashes, real
model snapshot verification, literal case matrices/draw counts, every command
result, and remaining risks. Do not commit or push.
