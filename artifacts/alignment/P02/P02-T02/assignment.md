# GPT-5.3 Codex Spark — P02-T02 Implementation Assignment

## Task

- Task ID: `P02-T02`
- Goal: implement Hugging Face/qwentts.cpp repetition penalty with exact sign,
  uniqueness, history and operation-order semantics.
- Phase contract: `tasks/P02-sampling-parity.md`
- Dependency: `P02-T01` must be `GATE_PASSED`.

## Allowed production files

- `src/talker/sampling.rs`
- `src/talker/talker.rs`
- `src/talker/code_predictor.rs`
- `src/text_frontend/candle_backend.rs`

## Allowed test, fixture, tool and documentation files

- `tests/repetition_penalty_test.rs`
- `tools/generate_repetition_penalty_vectors.py`
- `fixtures/alignment/p02_repetition_penalty_vectors.json`
- `docs/alignment/repetition-penalty.md`
- `config/fixtures.json`

No other file may be modified. Stop and report if another file is required.

## Read first

- `AGENTS.md`
- `tasks/P02-sampling-parity.md`
- `prompts/SPARK_IMPLEMENT.md`
- `src/talker/sampling.rs`
- `src/talker/talker.rs`
- `src/talker/code_predictor.rs`
- `E:\qwentts.cpp-reference\src\sampling.h`
- `E:\qwentts.cpp-reference\src\pipeline-tts.cpp`
- `E:\qwentts.cpp-reference\src\code-predictor-forward.h`

## Required behavior

1. Match the pinned qwentts.cpp/Hugging Face repetition rule exactly for unique valid
   history tokens:
   - `score < 0`: multiply by penalty;
   - `score >= 0`: divide by penalty;
   - duplicate history tokens are transformed once;
   - out-of-range history entries are ignored;
   - penalty `1.0` or empty history is a no-op.
2. Validate penalty inputs and fail clearly for non-finite or non-positive penalty.
   Do not panic and do not silently reinterpret invalid values.
3. Preserve the reference operation chain for stochastic sampling:
   `suppression -> repetition penalty -> f32 temperature divide -> top-k -> top-p
   -> softmax/multinomial`.
   Greedy (`temperature <= 0`) must bypass repetition penalty and consume no Philox draw.
4. Change the production sampler API so the caller supplies the relevant history.
   Integrate Talker codebook-0 history only:
   - history contains emitted non-EOS c0 tokens, in order;
   - it does not contain acoustic codebooks 1..15;
   - it is updated once per completed non-EOS frame;
   - prefill/text prompt IDs are not repetition history.
5. Code Predictor must pass empty history. In this task it shares the current sampling
   options, but repetition penalty must therefore have no effect on its 15 acoustic draws.
   Do not implement the separate Talker/Code Predictor configuration task early.
6. Use the official default repetition penalty `1.05` at the current Candle construction
   site. Do not expand the public `SynthesisOptions` surface in this task; P02-T03 owns
   configuration separation.
7. Preserve Philox state consumption from P02-T01: exactly one draw for a successful
   stochastic sample, none for greedy or an early no-candidate return.
8. Add an independent checked-in oracle (not produced by Rust under test) with literal
   f32 bit patterns before penalty, after penalty, after temperature, plus expected
   candidate/sample results. Record pinned qwentts.cpp commit
   `82cd05b9f3a175612dc89fd6943e610fab096ef5`.
9. Add tests that catch:
   - positive, negative and signed-zero logits;
   - duplicate and invalid history entries;
   - penalty before temperature/top-k/top-p;
   - greedy bypass;
   - Talker c0-only history progression;
   - Code Predictor empty-history behavior;
   - invalid penalty errors;
   - fixture missing/hash mismatch fail-closed behavior where applicable.
10. Keep this task scoped to repetition penalty. Do not implement P02-T03 config
    separation or P02-T04 EOS/suppression redesign.

## Required commands

```powershell
$env:PATH='C:\Users\steven\.cargo\bin;' + $env:PATH
python tools/generate_repetition_penalty_vectors.py --check fixtures/alignment/p02_repetition_penalty_vectors.json
cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo test --lib talker::sampling
cargo test --test repetition_penalty_test
cargo test --test philox_rng_test
cargo test --lib text_frontend
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
git diff --check -- src/talker/sampling.rs src/talker/talker.rs src/talker/code_predictor.rs src/text_frontend/candle_backend.rs tests/repetition_penalty_test.rs tools/generate_repetition_penalty_vectors.py fixtures/alignment/p02_repetition_penalty_vectors.json docs/alignment/repetition-penalty.md config/fixtures.json
```

## Acceptance

- Every oracle f32 bit pattern and operation-order assertion is exact.
- Talker penalty history is c0-only, unique transformation is correct and acoustic
  Code Predictor draws are unaffected by repetition history.
- Philox regression gates remain green.
- Fixture SHA-256 in `config/fixtures.json` matches the checked-in file.
- Independent reviewer returns `ACCEPT`.

## Final report

Return summary, exact files changed, oracle provenance, commands/results and risks.
Do not change task status and do not commit or push.
