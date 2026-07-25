# GPT-5.6 Luna — P02-T04 Implementation Assignment

## Task

- Task ID: `P02-T04`
- Goal: match official Talker token suppression, minimum-new-token EOS policy
  and EOS-frame termination while leaving Code Predictor logits unsuppressed.
- Phase contract: `tasks/P02-sampling-parity.md`
- Dependency: `P02-T03` is `GATE_PASSED`.

## Allowed production files

- `src/talker/sampling.rs`
- `src/talker/talker.rs`
- `src/talker/code_predictor.rs`
- `src/talker/config.rs`
- `src/text_frontend/candle_backend.rs`
- `src/text_frontend/model_catalog.rs`

## Allowed test, fixture, tool and documentation files

- `tests/suppression_eos_test.rs`
- `tools/generate_suppression_eos_vectors.py`
- `fixtures/alignment/p02_suppression_eos_vectors.json`
- `docs/alignment/suppression-eos.md`
- `config/fixtures.json`

No other file may be modified. Stop and report if another file is required.

## Authoritative references

1. Official installed Qwen implementation:
   `C:\Users\steven\Qwen3-TTS\.venv\Lib\site-packages\qwen_tts\core\models\modeling_qwen3_tts.py`
   - `generate(... min_new_tokens=2 ...)`;
   - `suppress_tokens = range(vocab_size - 1024, vocab_size)` except
     `codec_eos_token_id`;
   - postprocessing truncates before the first EOS frame.
2. Official Qwen source revision:
   `022e286b98fbec7e1e916cb940cdf532cd9f488e`.
3. qwentts.cpp revision:
   `82cd05b9f3a175612dc89fd6943e610fab096ef5`.
4. qwentts.cpp:
   - `src/sampling.h::apply_suppress`;
   - `src/pipeline-tts.cpp` c0 sampling and immediate EOS termination.

Where qwentts.cpp lacks the official `min_new_tokens=2` behavior, follow the
reference priority in `ACCEPTANCE_CRITERIA.md`: official Python first.

## Required behavior

1. Talker c0 masks `[vocab_size - 1024, vocab_size)` before greedy or sampled
   selection. The start is derived from validated runtime vocabulary size, not
   hard-coded `2048`.
2. Code Predictor codebooks 1–15 receive no Talker reserved-range suppression.
3. The official `min_new_tokens=2` policy suppresses EOS for generated c0 steps
   0 and 1. EOS becomes the sole allowed reserved token starting at step 2.
4. When c0 is EOS:
   - stop immediately;
   - do not run Code Predictor for that step;
   - do not append an EOS frame;
   - do not append EOS to repetition history;
   - consume no extra Philox draws beyond the c0 draw that selected EOS.
5. Greedy and sampled paths use identical suppression/EOS semantics. Remove or
   delegate duplicated greedy logic so it cannot bypass suppression.
6. Invalid model/runtime invariants fail closed:
   - vocabulary too small for the 1024-token reserved suffix;
   - EOS outside vocabulary or outside the reserved suffix;
   - suppression leaves no finite candidate.
   Do not silently return token `0`.
7. Keep P02-T03 independent sampling settings, shared Philox order and
   Talker-only public greedy override unchanged.
8. Do not implement the real-token deterministic parity corpus (`P02-T05`) or
   alter unrelated prompt, numerical-layer or decoder behavior.

## Required tests

- Exact literal mask/selection vectors for greedy and sampled paths.
- Dynamic suppression start with at least two vocabulary sizes.
- EOS forbidden on steps 0/1 and allowed on step 2.
- Actual synthetic Talker production run proving two non-EOS frames followed by
  EOS termination without a CP call or output row for EOS.
- Exact CP call count, shared Philox count and c0 history after termination.
- Actual Code Predictor run proving no reserved-range suppression.
- All-finite-candidates-suppressed/non-finite case returns an error without a
  Philox draw.
- P02-T01/T02/T03 regression gates remain green.
- Fixture metadata, pinned source revisions and SHA-256 are fail-closed and
  independently reviewed.

## Required commands

```powershell
$env:PATH='C:\Users\steven\.cargo\bin;' + $env:PATH
C:\Users\steven\Qwen3-TTS\.venv\Scripts\python.exe tools/generate_suppression_eos_vectors.py --check fixtures/alignment/p02_suppression_eos_vectors.json
cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo test --lib talker::sampling
cargo test --lib talker::talker
cargo test --lib talker::code_predictor
cargo test --test suppression_eos_test
cargo test --test sampling_config_test
cargo test --test repetition_penalty_test
cargo test --test philox_rng_test
cargo test --lib text_frontend
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
git diff --check -- src/talker/sampling.rs src/talker/talker.rs src/talker/code_predictor.rs src/talker/config.rs src/text_frontend/candle_backend.rs src/text_frontend/model_catalog.rs tests/suppression_eos_test.rs tools/generate_suppression_eos_vectors.py fixtures/alignment/p02_suppression_eos_vectors.json docs/alignment/suppression-eos.md config/fixtures.json
```

## Acceptance

- Official suppression suffix and minimum-new-token EOS semantics are literal,
  dynamic and production-tested.
- EOS position/frame count semantics match official postprocessing.
- Code Predictor remains unsuppressed.
- No-candidate paths fail closed.
- Independent reviewer returns `ACCEPT`.

## Final report

Return exact files changed, reference provenance, fixture SHA, commands/results
and risks. Do not change task status and do not commit or push.
