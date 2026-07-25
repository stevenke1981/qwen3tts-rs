# GPT-5.3 Codex Spark — P02-T01 Implementation Assignment

## Task

- Task ID: `P02-T01`
- Goal: replace the ad-hoc SplitMix sampling stream with qwentts.cpp-compatible
  Philox4x32-10 and gate it with independently derived known vectors.
- Phase contract: `tasks/P02-sampling-parity.md`
- Dependency: `P01-T05` and `P01-GATE` must be `GATE_PASSED`.

## Allowed production files

- `src/talker/sampling.rs`

## Allowed test, fixture and documentation files

- `tests/philox_rng_test.rs`
- `fixtures/alignment/p02_philox_vectors.json`
- `config/fixtures.json`
- `docs/alignment/philox-rng.md`
- `tools/generate_philox_vectors.py`

No other file may be modified. Stop and report if another file is required.

## Read first

- `AGENTS.md`
- `tasks/P02-sampling-parity.md`
- `prompts/SPARK_IMPLEMENT.md`
- `src/talker/sampling.rs`
- `E:\qwentts.cpp-reference\src\philox.h`
- `E:\qwentts.cpp-reference\src\sampling.h`
- `E:\qwentts.cpp-reference\src\pipeline-tts.cpp` sampling-state call sites

## Required behavior

1. Implement Philox4x32-10 with the exact qwentts.cpp/Random123 constants,
   wrapping key schedule and 32x32 high/low multiplication.
2. Match the reference counter convention exactly:
   - key = full 64-bit seed split low/high;
   - counter = `(ctr_lo, 0, subsequence_lo, subsequence_hi)`;
   - uniform = `((r.x as f32) + 0.5f32) * 2^-32` with f32 arithmetic;
   - one sampler draw consumes one subsequence and advances it exactly once;
   - greedy sampling consumes no random draw.
3. Integrate the Philox stream into production `Sampler`; remove SplitMix from the
   stochastic path. Preserve the existing public `Sampler::new(seed)` call contract.
4. Define overflow behavior with wrapping arithmetic, including seed zero and
   subsequence rollover. Do not use `unsafe`, external RNG crates, `unwrap`, or `expect`
   in library code.
5. Add exact known-vector tests for:
   - Random123 zero key/zero counter raw four-word output;
   - at least three non-zero seed/subsequence/counter cases produced by an independent
     implementation or the pinned qwentts.cpp reference;
   - exact `f32::to_bits()` uniform values;
   - deterministic sequence/reseed behavior;
   - greedy no-consumption and stochastic one-consumption behavior.
6. The checked-in vector fixture must record source, pinned qwentts.cpp commit
   `82cd05b9f3a175612dc89fd6943e610fab096ef5`, generator command and literal expected
   words/uniform bits. The generator must not call Rust code under test.
7. Keep this task limited to RNG parity. Do not implement repetition penalty, split
   Talker/Code Predictor configs, or change suppression/EOS ordering yet.

## Required commands

```powershell
$env:PATH='C:\Users\steven\.cargo\bin;' + $env:PATH
python tools/generate_philox_vectors.py --check fixtures/alignment/p02_philox_vectors.json
cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo test --lib talker::sampling
cargo test --test philox_rng_test
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
git diff --check -- src/talker/sampling.rs tests/philox_rng_test.rs tools/generate_philox_vectors.py fixtures/alignment/p02_philox_vectors.json docs/alignment/philox-rng.md config/fixtures.json
```

## Acceptance

- Every raw Philox word and uniform `f32` bit pattern equals the pinned reference.
- Production stochastic sampling consumes the Philox stream with documented state
  progression; greedy sampling leaves it untouched.
- Existing sampling tests remain green.
- Fixture SHA-256 in `config/fixtures.json` matches the checked-in file.
- Independent reviewer returns `ACCEPT`.

## Final report

Return summary, exact files changed, vector provenance, commands/results and risks.
Do not change task status and do not commit or push.
