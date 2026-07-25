# P02-T03 Root Pre-Gate Review

## Status

`REJECT` — implementation is not ready for independent review or task-gate closure.

## Verified passing evidence

- Sampling-config generator `--check`: pass.
- `cargo fmt --all -- --check`: pass.
- `cargo check --no-default-features --features cpu`: pass.
- `cargo test --lib talker::sampling`: 10 passed.
- `cargo test --test sampling_config_test`: 4 passed.
- `cargo test --test repetition_penalty_test`: 5 passed.
- `cargo test --test philox_rng_test`: 3 passed.
- `cargo test --lib text_frontend`: 32 passed.
- Both `candle-llm` synthesis examples compile.
- Scoped `git diff --check`: pass (line-ending warnings only).

## Blocking findings

1. **Five-model provenance is missing.**
   `fixtures/alignment/p02_sampling_config_matrix.json` contains only synthetic
   cases. It has no entry for the five public Qwen3-TTS models, no per-model
   immutable revision, no literal `generation_config.json` SHA-256, and no
   resolved Talker/subtalker pair. This contradicts assignment requirements 7–8
   and both fixture acceptance bullets.

2. **Talker and subtalker modes are not independently routed.**
   `CandleLLM::synthesize()` selects the whole generation path using only
   `generation_sampling.talker.do_sample`. When Talker is greedy but subtalker
   sampling is enabled, it calls `Talker::generate()`, which makes all Code
   Predictor steps greedy. The required `talker=false/subtalker=true`
   cross-product therefore cannot work.

3. **The existing Talker-only greedy override is broken.**
   `SynthesisOptions.temperature <= 0` is copied into Talker options while the
   original `generation_sampling.talker.do_sample=true` is still passed to
   `sample_with_mode()`. The sampler correctly rejects a non-positive
   temperature on a sampled path, so the documented Talker-only greedy override
   becomes an error instead of greedy inference.

4. **Cross-product RNG tests are missing.**
   Current tests cover parser values plus isolated sampled/greedy calls. They do
   not execute and lock all four Talker/subtalker mode combinations, the shared
   Philox call order (`c0`, then codebooks `1..15`), or the invariant that greedy
   calls consume no Philox draw while sampled calls consume exactly one.

5. **The current matrix test can pass despite findings 1–4.**
   It locally reimplements effective routing with a helper that checks only the
   Talker flag and does not exercise `CandleLLM`, `Talker`, or Code Predictor.
   The green test is therefore insufficient evidence for production routing.

## Required repair

- Add the five public models using revisions pinned in
  `fixtures/alignment/p01_prompt_id_matrix.json`.
- Download only each pinned `generation_config.json`, hash the exact bytes, and
  store its resolved Talker/subtalker pair in the generated fixture.
- Compute effective Talker sampling as:
  `generation_config.do_sample && synthesis_options.temperature > 0`.
  Do not alter the subtalker flag or options.
- Route through the explicit-flag generation path whenever either branch may
  sample, retaining one shared `Sampler`.
- Add production-facing tests for all four mode cross-products and exact shared
  Philox consumption/order.
- Regenerate the matrix, update `config/fixtures.json`, and rerun every command
  in `assignment.md`.

## Execution blocker

The required GPT-5.3 Codex Spark implementation model reports its account-wide
usage limit is exhausted until `2026-07-30 21:24`. The original implementer, a
fresh Spark recovery agent, and another existing Spark agent all returned the
same service error. No non-Spark implementation edits were made after this
review.
