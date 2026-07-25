# GPT-5.3 Codex Spark — Implementation Assignment

## Task

- Task ID: `P01-T02`
- Goal: load every prompt-affecting special token, language, speaker, and
  dialect table from official checkpoint metadata and install it into the
  native Talker runtime.
- Dependency: `P01-T01 GATE_PASSED`.

## Allowed files

- `src/text_frontend/model_catalog.rs`
- `src/talker/config.rs`
- `src/text_frontend/candle_backend.rs`
- `src/talker/input_builder.rs`
- `tests/model_runtime_metadata_real_test.rs` (new)
- `docs/alignment/runtime-token-metadata.md` (new)

No other files may be modified. Do not change RoPE math, prompt templates,
sampling, weights, codec decoding, or static speaker prose in this task.

## Required metadata contract

Extend the P01-T01 parser to require and own:

- top-level `assistant_token_id`, `im_start_token_id`, `im_end_token_id`,
  `tts_bos_token_id`, `tts_eos_token_id`, and `tts_pad_token_id`;
- Talker `codec_bos_id`, `codec_eos_token_id`, `codec_think_id`,
  `codec_nothink_id`, `codec_think_bos_id`, `codec_think_eos_id`, and
  `codec_pad_id`;
- the complete `codec_language_id`, `spk_id`, and `spk_is_dialect` maps.

`spk_is_dialect` is heterogeneous in official JSON: a value is either
`false` or a dialect language key such as `sichuan_dialect`. Do not model it
as `bool` only. Reject `true`, non-string/non-boolean values, duplicate
case-folded keys, empty keys, zero/invalid token IDs, speaker IDs outside the
Talker codec vocabulary, and dialect targets missing from
`codec_language_id`.

## Runtime installation and behavior

- The Candle model constructor must parse the same real `config.json` and
  install all metadata-derived tokens/maps into `TalkerConfig`; the default
  hard-coded table must not drive a loaded model.
- Keep tensor-shape inference for architecture dimensions only. Metadata is
  authoritative for all prompt-affecting IDs and maps.
- Explicit language lookup is case-insensitive and fail-closed. An unknown
  non-`auto` language must return `Error::Config`, never silently become Auto.
- Explicit speaker lookup is case-insensitive and fail-closed.
- Match official dialect behavior: for `language` equal to Chinese or Auto,
  a speaker mapped to a dialect substitutes that dialect's language token.
- Provide metadata-backed supported-language and supported-speaker accessors.
- Preserve deterministic map ordering for tests and user-facing output.

## Tests

- Synthetic table tests cover Base and CustomVoice metadata, including all
  special tokens, ten base languages, nine official speakers, and the two
  dialect mappings (`eric -> sichuan_dialect`, `dylan -> beijing_dialect`).
- Tests prove renamed/misleading directories do not affect tables.
- Missing fields, malformed dialect values, unknown dialect targets,
  duplicate case-folded keys, out-of-range IDs, unknown language, and unknown
  speaker fail closed.
- Input-builder tests assert Chinese/Auto dialect substitution and that
  English does not substitute a dialect.
- The ignored real-model test must require
  `QWEN3_TTS_REAL_MODEL_DIR`, run exactly one test when explicitly invoked,
  and assert the pinned 0.6B Base token/table values. It must not silently
  return when the variable is absent.

## Required commands

```text
cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo test --lib text_frontend::model_catalog
cargo test --lib talker::config
cargo test --lib talker::input_builder
cargo test --test model_runtime_metadata_real_test -- --ignored --nocapture
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
git diff --check -- src/text_frontend/model_catalog.rs src/talker/config.rs src/text_frontend/candle_backend.rs src/talker/input_builder.rs tests/model_runtime_metadata_real_test.rs docs/alignment/runtime-token-metadata.md
```

For the real test, use:

```text
QWEN3_TTS_REAL_MODEL_DIR=C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc
```

## Acceptance

- No loaded native model uses hard-coded prompt token or speaker/language
  tables.
- All required official metadata is represented without type loss.
- Unknown and malformed lookups fail closed.
- Dialect substitution matches the official implementation.
- Synthetic and real tests pass without silent skip.
- Diff stays inside the six allowed files.
