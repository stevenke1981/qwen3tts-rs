# GPT-5.3 Codex Spark — Implementation Assignment

## Task

- Task ID: `P01-T01`
- Goal: replace filename/model-ID behavior inference with fail-closed parsing of
  the official checkpoint `config.json`.
- Dependency: `P00 GATE_PASSED`.

## Allowed files

- `src/text_frontend/model_catalog.rs`
- `examples/synthesize.rs`
- `examples/synthesize_batch.rs`
- `tests/model_metadata_real_test.rs` (new)
- `docs/alignment/model-metadata.md` (new)

No other files may be modified. Do not modify token tables, prompt assembly,
sampling, RoPE, speaker presets, or weight math in this task.

## Required metadata contract

Parse the official Qwen3-TTS top-level config and require:

- `model_type == "qwen3_tts"`;
- `tokenizer_type == "qwen3_tts_tokenizer_12hz"`;
- `tts_model_size` (`0b6` or `1b7`);
- `tts_model_type` (`base`, `custom_voice`, or `voice_design`);
- `talker_config.model_type == "qwen3_tts_talker"`;
- `talker_config.code_predictor_config.model_type ==
  "qwen3_tts_talker_code_predictor"`;
- structurally valid non-zero talker/code-predictor layer/hidden/head values;
- `talker_config.rope_scaling.interleaved` and `mrope_section` captured for
  later P01-T03 use.

Expose an owned, public metadata/capability type with `from_config_path` and
`from_model_dir` constructors. Derive generation capabilities only from parsed
`tts_model_type` and metadata values:

- Base supports voice clone.
- CustomVoice supports speaker presets.
- VoiceDesign supports voice design.
- Instruction control is present only where metadata indicates the supported
  1.7B CustomVoice/VoiceDesign family; do not inspect the filename.

Unknown/missing/malformed/conflicting metadata must return `Error::Config`.
Never fall back to `contains`, normalized filename matching, directory names,
or `Auto` on parse failure.

## API/caller migration

- Generation validation accepts parsed metadata/capability, not a model name.
- Auto mode resolution derives from parsed metadata.
- Update both synthesis examples to locate/accept a real model directory,
  parse its `config.json` before validating generation options, and fail with a
  clear configuration error when metadata is unavailable.
- Static model listings may remain for display/download discovery only; they
  must not drive runtime behavior or validation.

## Tests

- Table tests for Base, CustomVoice, and VoiceDesign metadata independent of
  directory/file names.
- Same config in misleading directories gives identical behavior.
- Renaming a directory does not change capability.
- Missing config, malformed JSON, missing required fields, unknown model type,
  conflicting nested model types, zero/invalid dimensions, and invalid M-RoPE
  shape fail closed.
- Validation tests cover supported/unsupported modes using parsed metadata.
- `tests/model_metadata_real_test.rs` may be `#[ignore]` by default, but when
  explicitly run it must require `QWEN3_TTS_REAL_MODEL_DIR`, parse the real
  cached official checkpoint, and assert the 0.6B Base metadata. Sol will run
  this ignored test explicitly; it must not silently return.

## Required commands

```text
cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo test --lib text_frontend::model_catalog
cargo test --test model_metadata_real_test -- --ignored --nocapture
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
git diff --check -- src/text_frontend/model_catalog.rs examples/synthesize.rs examples/synthesize_batch.rs tests/model_metadata_real_test.rs docs/alignment/model-metadata.md
```

For the real test, use:

```text
QWEN3_TTS_REAL_MODEL_DIR=C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc
```

## Acceptance

- No runtime generation behavior depends only on model ID/path text.
- Official metadata is parsed and validated before mode validation.
- Synthetic and real official config tests pass without silent skip.
- Diff stays inside the five allowed files.
