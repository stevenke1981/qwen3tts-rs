# Prompt-ID Matrix (P01-T05)

## Scope

This artifact gates prompt assembly parity for the five Qwen3-TTS 12Hz variants:

- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`

The generator uses the official Python processor in
`C:\Users\steven\Qwen3-TTS\.venv\Scripts\python.exe` and writes/validates
`fixtures/alignment/p01_prompt_id_matrix.json` only with metadata and token IDs
(no model weights are downloaded).

## Generator behavior

- Reads immutable revisions from Hugging Face for each model repository.
- Downloads only `config.json` when absent in local cache.
- Loads the `Qwen2` tokenizer from local snapshot of
  `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (via
  `transformers.AutoTokenizer.from_pretrained(..., local_files_only=True, fix_mistral_regex=True)`)
  and wraps it with `Qwen3TTSProcessor`.
- Builds fixed fixtures:
  - main assistant prompt
  - reference assistant prompt
  - reference body slice `prompt_ids[3..len()-2]`
  - instruction prompt
- Verifies tokenizer identity across all five repos by remote file metadata (blob ids
  / LFS oid for canonical tokenizer artifacts).

## Matrix case profile enforced by Rust test

- `0.6B Base`: `x_vector_only`, `icl` (no instruction, no speaker preset)
- `1.7B Base`: `x_vector_only`, `icl` (no instruction, no speaker preset)
- `0.6B CustomVoice`: `custom_voice` (no instruction, `expected_speaker` present)
- `1.7B CustomVoice`: `custom_voice` + instruction (and `expected_speaker` present)
- `1.7B VoiceDesign`: `voice_design` + instruction (no speaker preset)

## Immutable artifacts recorded in fixture

- `model_revision` for each variant
- `config_sha256` for each variant
- `tokenizer.model_revision`
- `tokenizer.config_sha256`
- `tokenizer.tokenizer_sha256`
- `tokenizer.tokenizer_load` (loader and source path policy)
- canonical tokenizer file signature map for shared identity

## Required gate commands

```powershell
$env:PATH='C:\Users\steven\.cargo\bin;' + $env:PATH
$env:QWEN3_TTS_REAL_MODEL_DIR='C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc'

C:\Users\steven\Qwen3-TTS\.venv\Scripts\python.exe tools/generate_prompt_id_matrix.py --help
C:\Users\steven\Qwen3-TTS\.venv\Scripts\python.exe tools/generate_prompt_id_matrix.py --check fixtures/alignment/p01_prompt_id_matrix.json

cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo test --test prompt_id_matrix_real_test -- --ignored --nocapture
cargo test --lib text_frontend
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
```

## Oracle provenance

- Fixture path: `fixtures/alignment/p01_prompt_id_matrix.json`
- Generator: `tools/generate_prompt_id_matrix.py`
- `fixture_id`: `p01-prompt-id-matrix`
- Tokenizer source repo: `Qwen/Qwen3-TTS-Tokenizer-12Hz`
- `generated_by`: `tools/generate_prompt_id_matrix.py official-python`
