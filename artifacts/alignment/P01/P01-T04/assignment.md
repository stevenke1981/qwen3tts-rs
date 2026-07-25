# GPT-5.3 Codex Spark — P01-T04 Implementation Assignment

## Task

- Task ID: `P01-T04`
- Goal: match official Base, CustomVoice, VoiceDesign, x-vector and ICL prompt assembly.
- Phase contract: `tasks/P01-metadata-and-prompt-parity.md`

## Allowed production files

- `src/text_frontend/candle_backend.rs`
- `src/text_frontend/model_catalog.rs`
- `src/text_frontend/mod.rs`
- `src/text_frontend/prompt_templates.rs`
- `src/text_frontend/voice_clone.rs`
- `src/talker/input_builder.rs`

## Allowed test and documentation files

- `tests/prompt_assembly_test.rs`
- `tests/voice_clone_native_test.rs`
- `docs/alignment/prompt-assembly.md`

No other file may be modified. Stop and report if another file is required.

## Read first

- `AGENTS.md`
- `tasks/P01-metadata-and-prompt-parity.md`
- `prompts/SPARK_IMPLEMENT.md`
- official wrapper oracle:
  `C:\Users\steven\Qwen3-TTS\.venv\Lib\site-packages\qwen_tts\inference\qwen3_tts_model.py`
- official embedding-layout oracle:
  `C:\Users\steven\Qwen3-TTS\.venv\Lib\site-packages\qwen_tts\core\models\modeling_qwen3_tts.py`
- native reference:
  `E:\qwentts.cpp-reference\src\prompt-builder.h`

## Required behavior

1. Use exactly these chat wrappers, encoded with `add_special_tokens=false`:
   - main text: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
   - ICL reference text: `<|im_start|>assistant\n{ref_text}<|im_end|>\n`
   - instruction: `<|im_start|>user\n{instruct}<|im_end|>\n`
2. Reference text content is sliced exactly as official `ref_ids[:, 3:-2]`.
   The current `user` wrapper is a bug and must become `assistant`.
3. Do not invent or append instruction text from hard-coded speaker descriptions.
   CustomVoice speaker names select metadata speaker IDs only; the user instruction
   remains byte-for-byte unchanged apart from treating empty/whitespace as absent.
4. Enforce mode combinations from metadata at the direct Candle library boundary:
   - Base / VoiceClone: requires reference audio; rejects named speaker and instruction.
   - CustomVoice: requires non-empty metadata speaker; rejects reference audio; 0.6B
     rejects instruction, 1.7B accepts optional non-empty instruction.
   - VoiceDesign: requires non-empty instruction; rejects named speaker and reference audio.
   - unknown speaker/language fails closed.
5. Preserve official embedding layout:
   - optional instruction embeddings precede the three assistant-role embeddings;
   - automatic language uses nothink/think_bos/think_eos;
   - explicit language uses think/think_bos/language/think_eos;
   - speaker or x-vector embedding is inserted before codec_pad/codec_bos;
   - standard text body is `input_ids[3:-5]`, with EOS and trailing text exactly as official;
   - ICL uses reference body plus target body plus TTS EOS and reference codec frames.
6. Empty main text, empty required speaker/instruction/reference transcript, undersized
   templated token lists and malformed prompt geometry must return errors, never panic
   through unchecked subtraction/narrow.
7. Keep the public API compatible unless a minimal read-only helper is needed for exact tests.

## Required tests

- Exact wrapper string tests including Unicode and embedded newlines.
- Exact reference role and `[3:-2]` content extraction test.
- Full metadata-backed mode acceptance/rejection matrix for all five variants.
- Regression test proving a CustomVoice speaker does not synthesize an instruction.
- Deterministic synthetic-embedding layout tests for Base/x-vector, CustomVoice and
  VoiceDesign, covering auto and explicit language, speaker insertion, instruction
  prefix, standard body/trailing EOS and ICL geometry.
- Invalid/too-short token inputs return `Err` rather than panic.
- Expected values must be literal fixtures or independently calculated oracle values;
  do not generate expected values by calling the Rust function under test.

## Required commands

```powershell
$env:PATH='C:\Users\steven\.cargo\bin;' + $env:PATH
cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo test --lib text_frontend
cargo test --lib talker::input_builder
cargo test --test prompt_assembly_test -- --nocapture
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
git diff --check -- src/text_frontend/candle_backend.rs src/text_frontend/model_catalog.rs src/text_frontend/mod.rs src/text_frontend/prompt_templates.rs src/text_frontend/voice_clone.rs src/talker/input_builder.rs tests/prompt_assembly_test.rs tests/voice_clone_native_test.rs docs/alignment/prompt-assembly.md
```

## Acceptance

- Exact official wrapper and mode behavior.
- All required commands pass.
- No silent skip, no filename inference, no hard-coded speaker-to-instruction fallback.
- No changes outside the nine allowed production/test/documentation files.

## Final report

Return summary, files changed, design choices, commands, exact results, evidence paths,
and any remaining risk. Do not change task status and do not commit or push.
