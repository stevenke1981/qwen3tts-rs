# Talker Native Port TODO

## Current status

- `cargo check` passes for the newly added `src/talker` module.
- Existing codec decoder tests still pass.
- The talker path is not numerically aligned yet.
- 2026-06-03 update:
  - Added rank-3 `linear` helpers for PyTorch-style projection semantics.
  - Main talker generation now forwards each generated frame through `TalkerModel`.
  - Talker and code predictor attention now return K/V caches.
  - Code predictor now uses `talker_hidden + codebook_0_embed` as prefill and returns only codebooks 1-15.
  - Added a synthetic code predictor test that checks sub-codebook output shape and cache growth.
- 2026-06-03 native 1.7B update:
  - 1.7B `TalkerConfig` is now inferred from safetensors shapes.
  - Optional `talker.code_predictor.small_to_mtp_projection.{weight,bias}` is loaded for 2048 -> 1024 code predictor projection.
  - Practical synthesis uses deterministic temperature/top-k/top-p sampling when `temperature > 0`.
  - Main codec head sampling suppresses control tokens `>= 2048` while allowing EOS.
  - `cn_candle_1.7b_short.wav` was true silence: duration `1.280s`, RMS `0.0`, peak `0`.
  - Fixed sampled output `cn_candle_1.7b_short_sampled_final.wav`: duration `1.280s`, RMS `1837.7`, peak `7067`.
  - Added `examples/synthesize_batch.rs` to load the 1.7B Candle model and 12Hz decoder once, then synthesize multiple lines.
  - Batch smoke with two Chinese lines:
    - load once: `5.64s`
    - line 1: synth `7.31s`, decode `1.09s`, RMS `1837.7`, peak `7067`
    - line 2: synth `7.29s`, decode `1.05s`, RMS `1067.3`, peak `5620`
    - total wall time: `23.65s`
- 2026-06-03 VoiceDesign/native CLI update:
  - Added `SynthesisOptions::instruct` and CLI `--instruct` for single and batch synthesis.
  - Candle native path now tokenizes instruct as a separate `<|im_start|>user\n...<|im_end|>\n` prompt and prepends its text embedding before the normal TTS prompt, matching the upstream VoiceDesign/CustomVoice contract.
  - Python bridge fallback now forwards `--instruct` to `tools/generate_tokens.py`.
  - Single and batch CLIs now fallback to Base model `tokenizer.json` from local HuggingFace cache when a VoiceDesign snapshot does not include one.
  - Single CLI warns when generated frame count reaches a too-low `--max-new-tokens` for long Chinese text.
  - Release guides now document Base voice-control limits, `--speaker` limitations, `--instruct`, batch `.tokens`, and the Chinese length heuristic.
- 2026-06-03 v0.1.6 CLI usability update:
  - Added `--instruct-file` for single and batch synthesis.
  - Added `--seed` to control Candle sampler and Python fallback sampling.
  - Added batch `--no-save-tokens` as an explicit token-output off switch.
  - Batch now warns when Chinese text is likely to need a larger `--max-new-tokens`.
  - Release guides now show 0.6B Base usage for lower hardware requirements.
- 2026-06-03 v0.1.8 batch/cache update:
  - Batch now supports repeated `--instruct-file`, using one file globally or one file per text line.
  - Batch now supports repeated `--seed`, using one seed globally or one seed per text line.
  - Batch can auto-search the local HuggingFace cache for the default 0.6B Base model when `--model-dir` is omitted.
  - Tokenizer decoder auto-conversion now writes to a user-level cache by default, avoiding repeated >1GB conversion for each new release directory.
  - Remaining larger item: true model quantization (Q4/Q8) still requires calibration and numeric validation.
- 2026-06-04 v0.1.9 speaker preset update:
  - Added the 9 official Qwen CustomVoice speaker names: `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ryan`, `Aiden`, `Ono_Anna`, and `Sohee`.
  - `--speaker` is still passed through as a real speaker id for CustomVoice models with `spk_id`.
  - Base/VoiceDesign snapshots with empty speaker maps now translate known speaker names into a natural-language instruct fallback.
  - Added `--list-speakers` to single and batch CLIs.
- 2026-06-04 v0.1.10 quantization update:
  - Added Q8_0 and Q4_0 safetensors quantization payloads: `{tensor}.qweight`, `{tensor}.scales`, `{tensor}.meta`.
  - `WeightLoader` now auto-detects quantized payloads and dequantizes them back to F32 tensors before existing decoder construction.
  - Added `quantize_tokenizer.exe` with per-tensor cosine/max-error report and low-cosine F32 anchor preservation.
  - Local Q8 tokenizer decoder conversion: 99 tensors quantized, 137 preserved, stored/original ratio `0.266`; decoder smoke cosine vs base `0.99974333`.
  - Local Q4 tokenizer decoder with `group-size=32` and `min-cosine=0.995`: 12 tensors quantized, 224 preserved, stored/original ratio `0.947`; decoder smoke cosine vs base `0.99235672`. Treat Q4 as experimental until activation calibration and audio quality validation are stronger.
- 2026-06-04 v0.1.11 model capability update:
  - Added a Rust-native model capability catalog for 1.7B VoiceDesign, 1.7B CustomVoice, 1.7B Base, 0.6B CustomVoice, and 0.6B Base.
  - Added `--list-models`, `--mode`, and `--reference-audio` to single and batch CLIs.
  - `--mode custom-voice` now validates CustomVoice model usage and requires `--speaker`; 0.6B CustomVoice rejects `--instruct`.
  - `--mode voice-design` now requires a 1.7B VoiceDesign model plus `--instruct` or `--instruct-file`.
  - `--mode voice-clone` now validates Base model plus `--reference-audio`, then returns a clear pending-implementation error because native reference-audio conditioning is not implemented yet.
  - Release guides now document the model table, supported languages, streaming status, and Rust CLI equivalents for upstream `generate_custom_voice`, `generate_voice_design`, and `generate_voice_clone`.

## TODO

1. Restore autoregressive talker semantics.
   - Each generated codec frame must be fed back through `TalkerModel`.
   - Main talker KV cache must be updated after prefill and every generation step.
   - Status: implemented structurally; still needs PyTorch numeric fixtures.

2. Restore code predictor conditioning.
   - `talker_hidden` must be the prefill hidden state for code predictor generation.
   - Codebook 0 embedding is appended after `talker_hidden`.
   - Codebooks 1-15 must be generated using code predictor KV cache and step-specific embedding/head pairs.
   - Status: implemented structurally; still needs PyTorch numeric fixtures.

3. Return real KV cache from attention layers.
   - `TalkerAttention` and `StandardAttention` must return concatenated non-repeated K/V tensors.
   - Decoder layers and model forward calls must propagate updated caches.
   - Status: implemented.

4. Validate input builder against PyTorch processor.
   - Replace hard-coded prompt slicing assumptions with tokenizer/template-aware parsing.
   - Add short-input validation to avoid underflow.
   - Match language, speaker, dialect, and no-think/think codec conditioning.

5. Add PyTorch reference fixtures.
   - Text projection output.
   - Talker attention layer 0 output.
   - Talker model prefill hidden state.
   - Code predictor first-step logits.
   - Full generated codec frames for a short Chinese prompt.
   - Text projection fixture exported: `tests/fixtures/talker_text_projection.json`.
   - Rust ignored alignment test added: `tests/talker_alignment_test.rs`.
   - Text projection alignment passed: cosine `1.00000000`, max_abs `0.00000036`.
   - Codec embedding/head fixture exported: `tests/fixtures/talker_codec_embedding_head.json`.
   - Rust codec embedding/head alignment test added.
   - Codec embedding/head alignment passed:
     - embedding cosine `1.00000000`, max_abs `0.00000000`
     - codec head cosine `1.00000000`, max_abs `0.00000238`
   - TalkerAttention layer 0 fixture exported: `tests/fixtures/talker_attention_layer0.json`.
   - Rust TalkerAttention layer 0 alignment test added.
   - Fixed Rust 3D RoPE shape construction and interleaved section mapping.
   - TalkerAttention layer 0 alignment passed: cosine `1.00000000`, max_abs `0.00000358`.
   - Talker decoder layer 0 fixture exported: `tests/fixtures/talker_decoder_layer0.json`.
   - Rust decoder layer 0 alignment test added.
   - Talker decoder layer 0 alignment passed: cosine `1.00000000`, max_abs `0.00000057`.
   - TalkerModel prefill fixture exported: `tests/fixtures/talker_model_prefill.json`.
   - Rust TalkerModel prefill alignment test added.
   - TalkerModel prefill alignment passed: cosine `1.00000000`, max_abs `0.00008392`.
   - Code predictor first-step fixture exported: `tests/fixtures/code_predictor_first_step.json`.
   - Rust code predictor first-step alignment test added.
   - Code predictor first-step alignment passed: cosine `1.00000000`, max_abs `0.00002933`, next token `[1965]`.
   - Full greedy code predictor fixture exported: `tests/fixtures/code_predictor_greedy.json`.
   - PyTorch greedy sub-codebook sequence: `[1965, 1043, 1172, 1911, 95, 898, 555, 1013, 1986, 1371, 215, 695, 329, 560, 1527]`.
   - Rust full greedy code predictor alignment test added.
   - Code predictor greedy alignment passed with exact token match:
     `[1965, 1043, 1172, 1911, 95, 898, 555, 1013, 1986, 1371, 215, 695, 329, 560, 1527]`.
   - Full talker single-frame fixture exported: `tests/fixtures/talker_single_frame.json`.
   - PyTorch single-frame code sequence: `[1716, 1956, 980, 111, 1742, 186, 763, 122, 846, 232, 64, 956, 1741, 1449, 1606, 716]`.
   - Rust full talker single-frame alignment test added.
   - Fixed `compute_position_ids` to cast `attention_mask` to F32 before Candle `cumsum`; I64 cumsum hit unsupported CPU matmul.
   - Fixed Candle scalar arithmetic in `compute_position_ids` to use explicit `broadcast_sub`, `broadcast_mul`, and `broadcast_add`.
   - Fixed extra `squeeze(1)` calls after Candle `max(dim)` because Candle removes the reduced dimension.
   - Fixed `compute_position_ids` output dtype from I64 to U32 to match Rust RoPE lookup.
   - Talker single-frame alignment passed with exact token match:
     `[1716, 1956, 980, 111, 1742, 186, 763, 122, 846, 232, 64, 956, 1741, 1449, 1606, 716]`.
   - Two-frame autoregressive fixture exported: `tests/fixtures/talker_two_frame.json`.
   - PyTorch two-frame sequence:
     - frame 0: `[1716, 1956, 980, 111, 1742, 186, 763, 122, 846, 232, 64, 956, 1741, 1449, 1606, 716]`
     - frame 1: `[1706, 411, 568, 883, 183, 200, 1944, 211, 913, 749, 1046, 631, 1835, 396, 1313, 1138]`
   - Rust two-frame autoregressive alignment test added.
   - Talker two-frame autoregressive alignment passed with exact token match for both frames.
   - Non-ignored `cargo test` passed after alignment fixes:
     - 41 library tests passed.
     - Decoder/debug/integration test binaries passed.
     - 9 full talker alignment tests remain ignored by default because they load the 0.6B model.
   - Prompt input-builder fixture exported: `tests/fixtures/talker_prompt_input_builder.json`.
     - text: `你好`
     - template: `<|im_start|>assistant\n你好<|im_end|>\n<|im_start|>assistant\n`
     - token ids: `[151644, 77091, 198, 108386, 151645, 198, 151644, 77091, 198]`
     - expected `inputs_embeds` shape: `[1, 9, 1024]`
     - expected `trailing_text_hidden` shape: `[1, 1, 1024]`
   - Rust `InputBuilder` prompt alignment test added.
   - Rust `InputBuilder` prompt alignment passed:
     - inputs cosine `1.00000000`, max_abs `0.00000048`
     - trailing cosine `1.00000000`, max_abs `0.00000012`
     - pad cosine `1.00000000`, max_abs `0.00000024`
   - Final non-ignored `cargo test` passed after prompt fixture changes:
     - 41 library tests passed.
     - Decoder/debug/integration test binaries passed.
     - 10 full talker alignment tests remain ignored by default because they load the 0.6B model.
   - `cargo fmt --check` still reports pre-existing unrelated formatting diffs in:
     - `src/codec/mtp.rs`
     - `src/text_frontend/candle_backend.rs`
     - `src/text_frontend/python_bridge.rs`
     - `tests/debug_per_layer_compare.rs`
     The newly touched talker and alignment-test files were formatted.
   - Current step: prepare commit/push or continue to full prompt code generation fixtures.
   - Next step after this passes: export a prompt-driven codec generation fixture using PyTorch tokenizer/template inputs.

6. Add Rust alignment tests.
   - Cosine >= 0.999 for single-layer primitives.
   - Cosine >= 0.999 for prefill hidden state.
   - Exact token match or controlled sampling comparison for greedy code generation.

7. Integrate native talker into text frontend.
   - Load tokenizer and talker safetensors without Python.
   - Keep PythonBridge as a reference/fallback path until native tests pass.
   - Expose backend selection in the CLI.

8. Performance cleanup after correctness.
   - Remove hot-path allocations.
   - Preallocate KV caches.
   - Add latency benchmarks for first token and frame generation.
   - Status: load-once batch runner added; next bottleneck is per-frame talker/code-predictor generation and repeated allocation inside generation.
   - Next step: add timing around talker prefill, main codec head, code predictor sub-codebooks, and codec decode to identify the first hot path for optimization.

9. VoiceDesign follow-up validation.
   - Run a real 1.7B-VoiceDesign sample with `--instruct` on CUDA and compare style controllability against upstream PyTorch.
   - Add a PyTorch fixture for instruct prompt embedding once an official VoiceDesign checkpoint is available locally.
   - Check whether CustomVoice 1.7B requires the same instruct path for tone-only control.
   - Validate each built-in speaker preset against upstream CustomVoice audio once a CustomVoice snapshot is available locally.

10. Remaining release polish.
   - Shared tokenizer cache implemented in v0.1.8.
   - Q8/Q4-hybrid tokenizer quantization tooling implemented in v0.1.10.
   - Decide whether CUDA release zips should exclude Python scripts to reduce package size.
   - Keep batch token export as `--save-tokens-dir`; add `--tokens` batch replay only if a concrete workflow needs it.
   - Remaining quantization work: add activation calibration, objective audio metrics, and true int8/int4 compute kernels. Current v0.1.10 path dequantizes to F32 for compatibility.
