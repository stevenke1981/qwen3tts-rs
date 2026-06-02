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
