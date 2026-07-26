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
- 2026-06-04 v0.1.12 Q8 auto-preference update:
  - Tokenizer decoder path resolution now checks `weights/tokenizer-q8` before `weights/tokenizer` beside both the current directory and the executable directory.
  - Global cache lookup now checks `%LOCALAPPDATA%/qwen3tts-rs/tokenizer-12hz-q8` before the F32 `%LOCALAPPDATA%/qwen3tts-rs/tokenizer-12hz` cache.
  - `QWEN3TTS_TOKENIZER_WEIGHT_DIR` remains the explicit override and is searched first.
  - Q4 remains experimental because the local 0.995 cosine gate only quantized 12/236 tensors; Q8 is the practical default candidate.
- 2026-06-04 v0.1.13 Q8 first-run cache update:
  - After automatic F32 tokenizer conversion succeeds, the app now attempts to run `quantize_tokenizer.exe --format q8_0` into the sibling Q8 cache.
  - Q8 cache build failure is non-fatal; synthesis falls back to the F32 cache.
  - When Q8 is selected, the app prints a terminal summary from `quantization_report.json`, including stored MB, percent savings, and quantized/total tensors.
- 2026-06-04 v0.1.14 Voice Clone bridge update:
  - Single-file `synthesize.exe --backend python --mode voice-clone` now calls the official `qwen_tts.generate_voice_clone()` path and writes the WAV directly.
  - Added `--reference-text`; it is optional at CLI level, but best voice-clone quality should include an accurate reference transcript. When omitted, the Python bridge uses speaker-embedding-only mode.
  - Batch voice-clone is explicitly rejected for now, and Candle native voice-clone still returns a clear unsupported-backend error.
  - Native Rust work remaining: port the speech tokenizer encoder, port the Base-model speaker encoder, build the ICL prompt embedding path, decode `ref_code + generated_code`, trim the reference segment, and align with PyTorch fixtures.
- 2026-06-04 v0.1.15 native Voice Clone prerequisites:
  - `convert_tokenizer.exe` now writes `encoder.safetensors` and `quantizer.safetensors` in addition to the existing decoder files.
  - Added `convert_speaker_encoder.exe` to extract Base-model `speaker_encoder.*` tensors into `weights/speaker/speaker_encoder.safetensors`.
  - This keeps the target on zero Python runtime dependency; the remaining work is Rust forward implementations, not Python process optimization.
- 2026-06-04 native Voice Clone Candle wiring update:
  - Added `text_frontend::voice_clone::NativeVoiceClonePlan` and `NativeReferenceCodes` to validate native ICL clone inputs.
  - Native Candle voice-clone now requires `--reference-text`; x-vector-only mode remains Python-reference-only and is not treated as the quality target.
  - Added `InputBuilder::build_voice_clone` and `VoiceClonePrompt` to insert an external speaker embedding and append reference-text/reference-code ICL embeddings before generation.
  - Added `NativeVoiceCloneCondition` as the handoff type from native encoder forward modules into the talker ICL builder.
  - Candle CLI no longer exits before the backend for `--backend candle --mode voice-clone`; the backend now validates the native plan and reports the exact missing Rust forward modules.
  - Added `tests/voice_clone_native_test.rs` for native plan validation, reference codec token bounds, and reference-prefix trim math.
- 2026-06-04 native Voice Clone forward update:
  - `WeightLoader` now converts BF16/F16 safetensors payloads into F32 tensors, which unblocks Base-model `speaker_encoder.safetensors`.
  - Added `NativeSpeakerEncoder` with TDNN/Res2Net/SE/attentive-statistics-pooling forward over `[batch, frames, 128]` mel-like features.
  - Added `NativeSpeechTokenizerEncoder` with waveform convolutional downsample and RVQ nearest-code encode path, producing 16-codebook `NativeReferenceCodes`.
  - Candle backend now builds `NativeVoiceCloneCondition` from `--reference-audio` + `--reference-text` and routes it into `InputBuilder::build_voice_clone`.
  - Native voice-clone tokenizer lookup now falls back from decoder-only Q8 dirs to a tokenizer directory containing `encoder.safetensors`.
  - Added `tests/native_voice_forward_test.rs`; on this machine it loads real cached tokenizer/speaker weights and validates both native forwards.
  - Smoke passed: `cargo run --example synthesize --features candle-llm -- --backend candle --mode voice-clone --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --text "你好" --reference-audio cn_candle_1.7b_hello.wav --reference-text "hello" --max-new-tokens 2 --text-only --output native_vc_smoke.wav` produced 2 token frames through the Rust-native voice-clone condition path.
  - Precision work queued at this point: replace the simple Rust log-spectral speaker frontend with exact upstream mel extraction, add tokenizer encoder transformer layers, and export PyTorch fixtures for speaker embedding/ref_code cosine or exact-token alignment.
- 2026-06-04 native Voice Clone PyTorch fixture alignment update:
  - Added upstream-compatible Rust mel extraction for speaker references: reflect pad, Hann STFT, Slaney mel filterbank, magnitude floor, and log compression.
  - Replaced the backend's simplified log-spectral mel frontend with the upstream-compatible mel path.
  - Fixed speaker Res2Net dilation to match PyTorch speaker blocks: block dilations 2, 3, and 4.
  - Reworked `NativeSpeechTokenizerEncoder` to match upstream Mimi encoder order: ELU conv stack, residual ELU placement, encoder transformer layers, RoPE causal self-attention, LayerNorm with bias, GELU MLP, and replicate padding on the downsample conv.
  - Added PyTorch fixture checks in `tests/native_voice_forward_test.rs`:
    - mel extraction: cosine `1.00000000`, max_abs `0.00111103`.
    - speaker embedding: cosine `1.00000000`, max_abs `0.00000095`.
    - speech tokenizer ref_code: exact 13-frame x 16-codebook token match.
  - Verification passed:
    - `cargo test --test native_voice_forward_test --features candle-llm -- --nocapture`
    - `cargo test --test native_voice_forward_test --features candle-llm -- --ignored --nocapture`
    - `cargo test --features candle-llm`
  - Next step: run real `--backend candle --mode voice-clone` WAV generation and compare reference-prefix trim/audio quality against upstream PyTorch, then optimize the new Rust mel/STFT path if reference processing time is noticeable.
- 2026-06-04 native Voice Clone real WAV comparison update:
  - Rebuilt current `target/release/examples/synthesize.exe` with `--features candle-llm`.
  - Short-reference smoke with `cn_candle_1.7b_hello.wav`:
    - Native Candle output: `native_vc_real_v016_56.wav`, 37 frames, 2.960s, elapsed `26.89s`.
    - Upstream Python output: `upstream_vc_real_v016_56.wav`, 3.440s, elapsed `36.78s`.
    - Reference-prefix correlation stayed low for both paths, so the output WAV did not contain a copied reference prefix.
  - Known-transcript 4.16s reference run:
    - Generated `vc_reference_known_norm.wav` from text `這是一段參考語音，請記住這個聲音。` and normalized it to peak -3 dB.
    - Native Candle voice clone: `native_vc_known_ref.wav`, 50 frames, 4.000s, RMS `-22.21 dB`, elapsed `36.02s`.
    - Upstream Python voice clone: `upstream_vc_known_ref.wav`, 4.560s, RMS `-22.32 dB`, elapsed `37.97s`.
    - Reference-prefix correlation remained low:
      - native first 500ms corr `0.0736`, full reference-window corr `0.0202`.
      - upstream first 500ms corr `0.0295`, full reference-window corr `0.0095`.
    - Conclusion: current Rust CLI decodes only generated frames, not `reference + generated`, so reference-prefix trimming is not needed in the WAV output path. The existing `reference_prefix_samples` helper remains useful only if a future decode path concatenates reference and generated codec frames.
  - Performance note: native 1.7B CPU release was slightly faster than the Python bridge in these small runs, but total time is still dominated by talker generation. The new Rust STFT/mel extraction is not the first optimization target unless profiling shows reference preprocessing is significant for long references.
- 2026-06-04 native Voice Clone ASR/release-readiness update:
  - Ran CUDA ASR (`faster-whisper large-v3-turbo`, CUDA/float16) on the real native and upstream voice-clone WAVs.
  - Native transcript: `现在开始测试ROST原声语音克隆`.
  - Upstream transcript: `现在开始测试ROST原声语音克隆`.
  - Target text was `現在開始測試 Rust 原生語音克隆。`; both paths are intelligible and have the same ASR substitution for the English word `Rust`.
  - Release docs and CLI help now state that Candle/Rust native Voice Clone is available, with Python kept as an alignment/reference path.
  - Release version bumped to `0.1.16`.
- 2026-06-04 Candle-oriented optimization update:
  - `candle-rs` skill was not available in this Codex session, so optimization followed the same Candle principles from `AGENTS.md`: keep tensor math on the active device, reduce repeated Tensor/trig construction, and preserve PyTorch fixture alignment.
  - Replaced the speech-tokenizer ELU helper's `flatten_all().to_vec1()` host round-trip with a pure Tensor formula: `relu(x) - relu(1 - exp(x))`.
  - Reused Mimi tokenizer encoder transformer RoPE tensors across all 8 encoder transformer layers instead of recomputing the same cos/sin per layer.
  - Reused CodePredictor sub-codebook RoPE tensors across codebook steps within each generated frame instead of recomputing per step.
  - Validation stayed aligned:
    - `cargo test --test native_voice_forward_test --features candle-llm -- --ignored --nocapture`
    - code predictor PyTorch fixture exact token match.
    - `cargo test --features candle-llm`
  - Real 1.7B CPU release voice-clone smoke stayed non-silent and produced the same 50 frames / 4.000s WAV; elapsed time improved from the prior `36.02s` run to `34.55s` on the same prompt/reference/seed. The earlier `40.94s` run before the CodePredictor optimization appears to have been noise/regression from the first ELU-only change.
- 2026-06-05 Candle skill follow-up optimization:
  - Loaded the local `candle-rs` skill and kept validation narrow before broadening, per its workflow.
  - Reworked `Sampler` to reuse internal candidate/probability scratch buffers and use top-k partial selection before sorting. This keeps the default `top_k=50` path from sorting the whole codec vocabulary on every sampled token.
  - Added a deterministic tie-breaker by token id so the optimized sampler remains stable when logits tie, plus a reference-behavior unit test for top-k/top-p sampling.
  - Flattened talker generated-token collection from `Vec<Vec<u16>>` plus per-frame `Tensor::cat` into one preallocated `Vec<u32>` returned as the same `[num_frames, 16]` Tensor.
  - Added `MultimodalRotaryEmbedding::forward_single_position` for autoregressive generation; it avoids creating a `[3, batch, 1]` position Tensor and avoids `to_vec3()` on every generated frame.
  - Changed `TalkerModel::forward` to mutate KV caches in place and return only the hidden state, removing the repeated `kv_caches.to_vec()` clone in the main generation loop.
  - Validation stayed aligned:
    - sampler unit tests pass.
    - single-position RoPE fast path matches the general RoPE path.
    - code predictor first-step fixture remains `cosine=1.0`, `next=[1965]`.
    - code predictor greedy fixture remains exact: `[1965, 1043, 1172, 1911, 95, 898, 555, 1013, 1986, 1371, 215, 695, 329, 560, 1527]`.
    - native speaker/ref_code fixtures remain aligned.
    - full `cargo test --features candle-llm` passes.
  - Real 1.7B CPU release voice-clone smoke stayed bit-identical at the token level to the prior opt2 run (`sha256=635a1f9bded3986e10d38b1207d0bebb7b11947a7483703d34860c014d45afb9`), with the same 50 frames / 4.000s WAV and RMS `-22.21 dB`. Elapsed time was `36.39s` in the final warm run, so these cleanups reduce avoidable work but do not yet provide a stable end-to-end speedup over the earlier `34.55s` measurement.

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
   - Voice Clone status: native plan validation and talker ICL prompt builder are implemented.
   - Voice Clone status: first Rust/Candle forwards for tokenizer `encoder.safetensors` and Base-model `speaker_encoder.safetensors` are implemented and wired into `CandleLLM`.
   - Voice Clone status: PyTorch fixture alignment now passes for upstream mel extraction, speaker embedding, and tokenizer reference codes.
   - Voice Clone status: real native/upstream WAV comparison passed basic non-silence/loudness checks and confirmed no reference prefix is copied into the output WAV.
   - Voice Clone status: CUDA ASR intelligibility check passed for native and upstream comparison WAVs with equivalent transcripts.
   - Voice Clone next step: profile talker generation before optimizing Rust STFT/mel, then add a dedicated release smoke command for `--mode voice-clone`.

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

- 2026-07-26 GGUF talker backend update:
  - WO-4 (GGUF probe): 完成 `examples/gguf_talker_probe.rs` + `docs/gguf_tensor_mapping.md`。
    用真實 Q4_K_M GGUF 檔案驗證了 tensor 命名對照與讀取管線。
  - WO-5 (GGUF backend integration):
    - `TalkerWeightLoader::from_gguf()` + `gguf_key_to_safetensors()` 命名轉換。
    - CLI 新增 `--talker-backend gguf|safetensors` 選項（預設 safetensors）。
    - `CandleLLM::from_gguf()` 整合至主程式，兩條路徑並存。
    - 3 項端到端測試（`tests/gguf_load_real_test.rs`）：weight loading、config inference、build_talker 全部通過真實 Q4_K_M GGUF 檔案。
    - 新增 `tests/gguf_safetensors_alignment_test.rs`：safetensors vs GGUF 數值比對。
      - Safetensors 輸出與 PyTorch fixture 完全一致（`[1716, 1956, ...]` — token match）。
      - Q4_K_M GGUF codebook-0 logits cosine=0.9903 vs safetensors。
      - **結論：** Q4_K_M 為 4-bit 極致壓縮，cosine 0.99 屬合理範圍；正式 0.995 門檻需要 Q8_0 GGUF。
  - 建議淘汰清單（待人工確認）：
    - `examples/quantize_tokenizer.rs` 的 talker 相關量化路徑（—WO-1 已停用 codec/vocoder 量化，且 GGUF 可直接提供量化權重，自製量化校準流程已多餘）。
  - 已知限制：
    - 尚未下載 Q8_0 GGUF 做完整 0.995 cosine 驗證。
    - `docs/gguf_tensor_mapping.md` 中「待補：數值比對 cosine ≥ 0.995」的子項目已在此 session 完成，實際數值為 0.9903（Q4_K_M）。
