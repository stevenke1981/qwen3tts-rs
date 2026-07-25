# Current Implementation Audit

Audit date: 2026-07-25
Target baseline: `b08178964504d5a214565ffc4ff5ed592eb8f7ec`
Method: CBM graph inspection plus direct source, Cargo, test and workflow verification.

## Module Tree

```text
src/
  bin/qwen3tts-gui.rs
  codec/{activation,causal_conv,codebook,decoder_blocks,flow_matching,transformer}.rs
  talker/{config,model,talker,talker_attention,decoder_layer,code_predictor,
          input_builder,primitives,sampling,weight_loader}.rs
  text_frontend/{candle_backend,python_bridge,model_catalog,speaker_presets,
                 token_parser,voice_clone,voice_clone_speaker,
                 voice_clone_tokenizer}.rs
  tokenizer/mod.rs
  quantization/mod.rs
  decoder_12hz.rs
  decoder_25hz.rs
  vocoder/mod.rs
  gui.rs
  lib.rs
  paths.rs
  weights.rs
```

The repository also has integration/debug tests, three Criterion benches, conversion/reference
Python tools, and one GitHub Actions workflow.

## Implemented or Substantially Present

- Pure Rust/Candle library with CPU default and Cargo feature switches for CUDA and Metal.
- 12 Hz offline decoder components: RVQ/codebook lookup, pre-transformer, causal convolution,
  upsample/decoder blocks and vocoder path.
- Causal convolution ring-buffer primitives and step-level tests.
- Native Talker and Code Predictor model math with per-layer KV-cache parameters.
- Code Predictor uses frame-local KV caches across its sub-codebook steps.
- Native and Python-bridge text frontends.
- Native speaker encoder and tokenizer-encoder components used by voice-clone planning.
- Model catalog and hard-coded speaker presets for Base, CustomVoice, VoiceDesign and clone mode
  validation.
- Q8/Q4 safetensors conversion utilities and synthetic round-trip/cosine tests.
- egui GUI and synthesis examples.
- Criterion benches for causal convolution, codebook and 12 Hz decoder.

## Partial, Stubbed or Missing

### Metadata, Prompt and Special Tokens

- `src/text_frontend/model_catalog.rs` detects model family from model ID/path and holds a static
  model table. This is not metadata-driven parity.
- `src/text_frontend/speaker_presets.rs` hard-codes speaker names/instructions.
- Prompt construction exists in `src/text_frontend/candle_backend.rs` and
  `src/talker/input_builder.rs`, but exact official IDs and all model variants do not yet have a
  fail-closed parity gate.
- `src/tokenizer/mod.rs` still contains TODOs for loading `tokenizer.json`, encode and decode.

### M-RoPE

- Talker has multimodal rotary helpers and single-position generation.
- Exact official/reference multimodal position mapping and interleaving are not gated by known
  reference vectors. This remains P01 work.

### Sampling and RNG

- `src/talker/sampling.rs` implements temperature, top-k, top-p and multinomial selection.
- RNG state is a custom SplitMix64-style stream (`next_f64_state`), not Philox.
- Repetition penalty is absent from the sampling chain.
- Talker and Code Predictor receive the same `SamplingOptions` and same `Sampler`; no independent
  settings exist.
- Current order applies top-k selection before temperature probability calculation; exact
  Hugging Face/qwentts.cpp operation ordering is not yet established by parity tests.

### Talker and Code Predictor KV Cache

- Talker allocates a per-generation layer cache and reuses it during autoregressive Talker steps.
- Code Predictor creates a fresh frame-local cache for every Talker frame and reuses it for
  codebooks 1–15, matching the intended ownership shape but not yet numerically gated.
- Cache isolation exists only as local variables in the single-request generation call; there is
  no public multi-session runtime/session abstraction.

### Decoder Offline and Streaming

- Offline `Decoder12Hz::decode_frames` performs full-sequence decode.
- `decode_chunk_inner` is explicitly fake streaming: it appends every pre-convolution output to
  `pre_conv_buffer`, rebuilds an accumulated tensor, reruns pre-transformer, all upsample blocks,
  decoder blocks and final convolution, then slices the new PCM tail.
- Complexity is O(n²) over a long stream and state memory grows with history.
- Persistent transposed-convolution overlap and decoder transformer KV-ring state do not exist.
- `reset_state` clears accumulated buffers but is not evidence of true stateful codec streaming.
- `decoder_25hz.rs` remains a Phase 2 TODO and is outside the current 12 Hz parity target.

### End-to-End Streaming

- `CandleLLM::synthesize` generates a complete codes tensor, copies all frames into nested vectors,
  and only then returns a `TokenStream`.
- There is no Talker frame event, codec-session callback, backpressure or per-frame cancellation
  connection from Talker to PCM.
- GUI synthesis therefore cannot provide true Talker → Codec → PCM streaming.

### Quantization

- Packed Q8/Q4 files can be produced.
- `src/weights.rs::insert_safetensors_tensors` calls `load_quantized_f32`, builds a full Candle F32
  tensor and stores that tensor as the runtime weight.
- There is no packed resident Talker weight type or direct quantized matmul/embedding kernel.
- Current behavior is file compression followed by full dequantization, not native quantized
  runtime.

### Voice Features

- Voice clone planning, speaker embedding and native speech-tokenizer paths are present.
- CustomVoice behavior uses hard-coded presets rather than model metadata.
- VoiceDesign is represented in capability validation/instruction input, but exact official
  semantic and numerical parity is unverified.
- ICL priming is built as an input condition for batch generation; reusable streaming codec/Talker
  state snapshots are absent.

### Product Surfaces

- CLI: only `qwen3tts-gui` is declared as a binary. `qwen-tts` and `qwen-codec` do not exist.
- Server: no OpenAI-compatible HTTP server module/binary.
- Voice Registry: absent.
- C ABI: no `cdylib`, opaque handle API or C lifecycle tests.
- Continuous batching: absent.
- Multi-session isolation: no session scheduler/runtime; single-call local state only.

### Backends

- Cargo exposes `cpu`, `cuda` and `metal` features.
- This Windows baseline verified CPU only.
- CUDA hardware is available (RTX 3070 Ti, compute capability 8.6, CUDA 13.2), but CUDA build and
  model parity were not run in P00-T01.
- Metal cannot be run on this Windows host.
- Vulkan/ROCm runtime support is absent.

### CI and Real-Weight Tests

- `.github/workflows/ci.yml` contains `continue-on-error: true` for integration tests and benches.
- Numerous integration/debug tests print “Skipping” and return when weights/fixtures are missing.
- `cargo test --lib` passing is useful CPU unit evidence but is not real-weight parity evidence.
- The installed static checker correctly reports accumulated-history streaming, full
  dequantization, permissive CI and silent fixture skips.

## Hard-Coded and High-Risk Hotspots

- `src/text_frontend/model_catalog.rs`: model-family inference and static capability table.
- `src/text_frontend/speaker_presets.rs`: static speaker/dialect instructions.
- `src/talker/sampling.rs`: non-Philox RNG and incomplete sampling semantics.
- `src/talker/talker.rs`: one shared sampling configuration and full-result generation API.
- `src/decoder_12hz.rs`: accumulated-history fake streaming.
- `src/weights.rs` and `src/quantization/mod.rs`: packed-file-to-F32 runtime expansion.
- `tests/integration_test.rs`, `tests/talker_alignment_test.rs` and related real-weight tests:
  silent fixture skips.
- `.github/workflows/ci.yml`: permissive `continue-on-error`.

## First Ordered Gaps

1. Establish fail-closed fixture provenance and resolver (`P00-T02`).
2. Add trusted stage-dump schema/hooks and adapters (`P00-T03`/`P00-T04`).
3. Create a CPU F32 real-weight baseline corpus (`P00-T05`).
4. Replace metadata/name hard-coding and lock exact prompt/M-RoPE behavior (`P01`).
5. Implement Philox, repetition penalty and separate sampling controls (`P02`).

No P01+ implementation should begin before the P00 phase gate passes.
