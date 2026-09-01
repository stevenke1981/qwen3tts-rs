# Project: qwen3tts-rs Systemic Optimization & Refactoring

## Architecture
qwen3tts-rs is a pure Rust / Candle implementation of Qwen3-TTS.
The system consists of:
1. **Text Frontend / LLM (`src/text_frontend/`)**: Text normalization, tokenization via `tokenizers`, prompt assembly, CandleLLM backend.
2. **Talker (`src/talker/`)**: Autoregressive transformer predicting Codebook 0 + 15-codebook predictor (`CodePredictor`), M-RoPE 3D rotary embeddings, Philox RNG sampling.
3. **Codec Decoder (`src/codec/`, `src/decoder_12hz.rs`)**: 12Hz 16-codebook streaming neural audio decoder mapping token frames to 24kHz PCM audio.
   - Codebook embeddings (16 codebooks x 2048 x 512).
   - Pre-Conv (CausalConv1d 512 -> 1024).
   - Pre-Transformer (8 layers, sliding window 72, DeviceKvCache).
   - Upsample blocks (ConvTranspose1d + DW CausalConv: 1 -> 2 -> 4 frames).
   - Decoder Start & Decoder Blocks (ConvTranspose1d + Residual Units with dilated CausalConvs: 4 -> 32 -> 160 -> 640 -> 1920 audio samples).
   - SnakeBeta activation & Final CausalConv1d -> 1920 PCM samples per 12Hz frame.
4. **Weight Loading & Memory System (`src/weights.rs`, `src/talker/weight_loader.rs`)**: Zero-copy `memmap2` memory mapped Safetensors loading, Rayon-accelerated vectorized BF16-to-F32 transcoding.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| F1 | Clippy Gate Zero-Error Pass | Fix `tests/mrope_reference_test.rs:268` `deny(clippy::approx_constant)` false positive and clean warnings | M1 | ORIGINAL_REQUEST §R1 |
| F2 | TokenParser EOS Truncation Alignment | Align `CODEC_EOS_TOKEN_ID` (32767 -> 2150) with `TalkerConfig`, supporting dynamic EOS truncation | M1 | ORIGINAL_REQUEST §R1 |
| F3 | Repetition Penalty Test Contract Alignment | Update code contract test in `repetition_penalty_test.rs` to match P03-T04 on-device argmax | M1 | Survey Finding |
| F4 | Zero-Copy Mmap Weight Loading | Implement `memmap2` in `WeightLoader` and `TalkerWeightLoader` eliminating startup memory spikes | M2 | ORIGINAL_REQUEST §R3 |
| F5 | Vectorized BF16 Transcoding | Implement Rayon-accelerated parallel bit-shift BF16-to-F32 conversion with zero tensor byte copies | M2 | ORIGINAL_REQUEST §R3 |
| F6 | CausalConv1d Device-Resident Streaming State | Implement pure device-resident tensor state in `CausalConv1d::step_tensor`, eliminating all D2H/H2D copies | M3 | ORIGINAL_REQUEST §R2 |
| F7 | ConvTranspose1d Overlap-Add Streaming | Implement pure device Overlap-Add (OLA) state in `CausalTransConvNet::step` | M3 | ORIGINAL_REQUEST §R2 |
| F8 | PreTransformer Device-Resident KvCache | Replace CPU `Vec<f32>` `KvRing` with `DeviceKvCache` eliminating 32 host transfers/frame | M3 | ORIGINAL_REQUEST §R2 |
| F9 | Decoder12Hz Streaming O(1) Pipeline | Eliminate `step_buffer` history accumulation in `decode_chunk`, achieving true $O(1)$ latency | M3 | ORIGINAL_REQUEST §R2 |
| F10 | Talker Frame-by-Frame Streaming Interface | Implement `TalkerForConditionalGeneration::generate_streaming` with `FnMut(&[u16; 16])` callback | M4 | ORIGINAL_REQUEST §R4 |
| F11 | CandleLLM End-to-End PCM Streaming | Implement `CandleLLM::synthesize_streaming` connecting Talker frame callback to Decoder PCM stream | M4 | ORIGINAL_REQUEST §R4 |
| F12 | Tokenizer Module Clean Wrapper | Replace unimplemented stub in `src/tokenizer/mod.rs` with robust `tokenizers::Tokenizer` wrapper | M4 | ORIGINAL_REQUEST §R4 |
| F13 | Activation & Dead Code Cleanup | Remove dead `snake_beta_v2`/`SnakeBeta` in `activation.rs` and clean `convert_gguf.rs` warnings | M4 | ORIGINAL_REQUEST §R4 |
| F14 | Comprehensive Tier 1-4 Test Suite & Final Audit | Run 100% test matrix, benchmark verification (p99 <= 97ms, cosine >= 0.999), and victory audit | M5 | Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| M1 | Compilation Gates & TokenParser EOS Alignment | Fix mrope clippy false positive, repetition penalty test contract, TokenParser EOS 2150 alignment & tests | none | DONE |
| M2 | Zero-Copy Mmap & Fast BF16 Transcoding | Add memmap2, implement zero-copy weight loading in WeightLoader/TalkerWeightLoader, Rayon parallel BF16 transcode | none | DONE |
| M3 | Streaming Decoder O(1) & Pure Device Tensors | CausalConv1d device state, ConvTranspose1d Overlap-Add, DeviceKvCache, Decoder12Hz O(1) pipeline & alignment tests | none | DONE |
| M4 | E2E Streaming Pipeline & Code Cleanups | Talker generate_streaming callback, CandleLLM synthesize_streaming, Tokenizer wrapper, activation cleanup | M1, M3 | DONE |
| M5 | E2E Test Suite & Final Verification Hardening | Execute Tier 1-4 tests, latency/zero-allocation verification, forensic audit, Victory Audit preparation | M1, M2, M3, M4 | DONE |

## Interface Contracts
### TokenParser ↔ Text Frontend / Talker
- `pub const DEFAULT_CODEC_EOS_TOKEN_ID: u16 = 2150;`
- `TokenParser::with_eos(sample_rate: u32, codec_eos_token_id: u16) -> Self`
- `TokenParser::parse(&self, frames: &[Vec<u16>], options: &SynthesisOptions) -> Result<TokenStream>` (stops on `codec_eos_token_id`).

### WeightLoader / TalkerWeightLoader ↔ Storage
- `WeightLoader::from_file(path: impl AsRef<Path>, device: &Device) -> Result<Self>` (backed by `memmap2::Mmap`).
- `TalkerWeightLoader::from_safetensors(path: impl AsRef<Path>, device: &Device) -> Result<Self>` (backed by `memmap2::Mmap`).
- `bf16_bytes_to_f32_vec(data: &[u8]) -> Result<Vec<f32>>` (Rayon parallel bit-shift).

### Decoder12Hz Streaming Hot Path
- `CausalConv1d::step_tensor(&mut self, input: &Tensor) -> Result<Tensor>`: Input `(1, C_in, T_in)`, output `(1, C_out, T_in)` with pure device state concatenation.
- `CausalTransConvNet::step(&mut self, input: &Tensor) -> Result<Tensor>`: Input `(1, C_in, T_in)`, output `(1, C_out, T_in * stride)` with Overlap-Add buffer on device.
- `DeviceKvCache::step(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)>`: Sliding window 72 maintained purely on device.
- `Decoder12Hz::decode_chunk(&mut self, tokens: &[u16]) -> Result<Vec<f32>>`: Pure $O(1)$ single-chunk execution returning 1920 PCM f32 samples per 16-token frame.
- `Decoder12Hz::reset_state(&mut self)`: Zeroes/clears all internal device buffers.

### Talker ↔ CandleLLM Streaming Pipeline
- `TalkerForConditionalGeneration::generate_streaming<F>(&self, ..., mut frame_callback: F) -> Result<usize> where F: FnMut(&[u16; 16]) -> Result<()>`
- `CandleLLM::synthesize_streaming<F>(&self, text: &str, options: &SynthesisOptions, decoder: &mut Decoder12Hz, mut on_pcm_chunk: F) -> Result<usize> where F: FnMut(Vec<f32>) -> Result<()>`

## Code Layout
- `src/`
  - `codec/`
    - `causal_conv.rs` — `CausalConv1d` with device-resident `state_tensor`.
    - `decoder_blocks.rs` — `CausalTransConvNet` with Overlap-Add `overlap_buf`, `DecoderBlock`, `snake_beta`.
    - `transformer.rs` — `PreTransformer`, `DeviceKvCache` (replacing `KvRing`).
    - `activation.rs` — Cleaned activation exports.
    - `codebook.rs` — Codebook embeddings lookup.
  - `decoder_12hz.rs` — `Decoder12Hz` streaming $O(1)$ pipeline & state reset.
  - `talker/`
    - `talker.rs` — `TalkerForConditionalGeneration`, `generate_streaming`.
    - `weight_loader.rs` — `TalkerWeightLoader` with `memmap2` & Rayon BF16 transcoding.
  - `text_frontend/`
    - `token_parser.rs` — `TokenParser` with dynamic EOS token ID (2150).
    - `candle_backend.rs` — `CandleLLM`, `synthesize_streaming`.
  - `tokenizer/`
    - `mod.rs` — Safe wrapper around `tokenizers::Tokenizer`.
  - `weights.rs` — `WeightLoader` with `memmap2` zero-copy.
  - `vocoder/mod.rs` — `HifiGanVocoder`.
- `tests/`
  - `mrope_reference_test.rs` — M-RoPE reference test (clippy deny resolved).
  - `repetition_penalty_test.rs` — Repetition penalty test contract.
  - `streaming_alignment_test.rs` — Numerical alignment: batch `decode_frames` vs streaming `decode_chunk`.
  - `streaming_e2e_test.rs` — End-to-end streaming synthesis test.
