# Qwen3-TTS Rust Decoder — Debug Status

Last updated: 2026-06-02

## Project Goal
Port Qwen3-TTS Codec Decoder (12Hz streaming mode) from PyTorch to pure Rust / Candle.

## Pipeline Verification (3 frames × 16 tokens)

| Stage                          | Cosine vs PT | Status      |
|--------------------------------|-------------|-------------|
| Codebook lookup                | 1.000       | ✅ MATCH    |
| Pre-conv (CausalConv1d)       | 1.000       | ✅ MATCH    |
| Pre-transformer                | 0.965       | ⚠️ DIVERGED |
| Upsample.0.ct (ConvTranspose1d)| 0.970       | ⚠️ DIVERGED |
| Upsample.0.cn (ConvNeXtBlock)  | 0.998       | ✅ OK       |
| Upsample.1.ct                  | 0.644       | ❌ AMPLIFIED|
| Upsample.1.cn                  | 0.451       | ❌ AMPLIFIED|
| Decoder blocks 0-4             | < 0.3       | ❌ AMPLIFIED|
| Final PCM                      | -0.018      | ❌ AMPLIFIED|

## Headline: Candle grouped Conv1d IS correct

We spent ~24 hours tracing a dwconv divergence to find that **Candle 0.10.2's grouped Conv1d is correct**.
The reference `.npy` files for ConvNeXtBlock sub-steps were generated incorrectly (possibly with a
different PyTorch version or model checkpoint). After regenerating them with the correct
`nn.Conv1d(padding=3, groups=1024, bias=True)`, **all ConvNeXtBlock sub-steps show cos=1.000**.

## Remaining Issues

### 1. Pre-Transformer (cos=0.965) — ROOT CAUSE
- 8-layer Transformer with causal multi-head attention + SwiGLU FFN
- Per-layer reference files exist in `weights/pt_pre_trans_layers.{0..7}.npy`
- Likely a bug in attention implementation (RoPE? KV-cache? mask?) or FFN.

### 2. ConvTranspose1d (cos=0.970)
- Input is pre-transformer output (already diverged at cos=0.965)
- The cos=0.970 is close to input cos — likely just error propagation
- Need separate verification with identical input

### 3. Everything downstream diverge
- Caused by error amplification through the pipeline

## Setup
- Test: `cargo test --test debug_per_layer_compare -- --nocapture`
- Weights: `weights/tokenizer/` (safetensors files)
- Reference PW npy: `weights/pt_*.npy`
- Hardcoded 3 frames of 16 tokens each from `pt_full_trace.py`
- Device: CPU (~28s runtime, ~2GB RAM)
