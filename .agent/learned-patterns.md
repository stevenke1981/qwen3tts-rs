# Learned Patterns — Qwen3-TTS Rust

## Debug Strategies

### 1. Isolate Before Blaming
When a cos mismatch appears:
1. First verify with same-input isolation test (feed PT input into Rust)
2. If still diverges, test the specific operator in isolation with random data
3. If random data passes but real data fails, suspect **reference data quality**

### 2. Per-Channel Diagnostics
For grouped ops (depthwise conv, grouped matmul):
- Don't just compare global cos — compare per-channel cos
- A shift pattern (`Rust[i] = PT[i+k]`) suggests reference data issue, not implementation bug
- Best/worst channel analysis reveals whether the error is systematic or isolated

### 3. Reference File Verification
- Always verify reference .npy files against actual PyTorch forward pass
- Don't trust files generated in a different session/script without verification
- When regenerating refs, use the exact same weight loading method as Rust

### 4. Minimal Reproduction
- For suspected Candle bugs: create a standalone test with `Device::Cpu`
- Compare with exact same shapes/types in PyTorch
- Dump raw values for both sides → compute cos per-element

## Effective Tools
- `tests/debug_per_layer_compare.rs` — comprehensive per-layer diagnostic
- `tests/depthwise_conv_test.rs` — isolated depthwise conv test
- Python scripts in `tests/` directory for reference verification

## Code Organization
- Rust source: `src/codec/`
- Test harness: `tests/debug_per_layer_compare.rs`
- Reference data: `weights/pt_*.npy`
- Weight storage: `weights/tokenizer/*.safetensors`
