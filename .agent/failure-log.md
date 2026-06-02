# Failure Log — Qwen3-TTS Rust Decoder

## FAILURE_LOG #1 — ConvNeXtBlock/DWConv cos=0.889

- **Timestamp**: 2026-06-02
- **Error**: `CN (PT inp) dwconv` cos=0.889, pattern: Rust[0..3] = PT[3..5]
- **Approach**: Spent ~12h tracing Candle grouped conv1d source code, analyzing Im2Col1D, confirmed it's mathematically correct. Wrote isolated 1024-group test that passes (cos=1.000).
- **Root Cause**: Reference .npy files were incorrect — didn't match actual PyTorch `nn.Conv1d` output.
- **Prevention**: Always verify reference data against actual PyTorch forward pass before assuming implementation bug.

## FAILURE_LOG #2 — Pre-Transformer cos=0.965

- **Timestamp**: 2026-06-02
- **Error**: PreTransformer output cos=0.965 vs PT reference
- **Status**: Investigation pending
- **Hypothesis**: Causal attention mask, RoPE frequency, or SwiGLU FFN implementation issue
- **Next Step**: Enable per-layer comparison in section 3a of debug_per_layer_compare.rs
