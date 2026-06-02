# Pre-Transformer Bug Investigation

**Status**: ⚠️ cos=0.965 — FIRST divergence in the pipeline
**Severity**: ROOT CAUSE — everything downstream amplifies this error

## Architecture
- 8-layer PreTransformer
- Causal multi-head attention (sliding window + RoPE)
- SwiGLU FFN
- Input: [1, 1024, 3] from CausalConv1d, internally transposed to [1, 3, 1024]
- Output: [1, 3, 1024]

## What We Know
- Codebook and pre-conv match perfectly (cos=1.000)
- Pre-transformer is the first stage that diverges (cos=0.965)
- 8 per-layer reference files exist: `weights/pt_pre_trans_layers.{0..7}.npy`
- The test prints "Layer {i}: ref exists" but doesn't currently compare them (just checks existence)

## Likely Culprits
1. **Attention mask**: causal mask might be off-by-one or wrong shape
2. **RoPE**: frequency computation or apply_rotary_emb implementation
3. **SwiGLU FFN**: gate_proj / up_proj / down_proj ordering or shapes
4. **Sliding window**: window size or padding
5. **LayerNorm**: pre-attention and pre-FFN layernorms
6. **Residual connection**: gamma scaling or wrong addition

## Reference Data
- `weights/pt_pre_trans_out.npy` — full pre-transformer output [3, 1024]
- `weights/pt_pre_trans_layers.{0..7}.npy` — per-layer outputs
- If per-layer refs were saved BEFORE the residual-add, that's key

## How to Debug
1. Modify `debug_per_layer_compare.rs` section 3a to actually compare per-layer refs
2. The cos of each layer will show which layer first diverges
3. Then isolate that layer's sub-steps: Q/K/V projections → attention → FFN → residual

## Implementation Files
- `src/codec/pre_transformer.rs` — PreTransformer, DecoderLayer, Attention, SwiGLU
