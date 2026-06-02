# ConvNeXtBlock / Depthwise Conv Resolution

## What We Thought Was Wrong
- Candle 0.10.2's grouped `Conv1d` with `groups>1` produces wrong output
- `CN (PT inp) dwconv` showed cos=0.889
- The error pattern was: `Rust[0..3] = PT[3..5]` — a 3-position shift

## What Was Actually Wrong
**The reference `.npy` files were incorrect.** They did not match what PyTorch's
`nn.Conv1d(padding=3, groups=1024, bias=True)` actually produces.

## How We Found It
1. Wrote isolated 1024-group depthwise conv test (random data): **cos=1.000** ✅
2. Added per-channel diagnostic: worst channels showed shift pattern, best channels showed cos≈1.000
3. Wrote Python script to run actual PyTorch `nn.Conv1d` with real weights:
   - Without bias: cos=0.754 vs reference
   - With bias: cos=0.889 vs reference
   - **Rust output exactly matches PyTorch with bias**
4. Regenerated reference files using correct PyTorch computation → all ConvNeXtBlock sub-steps cos=1.000

## Key Files
- `tests/depthwise_conv_test.rs` — isolated depthwise conv tests (2 groups and 1024 groups)
- `tests/_extract_weight.py` — Python script to extract real weights from safetensors
- `tests/_verify_conv.py` — Python script that proved Candle matches PyTorch
- `tests/_regenerate_refs.py` — regenerated all ConvNeXtBlock reference .npy files

## Lesson
- **When a cos mismatch looks like a shift by kernel_size//2, suspect the reference data, not the implementation**
- Always verify with a minimal isolated test first (we should have done this earlier)
- Always check against actual PyTorch forward pass, not just saved npy files

## Current Status
- ✅ Candle grouped Conv1d: **correct**
- ✅ ConvNeXtBlock with PT input: **cos=1.000** for all sub-steps
- ✅ Candle ConvNeXtBlock implementation: **correct**
- ⚠️ ConvNeXtBlock with Rust ct input: cos=0.998 (the 0.002 error comes from ct input divergence)
