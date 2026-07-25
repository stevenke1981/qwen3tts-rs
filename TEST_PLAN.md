# Test Plan

## Test Layers

### L0 Static and Unit

- metadata parsing
- token tables
- prompt assembly
- Philox known vectors
- repetition penalty ordering
- top-k/top-p edge cases
- quantized block packing/depacking
- state reset and buffer boundary logic
- C ABI null/invalid handle behavior

### L1 Synthetic Tensor Parity

Small deterministic tensors compare each Rust operator with a Python or qwentts.cpp reference dump:

- RMSNorm
- Q/K normalization
- NEOX RoPE / multimodal position mapping
- attention with KV append/ring
- SwiGLU
- causal convolution
- causal transposed convolution with overlap
- SnakeBeta
- RVQ lookup and projection
- quantized linear and embedding

### L2 Real-Weight Stage Parity

Required model fixtures:

- 0.6B Base
- 1.7B Base
- 1.7B CustomVoice
- 1.7B VoiceDesign
- tokenizer

Corpus contains:

- Chinese, English, Japanese, Korean, French, German, Italian, Spanish, Portuguese, Russian
- Mandarin dialect speakers
- punctuation, numbers, mixed language, long text, emoji/unsupported text
- clone clips of short and long reference duration
- empty/invalid requests

Stage outputs are stored as metadata plus binary arrays with SHA-256.

### L3 End-to-End

- default voice
- x-vector clone
- ICL clone
- named speaker
- voice design
- long-form streaming
- cancellation mid-generation
- reset and repeated synthesis
- same model, 8 concurrent sessions

### L4 Product

- CLI snapshots and exit codes
- HTTP request/response compatibility
- chunked streaming
- voice registry persistence and sanitization
- C ABI lifecycle from C test program

### L5 Performance

- CPU F32 baseline
- CUDA BF16
- Metal
- Q8/Q4
- single and batched sessions
- 1200-frame streaming complexity

## Fixture Policy

A test requiring weights must fail with a clear `FIXTURE_MISSING` result in parity CI.
It must not silently pass. Lightweight developer CI may mark such jobs as not-run, but the
release gate requires the full fixture job.

Every fixture has:

```json
{
  "source": "...",
  "revision": "...",
  "sha256": "...",
  "license": "...",
  "generated_by": "...",
  "command": "..."
}
```
