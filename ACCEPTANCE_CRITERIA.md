# Acceptance Criteria

## Numerical Reference Order

1. Official Qwen3-TTS Python implementation and official checkpoint config.
2. qwentts.cpp CPU F32 reference behavior.
3. qwen3tts-rs CPU F32 implementation.
4. accelerated and quantized variants compared with the CPU F32 Rust baseline.

## Required Thresholds

### Prompt and Tokens

- BPE prompt IDs: exact equality.
- language, speaker and special token IDs: exact equality.
- fixed seed, CPU F32 generation: exact codec token sequence for the parity corpus.
- EOS position: exact equality.
- generated frame count: exact equality.

### Tensor Stages

Unless a stricter task card applies:

- embeddings and normalization outputs: cosine ≥ 0.99999
- attention/MLP layer checkpoints: cosine ≥ 0.9995
- logits: cosine ≥ 0.9990 and matching top-10 ordering for ≥ 99.9% of checked steps
- decoder intermediate tensors: cosine ≥ 0.9990
- CPU F32 final waveform: cosine ≥ 0.999 and max absolute error ≤ 1e-4

Thresholds cannot be lowered without an ADR and reference evidence.

### Streaming Correctness

For 1, 3, 13, 128, 300 and 1200 frames:

- streamed sample count equals offline sample count
- CPU F32 waveform max absolute difference ≤ 1e-5
- reset + replay reproduces the first run within the same tolerance
- chunk callback begins before generation completes
- no boundary discontinuity beyond the offline reference

### Streaming Complexity

- No input/history buffer grows linearly solely to re-run prior frames.
- Median decode time for the last 25% of a 1200-frame run ≤ 1.20 × first 25%
  after warmup on the same backend.
- Resident per-session state reaches a bounded plateau.
- Static check finds no production call from `decode_frame` to offline full-sequence decode.

### Quantization

- No full F32 materialization of the quantized Talker backbone.
- Q8 model resident weight memory ≤ 65% of BF16 baseline.
- Q4 model resident weight memory ≤ 40% of BF16 baseline.
- Q8 codec-token agreement with F32 ≥ 99.5% on deterministic greedy corpus, or the
  documented stochastic quality gate for sampled output.
- RVQ-sensitive tensors follow the protected F32/F16 policy.
- Audio quality corpus passes cosine and perceptual checks defined in `TEST_PLAN.md`.

### Product API

- CLI exit codes and argument validation are covered by tests.
- `/v1/audio/speech` supports standard non-streaming and streaming responses.
- C ABI passes create/load/synthesize/cancel/free lifecycle tests.
- Session isolation tests pass under at least 8 concurrent requests.
- Malformed inputs never panic across the FFI boundary or server handler.

## Overall Alignment Gate

All required tasks P00-P13 must be `GATE_PASSED`, all P0 and P1 requirements must pass, and
`artifacts/alignment/release/final-report.json` must exist and validate against its schema.
