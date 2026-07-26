# Alignment TODOs

Overall status: `IN_PROGRESS`

## P00 — Baseline and parity harness

- [x] `P00-T01` Pin target/reference/upstream revisions and write baseline delta report
- [x] `P00-T02` Create fixture manifest and fail-closed fixture resolver
- [x] `P00-T03` Add stage dump format and dump hooks behind a feature flag
- [x] `P00-T04` Build reference command adapters for Python and qwentts.cpp
- [x] `P00-T05` Create CPU F32 smoke corpus and baseline report
- [x] `P00-GATE` Phase gate passed

## P01 — Metadata and prompt parity

- [x] `P01-T01` Replace model-name inference with metadata/config parsing
- [x] `P01-T02` Load all special token, language, speaker and dialect tables from metadata
- [x] `P01-T03` Correct M-RoPE semantics and add exact position/rotation tests
- [x] `P01-T04` Match prompt assembly for Base, CustomVoice and VoiceDesign
- [x] `P01-T05` Gate exact prompt IDs across the model matrix
- [x] `P01-GATE` Phase gate passed

## P02 — Sampling parity

- [x] `P02-T01` Implement Philox RNG with known-vector tests
- [x] `P02-T02` Implement repetition penalty with exact operation ordering (`GATE_PASSED`)
- [x] `P02-T03` Separate Talker and Code Predictor sampling configs (`GATE_PASSED`)
- [x] `P02-T04` Match token suppression and EOS handling (`GATE_PASSED`)
- [x] `P02-T05` Gate deterministic token sequence parity (`GATE_PASSED`)
- [x] `P02-GATE` Phase gate passed

## P03 — Talker and Code Predictor numerical parity

- [x] `P03-T01` Instrument embeddings, norms, RoPE and layer outputs (`GATE_PASSED`)
- [x] `P03-T02` Verify Talker prefill and single-step KV cache (`GATE_PASSED`)
- [x] `P03-T03` Verify Code Predictor frame-local prefill and 14 decode steps (`GATE_PASSED`)
- [ ] `P03-T04` Eliminate avoidable host transfers in acoustic prediction
  (`IN_PROGRESS`: external DeepSeek V4 Flash handoff prepared; current attempt
  has blocking telemetry/evidence defects)
- [ ] `P03-T05` Gate stage cosine and logit ranking thresholds
- [ ] `P03-GATE` Phase gate passed

## P04 — Tokenizer decoder offline parity

- [ ] `P04-T01` Verify RVQ split projections and codebook policy
- [ ] `P04-T02` Verify decoder transformer and sliding-window semantics
- [ ] `P04-T03` Verify ConvNeXt upsample and DAC blocks
- [ ] `P04-T04` Match offline waveform on short/medium/long corpora
- [ ] `P04-T05` Create buffered chunk decode with left-context trimming
- [ ] `P04-GATE` Phase gate passed

## P05 — True stateful codec streaming

- [ ] `P05-T01` Define CodecStreamState and state ownership
- [ ] `P05-T02` Implement persistent causal-convolution contexts
- [ ] `P05-T03` Implement transposed-convolution overlap state
- [ ] `P05-T04` Implement transformer KV ring and absolute RoPE position
- [ ] `P05-T05` Implement one-frame graph/buffer reuse
- [ ] `P05-T06` Implement reset, ICL prime and optional state snapshots
- [ ] `P05-T07` Gate offline-equivalent output and bounded complexity
- [ ] `P05-GATE` Phase gate passed

## P06 — End-to-end generation streaming

- [ ] `P06-T01` Expose frame events from Talker generation
- [ ] `P06-T02` Connect generated codes directly to codec stream session
- [ ] `P06-T03` Add audio callback, backpressure and cancellation
- [ ] `P06-T04` Update GUI to consume streaming events
- [ ] `P06-T05` Gate TTFA-before-completion and long-form output
- [ ] `P06-GATE` Phase gate passed

## P07 — Tokenizer encoder and qwen-codec

- [ ] `P07-T01` Complete 24 kHz audio preprocessing/resampling contract
- [ ] `P07-T02` Verify SEANet and encoder transformer
- [ ] `P07-T03` Implement RVQ encode argmin path
- [ ] `P07-T04` Define versioned RVQ code file format
- [ ] `P07-T05` Implement qwen-codec encode/decode/stream CLI
- [ ] `P07-T06` Gate round-trip and reference code parity
- [ ] `P07-GATE` Phase gate passed

## P08 — Native quantized runtime

- [ ] `P08-T01` Define quantized tensor types and protected tensor policy
- [ ] `P08-T02` Implement direct Q8 linear and embedding operations
- [ ] `P08-T03` Implement Q4_K_M-class block layout and kernels
- [ ] `P08-T04` Add backend-resident packed weight loader
- [ ] `P08-T05` Integrate quantized Talker and Code Predictor
- [ ] `P08-T06` Gate memory, token and quality metrics
- [ ] `P08-GATE` Phase gate passed

## P09 — Model format and conversion

- [ ] `P09-T01` Define metadata-complete Rust model container strategy
- [ ] `P09-T02` Implement GGUF reader compatibility or lossless converter
- [ ] `P09-T03` Implement official checkpoint conversion with provenance
- [ ] `P09-T04` Implement quantization command and protected tensor rules
- [ ] `P09-T05` Gate five talker variants plus shared tokenizer
- [ ] `P09-GATE` Phase gate passed

## P10 — CLI and library product surface

- [ ] `P10-T01` Implement stable high-level Rust synthesis/session API
- [ ] `P10-T02` Implement qwen-tts CLI with streaming and WAV output
- [ ] `P10-T03` Add model discovery/download and cache policy
- [ ] `P10-T04` Add structured logs, JSON metrics and exit codes
- [ ] `P10-T05` Gate compatibility corpus and cancellation
- [ ] `P10-GATE` Phase gate passed

## P11 — OpenAI server and voice registry

- [ ] `P11-T01` Implement /v1/audio/speech non-streaming endpoint
- [ ] `P11-T02` Implement chunked/streaming audio response
- [ ] `P11-T03` Implement safe cloned-voice registry
- [ ] `P11-T04` Implement request limits, cancellation and error mapping
- [ ] `P11-T05` Gate API and concurrency tests
- [ ] `P11-GATE` Phase gate passed

## P12 — C ABI and continuous batching

- [ ] `P12-T01` Define stable opaque-handle C ABI
- [ ] `P12-T02` Implement callback ownership and cancellation
- [ ] `P12-T03` Implement per-session scheduler state
- [ ] `P12-T04` Implement bounded multi-lane Talker/Predictor batching
- [ ] `P12-T05` Implement per-slot codec streams and isolation
- [ ] `P12-T06` Gate C lifecycle, 8-session isolation and throughput
- [ ] `P12-GATE` Phase gate passed

## P13 — Backends, CI and release

- [ ] `P13-T01` CPU/CUDA/Metal full matrix and performance report
- [ ] `P13-T02` Vulkan/ROCm strategy and implementation gate
- [ ] `P13-T03` Replace permissive CI with fail-closed parity workflow
- [ ] `P13-T04` Generate SBOM, notices, model manifest and reproducible build notes
- [ ] `P13-T05` Run final audit and produce alignment release report
- [ ] `P13-GATE` Phase gate passed
