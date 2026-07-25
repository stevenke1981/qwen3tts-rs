# Source Baseline

## Target

- Repository: https://github.com/stevenke1981/qwen3tts-rs
- Branch: `master`
- Observed commit: `b08178964504d5a214565ffc4ff5ed592eb8f7ec`
- Observed date: `2026-06-12`

Observed implementation characteristics:

- Candle-based Talker and Code Predictor.
- Native and Python-bridge text frontends.
- 12 Hz decoder batch path.
- Existing `decode_chunk` rebuilds accumulated history and is O(n²).
- Quantized safetensors are restored to F32 tensors before compute.
- GUI exists; deployment-compatible CLI/server/C ABI parity is incomplete.
- 25 Hz decoder is not part of the 12 Hz qwentts.cpp parity target and must not block P0-P13.

## Behavioral Reference

- Repository: https://github.com/ServeurpersoCom/qwentts.cpp
- Branch: `master`
- Observed commit: `82cd05b9f3a175612dc89fd6943e610fab096ef5`
- Observed date: `2026-07-21`

Reference capabilities to reproduce:

- Stateful frame-by-frame codec decode with offline-equivalent output.
- Talker and 15-code acoustic predictor with KV caches.
- HF-aligned sampling chain:
  repetition penalty → temperature → top-k → top-p → multinomial.
- Seedable Philox RNG.
- GGUF conversion and native Q8_0 / Q4_K_M inference.
- `qwen-tts`, `qwen-codec`, and OpenAI-compatible `tts-server`.
- Voice clone, CustomVoice, VoiceDesign.
- CPU, CUDA, Metal and Vulkan-class deployment coverage.
- Multi-lane batching and per-session codec streams.

## Official Semantics Reference

- https://github.com/QwenLM/Qwen3-TTS
- Observed commit: `022e286b98fbec7e1e916cb940cdf532cd9f488e`
- Observed date: `2026-07-25` (remote HEAD inspection)
- Use official model configs and Python outputs as the final authority when qwentts.cpp
  and the current Rust implementation disagree.
- Never infer a token ID, model dimension, speaker table, language table, or generation
  default from a hard-coded fallback when it is available from model metadata.

## Baseline Refresh Rule

At the start of every work session:

```text
1. Fetch current target and reference commit SHAs.
2. Compare against this file.
3. If either changed, generate docs/alignment/baseline-delta-YYYYMMDD.md.
4. Re-run P00 probes before modifying production code.
5. Do not silently move the parity target during an active phase.
```
