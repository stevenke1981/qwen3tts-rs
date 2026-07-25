# Gap Matrix

Legend: P0 = required for core parity, P1 = product parity, P2 = extended backend parity.

| Area | Current qwen3tts-rs condition | Required target | Priority |
|---|---|---|---|
| Metadata | shape inference plus hard-coded defaults | config/metadata-driven model variants | P0 |
| M-RoPE | default interleaving can disagree with model metadata | exact official/reference semantics | P0 |
| Sampling RNG | SplitMix64 | Philox-compatible deterministic stream | P0 |
| Sampling chain | temperature/top-k/top-p | repetition penalty then temperature/top-k/top-p | P0 |
| Sampling controls | shared settings | independent Talker and Code Predictor settings | P0 |
| Stage parity | limited final-output checks | prompt/token/layer/logit/code/audio dumps | P0 |
| Codec streaming | accumulated-history recompute O(n²) | persistent causal states, O(1) work/frame wrt history | P0 |
| TTS streaming | complete token generation then batch decode | token frame → PCM callback during generation | P0 |
| ICL priming | full/batch path | exact streaming-state prime and reusable snapshot | P0 |
| Tokenizer encoder | partial/native components | complete WAV → 16 RVQ code path and CLI | P1 |
| Quantization | dequantize packed weights to F32 | native quantized matmul/embedding storage and compute | P0 |
| Model format | safetensors directories | GGUF-compatible loader/converter or equivalent metadata format | P1 |
| CLI | GUI/examples dominant | qwen-tts and qwen-codec compatible tools | P1 |
| Server | absent | OpenAI-compatible streaming HTTP server | P1 |
| Voice registry | absent | named/cloned voice registry with safe persistence | P1 |
| C ABI | absent | stable cdylib, ownership-safe handles and callbacks | P1 |
| Batching | single request | bounded multi-session continuous batching | P1 |
| CUDA | basic Candle support | tested optimized path and bounded syncs | P1 |
| Metal | feature exists | tested model/runtime parity | P1 |
| Vulkan/ROCm | absent | backend strategy and parity implementation | P2 |
| CI | real-weight tests can skip/continue | mandatory gated parity jobs with fixture provenance | P0 |
| Release evidence | ad hoc | reproducible matrix, SBOM, licenses, benchmark report | P1 |
