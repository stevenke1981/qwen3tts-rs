# Performance Budget

Hardware-independent gates are preferred; hardware results must record exact device and build.

## Primary Metrics

- TTFT: time to first codec frame.
- TTFA: time to first audio callback.
- RTF: wall time / generated audio duration.
- codec frame latency p50/p95/p99.
- model resident memory and peak memory.
- GPU→CPU synchronization count per generated frame.
- per-session state memory.
- batch throughput at 1/2/4/8 sessions.

## Budgets

### Correctness Build, CPU F32

- Streaming work per frame must be bounded with history.
- 1200-frame memory growth after state warmup: < 5% excluding output buffering.
- No per-frame model reload or graph rebuild.

### CUDA

- persistent model allocations
- no 15 separate acoustic token readbacks per frame
- target no more than one mandatory small EOS/control synchronization per frame
- stateful codec graph topology reused where backend permits
- streaming overhead vs non-streaming RTF target ≤ 20%

### Metal

- no CPU fallback for core Talker/decoder math unless explicitly reported
- buffer reuse and bounded allocation count

### Quantization

Performance claims require:
- actual packed-weight compute path
- model-resident memory measurement
- comparison against identical text/model/seed/backend
