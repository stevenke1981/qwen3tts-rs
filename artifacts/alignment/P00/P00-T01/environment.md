# P00-T01 Environment

Captured: 2026-07-25 (Asia/Taipei)

## Source Revisions

- Target repository: `E:\qwen3tts-rs`
- Target baseline branch: `master`
- Working branch: `alignment/full-qwentts-parity`
- Target commit: `b08178964504d5a214565ffc4ff5ed592eb8f7ec`
- Target commit date: `2026-06-12 02:36:03 +0800`
- Target subject: `perf(codec): flatten Vec<RingBuffer> into single array + add benchmarks & CI`
- qwentts.cpp reference: `E:\qwentts.cpp-reference`
- qwentts.cpp branch: `master`
- qwentts.cpp commit: `82cd05b9f3a175612dc89fd6943e610fab096ef5`
- qwentts.cpp commit date: `2026-07-21 19:34:30 +0200`
- qwentts.cpp subject: `ggml: fork update`
- Official Qwen3-TTS remote HEAD: `022e286b98fbec7e1e916cb940cdf532cd9f488e`

Target and qwentts.cpp commits match `SOURCE_BASELINE.md`; therefore
`docs/alignment/baseline-delta-20260725.md` was not required.

## Toolchain and OS

- OS: Microsoft Windows 10 Pro, version 10.0.19045, build 19045, x86_64
- rustc: `1.97.1 (8bab26f4f 2026-07-14)`
- Rust host: `x86_64-pc-windows-msvc`
- LLVM: `22.1.6`
- cargo: `1.97.1 (c980f4866 2026-06-30)`

## CPU and GPU

- CPU: Intel Core i7-11700 @ 2.50 GHz
- CPU topology: 8 cores / 16 logical processors
- GPU: NVIDIA GeForce RTX 3070 Ti
- NVIDIA driver: 596.36
- VRAM: 8192 MiB (reported by `nvidia-smi`)
- Compute capability: 8.6
- CUDA toolkit: 13.2, nvcc 13.2.78
- Metal: unavailable on this Windows host

P00-T01 validated the CPU build only. CUDA and Metal numerical/performance validation remain
later backend gates and are not claimed here.
