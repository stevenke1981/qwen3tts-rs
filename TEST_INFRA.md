# E2E Test Infra: qwen3tts-rs

## Test Philosophy
- Opaque-box & unit verification, requirement-driven per `ORIGINAL_REQUEST.md`, `optimization_suggestions_2026-09-01.md`, and `AGENTS.md`.
- Methodology: Category-Partition + Boundary Value Analysis + Pairwise Combinatorial + Real-World Workload Testing.

## Feature Inventory & Test Mapping
| # | Feature | Source | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---|---|:---:|:---:|:---:|:---:|
| F1 | Clippy Gate Zero-Error Pass | ORIGINAL_REQUEST §R1 | ✓ | ✓ | ✓ | ✓ |
| F2 | TokenParser EOS Truncation Alignment | ORIGINAL_REQUEST §R1 | ✓ | ✓ | ✓ | ✓ |
| F3 | Repetition Penalty Contract | Survey Finding | ✓ | ✓ | ✓ | ✓ |
| F4 | Zero-Copy Mmap Weight Loading | ORIGINAL_REQUEST §R3 | ✓ | ✓ | ✓ | ✓ |
| F5 | Vectorized BF16 Transcoding | ORIGINAL_REQUEST §R3 | ✓ | ✓ | ✓ | ✓ |
| F6 | CausalConv1d Device Streaming State | ORIGINAL_REQUEST §R2 | ✓ | ✓ | ✓ | ✓ |
| F7 | ConvTranspose1d Overlap-Add Streaming | ORIGINAL_REQUEST §R2 | ✓ | ✓ | ✓ | ✓ |
| F8 | PreTransformer Device KvCache | ORIGINAL_REQUEST §R2 | ✓ | ✓ | ✓ | ✓ |
| F9 | Decoder12Hz Streaming O(1) Pipeline | ORIGINAL_REQUEST §R2 | ✓ | ✓ | ✓ | ✓ |
| F10 | Talker Frame Streaming Interface | ORIGINAL_REQUEST §R4 | ✓ | ✓ | ✓ | ✓ |
| F11 | CandleLLM End-to-End PCM Streaming | ORIGINAL_REQUEST §R4 | ✓ | ✓ | ✓ | ✓ |
| F12 | Tokenizer Module Wrapper | ORIGINAL_REQUEST §R4 | ✓ | ✓ | ✓ | ✓ |
| F13 | Activation & Dead Code Cleanup | ORIGINAL_REQUEST §R4 | ✓ | ✓ | ✓ | ✓ |
| F14 | Comprehensive Quality & Victory Verification | Acceptance Criteria | ✓ | ✓ | ✓ | ✓ |

## Test Architecture
- **Runner**: `cargo test --all-targets`, `cargo clippy --all-targets`, `cargo check --all-targets`.
- **Latency & Streaming Benchmark**: `cargo bench --bench bench_decoder_12hz`.
- **Numerical Alignment Target**: Cosine similarity $\ge 0.999$, MSE $\le 10^{-4}$.
- **Zero Allocation Guard**: Device-resident tensors in hot path, zero CPU-GPU buffer transfers per chunk.

## Coverage Thresholds
- Tier 1: Fast unit & mathematical alignment tests (<1s).
- Tier 2: Module integration & micro-benchmarks (1-5s).
- Tier 3: End-to-end streaming & cross-module integration tests (5-30s).
- Tier 4: Real-weight hardware alignment & memory leakage stress tests.
