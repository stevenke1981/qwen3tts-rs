# Risk Register

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| R01 | Official and qwentts.cpp behavior diverge | wrong parity target | official Python/config wins; record ADR |
| R02 | Hard-coded token/model defaults | silent wrong voices/tokens | metadata-first loader and exact tests |
| R03 | Streaming output matches but remains O(n²) | unusable long-form runtime | complexity and static gates |
| R04 | Transposed-conv state produces seams | audible clicks | offline exactness tests at many lengths |
| R05 | Quantized files are dequantized at load | false memory/performance claim | resident memory and type inspection |
| R06 | RNG differs | token mismatch | Philox known vectors and full-chain tests |
| R07 | Candle backend lacks required custom op | fallback or large slowdown | isolated backend trait and custom kernels |
| R08 | 1.7B projection dimensions mishandled | incorrect Code Predictor | model-matrix tests |
| R09 | Voice clone preprocessing mismatch | identity drift | reference mel/x-vector/code dumps |
| R10 | Real-weight tests silently skip | false confidence | release fixture jobs fail closed |
| R11 | Large agent task causes broad regressions | hard-to-review changes | bounded Spark tasks and small commits |
| R12 | Server/FFI added before core parity | product hides model errors | phase dependencies enforce P00-P08 first |
| R13 | Cross-session state contamination | privacy/correctness issue | randomized concurrency isolation tests |
| R14 | Copied code violates attribution | legal/release risk | NOTICE, preserved licenses, clean-room records |
| R15 | Reference repo changes mid-phase | moving target | pinned baseline and delta review |
