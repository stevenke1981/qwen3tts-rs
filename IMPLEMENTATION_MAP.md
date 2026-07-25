# Current-to-Target Implementation Map

This map tells Sol where to start inspecting. Paths may change after the baseline refresh.

| Existing area | Target responsibility | Initial action |
|---|---|---|
| `src/talker/` | Talker and Code Predictor model math | add stage-dump hooks and exact metadata |
| `src/talker/sampling.rs` | deterministic sampling | replace/extend with Philox and repetition penalty |
| `src/text_frontend/candle_backend.rs` | full native generation | convert from full-result API to frame/event API |
| `src/decoder_12hz.rs` | codec decode | separate offline decoder from stateful stream runtime |
| `src/text_frontend/voice_clone/` | x-vector and ICL conditioning | verify exact audio preprocessing and priming |
| `src/quantization/` | file quantization utilities | add true runtime quantized storage and kernels |
| `src/gui.rs` | existing product UI | consume streaming callback without owning inference logic |
| `tests/integration_test.rs` | current real-weight checks | split into mandatory fixture-aware parity suites |
| `tools/` | conversion helpers | create provenance-producing conversion and dump tools |

## Rule

Do not rename or rewrite all modules at once. Add contracts and adapters first, move one path at
a time, then remove obsolete paths after parity and performance gates pass.
