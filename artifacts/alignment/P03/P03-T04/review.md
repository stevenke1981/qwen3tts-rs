# P03-T04 Independent Review

## Result

Independent Luna review accepts P03-T04 with no remaining Gate-blocking
correctness issue.

## Findings resolved

- Restored the public Code Predictor `(codes, caches)` return contract while
  Talker uses internal no-copy methods.
- Kept existing `StageDumpObserver` implementations source compatible by
  separating stage and transfer observers.
- Gated every transfer callback with `wants_transfer_capture()`.
- Added exact direction, element-count, and order assertions for greedy,
  sampled Code Predictor, and sampled Talker paths.
- Replaced 14 growing Code Predictor concatenations with a fixed 15-element
  tensor array and one final `Tensor::cat`.
- Validated the all-invalid sentinel before embedding, preventing a possible
  out-of-range CUDA gather.

## Accepted safety exception

An ordinary greedy completed frame transfers one codebook-0 scalar. The
incomplete terminal draw transfers one additional scalar solely to validate
the fail-closed sentinel before any backend embedding lookup. The combined
one-frame-plus-terminal scenario is therefore two device-to-host scalar events.
This preserves pinned terminal-draw semantics and backend safety.

## Evidence

- 135 library tests passed.
- 21 acoustic host-transfer tests passed.
- 8 Code Predictor frame tests passed.
- The pinned real Code Predictor oracle passed with minimum cosine
  `0.999999999939` and maximum absolute error `0.000438690186`.
- 12 integration tests passed.
