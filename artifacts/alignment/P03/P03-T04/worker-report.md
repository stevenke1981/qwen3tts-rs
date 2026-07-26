# P03-T04 Worker Report

## Implementation

- Greedy Talker selection now masks suppression and non-finite values on device.
- Greedy Code Predictor keeps all 15 argmax tokens on device.
- Talker and Code Predictor assemble output tensors on device.
- Talker's production path uses internal no-copy Code Predictor methods.
  Existing public `(codes, caches)` wrappers remain source compatible and keep
  their legacy cache-container return.
- Sampled transfer telemetry is emitted inside `Sampler` after the actual
  full-logit host materialization.
- Scalar upload telemetry is emitted only after successful tensor creation.
- Greedy terminal-cap attempts read one scalar to validate the all-invalid
  sentinel before any embedding lookup; this avoids unsafe CUDA out-of-range
  gather behavior while preserving terminal-draw semantics.
- Stage dump writers implement allocation-free default-noop transfer telemetry.

## Sol Gate-Owner Corrections

The first external-agent result was not accepted because:

- `stage_instrumentation_test` did not compile;
- the transfer test contained stale inference-based documentation;
- greedy masking did not preserve non-finite behavior;
- terminal-cap transfer accounting initially omitted the safety/semantic
  tradeoff;
- sampled telemetry was adjacent to, rather than inside, the real sampler
  boundary;
- task evidence files were absent.

These issues were corrected before final verification.

Independent review later identified that embedding the sentinel before
validation was unsafe on CUDA. The final implementation validates it with one
scalar read first, and the evidence records the terminal draw separately from
ordinary completed-frame cost.

## Verification

See `commands.txt` and `test-results.txt`. The pinned real 0.6B Code Predictor
oracle passed with minimum cosine `0.999999999939` and maximum absolute error
`0.000438690186`.

## Remaining Risks

- Repository-wide formatting and strict Clippy are red because of existing
  changes in three earlier local commits, outside P03-T04.
- The aggregate `cargo test --tests` command exceeded the 120-second execution
  window; required focused, integration, stage, and real-oracle commands passed
  independently.
- Eleven legacy Talker alignment cases remain ignored and are not counted.
- MSVC continues to emit the pre-existing `LNK4098` warning.
