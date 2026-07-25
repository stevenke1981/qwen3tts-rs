# P00-T03 Worker Report

Status: `IMPLEMENTED`

## Summary

- Added the non-default `stage-dump` feature and an explicitly owned observer API.
- Added a fail-closed F32 little-endian stage writer with manifest v1, SHA-256, safe unique
  names, create-new file semantics, and no global recorder.
- Added frame-aware Talker and Code Predictor hooks plus offline 12 Hz codec input/output hooks.
- Preserved existing public generation/decode entry points through a no-op observer.
- Added manifest schema, dump comparator, overwrite/name/order/byte/hash tests, and regression
  comparison coverage.

## Files

- `Cargo.toml`
- `src/lib.rs`
- `src/alignment_stage_dump.rs`
- `src/talker/talker.rs`
- `src/talker/code_predictor.rs`
- `src/decoder_12hz.rs`
- `tests/stage_dump_test.rs`
- `schemas/stage-dump.schema.json`
- `tools/compare_stage_dumps.py`

## Hook names

- `talker_codebook0_logits_{frame:04}`
- `talker_final_code_matrix`
- `code_predictor_step_logits_{frame:04}_{step:04}`
- `code_predictor_final_code_matrix_{frame:04}`
- `codec_input_code_matrix`
- `codec_final_pcm`

## Notes

- No-observer codec input capture is guarded before allocating the capture tensor/vector.
- The stage-dump writer exists only when the Cargo feature is enabled.
- Synthetic tensors validate plumbing and format only; they are not numerical-parity evidence.
