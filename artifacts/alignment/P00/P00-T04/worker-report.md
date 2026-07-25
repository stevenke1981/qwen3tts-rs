# P00-T04 Worker Report

Status: `IMPLEMENTED`

## Summary

- Added fail-closed `official-python` and `qwentts-cpp` subprocess adapters.
- Added dry-run plans with exact argv and no child execution or output creation.
- Added revisioned, hashed `reference-run.json` manifests for successful and child-failure runs.
- Added prompt/token/stdout/stderr/audio/tensor artifact collection with safe relative paths.
- Added qwentts.cpp raw header validation and P00-T03-compatible F32 stage manifests.
- Added overwrite, reserved-argument, sensitive-argument, malformed-output and symlink guards.

## Files

- `tools/reference_adapter.py`
- `tools/test_reference_adapter.py`
- `schemas/reference-run.schema.json`
- `docs/alignment/reference-adapters.md`

## Notes

- Real model inference is intentionally deferred to P00-T05.
- The tests use tiny fake runners and do not claim numerical parity.
