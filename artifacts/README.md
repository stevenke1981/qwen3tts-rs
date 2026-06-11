# Artifacts

This directory contains checked-in artifacts that are useful for debugging,
alignment, or release history, but are not source code.

- `tokens/`: token streams produced during native and Python comparison runs.
- `asr/`: ASR transcripts/manifests used to compare native and upstream WAVs.
- `reference-audio/`: reference media used for manual voice-clone validation.
- `logs/`: captured command output from earlier troubleshooting sessions.
- `text/`: prompt/story text used for local synthesis checks.
- `inspection/`: temporary upstream inspection dumps kept for traceability.
- `local/`: ignored local-only generated audio, release extracts, and batch runs.

New large local outputs should stay untracked unless they are intentionally
promoted into this directory with a clear reason.
