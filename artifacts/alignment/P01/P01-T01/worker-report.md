# P01-T01 Worker Report

Status: `IMPLEMENTED`

The GPT-5.3 Codex Spark implementer replaced runtime filename/model-ID
capability inference with fail-closed parsing of the official `config.json`.
The implementation exposes owned metadata, validates the nested Talker and
Code Predictor identities and dimensions, captures M-RoPE metadata, and makes
Auto mode resolve through the parsed model family.

Both synthesis examples now locate a real model directory, parse metadata for
every backend, and use the resolved mode for validation and runtime branches.
The real-model integration test requires an explicit environment variable and
cannot silently pass without a checkpoint.
