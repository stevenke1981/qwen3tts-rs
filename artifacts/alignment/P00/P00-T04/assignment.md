# GPT-5.3 Codex Spark — Implementation Assignment

## Task

- Task ID: `P00-T04`
- Goal: build fail-closed command adapters for the official Python implementation and the pinned
  qwentts.cpp CLI, producing revisioned and hashed prompt/token/tensor/audio evidence.
- Phase contract: `tasks/P00-baseline-and-parity-harness.md`

## Allowed files

- `tools/reference_adapter.py` (new)
- `tools/test_reference_adapter.py` (new)
- `schemas/reference-run.schema.json` (new)
- `docs/alignment/reference-adapters.md` (new)

No other file may be modified. Do not modify the external reference checkout.

## Verified external contracts

- qwentts.cpp revision: `82cd05b9f3a175612dc89fd6943e610fab096ef5`
- Official Qwen3-TTS revision: `022e286b98fbec7e1e916cb940cdf532cd9f488e`
- qwentts.cpp `qwen-tts` reads UTF-8 text from stdin and accepts:
  `--model`, `--codec`, `--lang`, `--speaker`, `--instruct`, `--ref-wav`, `--ref-text`,
  `--seed`, `--greedy`, `--max-new`, `--dump`, and `-o`.
- qwentts.cpp raw tensor dump format is:
  `[ndims:i32-le][shape:i32-le * ndims][contiguous f32-le data]`.
- The existing official Python token runner is `tools/generate_tokens.py`; the adapter must also
  support an explicitly supplied official runner that writes parity evidence into an output
  directory.

## Required interface

Implement `tools/reference_adapter.py` with two subcommands:

1. `official-python`
   - explicit `--python`, `--runner`, `--source-revision`, `--model`, `--text` or `--text-file`,
     `--output-dir`, seed/language/mode options, and repeatable extra arguments;
   - execute with an argv list, never through a shell;
   - preserve UTF-8 and capture stdout/stderr without mixing binary token output with logs;
   - support the existing token runner by writing captured stdout as a token artifact;
   - collect explicitly declared prompt/token/tensor/audio outputs from the session directory.
2. `qwentts-cpp`
   - explicit `--executable`, `--source-revision`, `--model`, `--codec`, input text,
     `--output-dir`, sampling/mode options, and repeatable extra arguments;
   - pipe text through UTF-8 stdin;
   - always request a raw dump directory and WAV output inside the new session directory;
   - normalize every valid qwentts.cpp tensor dump into the P00-T03 manifest/F32 file contract.

Common rules:

- The output directory must not already exist; do not overwrite.
- Missing executable/runner/model/codec/input or missing declared outputs is a non-zero failure.
- A child non-zero exit is a non-zero adapter failure with captured stderr evidence.
- Require a non-empty pinned revision; never infer `HEAD`.
- Write a version-1 `reference-run.json` with adapter kind, source, revision, model/case/seed,
  exact argv excluding secrets, child exit status, and every produced artifact's kind, relative
  path, byte length, and SHA-256.
- Record prompt, token, tensor and audio artifacts when produced. Do not claim a kind that was not
  produced.
- Reject unsafe relative paths, duplicates, symlinks escaping the session, malformed tensor
  headers, negative/overflowing shapes, trailing bytes, and empty required artifacts.
- Use stable error prefixes: `REFERENCE_CONFIG`, `REFERENCE_EXEC`, `REFERENCE_OUTPUT`,
  `REFERENCE_FORMAT`.
- Provide `--dry-run` that validates inputs and prints the exact JSON execution plan without
  creating files or launching a child.
- Do not download models, install packages, or use PyTorch inside the adapter/tests.

## Tests

Use temporary directories and tiny fake Python/executable scripts. Cover:

- exact argv construction and UTF-8 stdin for both adapters;
- output-directory overwrite refusal;
- missing executable/runner/model/codec/revision/input failure;
- child non-zero propagation and stderr capture;
- official binary token stdout capture and SHA-256 catalog entry;
- qwentts raw tensor header normalization with exact shape/F32 bytes/hash;
- malformed/truncated/overflow/trailing tensor dump rejection;
- required declared output missing or empty rejection;
- duplicate/unsafe/symlink-escape output rejection;
- dry-run performs no execution and no file I/O;
- manifest schema fields, deterministic artifact ordering and relative paths.

## Required commands

```text
python -m unittest tools.test_reference_adapter -v
python tools/reference_adapter.py official-python --help
python tools/reference_adapter.py qwentts-cpp --help
cargo fmt --all -- --check
cargo check --no-default-features --features cpu
```

## Acceptance

- Both real command shapes match the pinned source contracts.
- Tests prove fail-closed behavior without real weights.
- The adapters can catalog prompt/token/tensor/audio evidence and normalize qwentts tensors.
- No shell interpolation, implicit revision, silent skip, overwrite, absolute manifest path, or
  false success exists.
- Diff stays inside the four allowed files.

## Final report

Return summary, exact files, design choices, commands with exit codes, evidence, and risks.
