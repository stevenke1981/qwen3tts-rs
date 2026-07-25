# Reference Command Adapters (P00-T04)

This task adds two fail-closed adapters in `tools/reference_adapter.py`:

- `official-python`
- `qwentts-cpp`

Each adapter writes one manifest per run: `reference-run.json` in `--output-dir`.

## Common contract

- `--source-revision` must be present and non-empty.
- `--output-dir` must not already exist.
- Child processes are executed via explicit `argv`, never through a shell.
- Extra arguments that could override adapter-owned paths or expose credentials are rejected.
- Artifact paths are validated as safe relative paths (no traversal, no absolute paths, no unsafe chars, no separator injection, no symlink escape).
- `--dry-run` validates inputs and prints exact execution plan JSON without creating files or launching a child process.
- Stable failure prefixes:
  - `REFERENCE_CONFIG`
  - `REFERENCE_EXEC`
  - `REFERENCE_OUTPUT`
  - `REFERENCE_FORMAT`
- Failure paths persist non-empty captured logs:
  - `logs/stdout.bin`
  - `logs/stderr.txt`
- Successful runs persist stderr log only when stderr is non-empty and not declared via `--artifact stderr`. Official success runs usually write `token`/`stdout` as primary output, so `stdout` log is not duplicated.

Manifest fields (`reference-run.json`):

- `schema_version`, `kind`, `source`, `revision`, `model`, `case_id`, `seed`,
  `argv`, `exit_status`, `artifacts`, `stages`
- `artifacts` include kind, `relative_path`, `byte_length`, and `sha256`; tensor artifacts also include `shape`, `dtype`, `layout`.
- On non-zero child exit, manifest is still written with `exit_status` and log artifacts (`kind` `stdout`, `stderr`) under `logs/*`.
- `stages` lists normalized tensor manifest entries from tensor dumps.

## `official-python`

Command form:

```text
python <runner> --model ... --language ... [--seed ...] [--max-new-tokens|--max-new ...]
                [--top-k ...] [--top-p ...] [--temperature ...] [--greedy]
                [--speaker ...] [--instruct ...]
                (--text ... | --text-file ...)
                [--output-wav <path> if audio declared]
                [--dump <session>/raw_tensor_dump if tensor declared]
                [extra args...]
```

- stdin is not used; text is passed through args.
- stdout is captured as the `stdout` artifact (binary).
- `token` is accepted as alias for `stdout`.
- prompt artifact is the adapter input text.
- stderr is included in `REFERENCE_EXEC` messages on failure.

## `qwentts-cpp`

Command form:

```text
<executable> --model ... --codec ... [--lang ...] [--seed ...] [--max-new ...] [--greedy]
  [--speaker ...] [--instruct ...] [--ref-wav ...] [--ref-text ...]
  [extra args...]
  --dump <session>/raw_tensor_dump -o <session>/output.wav
```

- text is piped through UTF-8 stdin.
- the adapter creates the raw dump directory before launch because qwentts.cpp expects it to exist.
- when tensor output is present, `raw_tensor_dump` is normalized to `.f32.bin` files and each dump
  also gets a per-directory `manifest.json` with a `stages` list.

## Declared outputs

- declare with `--artifact kind=path`, where `kind` is one of:
  - `prompt`
  - `stdout` / `token` (alias)
  - `stderr`
  - `audio`
  - `tensor`
- Defaults (when not declared):
  - official-python: `prompt.txt`, `tokens.bin`
  - qwentts-cpp: `prompt.txt`, `output.wav`, `tensors`
- if declared but not produced/empty, adapter exits with `REFERENCE_OUTPUT`.
- if a declared artifact has duplicate kind/path, adapter exits with `REFERENCE_OUTPUT`.

## Determinism

- artifacts are deterministic and sorted by `relative_path`.
- manifest entries in `stages` keep discovery order and normalized file names.
