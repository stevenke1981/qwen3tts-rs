# Automatic release verification

Scope: spec.md 6.6 (product binaries) and 6.7 (verification and external weights).
Baseline: `a4f49ea` on GitHub `master`.

## Changes

- Build Windows x64 CPU GUI/CLI/converter and Linux x64 CPU CLI/converter.
- Test libraries and launch both packaged command-line tools before upload.
- Publish only after both builds succeed and archive hashes verify.
- Pushes to master produce unique prereleases; version tags produce releases.
- Fix CPU-only builds selecting an old CUDA executable from a prior build.
- Add converter help without loading model files.

## Local evidence

- `pwsh -NoProfile -File tests/build_cpu_only_test.ps1`: PASS. The actual build
  script runs with a fake compiler and an existing stale CUDA artifact; the
  selected default executable is the freshly generated CPU artifact.
- `cargo test --locked --lib`: 163 passed, 0 failed, 0 ignored.
- `cargo run --locked --bin convert-gguf -- --help`: exit 0.
- `cargo clippy --locked --lib --bins --examples`: exit 0 with existing lint debt.
  This is advisory, not a warnings-free certification.
- actionlint 1.7.7, YAML parsing and four embedded PowerShell blocks: PASS.
- Independent code review: accepted, no blocking findings.

## Remote evidence

Candidate workflow and CI runs are linked from the GitHub commit checks.
Every release includes its exact commit and run URL, archive checksums, and
`build-info.json` inside each archive. These are the authoritative records of
the binaries that were actually built and tested.

Release validation does not run model-dependent oracles, GUI interaction, CUDA,
speech-quality, or latency gates and does not advance the alignment status.
