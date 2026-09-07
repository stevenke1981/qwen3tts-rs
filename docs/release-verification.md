# Automatic release verification

Scope: spec.md 6.6 (product binaries) and 6.7 (verification and external weights).
Baseline: `a4f49ea` on GitHub `master`.

## Changes

- Build Windows x64 CPU and CUDA GUI/CLI/converter, plus Linux x64 CPU CLI/converter.
- Test libraries and launch both packaged command-line tools before upload.
- Publish only after all three builds and the Windows automatic bundle succeed
  and archive hashes verify.
- Pushes to master produce unique prereleases; version tags produce releases.
- Fix CPU-only builds selecting an old CUDA executable from a prior build.
- Add converter help without loading model files.
- Add asynchronous Auto/CPU/CUDA selection and actual synthesis device reporting.
- Probe a real CUDA kernel before accepting GPU selection; launch an independent
  CPU executable when the CUDA executable cannot load or its probe fails.

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
- `powershell.exe -NoProfile -ExecutionPolicy Bypass -File
  tests/automatic_launcher_test.ps1`: PASS on Windows PowerShell 5.1, including
  missing CUDA executable, failing probe, successful probe and explicit CPU.
- GitHub CI run `34093639233`: all four jobs passed, including the GUI device
  selection regression suite.
- `./build_all.ps1 -CudaOnly -ComputeCapability 86`: release build passed.
  `target/release/qwen3tts-gui-cuda.exe --probe-cuda`: exit 0 on RTX 3070 Ti,
  driver 596.36, CUDA toolkit 13.2. The probe executes and checks a small Candle
  CUDA affine operation, not just device-context creation. No model was loaded.
- `./build_all.ps1 -CpuOnly`: release build passed.
  `target/release/qwen3tts-gui-cpu.exe --probe-device`: exit 0, reports CPU.
  Both `qwen3tts-gui-cpu.exe` and `qwen3tts-gui-cuda.exe` are retained.
- Real release executables staged with `installer/Start-Qwen3TTS.ps1`:
  Windows PowerShell 5.1 `-ProbeOnly` selected CUDA successfully, and
  `-Device Cpu -ProbeOnly` selected the independent CPU executable.
- `cargo test --locked --no-default-features --features cpu,candle-llm,gui
  --test gui_device_selection_test`: 7 passed, 0 failed, 0 ignored locally.
- Candidate run `34094160652`: all three builds and automatic bundle passed.
  Downloaded Windows CPU and CUDA archives passed SHA-256 verification.
  Packaged CPU and CUDA GUI probes both exited 0. With CUDA Toolkit entries
  removed from the test process PATH, the packaged CUDA DLLs still supported
  the real-kernel probe and the launcher selected CUDA successfully.
- The MSVC environment command was independently reviewed and run locally:
  exit 0, 133 environment entries. This corrects the remote CUDA job's quoting
  failure; the corrected remote initialization passed in run `34094160652`.

## Remote evidence

- Mainline CI `34095044017`: success.
- Mainline release `34095044057`: all three builds, automatic bundle and publish
  succeeded. Source commit: `9e32568c2592f9aaa2f632add8bb620a6714f74d`.
- Published release: https://github.com/stevenke1981/qwen3tts-rs/releases/tag/preview-9e32568c2592-34095044057-1
- Downloaded runtime checks above used candidate `a4c666e`; the published source
  differs only in documentation. Published artifacts are certified by their
  own workflow checks, not claimed to be the exact locally executed binaries.

Candidate workflow and CI runs are linked from the GitHub commit checks.
Every release includes its exact commit and run URL, archive checksums, and
`build-info.json` inside each archive. These are the authoritative records of
the binaries that were actually built and tested.

Hosted release validation does not run model-dependent oracles, GUI interaction,
GPU runtime, speech-quality, or latency gates and does not advance alignment.
Local GUI process launch succeeded, but desktop screenshot capture failed with
an invalid-handle error; visual interaction is not certified.
