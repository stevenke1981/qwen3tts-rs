# Alignment Status

- Overall: `IN_PROGRESS`
- Current phase: `P02`
- Current task: `P02-T03` (`READY`, assignment prepared)
- Target baseline commit: `b08178964504d5a214565ffc4ff5ed592eb8f7ec`
- qwentts.cpp reference commit: `82cd05b9f3a175612dc89fd6943e610fab096ef5`
- Official Qwen3-TTS commit: `022e286b98fbec7e1e916cb940cdf532cd9f488e`
- Working branch: `alignment/full-qwentts-parity`
- Last task gate: `P02-T02 GATE_PASSED`
- Last full phase gate: `P01 GATE_PASSED`

## Last Commands

```text
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -TargetRepo E:\qwen3tts-rs
cargo fmt --all
cargo fmt --all -- --check
cargo check --all-targets
cargo test --lib
python tools/check_alignment_static.py . --report artifacts/alignment/P00/P00-T01/static-gap-report.json
git clone https://github.com/ServeurpersoCom/qwentts.cpp.git E:\qwentts.cpp-reference
git ls-remote https://github.com/QwenLM/Qwen3-TTS.git HEAD
$env:PATH='C:\Users\steven\.cargo\bin;' + $env:PATH
$env:QWEN3_TTS_REAL_MODEL_DIR='C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc'
C:\Users\steven\Qwen3-TTS\.venv\Scripts\python.exe tools/generate_prompt_id_matrix.py --help
C:\Users\steven\Qwen3-TTS\.venv\Scripts\python.exe tools/generate_prompt_id_matrix.py --check fixtures/alignment/p01_prompt_id_matrix.json
cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo test --test prompt_id_matrix_real_test -- --ignored --nocapture
cargo test --lib text_frontend
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
git diff --check -- src/text_frontend/prompt_templates.rs src/text_frontend/model_catalog.rs tests/prompt_id_matrix_real_test.rs tools/generate_prompt_id_matrix.py fixtures/alignment/p01_prompt_id_matrix.json docs/alignment/prompt-id-matrix.md config/fixtures.json
```

## Results

- `cargo fmt --all -- --check`: PASS after mechanical rustfmt normalization.
- `cargo check --all-targets`: PASS with one existing `unused_mut` warning in
  `tests/debug_per_layer_compare.rs`.
- `cargo test --lib`: PASS, 82 passed, 0 failed, with MSVC `LNK4098` warning.
- Static gap scan: completed; expected P0 gaps remain and are documented in
  `docs/alignment/current-implementation-audit.md`.
- P00-T02 fixture resolver: 11/11 tests passed; the empty real-fixture manifest fails closed with
  `FIXTURE_MISSING` as required.
- P00-T03 stage dump: feature-off check passed, 82/82 library tests passed, and 4/4
  stage-dump tests passed. Independent GPT-5.3 Codex Spark review returned `ACCEPT`.
- P00-T04 reference adapters: 18/18 unit tests passed with one platform-only symlink
  creation skip; both CLI help contracts, Rust format/CPU checks, and installed-runner
  dry-run passed. Independent GPT-5.3 Codex Spark review returned `ACCEPT`.
- P00-T05 CPU/F32 baseline: official Python and qwentts.cpp real-weight runs produced
  hashed prompt/token/tensor/audio evidence. The baseline report passed schema validation,
  the fixture release check verified 8/8 required entries, and independent GPT-5.3
  Codex Spark review returned `ACCEPT`.
- P01-T01 metadata parsing: runtime capability and Auto-mode resolution now come from
  fail-closed official `config.json` metadata. The real 0.6B Base test ran 1/1 and both
  synthesis examples compiled. Independent GPT-5.3 Codex Spark review returned `ACCEPT`.
- P01-T02 runtime metadata: every prompt-affecting token, language, speaker, and
  heterogeneous dialect entry is validated and installed into loaded Candle models.
  Tests passed 16 metadata, 5 config, 7 input-builder, and 1 real checkpoint case.
  Independent GPT-5.3 Codex Spark review returned `ACCEPT`.
- P01-T03 M-RoPE parity: metadata `rope_theta`, interleaving and section values now reach
  the Candle runtime; prefill positions, padding, deltas, cached decode positions, and
  both channel layouts have exact tests. CPU checks, 7 integration tests, both examples,
  formatting and scoped diff checks passed. Independent GPT-5.3 Codex Spark review
  returned `ACCEPT`.
- P01-T04 prompt assembly: production prompt helpers now match all three official wrappers;
  reference text uses the official assistant role and `[3:-2]` slice; direct Candle calls
  enforce metadata-backed mode combinations; Base x-vector-only and ICL paths are distinct.
  Root gates passed 32 text-frontend tests, 7 input-builder tests, 8 prompt tests, 7 native
  voice-clone tests and both Candle examples. Independent review found no blockers.
- P01-T05 prompt-ID matrix: official oracle now checkpoints five-model matrix case IDs
  (main/reference/instruction/body-slice) directly against Rust production helpers under
  real-tokenizer gate; Python oracle help/check, format, CPU compile, real-tokenizer test,
  text_frontend tests, both examples, and scoped diff check passed. Independent review
  returned `ACCEPT`.
- Source revisions: target and qwentts.cpp match the packaged baseline; no baseline delta file
  was required. Official semantics revision is now pinned.

## Failure Records

- F1 Toolchain: first Bootstrap attempt could not find `rustc` on process `PATH`.
  Reproduced by the original Bootstrap command. Resolved non-globally by prepending
  `C:\Users\steven\.cargo\bin` to the child process `PATH`.
- F7 Product/Tooling: `scripts/bootstrap.ps1` does not fail closed on native command non-zero
  exit codes. Independent commands were rerun with explicit `$LASTEXITCODE` checks.
- F8 Fixture: real-weight tests currently contain silent skip paths; not accepted as parity
  evidence. P00-T02 established the resolver, but real fixture entries and migration of existing
  test callers remain P00-T04/P00-T05 work.
- F6 Backend: the mandated `--all-features` commit gate cannot compile on Windows because enabling
  the Metal feature pulls `objc2`, which rejects non-Apple targets. `cargo fmt` passed, while
  clippy/test both exited 101 at `objc2`. Do not interpret this as a CPU or CUDA regression;
  platform-aware feature gating must be fixed before the final commit gate.

## Blockers

- No blocker for P01-T05 implementation.
- No blocker for P01-T05 gate.
- The P00 phase gate is passed. The 3-vs-4 generated-frame baseline delta remains
  intentional evidence for the later prompt/sampling parity phases.

## Evidence

- `artifacts/alignment/P00/P00-T01/`
- `docs/alignment/current-implementation-audit.md`
- `artifacts/alignment/P00/P00-T02/assignment.md`
- `artifacts/alignment/P00/P00-T02/gate.json`
- `artifacts/alignment/P00/P00-T03/gate.json`
- `artifacts/alignment/P00/P00-T04/gate.json`
- `artifacts/alignment/P00/P00-T05/gate.json`
- `artifacts/alignment/P00/P00-T05/cpu-f32-baseline.json`
- `artifacts/alignment/P00/P00-T05/model-provenance.json`
- `artifacts/alignment/P01/P01-T01/gate.json`
- `artifacts/alignment/P01/P01-T02/gate.json`
- `artifacts/alignment/P01/P01-T03/gate.json`
- `artifacts/alignment/P01/P01-T04/gate.json`
- `artifacts/alignment/P01/P01-T05/gate.json`
- `artifacts/alignment/P01/P01-T05/commands.txt`
- `artifacts/alignment/P01/P01-T05/test-results.txt`
- `artifacts/alignment/P01/P01-T05/worker-report.md`
- `artifacts/alignment/P01/P01-T05/review.md`
- `fixtures/alignment/p01_prompt_id_matrix.json`
- `tools/generate_prompt_id_matrix.py`
- `src/text_frontend/model_catalog.rs`
- `src/text_frontend/prompt_templates.rs`
- `tests/prompt_id_matrix_real_test.rs`
- `config/fixtures.json`
- `config/model-matrix.json`
