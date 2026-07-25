param(
  [string]$TargetRepo = "."
)
$ErrorActionPreference = "Stop"
Push-Location $TargetRepo
try {
  if (-not (Test-Path "Cargo.toml")) { throw "Run inside qwen3tts-rs." }
  New-Item -ItemType Directory -Force "artifacts/alignment/P00/P00-T01" | Out-Null
  git rev-parse HEAD | Set-Content "artifacts/alignment/P00/P00-T01/target-commit.txt"
  git status --short | Set-Content "artifacts/alignment/P00/P00-T01/git-status.txt"

  rustc --version | Set-Content "artifacts/alignment/P00/P00-T01/rustc.txt"
  cargo --version | Set-Content "artifacts/alignment/P00/P00-T01/cargo.txt"

  cargo fmt --all -- --check
  cargo check --all-targets
  cargo test --lib

  python tools/check_alignment_static.py . `
    --report artifacts/alignment/P00/P00-T01/static-gap-report.json

  Write-Host "Bootstrap completed. Sol must refresh the qwentts.cpp reference revision separately."
} finally {
  Pop-Location
}
