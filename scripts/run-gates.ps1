param(
  [Parameter(Mandatory=$true)][ValidatePattern("^P\d\d$")][string]$Phase,
  [string]$Repo = "."
)
$ErrorActionPreference = "Stop"
Push-Location $Repo
try {
  cargo fmt --all -- --check
  cargo clippy --all-targets --all-features -- -D warnings
  cargo test --all-targets --all-features

  $phaseNumber = [int]$Phase.Substring(1)
  if ($phaseNumber -ge 5) {
    if (-not (Test-Path "artifacts/alignment/$Phase")) {
      throw "Missing evidence directory for $Phase"
    }
  }
  if ($phaseNumber -ge 8) {
    python tools/check_alignment_static.py . --report "artifacts/alignment/$Phase/static-report.json"
  }
  Write-Host "$Phase base gate passed. Run the phase-specific commands in its task card."
} finally {
  Pop-Location
}
