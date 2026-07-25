param(
  [Parameter(Mandatory=$true)][string]$TargetRepo,
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $PSScriptRoot
$TargetRepo = (Resolve-Path $TargetRepo).Path

if (-not (Test-Path (Join-Path $TargetRepo "Cargo.toml"))) {
  throw "TargetRepo does not look like qwen3tts-rs: Cargo.toml missing."
}

$marker = Join-Path $TargetRepo ".alignment-devpack"
if ((Test-Path $marker) -and -not $Force) {
  throw "Alignment package is already installed. Use -Force to refresh package files."
}

$copyItems = @(
  "AGENTS.md","TEAM.md","CONTROLLED_WORKFLOW.md","SOURCE_BASELINE.md","GAP_MATRIX.md",
  "SPEC.md","TARGET_ARCHITECTURE.md","IMPLEMENTATION_MAP.md","ACCEPTANCE_CRITERIA.md",
  "PERFORMANCE_BUDGET.md","TEST_PLAN.md","RISK_REGISTER.md","CLEAN_ROOM_RULES.md",
  "NOTICE.md","PLAN.md","TODOS.md","STATUS.md","agents","prompts","tasks","config",
  "schemas","tools","templates","docs",".github"
)

foreach ($item in $copyItems) {
  $src = Join-Path $PackageRoot $item
  if (-not (Test-Path $src)) { continue }
  $dst = Join-Path $TargetRepo $item
  if ((Test-Path $dst) -and ($item -in @("AGENTS.md","PLAN.md","TODOS.md","STATUS.md")) -and -not $Force) {
    Copy-Item $src "$dst.alignment-new" -Recurse -Force
    Write-Warning "$item exists; wrote $item.alignment-new instead."
  } else {
    Copy-Item $src $dst -Recurse -Force
  }
}

@"
installed_at=$(Get-Date -Format o)
package=qwen3tts-rs-full-alignment-devpack-v1.0
"@ | Set-Content -Encoding UTF8 $marker

Write-Host "Installed alignment package into $TargetRepo"
Write-Host "Next: cd $TargetRepo; powershell -File .\scripts\bootstrap.ps1"
