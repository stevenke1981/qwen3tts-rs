param(
    [string]$Version = "0.1.3",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DistRoot = Join-Path $RepoRoot "dist"
$PackageName = "qwen3tts-rs-v$Version-windows-x64"
$PackageDir = Join-Path $DistRoot $PackageName
$ZipPath = Join-Path $DistRoot "$PackageName.zip"

if (-not $SkipBuild) {
    Push-Location $RepoRoot
    try {
        cargo build --release --features candle-llm --example synthesize --example synthesize_batch --example convert_tokenizer
    } finally {
        Pop-Location
    }
}

New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null

if (Test-Path $PackageDir) {
    $resolvedDist = (Resolve-Path $DistRoot).Path
    $resolvedPackage = (Resolve-Path $PackageDir).Path
    if (-not $resolvedPackage.StartsWith($resolvedDist, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove package path outside dist: $resolvedPackage"
    }
    Remove-Item -LiteralPath $PackageDir -Recurse -Force
}
if (Test-Path $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageDir "tools") | Out-Null

$exampleDir = Join-Path $RepoRoot "target\release\examples"
$files = @(
    @{ Source = Join-Path $exampleDir "synthesize.exe"; Target = "synthesize.exe" },
    @{ Source = Join-Path $exampleDir "synthesize_batch.exe"; Target = "synthesize_batch.exe" },
    @{ Source = Join-Path $exampleDir "convert_tokenizer.exe"; Target = "convert_tokenizer.exe" },
    @{ Source = Join-Path $RepoRoot "docs\release_usage_zh.md"; Target = "README.zh-TW.md" },
    @{ Source = Join-Path $RepoRoot "docs\release_usage_en.md"; Target = "README.en-US.md" },
    @{ Source = Join-Path $RepoRoot "docs\agent_voice_failure_playbook.md"; Target = "AGENT_VOICE_FAILURE_PLAYBOOK.md" },
    @{ Source = Join-Path $RepoRoot "tools\convert_weights.py"; Target = "tools\convert_weights.py" }
)

foreach ($file in $files) {
    if (-not (Test-Path $file.Source)) {
        throw "Missing release input: $($file.Source)"
    }
    Copy-Item -LiteralPath $file.Source -Destination (Join-Path $PackageDir $file.Target) -Force
}

@"
qwen3tts-rs v$Version Windows x64

Included:
- synthesize.exe
- synthesize_batch.exe
- convert_tokenizer.exe
- README.zh-TW.md
- README.en-US.md
- AGENT_VOICE_FAILURE_PLAYBOOK.md
- tools/convert_weights.py

Large weights are not bundled. If converted tokenizer decoder weights are
missing, the app can attempt to run convert_tokenizer.exe automatically.
See README files before running.
"@ | Set-Content -LiteralPath (Join-Path $PackageDir "VERSION.txt") -Encoding UTF8

$hashes = Get-ChildItem -LiteralPath $PackageDir -File |
    Sort-Object Name |
    ForEach-Object {
        $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
        "$($hash.Hash)  $($_.Name)"
    }
$hashes | Set-Content -LiteralPath (Join-Path $PackageDir "SHA256SUMS.txt") -Encoding ASCII

Compress-Archive -LiteralPath $PackageDir -DestinationPath $ZipPath -Force

Write-Host "Release package created:"
Write-Host "  $PackageDir"
Write-Host "  $ZipPath"
