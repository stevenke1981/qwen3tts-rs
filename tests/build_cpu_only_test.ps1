# Regression: a CPU-only build must not restore a stale CUDA executable.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$sandbox = Join-Path ([IO.Path]::GetTempPath()) ('qwen-build-test-' + [guid]::NewGuid().ToString('N'))
$originalPath = $env:PATH
try {
    New-Item -ItemType Directory -Path (Join-Path $sandbox 'target/release') -Force | Out-Null
    Copy-Item (Join-Path $PSScriptRoot '../build_all.ps1') (Join-Path $sandbox 'build_all.ps1')
    Set-Content (Join-Path $sandbox 'target/release/qwen3tts-gui-cuda.exe') 'STALE_CUDA'
    # Fake only the compiler boundary; execute the actual build script.
    @'
@echo off
echo CPU>target\release\qwen3tts-gui.exe
echo CLI>target\release\qwen3tts-rs.exe
echo CONVERTER>target\release\convert-gguf.exe
exit /b 0
'@ | Set-Content (Join-Path $sandbox 'cargo.cmd')
    $env:PATH = $sandbox + [IO.Path]::PathSeparator + $originalPath
    & (Join-Path $sandbox 'build_all.ps1') -CpuOnly
    $actual = (Get-Content -Raw (Join-Path $sandbox 'target/release/qwen3tts-gui.exe')).Trim()
    if ($actual -ne 'CPU') { throw "CPU-only build selected wrong executable: $actual" }
    if ((Get-Content -Raw (Join-Path $sandbox 'target/release/qwen3tts-gui-cuda.exe')).Trim() -ne 'STALE_CUDA') {
        throw 'Existing CUDA artifact was unexpectedly modified.'
    }
    Write-Output 'PASS: CPU-only build selects freshly built CPU executable despite stale CUDA output.'
} finally {
    $env:PATH = $originalPath
    # Only this invocation's newly allocated test directory is removed.
    if (Test-Path -LiteralPath $sandbox) { Remove-Item -LiteralPath $sandbox -Recurse -Force }
}
