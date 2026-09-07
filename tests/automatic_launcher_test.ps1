Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$sandbox = Join-Path ([IO.Path]::GetTempPath()) ('qwen-launch-test-' + [guid]::NewGuid().ToString('N'))
try {
    $cpuDir = Join-Path $sandbox 'qwen3tts-rs-x86_64-pc-windows-msvc-cpu'
    New-Item -ItemType Directory -Path $cpuDir -Force | Out-Null
    Copy-Item (Join-Path $PSScriptRoot '../installer/Start-Qwen3TTS.ps1') $sandbox
    # ProbeOnly must select the CPU fallback without launching this marker.
    Set-Content (Join-Path $cpuDir 'qwen3tts-gui.exe') 'not an executable'
    $launcher = Join-Path $sandbox 'Start-Qwen3TTS.ps1'
    $result = & $launcher -ProbeOnly
    if (($result -join "`n") -notmatch 'CPU' -or ($result -join "`n") -notmatch '-cpu') {
        throw 'Auto failed to select CPU with absent CUDA binary.'
    }
    & $launcher -Device Cpu -ProbeOnly
    $rejected = $false
    try { & $launcher -Device Cuda -ProbeOnly } catch { $rejected = $true }
    if (-not $rejected) { throw 'Forced CUDA must reject missing CUDA binary.' }
    Write-Output 'PASS: automatic and explicit CPU selection; forced CUDA rejects unavailable executable.'
} finally {
    if (Test-Path -LiteralPath $sandbox) { Remove-Item -LiteralPath $sandbox -Recurse -Force }
}
