Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$sandbox = Join-Path ([IO.Path]::GetTempPath()) ('qwen-launch-test-' + [guid]::NewGuid().ToString('N'))
$oldExitCode = $env:QWEN_LAUNCHER_TEST_EXITCODE
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
    $cudaDir = Join-Path $sandbox 'qwen3tts-rs-x86_64-pc-windows-msvc-cuda-sm86'
    New-Item -ItemType Directory -Path $cudaDir | Out-Null
    $compiler = Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    & $compiler /nologo /target:exe "/out:$cudaDir/qwen3tts-gui.exe" (Join-Path $PSScriptRoot 'fixtures/launcher_probe.cs')
    if ($LASTEXITCODE -ne 0) { throw 'Probe test fixture compilation failed.' }
    $env:QWEN_LAUNCHER_TEST_EXITCODE = '1'
    $result = & $launcher -ProbeOnly
    if (($result -join "`n") -notmatch 'msvc-cpu') { throw 'Failed CUDA probe did not select CPU.' }
    $env:QWEN_LAUNCHER_TEST_EXITCODE = '0'
    $result = & $launcher -ProbeOnly
    if (($result -join "`n") -notmatch 'msvc-cuda-sm86') { throw 'Successful CUDA probe did not select CUDA.' }
    Write-Output 'PASS: missing and failing CUDA fall back; successful CUDA is selected; explicit modes respected.'
} finally {
    $env:QWEN_LAUNCHER_TEST_EXITCODE = $oldExitCode
    if (Test-Path -LiteralPath $sandbox) { Remove-Item -LiteralPath $sandbox -Recurse -Force }
}
