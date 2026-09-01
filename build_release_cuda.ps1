# Build Qwen3-TTS Release with NVIDIA CUDA acceleration
$ErrorActionPreference = "Stop"

Write-Host "=== Configuring MSVC C++ compiler for nvcc ===" -ForegroundColor Cyan
$msvcHostX64 = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64"
if (-not (Test-Path $msvcHostX64)) {
    $found = Get-ChildItem -Path "C:\Program Files*\Microsoft Visual Studio\**\Hostx64\x64\cl.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
        $msvcHostX64 = $found.DirectoryName
    }
}

Write-Host "Using MSVC at: $msvcHostX64"
$env:NVCC_CCBIN = $msvcHostX64
$env:PATH = "$msvcHostX64;" + $env:PATH
$env:CUDA_COMPUTE_CAP = "86"

Write-Host "=== Building qwen3tts-gui with CUDA feature ===" -ForegroundColor Green
cargo build --release --features cuda

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Release CUDA build successful: target\release\qwen3tts-gui.exe" -ForegroundColor Green
} else {
    Write-Host "❌ Build failed!" -ForegroundColor Red
}
