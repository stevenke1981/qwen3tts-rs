# ==============================================================================
# Qwen3-TTS 一鍵編譯腳本 (CUDA GPU + 純 CPU 雙版本)
# ==============================================================================
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " 🚀 Qwen3-TTS 雙版本編譯 (CUDA GPU 加速版 + 純 CPU 輕量版)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# 1. 配置 MSVC C++ 編譯器環境（供 nvcc 使用）
# ------------------------------------------------------------------------------
Write-Host "`n[1/3] 偵測並配置 MSVC C++ 編譯器環境..." -ForegroundColor Yellow
$msvcHostX64 = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64"
if (-not (Test-Path $msvcHostX64)) {
    $found = Get-ChildItem -Path "C:\Program Files*\Microsoft Visual Studio\**\Hostx64\x64\cl.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
        $msvcHostX64 = $found.DirectoryName
    }
}
Write-Host "使用 MSVC 路徑: $msvcHostX64"
$env:NVCC_CCBIN = $msvcHostX64
$env:PATH = "$msvcHostX64;" + $env:PATH
$env:CUDA_COMPUTE_CAP = "86"

# ------------------------------------------------------------------------------
# 2. 編譯 NVIDIA CUDA GPU 加速版本
# ------------------------------------------------------------------------------
Write-Host "`n[2/3] 編譯 CUDA GPU 加速版 (Release)..." -ForegroundColor Green
cargo build --release --bin qwen3tts-gui --features cuda
if ($LASTEXITCODE -eq 0) {
    Copy-Item -Path "target\release\qwen3tts-gui.exe" -Destination "target\release\qwen3tts-gui-cuda.exe" -Force
    Write-Host "✅ CUDA GPU 版本產出成功: target\release\qwen3tts-gui-cuda.exe" -ForegroundColor Green
} else {
    Write-Host "❌ CUDA 版本編譯失敗!" -ForegroundColor Red
    exit 1
}

# ------------------------------------------------------------------------------
# 3. 編譯 純 CPU 輕量版本（無 GPU 依賴，可在任何 x64 Windows 執行）
# ------------------------------------------------------------------------------
Write-Host "`n[3/3] 編譯 純 CPU 輕量版 (Release, No-CUDA)..." -ForegroundColor Green
cargo build --release --bin qwen3tts-gui --no-default-features --features "cpu,candle-llm"
if ($LASTEXITCODE -eq 0) {
    Copy-Item -Path "target\release\qwen3tts-gui.exe" -Destination "target\release\qwen3tts-gui-cpu.exe" -Force
    Write-Host "✅ 純 CPU 版本產出成功: target\release\qwen3tts-gui-cpu.exe" -ForegroundColor Green
} else {
    Write-Host "❌ 純 CPU 版本編譯失敗!" -ForegroundColor Red
    exit 1
}

# ------------------------------------------------------------------------------
# 4. 再次確保預設 qwen3tts-gui.exe 為具備自動回退的 CUDA 版本
# ------------------------------------------------------------------------------
Copy-Item -Path "target\release\qwen3tts-gui-cuda.exe" -Destination "target\release\qwen3tts-gui.exe" -Force

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " 🎉 全部編譯完成！產出檔案清單：" -ForegroundColor Cyan
Write-Host " 1. target\release\qwen3tts-gui.exe      (預設版：具備 GPU/CPU 自動偵測與平滑回退)" -ForegroundColor White
Write-Host " 2. target\release\qwen3tts-gui-cuda.exe (專屬版：CUDA GPU 硬體加速)" -ForegroundColor White
Write-Host " 3. target\release\qwen3tts-gui-cpu.exe  (輕量版：純 CPU，免安裝 CUDA 驅動)" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan
