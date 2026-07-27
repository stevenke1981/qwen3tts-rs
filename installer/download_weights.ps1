# Qwen3-TTS Rust — 模型權重下載腳本
# 用法: powershell -ExecutionPolicy Bypass -File download_weights.ps1 [-InstallDir <path>] [-Model <0.6b-base|0.6b-customvoice|1.7b-voicedesign|all>]

param(
    [string]$InstallDir = "$env:LOCALAPPDATA\qwen3tts-rs",
    [string]$Model = "0.6b-base"
)

$ErrorActionPreference = "Stop"

# HuggingFace model IDs
$MODELS = @{
    "0.6b-base"         = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    "0.6b-customvoice"  = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    "1.7b-base"         = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    "1.7b-customvoice"  = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    "1.7b-voicedesign"  = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
}

$TOKENIZER_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

function Download-HFModel {
    param([string]$ModelId, [string]$DestDir)

    Write-Host ""
    Write-Host "=== 下載 $ModelId ===" -ForegroundColor Cyan

    # Use huggingface-cli if available, otherwise use Python
    $hfCli = Get-Command huggingface-cli -ErrorAction SilentlyContinue
    if ($hfCli) {
        Write-Host "使用 huggingface-cli..."
        huggingface-cli download $ModelId --local-dir $DestDir
    } else {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python) {
            Write-Host "錯誤: 找不到 python 或 huggingface-cli" -ForegroundColor Red
            Write-Host "請安裝: pip install huggingface_hub" -ForegroundColor Yellow
            exit 1
        }
        Write-Host "使用 Python huggingface_hub..."
        python -c @"
from huggingface_hub import snapshot_download
snapshot_download('$ModelId', local_dir=r'$DestDir')
print('Done: $DestDir')
"@
    }
}

# Create install directory
$modelsDir = Join-Path $InstallDir "models"
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
}

Write-Host ""
Write-Host "╔══════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   Qwen3-TTS Rust 權重下載           ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "安裝目錄: $InstallDir"
Write-Host "模型目錄: $modelsDir"
Write-Host "下載模型: $Model"

# Download selected model(s)
if ($Model -eq "all") {
    foreach ($key in $MODELS.Keys | Sort-Object) {
        $modelId = $MODELS[$key]
        $destDir = Join-Path $modelsDir ($key -replace '\.', '-')
        Download-HFModel -ModelId $modelId -DestDir $destDir
    }
} elseif ($MODELS.ContainsKey($Model)) {
    $modelId = $MODELS[$Model]
    $destDir = Join-Path $modelsDir ($Model -replace '\.', '-')
    Download-HFModel -ModelId $modelId -DestDir $destDir
} else {
    Write-Host "錯誤: 未知模型 '$Model'" -ForegroundColor Red
    Write-Host "可用模型: $($MODELS.Keys -join ', '), all" -ForegroundColor Yellow
    exit 1
}

# Download tokenizer (always needed)
$tokenizerDir = Join-Path $InstallDir "tokenizer-12hz"
if (-not (Test-Path (Join-Path $tokenizerDir "model.safetensors"))) {
    Download-HFModel -ModelId $TOKENIZER_ID -DestDir $tokenizerDir
} else {
    Write-Host ""
    Write-Host "=== Tokenizer 已存在，跳過 ===" -ForegroundColor Yellow
}

# Create environment setup script
$envScript = Join-Path $InstallDir "qwen3tts-env.ps1"
@"
# Qwen3-TTS Rust 環境變數
`$env:QWEN3_TTS_MODEL_DIR = "$modelsDir\$($Model -replace '\.', '-')"
`$env:QWEN3_TTS_TOKENIZER_DIR = "$tokenizerDir"
Write-Host "Qwen3-TTS 環境已設定" -ForegroundColor Green
Write-Host "  模型: `$env:QWEN3_TTS_MODEL_DIR"
Write-Host "  Tokenizer: `$env:QWEN3_TTS_TOKENIZER_DIR"
"@ | Out-File -FilePath $envScript -Encoding utf8

Write-Host ""
Write-Host "✅ 下載完成！" -ForegroundColor Green
Write-Host ""
Write-Host "使用方式:" -ForegroundColor Cyan
Write-Host "  1. 設定環境: . '$envScript'"
Write-Host "  2. 執行 TTS: qwen3tts-synthesize.exe --text `"你好`" --backend candle --model-dir `$env:QWEN3_TTS_MODEL_DIR"
Write-Host ""
