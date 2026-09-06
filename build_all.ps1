[CmdletBinding()]
param(
    [ValidatePattern('^[0-9]{2,3}$')]
    [string]$ComputeCapability = $(if ($env:CUDA_COMPUTE_CAP) { $env:CUDA_COMPUTE_CAP } else { '86' }),

    [switch]$CpuOnly,
    [switch]$CudaOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($CpuOnly -and $CudaOnly) {
    throw '-CpuOnly 與 -CudaOnly 不可同時使用。'
}

$repoRoot = $PSScriptRoot
$cargoCommand = Get-Command cargo -ErrorAction Stop
$cargoPath = $cargoCommand.Source

function Invoke-CargoBuild {
    param([Parameter(Mandatory)][string[]]$Arguments)

    Write-Host "`n> cargo $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $script:cargoPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Cargo command failed with exit code $LASTEXITCODE."
    }
}

function Import-VisualStudioEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return
    }

    $vswhereCommand = Get-Command vswhere.exe -ErrorAction SilentlyContinue
    $vswhereExecutable = if ($vswhereCommand) { $vswhereCommand.Source } else { $null }
    if (-not $vswhereExecutable) {
        $candidates = @()
        if (${env:ProgramFiles(x86)}) {
            $candidates += Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
        }
        if ($env:ProgramFiles) {
            $candidates += Join-Path $env:ProgramFiles 'Microsoft Visual Studio\Installer\vswhere.exe'
        }
        $vswhereExecutable = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    }

    if (-not $vswhereExecutable) {
        throw '找不到 vswhere.exe。請安裝 Visual Studio 2022 Build Tools 與「Desktop development with C++」。'
    }

    $installationPath = & $vswhereExecutable -latest -products '*' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $installationPath) {
        throw '找不到含 MSVC x64 工具鏈的 Visual Studio 安裝。'
    }

    $vcvars = Join-Path ($installationPath | Select-Object -First 1) 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) {
        throw "找不到 vcvars64.bat：$vcvars"
    }

    Write-Host "載入 MSVC 開發環境：$vcvars" -ForegroundColor DarkGray
    $environmentLines = & $env:ComSpec /d /s /c "`"$vcvars`" >nul && set"
    if ($LASTEXITCODE -ne 0) {
        throw "vcvars64.bat 執行失敗，exit code $LASTEXITCODE。"
    }

    foreach ($line in $environmentLines) {
        $parts = $line -split '=', 2
        if ($parts.Count -eq 2 -and $parts[0] -and -not $parts[0].StartsWith('=')) {
            [Environment]::SetEnvironmentVariable($parts[0], $parts[1], 'Process')
        }
    }

    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw 'MSVC 環境已載入，但仍找不到 cl.exe。'
    }
}

Push-Location $repoRoot
try {
    Write-Host '============================================================' -ForegroundColor Cyan
    Write-Host ' Qwen3-TTS Rust release builder' -ForegroundColor Cyan
    Write-Host '============================================================' -ForegroundColor Cyan

    # CLI 與轉換工具不依賴 GUI，可先建立可攜 CPU 版本。
    Write-Host "`n[Common] 建置 CLI 與 GGUF 轉換工具..." -ForegroundColor Yellow
    Invoke-CargoBuild -Arguments @(
        'build', '--locked', '--release',
        '--no-default-features', '--features', 'cpu,candle-llm',
        '--bin', 'qwen3tts-rs', '--bin', 'convert-gguf'
    )

    $releaseDir = Join-Path $repoRoot 'target\release'
    $cpuOutput = Join-Path $releaseDir 'qwen3tts-gui-cpu.exe'
    $cudaOutput = Join-Path $releaseDir 'qwen3tts-gui-cuda.exe'
    $defaultOutput = Join-Path $releaseDir 'qwen3tts-gui.exe'

    if (-not $CpuOnly) {
        Write-Host "`n[CUDA] 檢查 CUDA Toolkit 與 MSVC..." -ForegroundColor Yellow
        Import-VisualStudioEnvironment

        $nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
        if (-not $nvcc) {
            $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
        }
        if (-not $nvcc) {
            throw '找不到 nvcc。請安裝 CUDA Toolkit，並將其 bin 目錄加入 PATH。'
        }

        $cl = Get-Command cl.exe -ErrorAction Stop
        $env:NVCC_CCBIN = Split-Path -Parent $cl.Source
        $env:CUDA_COMPUTE_CAP = $ComputeCapability

        Write-Host "CUDA compiler：$($nvcc.Source)" -ForegroundColor DarkGray
        Write-Host "MSVC compiler：$($cl.Source)" -ForegroundColor DarkGray
        Write-Host "Compute capability：$ComputeCapability" -ForegroundColor DarkGray

        Invoke-CargoBuild -Arguments @(
            'build', '--locked', '--release',
            '--no-default-features', '--features', 'cpu,candle-llm,gui,cuda',
            '--bin', 'qwen3tts-gui'
        )
        Copy-Item $defaultOutput $cudaOutput -Force
        Write-Host "CUDA GUI：$cudaOutput" -ForegroundColor Green
    }

    if (-not $CudaOnly) {
        Write-Host "`n[CPU] 建置不依賴 CUDA 的 GUI..." -ForegroundColor Yellow
        Invoke-CargoBuild -Arguments @(
            'build', '--locked', '--release',
            '--no-default-features', '--features', 'cpu,candle-llm,gui',
            '--bin', 'qwen3tts-gui'
        )
        Copy-Item $defaultOutput $cpuOutput -Force
        Write-Host "CPU GUI：$cpuOutput" -ForegroundColor Green
    }

    # CPU build 會覆寫原始檔名；有 CUDA 產物時，讓預設檔名恢復為 CUDA/CPU 自動回退版本。
    if (Test-Path $cudaOutput) {
        Copy-Item $cudaOutput $defaultOutput -Force
    } elseif (Test-Path $cpuOutput) {
        Copy-Item $cpuOutput $defaultOutput -Force
    }

    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host ' 建置完成' -ForegroundColor Green
    Write-Host " CLI：$(Join-Path $releaseDir 'qwen3tts-rs.exe')"
    Write-Host " GGUF converter：$(Join-Path $releaseDir 'convert-gguf.exe')"
    if (Test-Path $defaultOutput) { Write-Host " GUI：$defaultOutput" }
    if (Test-Path $cudaOutput) { Write-Host " CUDA GUI：$cudaOutput" }
    if (Test-Path $cpuOutput) { Write-Host " CPU GUI：$cpuOutput" }
    Write-Host '============================================================' -ForegroundColor Cyan
}
finally {
    Pop-Location
}
