param(
    [string]$Version = "0.1.8",
    [switch]$Cuda,
    [string]$CudaComputeCap = "86",
    [string]$CudaToolkitRoot = "",
    [string]$VcVarsVersion = "14.29",
    [string]$CudaForgeThreads = "1",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DistRoot = Join-Path $RepoRoot "dist"
$TargetName = if ($Cuda) { "windows-x64-cuda" } else { "windows-x64" }
$FeatureList = if ($Cuda) { "candle-llm cuda" } else { "candle-llm" }
$PackageName = "qwen3tts-rs-v$Version-$TargetName"
$PackageDir = Join-Path $DistRoot $PackageName
$ZipPath = Join-Path $DistRoot "$PackageName.zip"

function Get-VcVars64Path {
    $preferred = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $preferred) {
        return $preferred
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($install) {
            $candidate = Join-Path $install "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $candidate) {
                return $candidate
            }
        }
    }

    $roots = @(
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio"),
        (Join-Path ${env:ProgramFiles} "Microsoft Visual Studio")
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) {
            continue
        }
        $candidate = Get-ChildItem -LiteralPath $root -Recurse -Filter vcvars64.bat -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty FullName
        if ($candidate) {
            return $candidate
        }
    }
    return $null
}

function Get-CudaToolkitRoot {
    if ($CudaToolkitRoot) {
        if (-not (Test-Path (Join-Path $CudaToolkitRoot "bin\nvcc.exe"))) {
            throw "CUDA toolkit root does not contain bin\nvcc.exe: $CudaToolkitRoot"
        }
        return $CudaToolkitRoot
    }

    $preferred = Join-Path ${env:ProgramFiles} "NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    if (Test-Path (Join-Path $preferred "bin\nvcc.exe")) {
        return $preferred
    }
    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) {
        return $env:CUDA_PATH
    }
    $nvcc = (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source
    if ($nvcc) {
        return (Resolve-Path (Join-Path (Split-Path $nvcc -Parent) "..")).Path
    }
    throw "CUDA toolkit not found. Pass -CudaToolkitRoot <path>."
}

function Get-ClPath {
    $roots = @(
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"),
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC")
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) {
            continue
        }
        $patterns = if ($VcVarsVersion) { @("$VcVarsVersion*") } else { @("*") }
        foreach ($pattern in $patterns) {
            $candidate = Get-ChildItem -LiteralPath $root -Directory -Filter $pattern -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                ForEach-Object { Join-Path $_.FullName "bin\HostX64\x64\cl.exe" } |
                Where-Object { Test-Path $_ } |
                Select-Object -First 1
            if ($candidate) {
                return $candidate
            }
        }
    }
    return $null
}

function Invoke-ReleaseBuild {
    Push-Location $RepoRoot
    try {
        if ($Cuda) {
            $vcvars = Get-VcVars64Path
            if (-not $vcvars) {
                throw "CUDA build requires MSVC Build Tools with vcvars64.bat in PATH or Visual Studio installation."
            }
            $cudaRoot = Get-CudaToolkitRoot
            $nvcc = Join-Path $cudaRoot "bin\nvcc.exe"
            $cl = Get-ClPath
            if (-not $cl) {
                throw "CUDA build requires cl.exe. Install MSVC Build Tools or pass a compatible -VcVarsVersion."
            }
            $clDir = Split-Path $cl -Parent
            $vcvarsArgs = if ($VcVarsVersion) { " -vcvars_ver=$VcVarsVersion" } else { "" }
            Write-Host "Using MSVC dev shell: $vcvars"
            Write-Host "Using cl.exe: $cl"
            Write-Host "Using CUDA toolkit: $cudaRoot"
            Write-Host "CUDA_COMPUTE_CAP=$CudaComputeCap"
            $buildCommand = "call `"$vcvars`"$vcvarsArgs && set CUDA_COMPUTE_CAP=$CudaComputeCap&& set CUDAFORGE_THREADS=$CudaForgeThreads&& set RAYON_NUM_THREADS=$CudaForgeThreads&& set CUDA_ROOT=$cudaRoot&& set CUDA_PATH=$cudaRoot&& set NVCC=$nvcc&& set NVCC_CCBIN=$cl&& set PATH=$clDir;$cudaRoot\bin;%PATH%&& cargo build --release --features `"$FeatureList`" --example synthesize --example synthesize_batch --example convert_tokenizer"
            & cmd.exe /d /c $buildCommand
            if ($LASTEXITCODE -ne 0) {
                throw "cargo CUDA release build failed with exit code $LASTEXITCODE"
            }
        } else {
            cargo build --release --features $FeatureList --example synthesize --example synthesize_batch --example convert_tokenizer
        }
    } finally {
        Pop-Location
    }
}

if (-not $SkipBuild) {
    Invoke-ReleaseBuild
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
qwen3tts-rs v$Version $TargetName

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
CUDA enabled: $([bool]$Cuda)
CUDA compute capability: $(if ($Cuda) { $CudaComputeCap } else { "n/a" })
CUDA toolkit: $(if ($Cuda) { Get-CudaToolkitRoot } else { "n/a" })
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
