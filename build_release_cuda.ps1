[CmdletBinding()]
param(
    [ValidatePattern('^[0-9]{2,3}$')]
    [string]$ComputeCapability = $(if ($env:CUDA_COMPUTE_CAP) { $env:CUDA_COMPUTE_CAP } else { '86' })
)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'build_all.ps1') -CudaOnly -ComputeCapability $ComputeCapability
exit $LASTEXITCODE
