[CmdletBinding()]
param(
    [ValidateSet('Auto', 'Cpu', 'Cuda')]
    [string]$Device = 'Auto',
    [switch]$ProbeOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$cpu = Join-Path $PSScriptRoot 'qwen3tts-rs-x86_64-pc-windows-msvc-cpu/qwen3tts-gui.exe'
$cuda = Join-Path $PSScriptRoot 'qwen3tts-rs-x86_64-pc-windows-msvc-cuda-sm86/qwen3tts-gui.exe'
$selected = $null
$reason = 'CPU selected by request.'

if ($Device -ne 'Cpu') {
    if (Test-Path -LiteralPath $cuda -PathType Leaf) {
        try {
            # Probe in a child process: a missing CUDA/driver DLL must not take
            # down the launcher. A bounded wait also covers a stalled driver.
            $probe = Start-Process -FilePath $cuda -ArgumentList '--probe-cuda' -WindowStyle Hidden -PassThru
            if (-not $probe.WaitForExit(30000)) {
                $probe.Kill()
                $probe.WaitForExit()
                $reason = 'CUDA initialization timed out; using CPU.'
            } elseif ($probe.ExitCode -eq 0) {
                $selected = $cuda
                $reason = 'CUDA GPU initialized successfully.'
            } else {
                $reason = "CUDA probe failed (exit $($probe.ExitCode)); using CPU."
            }
            $probe.Dispose()
        } catch {
            $reason = "CUDA executable could not start; using CPU. $($_.Exception.Message)"
        }
    } else {
        $reason = 'CUDA executable is missing; using CPU.'
    }
}

if (-not $selected) {
    if ($Device -eq 'Cuda') { throw $reason }
    if (-not (Test-Path -LiteralPath $cpu -PathType Leaf)) {
        throw 'CPU fallback executable is missing. Extract the complete automatic Windows bundle again.'
    }
    $selected = $cpu
}
Write-Output $reason
Write-Output "Executable: $selected"
if (-not $ProbeOnly) {
    # This is the user-facing GUI explicitly requested by launching this script.
    Start-Process -FilePath $selected -WorkingDirectory (Split-Path $selected) | Out-Null
}
