@echo off
setlocal

REM Compatibility wrapper. The PowerShell builder auto-detects Visual Studio,
REM validates nvcc, and accepts -ComputeCapability (default: 86).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_all.ps1" -CudaOnly %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
