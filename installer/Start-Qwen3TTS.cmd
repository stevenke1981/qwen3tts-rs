@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Qwen3TTS.ps1" %*
if errorlevel 1 pause
