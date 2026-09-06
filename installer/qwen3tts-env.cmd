@echo off
REM Qwen3-TTS Rust 環境設定
REM 用法: qwen3tts-env.cmd

set "QWEN3_TTS_HOME=%~dp0"
set "PATH=%QWEN3_TTS_HOME%;%PATH%"

echo Qwen3-TTS Rust 環境已設定
echo   安裝目錄: %QWEN3_TTS_HOME%
echo.
echo 可用命令:
echo   qwen3tts-rs.exe --help
echo   qwen3tts-rs.exe --list-models
echo   qwen3tts-gui.exe
echo   convert-gguf.exe talker ^<model-dir^> ^<output.gguf^>
