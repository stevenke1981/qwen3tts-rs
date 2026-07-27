@echo off
REM Qwen3-TTS Rust Windows 安裝檔建置腳本
REM
REM 前置需求:
REM   1. Rust toolchain (rustup.rs)
REM   2. Inno Setup 6 (jrsoftware.org/isinfo.php)
REM
REM 用法:
REM   installer\build_installer.cmd

setlocal enabledelayedexpansion

echo ╔══════════════════════════════════════╗
echo ║   Qwen3-TTS Rust 安裝檔建置         ║
echo ╚══════════════════════════════════════╝
echo.

REM ── Step 1: Build release binaries ──
echo [1/3] 建置 release 二進位檔...
cargo build --release --features candle-llm --bin qwen3tts-gui --bin convert-gguf --example synthesize
if errorlevel 1 (
    echo 錯誤: cargo build 失敗
    exit /b 1
)
echo   ✓ qwen3tts-gui.exe
echo   ✓ convert-gguf.exe
echo   ✓ synthesize.exe

REM ── Step 2: Verify binaries exist ──
echo.
echo [2/3] 驗證二進位檔...
set "MISSING=0"
for %%f in (
    target\release\qwen3tts-gui.exe
    target\release\convert-gguf.exe
    target\release\examples\synthesize.exe
) do (
    if not exist "%%f" (
        echo   ✗ 缺少: %%f
        set "MISSING=1"
    ) else (
        echo   ✓ %%f
    )
)
if "%MISSING%"=="1" (
    echo 錯誤: 缺少二進位檔
    exit /b 1
)

REM ── Step 3: Compile Inno Setup installer ──
echo.
echo [3/3] 編譯 Inno Setup 安裝檔...

REM Find Inno Setup
set "ISCC="
for %%d in (
    "%PROGRAMFILES(X86)%\Inno Setup 6\ISCC.exe"
    "%PROGRAMFILES%\Inno Setup 6\ISCC.exe"
    "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
) do (
    if exist %%d (
        set "ISCC=%%~d"
        goto :found_iscc
    )
)

echo 錯誤: 找不到 Inno Setup 6
echo 請從 https://jrsoftware.org/isdl.php 下載安裝
exit /b 1

:found_iscc
echo   使用: %ISCC%

REM Create dist directory
if not exist dist mkdir dist

REM Compile installer
"%ISCC%" installer\setup.iss
if errorlevel 1 (
    echo 錯誤: Inno Setup 編譯失敗
    exit /b 1
)

echo.
echo ╔══════════════════════════════════════╗
echo ║   ✅ 安裝檔建置完成！               ║
echo ╚══════════════════════════════════════╝
echo.
echo 輸出: dist\qwen3tts-rs-0.2.0-setup.exe
echo.
echo 散佈方式:
echo   將 dist\qwen3tts-rs-0.2.0-setup.exe 提供給使用者
echo   安裝程式會引導下載模型權重（約 1.8 GB）

endlocal
