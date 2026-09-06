@echo off
REM Qwen3-TTS Rust Windows 安裝檔建置腳本
REM
REM 前置需求:
REM   1. Rust 1.85+ toolchain (rustup.rs)
REM   2. Visual Studio C++ runtime/build tools
REM   3. Inno Setup 6 (jrsoftware.org/isinfo.php)
REM
REM 用法:
REM   installer\build_installer.cmd

setlocal enabledelayedexpansion
pushd "%~dp0.."

echo ============================================================
echo  Qwen3-TTS Rust Windows installer builder
echo ============================================================
echo.

REM -- Step 1: Build release binaries --
echo [1/3] 建置 CPU release 二進位檔...
cargo build --locked --release --no-default-features --features "cpu,candle-llm,gui" --bin qwen3tts-gui --bin qwen3tts-rs --bin convert-gguf
if errorlevel 1 (
    echo 錯誤: cargo build 失敗
    popd
    exit /b 1
)
echo   [OK] qwen3tts-gui.exe
echo   [OK] qwen3tts-rs.exe
echo   [OK] convert-gguf.exe

REM -- Step 2: Verify binaries exist --
echo.
echo [2/3] 驗證二進位檔...
set "MISSING=0"
for %%f in (
    target\release\qwen3tts-gui.exe
    target\release\qwen3tts-rs.exe
    target\release\convert-gguf.exe
) do (
    if not exist "%%f" (
        echo   [MISSING] %%f
        set "MISSING=1"
    ) else (
        echo   [OK] %%f
    )
)
if "%MISSING%"=="1" (
    echo 錯誤: 缺少二進位檔
    popd
    exit /b 1
)

REM -- Step 3: Compile Inno Setup installer --
echo.
echo [3/3] 編譯 Inno Setup 安裝檔...
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
popd
exit /b 1

:found_iscc
echo   使用: %ISCC%
if not exist dist mkdir dist
"%ISCC%" installer\setup.iss
if errorlevel 1 (
    echo 錯誤: Inno Setup 編譯失敗
    popd
    exit /b 1
)

echo.
echo ============================================================
echo  安裝檔建置完成
echo  輸出: dist\qwen3tts-rs-0.2.0-setup.exe
echo ============================================================

popd
endlocal
