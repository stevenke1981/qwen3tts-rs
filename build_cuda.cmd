@echo off
REM Force MSVC 14.29 (compatible with CUDA 13.2 NVCC), NOT 14.44
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.29.30133 > nul
set "MSVC_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64"
set "NVCC_CCBIN=%MSVC_BIN%"

REM Override CUDA_PATH to v13.2 (matches RTX 3070 Ti) — vcvars64.bat sets v12.1
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "CUDA_PATH_V13_2=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

REM Ensure NVCC can find cl.exe in PATH (cargo strips PATH but NVCC needs it for -c mode)
set "PATH=%MSVC_BIN%;%PATH%"
cargo clean -p candle-kernels 2>&1
cargo build --features cuda %*
