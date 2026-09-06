# qwen3tts-rs

以 Rust 與 Candle 實作的 Qwen3-TTS 推理與音訊解碼專案，提供命令列工具、Windows 桌面 GUI、CPU/CUDA 執行路徑、Safetensors/GGUF Talker 載入，以及長文分段合成。

> **專案狀態：開發中。** 目前主要維護 12Hz 推理與解碼路徑。模型權重不包含在此倉庫內；首次使用前必須下載對應 Qwen3-TTS 模型及 Tokenizer/Codec 權重。正式工作負載請先以自己的模型、硬體與文字內容完成音質及效能驗證。

**English:** A Rust/Candle implementation of the Qwen3-TTS inference and codec pipeline with a CLI, an optional desktop GUI, CPU/CUDA backends, model utilities, and long-text synthesis support.

## 主要功能

- `qwen3tts-rs`：正式 CLI binary，可全域安裝，不再只能透過 Cargo example 執行。
- `qwen3tts-gui`：以 egui/eframe 製作的桌面介面，支援背景下載、CPU/CUDA 選擇、語言偵測及長文逐句合成。
- 純 Rust/Candle Talker 與 12Hz Codec Decoder；亦保留 Python bridge 供相容與對照用途。
- CustomVoice、Voice Clone（Base）與 VoiceDesign 模式驗證。
- Safetensors 權重、Talker GGUF 載入與 `convert-gguf` 轉換工具。
- Stateful streaming、mmap 權重載入、裝置端狀態與對齊測試基礎設施。

## 前置需求

- Rust **1.85 或更新版本**。
- Windows 10/11 x64 為目前 GUI 與安裝程式的主要支援平台。
- CPU 版本不需要 CUDA。
- CUDA 版本需要相容的 NVIDIA Driver、CUDA Toolkit、MSVC C++ Build Tools，以及正確的 GPU compute capability。
- 使用自動下載時，系統需有 `hf`、安裝了 `huggingface_hub` 的 Python，或舊版 `huggingface-cli` 其中之一。

## 快速開始

### 1. 取得原始碼並檢查

```powershell
git clone https://github.com/stevenke1981/qwen3tts-rs.git
cd qwen3tts-rs
cargo check --locked
cargo test --locked --lib
```

### 2. 啟動 GUI（CPU）

GUI 為可選 feature，不會再拖慢純 library/CLI 的預設建置。

```powershell
cargo run --locked --release --bin qwen3tts-gui `
  --no-default-features --features "cpu,candle-llm,gui"
```

GUI 預設會在模型缺失時引導或自動下載所需資源。模型預設存放在執行檔旁或專案的 `models` 目錄，也可在介面中另行指定。

### 3. 啟動 GUI（NVIDIA CUDA）

```powershell
cargo run --locked --release --bin qwen3tts-gui `
  --no-default-features --features "cpu,candle-llm,gui,cuda"
```

Windows 也可使用整合腳本：

```powershell
# 同時建立 CLI、轉換工具、CPU GUI 與 CUDA GUI
.\build_all.ps1

# 只建立 CUDA GUI；Ampere 預設 compute capability 為 86
.\build_all.ps1 -CudaOnly -ComputeCapability 86

# 只建立 CPU GUI
.\build_all.ps1 -CpuOnly
```

### 4. 建置或全域安裝 CLI

```powershell
cargo build --locked --release --bin qwen3tts-rs
cargo install --locked --path . --bin qwen3tts-rs
qwen3tts-rs --help
```

CLI 會從 `--model-dir`、`QWEN3_TTS_MODEL_DIR` 或 Hugging Face cache 尋找模型。以下為 CustomVoice 範例；請先準備模型與 Codec 權重：

```powershell
qwen3tts-rs `
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice `
  --model-dir D:\Models\Qwen3-TTS-12Hz-0.6B-CustomVoice `
  --backend candle `
  --speaker Vivian `
  --language Chinese `
  --text "你好，這是 qwen3tts-rs 語音合成測試。" `
  --output output.wav
```

完整權重準備、Voice Clone、VoiceDesign、GGUF 與發行版用法請參閱：

- [繁體中文使用指南](docs/release_usage_zh.md)
- [English usage guide](docs/release_usage_en.md)
- [模型模式與必要參數](docs/model-modes.md)
- [GGUF tensor mapping](docs/gguf_tensor_mapping.md)

## 模型模式

| 模式 | 建議模型 | 必要輸入 | 常見用途 |
|---|---|---|---|
| CustomVoice | `*-CustomVoice` | `--speaker` | 使用內建說話者快速合成 |
| Voice Clone | `*-Base` | `--reference-audio`；建議同時提供 `--reference-text` | 複製參考音色 |
| VoiceDesign | `1.7B-VoiceDesign` | `--instruct` 或 `--instruct-file` | 以文字描述設計音色與表演方式 |

內建 speaker 名稱可用下列指令查看：

```powershell
qwen3tts-rs --list-speakers
qwen3tts-rs --list-models
```

## Cargo features

| Feature | 預設 | 用途 |
|---|---:|---|
| `cpu` | 是 | CPU 執行標記 |
| `candle-llm` | 是 | 純 Rust/Candle Talker 路徑 |
| `gui` | 否 | 桌面 GUI、檔案對話框與音訊播放 |
| `cuda` | 否 | NVIDIA CUDA backend |
| `metal` | 否 | Apple Metal backend |
| `stage-dump` | 否 | 對齊與中間張量記錄 |

只檢查最小核心 library：

```bash
cargo check --locked --no-default-features --lib
```

## 模型與輸出目錄

- `QWEN3_TTS_MODEL_DIR`：覆寫 CLI/程式尋找模型的基礎目錄。
- `models/`：GUI 預設模型目錄之一。
- `weights/`：轉換後的 Codec/Tokenizer 權重慣用位置。
- `output.wav`：CLI 預設輸出檔案。

模型與大型產物已列入 `.gitignore`。請勿將模型權重、測試音檔、使用者聲音資料或本機 agent/runtime database 提交到 Git。

## 開發與驗證

提交前建議執行：

```bash
cargo fmt --all -- --check
cargo check --locked --lib --bins --examples
cargo clippy --locked --lib --bins --examples -- -D warnings
cargo test --locked --lib
cargo test --locked --tests --no-run
```

Windows GUI feature 驗證：

```powershell
cargo check --locked --no-default-features `
  --features "cpu,candle-llm,gui" --bin qwen3tts-gui
```

CUDA correctness、峰值 VRAM、首包延遲與 throughput 必須在實際 NVIDIA GPU 上另行測試；GitHub hosted runner 的一般 CI 不宣稱完成 CUDA 硬體驗證。

## 專案結構

```text
src/
  bin/                  CLI、GUI、GGUF 轉換工具
  talker/               Talker 與 Code Predictor
  codec/                Codec、因果卷積與 Transformer
  text_frontend/        模型模式、Tokenizer、Python/Candle 前端
  gui.rs                可選桌面 GUI
  downloader.rs         模型與 Tokenizer 資源下載
examples/               探針、轉換及相容範例
installer/              Windows Inno Setup 安裝程式
patches/candle-kernels/  專案使用的 Candle CUDA kernel patch
tests/                  單元、對齊、壓力與整合測試
```

## 授權與模型條款

本倉庫程式碼採用 [MIT License](LICENSE)。Qwen3-TTS 模型、資料及其他第三方元件各自適用其原始授權；下載與散布前請自行確認並遵守相關條款。本專案並非 Qwen 官方發行版。
