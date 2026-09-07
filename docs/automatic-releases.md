# 自動編譯與 GitHub Releases

`.github/workflows/release.yml` 對應 `spec.md` §4 平台支援與 §6.6 CLI / optional GUI 產品介面。每次建置同時提供 Windows CPU、Windows CUDA 與 Linux CPU 三種套件；數值對齊、語音品質與效能驗收仍依原有 Alignment Gate 執行。

| 事件 | 編譯與測試 | 發布 |
| --- | --- | --- |
| 原始倉庫 `master` push | Windows + Linux | 唯一 `preview-<sha12>-<run_id>-<run_attempt>` 預覽 release |
| 原始倉庫 `v*` tag push | Windows + Linux | 使用既有 tag 的正式 release |
| 對 `master` 的 pull request | Windows + Linux | 不發布，提供 Actions artifacts |
| `codex/automatic-releases` push | Windows + Linux | 不發布，用於發布流程首次遠端演練 |
| Actions 手動執行 | Windows + Linux | 不發布，提供 Actions artifacts |
| fork | 依相同事件編譯 | 不發布 |

預覽版設定為 prerelease，不更動 GitHub 的 latest release。每次執行使用不同 tag，流程不覆寫既有 tag、release 或附件。重新執行正式版若 release 已存在會失敗，請檢查現有資產並使用新的版本 tag；不要刪除或覆寫已發行版本。

## 下載內容

- 建議 Windows 自動包 `qwen3tts-rs-windows-x64-auto.zip`：同時包含完整 CPU 與 CUDA 子資料夾。解壓後執行 `Start-Qwen3TTS.cmd`，啟動器檢測 CUDA 可用性後選擇 GUI，CUDA 無法使用時啟動獨立 CPU 版。請保留兩個子資料夾與啟動器的相對位置。
- Windows x86_64 CPU ZIP：`qwen3tts-rs.exe`、`qwen3tts-gui.exe`、`convert-gguf.exe`。
- Windows x86_64 CUDA ZIP：同樣三個執行檔，另附 CUDA 13.2 的 cuBLAS、cuBLASLt、cuRAND、NVRTC、NVRTC builtins 與 CUDA runtime DLL。需要支援 CUDA 13.2 的 NVIDIA 驅動程式；編譯目標 `CUDA_COMPUTE_CAP=86`，適用 compute capability 8.6 或更新 GPU（例如 RTX 30 系列以上），較舊 GPU 請使用 CPU 套件或自行重新編譯。
- Linux x86_64 CPU tar.gz：`qwen3tts-rs`、`convert-gguf`，建置於 Ubuntu 22.04，需要 glibc 2.35 或更新環境。
- 三種套件均含 README、LICENSE、本說明及 `build-info.json`（來源 commit、target、variant、features、run、Rust 版本、驗證範圍與 CUDA 編譯資訊）。
- 每個壓縮檔附獨立 `.sha256` 校驗檔。模型權重與 NVIDIA 驅動程式不包含在套件內，尤其不複製 `nvcuda.dll`；模型設定與使用方式請參閱 README。

PowerShell 驗證下載的 ZIP：

```powershell
$archive = 'qwen3tts-rs-x86_64-pc-windows-msvc-cpu.zip'
$expected = (Get-Content "$archive.sha256" -Raw).Trim().Split(' ')[0]
if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
    throw 'SHA256 mismatch'
}
```

Linux 可在下載目錄執行：

```bash
sha256sum -c qwen3tts-rs-x86_64-unknown-linux-gnu-cpu.tar.gz.sha256
```

## 發布前驗證與權限

CPU 兩個平台先以明確 CPU features 執行 `cargo test --locked --lib`，再執行 `cargo build --release --locked --bins`，最後實際執行 CLI 與 convert-gguf 的 `--help`。Windows 額外啟用 `gui` feature。

CUDA 使用 Windows Server 2022 runner，安裝固定 CUDA 13.2.0 toolkit、初始化 MSVC x64 環境，使用 `cpu,cuda,candle-llm,gui` features 編譯。CUDA target cache 與 CPU 完全分開；必要 runtime DLL 缺失會直接失敗。GitHub hosted runner 沒有 NVIDIA GPU/driver，因此 CUDA 列僅編譯及檢查 DLL，**不執行 CUDA library tests 或 CLI smoke，也不宣稱 GPU 執行驗收通過**。本機 RTX 3070 Ti 的實際 GPU 驗收由 Gate Owner 另外執行並記錄，不能從雲端成功推論。

任何 CPU 測試、建置、smoke 或封裝失敗都會阻止發布。三列 matrix 成功後，獨立 bundle job 驗證 Windows CPU/CUDA checksum 並組合自動包；publish job 必須等 build 與 bundle 全部成功，再次驗證四個附件的 checksum。Cargo 下載與 target 編譯輸出有快取；target key 分隔平台、variant、依賴及 commit。快取命中仍會執行各列要求的驗證與建置，Cargo 負責工具鏈與來源的 freshness 檢查。

建置 job 僅有 `contents: read`，checkout 不保留 token。獨立 publish job 才取得 `contents: write`，不執行倉庫程式碼，且只有原始倉庫 push 事件可以進入。流程透過 `gh release create --target <github.sha>` 明確綁定來源；正式版另要求 tag 已存在。

此流程不載入大型模型、不測量 p99 延遲，也不執行 GUI 真人操作或完整數值對齊。因此 Actions 成功只代表打包流程通過，不能將專案或 Phase 狀態改成 `ALIGNED`。

## 手動演練與正式版

在 GitHub Actions 選擇 **Release binaries → Run workflow**，選取要測試的分支。成功後可下載三種獨立編譯 artifacts 與 Windows 自動包（保留 14 天）；手動執行不會建立 release。PR artifacts 同樣只用於驗證。

要發布正式版，維護者先確認欲發行 commit 已通過產品與 Alignment 驗收，再在該 commit 建立並推送版本 tag，例如 `v0.2.1`。`v*` tag 是正式發布的明確入口，應與 Cargo 套件版本保持一致；自動化不會修改 Cargo.toml 或替代維護者的版本決策。
