# 自動編譯與 GitHub Releases

`.github/workflows/release.yml` 對應 `spec.md` §4 平台支援與 §6.6 CLI / optional GUI 產品介面。它提供可下載的 CPU 套件；數值對齊、語音品質與效能驗收仍依原有 Alignment Gate 執行。

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

- Windows x86_64 CPU ZIP：`qwen3tts-rs.exe`、`qwen3tts-gui.exe`、`convert-gguf.exe`。
- Linux x86_64 CPU tar.gz：`qwen3tts-rs`、`convert-gguf`，建置於 Ubuntu 22.04，需要 glibc 2.35 或更新環境。
- 兩種套件均含 README、LICENSE、本說明及 `build-info.json`（來源 commit、target、features、run、Rust 版本）。
- 每個壓縮檔附獨立 `.sha256` 校驗檔。模型權重、CUDA runtime 與 CUDA binaries 不包含在套件內；模型設定與使用方式請參閱 README。

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

每個平台都先以明確 CPU features 執行 `cargo test --locked --lib`，再執行 `cargo build --release --locked --bins`，最後實際執行 CLI 與 convert-gguf 的 `--help`。Windows 額外啟用 `gui` feature。任何測試、建置、smoke 或封裝失敗都會阻止發布；publish job 必須等整個 matrix 成功，並再次驗證附件 checksum。Cargo 下載與 target 編譯輸出有快取；target key 分隔平台、features、依賴及 commit。快取命中仍會執行所有測試與建置，Cargo 負責工具鏈與來源的 freshness 檢查。

建置 job 僅有 `contents: read`，checkout 不保留 token。獨立 publish job 才取得 `contents: write`，不執行倉庫程式碼，且只有原始倉庫 push 事件可以進入。流程透過 `gh release create --target <github.sha>` 明確綁定來源；正式版另要求 tag 已存在。

此流程不載入大型模型、不測量 p99 延遲，也不執行 GUI 真人操作或完整數值對齊。因此 Actions 成功只代表打包流程通過，不能將專案或 Phase 狀態改成 `ALIGNED`。

## 手動演練與正式版

在 GitHub Actions 選擇 **Release binaries → Run workflow**，選取要測試的分支。成功後可下載兩平台 artifacts（保留 14 天）；手動執行不會建立 release。PR artifacts 同樣只用於驗證。

要發布正式版，維護者先確認欲發行 commit 已通過產品與 Alignment 驗收，再在該 commit 建立並推送版本 tag，例如 `v0.2.1`。`v*` tag 是正式發布的明確入口，應與 Cargo 套件版本保持一致；自動化不會修改 Cargo.toml 或替代維護者的版本決策。
