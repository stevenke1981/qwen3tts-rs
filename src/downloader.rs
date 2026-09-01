//! # Qwen3-TTS 模型自動下載模組
//!
//! 提供 HuggingFace 模型與 Tokenizer 解碼器權重的自動偵測、鏡像源支援與背景執行緒下載功能：
//! - 支援官方站 (huggingface.co) 與國內鏡像站 (hf-mirror.com)
//! - 依序自動探測 CLI 工具：`hf` -> `python -c huggingface_hub` -> `huggingface-cli`
//! - 下載過程非阻塞並可即時回報進度日誌
//! - 下載完成自動觸發純 Rust Tokenizer 權重轉換，開箱即用

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::tokenizer_converter::{self, ConverterOptions};
use crate::{Error, Result};

pub const HF_OFFICIAL_ENDPOINT: &str = "https://huggingface.co";
pub const HF_MIRROR_ENDPOINT: &str = "https://hf-mirror.com";
pub const TOKENIZER_MODEL_ID: &str = "Qwen/Qwen3-TTS-Tokenizer-12Hz";

/// 可用的下載工具種類
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadTool {
    Hf(PathBuf),
    Python(PathBuf),
    LegacyHfCli(PathBuf),
}

impl DownloadTool {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Hf(_) => "hf",
            Self::Python(_) => "python (huggingface_hub)",
            Self::LegacyHfCli(_) => "huggingface-cli",
        }
    }

    pub fn path(&self) -> &Path {
        match self {
            Self::Hf(p) | Self::Python(p) | Self::LegacyHfCli(p) => p,
        }
    }
}

/// 偵測系統中可用的 HuggingFace 下載工具
pub fn detect_download_tool() -> Result<DownloadTool> {
    // 1. 優先嘗試 hf (官方最新推薦 CLI)
    if let Some(path) = find_hf_cli() {
        return Ok(DownloadTool::Hf(path));
    }

    // 2. 次選 python (內建 huggingface_hub)
    if let Some(path) = find_python() {
        return Ok(DownloadTool::Python(path));
    }

    // 3. 嘗試舊版 huggingface-cli
    if let Some(path) = find_legacy_hf_cli() {
        return Ok(DownloadTool::LegacyHfCli(path));
    }

    Err(Error::Config(
        "系統中找不到合適的下載工具 (hf / python / huggingface-cli)。\n\
         請先安裝 huggingface_hub：pip install huggingface_hub"
            .into(),
    ))
}

fn find_hf_cli() -> Option<PathBuf> {
    if let Ok(path) = which("hf") {
        return Some(path);
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        let p = PathBuf::from(home).join(".local").join("bin").join("hf.exe");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn find_python() -> Option<PathBuf> {
    if let Ok(path) = which("python") {
        return Some(path);
    }
    if let Ok(path) = which("python3") {
        return Some(path);
    }
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let p = PathBuf::from(local)
            .join("Programs")
            .join("Python")
            .join("Python312")
            .join("python.exe");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn find_legacy_hf_cli() -> Option<PathBuf> {
    if let Ok(path) = which("huggingface-cli") {
        return Some(path);
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        let p = PathBuf::from(home)
            .join(".local")
            .join("bin")
            .join("huggingface-cli.exe");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// 簡單的跨平台 which 搜尋
fn which(name: &str) -> std::result::Result<PathBuf, ()> {
    let output = if cfg!(windows) {
        Command::new("where.exe")
            .arg(name)
            .output()
            .map_err(|_| ())?
    } else {
        Command::new("which")
            .arg(name)
            .output()
            .map_err(|_| ())?
    };

    if output.status.success() {
        let text = String::from_utf8_lossy(&output.stdout);
        if let Some(first) = text.lines().next() {
            let p = PathBuf::from(first.trim());
            if p.exists() {
                return Ok(p);
            }
        }
    }
    Err(())
}

/// 取得 HuggingFace hub 預設快取目錄
pub fn hf_hub_cache_dir() -> Option<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()
        .map(PathBuf::from)?;
    Some(home.join(".cache").join("huggingface").join("hub"))
}

/// 在指定目錄中搜尋模型檔案（支援直接存放、repo_name 資料夾或 HF models-- 快取結構）
pub fn locate_model_in_dir(model_id: &str, base_dir: &Path) -> Option<PathBuf> {
    if !base_dir.exists() {
        return None;
    }
    // 1. base_dir 本身直接包含 model.safetensors
    if base_dir.join("model.safetensors").exists() {
        return Some(base_dir.to_path_buf());
    }
    let repo_name = model_id.split('/').last().unwrap_or(model_id);
    // 2. base_dir / <repo_name> (例如 models/Qwen3-TTS-12Hz-0.6B-Base)
    let candidate1 = base_dir.join(repo_name);
    if candidate1.join("model.safetensors").exists() {
        return Some(candidate1);
    }
    // 3. base_dir / <model_id> (例如 models/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
    let candidate2 = base_dir.join(model_id);
    if candidate2.join("model.safetensors").exists() {
        return Some(candidate2);
    }
    // 4. base_dir / models--<owner>--<repo> (HF Hub cache 結構)
    let candidate3 = base_dir.join(format!("models--{}", model_id.replace('/', "--")));
    if candidate3.exists() {
        let snapshots = candidate3.join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            for entry in entries.flatten() {
                if entry.path().join("model.safetensors").exists() {
                    return Some(entry.path());
                }
            }
        }
    }
    // 5. 掃描 base_dir 下所有第一層子目錄
    if let Ok(entries) = std::fs::read_dir(base_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.join("model.safetensors").exists() {
                let dir_name = path.file_name().unwrap_or_default().to_string_lossy();
                if dir_name.contains(repo_name) || repo_name.contains(&*dir_name) {
                    return Some(path);
                }
            }
        }
    }
    None
}

/// 尋找本地已存在的模型快照目錄（依序搜尋自訂 models 目錄、環境變數、HF 快取）
pub fn locate_model_snapshot(model_id: &str, models_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = models_dir {
        if let Some(found) = locate_model_in_dir(model_id, dir) {
            return Some(found);
        }
    }
    if let Ok(env_dir) = std::env::var("QWEN3_TTS_MODEL_DIR") {
        let p = PathBuf::from(env_dir);
        if let Some(found) = locate_model_in_dir(model_id, &p) {
            return Some(found);
        }
    }
    locate_hf_snapshot(model_id)
}

/// 尋找 HuggingFace hub 快取中的模型快照目錄
pub fn locate_hf_snapshot(model_id: &str) -> Option<PathBuf> {
    let hub = hf_hub_cache_dir()?;
    let repo_dir = hub.join(format!("models--{}", model_id.replace('/', "--")));
    let snapshots = repo_dir.join("snapshots");
    let entries = std::fs::read_dir(&snapshots).ok()?;
    for entry in entries.flatten() {
        let candidate = entry.path().join("model.safetensors");
        if candidate.exists() {
            return Some(entry.path());
        }
    }
    None
}

/// 判斷指定模型是否已經就緒（快取中或自訂路徑中存在 model.safetensors）
pub fn is_model_ready(model_id: &str, custom_dir: Option<&Path>) -> bool {
    locate_model_snapshot(model_id, custom_dir).is_some()
}

/// 判斷 Tokenizer 解碼器權重是否已經就緒
pub fn is_tokenizer_ready() -> bool {
    crate::paths::find_existing_tokenizer_weight_dir().is_ok()
}

/// 下載 HuggingFace 儲存庫（支援指定目標本地目錄）
pub fn download_repo(
    repo_id: &str,
    target_dir: Option<&Path>,
    endpoint: Option<&str>,
    cancel: Option<&AtomicBool>,
    on_log: Arc<dyn Fn(&str) + Send + Sync>,
) -> Result<()> {
    let tool = detect_download_tool()?;
    let mut cmd = match &tool {
        DownloadTool::Hf(path) => {
            on_log(&format!("使用下載工具: hf ({})", path.display()));
            let mut c = Command::new(path);
            c.arg("download").arg(repo_id);
            if let Some(target) = target_dir {
                c.arg("--local-dir").arg(target);
            }
            c
        }
        DownloadTool::Python(path) => {
            on_log(&format!(
                "使用下載工具: Python huggingface_hub ({})",
                path.display()
            ));
            let mut c = Command::new(path);
            let script = if let Some(target) = target_dir {
                let target_str = target.to_string_lossy().replace('\\', "/");
                format!(
                    "from huggingface_hub import snapshot_download; print('正在下載', '{repo_id}', '至', r'{target_str}'); snapshot_download('{repo_id}', local_dir=r'{target_str}'); print('下載完成')"
                )
            } else {
                format!(
                    "from huggingface_hub import snapshot_download; print('正在下載', '{repo_id}'); snapshot_download('{repo_id}'); print('下載完成')"
                )
            };
            c.arg("-u").arg("-c").arg(script);
            c
        }
        DownloadTool::LegacyHfCli(path) => {
            on_log(&format!("使用下載工具: huggingface-cli ({})", path.display()));
            let mut c = Command::new(path);
            c.arg("download").arg(repo_id);
            if let Some(target) = target_dir {
                c.arg("--local-dir").arg(target);
            }
            c
        }
    };

    if let Some(ep) = endpoint {
        let ep = ep.trim();
        if !ep.is_empty() && ep != HF_OFFICIAL_ENDPOINT {
            on_log(&format!("使用下載鏡像端點: {ep}"));
            cmd.env("HF_ENDPOINT", ep);
        }
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| {
        Error::Config(format!(
            "無法啟動下載工具 {}：{e}",
            tool.name()
        ))
    })?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let log_1 = on_log.clone();
    let handle_out = std::thread::spawn(move || {
        if let Some(out) = stdout {
            let reader = BufReader::new(out);
            for line in reader.lines().flatten() {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    log_1(trimmed);
                }
            }
        }
    });

    let log_2 = on_log.clone();
    let handle_err = std::thread::spawn(move || {
        if let Some(err) = stderr {
            let reader = BufReader::new(err);
            for line in reader.lines().flatten() {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    log_2(trimmed);
                }
            }
        }
    });

    // 監控子行程與取消信號
    loop {
        if let Some(cancel_flag) = cancel {
            if cancel_flag.load(Ordering::SeqCst) {
                let _ = child.kill();
                let _ = handle_out.join();
                let _ = handle_err.join();
                return Err(Error::Config("下載已被使用者手動取消".into()));
            }
        }

        match child.try_wait() {
            Ok(Some(status)) => {
                let _ = handle_out.join();
                let _ = handle_err.join();
                if status.success() {
                    return Ok(());
                } else {
                    return Err(Error::Config(format!(
                        "下載行程異常結束 (狀態碼: {status})"
                    )));
                }
            }
            Ok(None) => {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => {
                let _ = handle_out.join();
                let _ = handle_err.join();
                return Err(Error::Config(format!("等待下載行程失敗：{e}")));
            }
        }
    }
}

/// 下載指定模型並回傳其快照目錄路徑（支援儲存至自訂 models 目錄）
pub fn download_model<F>(
    model_id: &str,
    models_base_dir: Option<&Path>,
    endpoint: Option<&str>,
    cancel: Option<&AtomicBool>,
    on_log: F,
) -> Result<PathBuf>
where
    F: Fn(&str) + Send + Sync + 'static,
{
    let on_log: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(on_log);
    if let Some(snapshot) = locate_model_snapshot(model_id, models_base_dir) {
        on_log(&format!("模型 {model_id} 已存在於本地：{}", snapshot.display()));
        return Ok(snapshot);
    }

    let target_dir = models_base_dir.map(|base| {
        let repo_name = model_id.split('/').last().unwrap_or(model_id);
        base.join(repo_name)
    });

    if let Some(ref target) = target_dir {
        on_log(&format!("開始下載模型：{model_id} 至目錄 {}…", target.display()));
        std::fs::create_dir_all(target)
            .map_err(|e| Error::Config(format!("無法建立模型儲存目錄 {}: {e}", target.display())))?;
    } else {
        on_log(&format!("開始下載模型：{model_id}…"));
    }

    download_repo(model_id, target_dir.as_deref(), endpoint, cancel, on_log.clone())?;

    if let Some(snapshot) = locate_model_snapshot(model_id, models_base_dir) {
        on_log(&format!("✅ 模型下載並驗證完成：{}", snapshot.display()));
        Ok(snapshot)
    } else if let Some(target) = target_dir {
        if target.join("model.safetensors").exists() {
            on_log(&format!("✅ 模型下載並驗證完成：{}", target.display()));
            Ok(target)
        } else {
            Err(Error::Config(format!(
                "模型 {model_id} 下載完成，但在 {} 中未能定位到 model.safetensors。",
                target.display()
            )))
        }
    } else {
        Err(Error::Config(format!(
            "模型 {model_id} 下載完成，但未能於快取中定位到 model.safetensors。"
        )))
    }
}

/// 確保 Tokenizer 解碼器權重已下載並轉換為 Candle safetensors（支援存放在自訂 models 目錄）
pub fn ensure_tokenizer_weights<F>(
    models_base_dir: Option<&Path>,
    endpoint: Option<&str>,
    cancel: Option<&AtomicBool>,
    on_log: F,
) -> Result<PathBuf>
where
    F: Fn(&str) + Send + Sync + 'static,
{
    let on_log: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(on_log);

    // 1. 如果自訂 models_base_dir 下已有 tokenizer 或 tokenizer-q8 權重
    if let Some(base) = models_base_dir {
        for sub in ["tokenizer", "tokenizer-q8"] {
            let candidate = base.join(sub);
            if candidate.join("codebook.safetensors").exists()
                && candidate.join("pre_transformer.safetensors").exists()
            {
                on_log(&format!("Tokenizer 解碼器權重已就緒：{}", candidate.display()));
                return Ok(candidate);
            }
        }
    }

    // 2. 如果預設 weights 目錄已存在有效的 Rust 解碼器權重，直接返回
    if let Ok(dir) = crate::paths::find_existing_tokenizer_weight_dir() {
        on_log(&format!("Tokenizer 解碼器權重已就緒：{}", dir.display()));
        return Ok(dir);
    }

    // 3. 檢查 snapshot 是否存在
    let mut hf_snapshot = locate_model_snapshot(TOKENIZER_MODEL_ID, models_base_dir);
    if hf_snapshot.is_none() {
        on_log("本地未找到 Tokenizer 原始權重，開始下載 Qwen/Qwen3-TTS-Tokenizer-12Hz…");
        let target_tok = models_base_dir.map(|b| b.join("Qwen3-TTS-Tokenizer-12Hz"));
        if let Some(ref t) = target_tok {
            let _ = std::fs::create_dir_all(t);
        }
        download_repo(TOKENIZER_MODEL_ID, target_tok.as_deref(), endpoint, cancel, on_log.clone())?;
        hf_snapshot = locate_model_snapshot(TOKENIZER_MODEL_ID, models_base_dir);
    }

    let hf_snapshot = hf_snapshot.ok_or_else(|| {
        Error::Config("Tokenizer 下載完成但未能定位快照目錄".into())
    })?;

    on_log(&format!(
        "正在使用 Tokenizer snapshot: {}，執行純 Rust 權重轉換…",
        hf_snapshot.display()
    ));

    // 4. 轉換權重至 models/tokenizer 或 weights/tokenizer
    let output_dir = if let Some(base) = models_base_dir {
        base.join("tokenizer")
    } else {
        crate::paths::default_tokenizer_cache_dir()
            .unwrap_or_else(|| PathBuf::from("weights").join("tokenizer"))
    };

    let options = ConverterOptions {
        input_dir: Some(hf_snapshot),
        output_dir: output_dir.clone(),
    };

    let converted = tokenizer_converter::convert_tokenizer_weights(&options)?;
    on_log(&format!(
        "✅ Tokenizer 權重轉換完成！產出 {} 個權重檔案至 {}",
        converted.files.len(),
        output_dir.display()
    ));

    Ok(output_dir)
}

/// 同時確保模型與 Tokenizer 權重皆已就緒
pub fn ensure_synthesis_resources<F>(
    model_id: &str,
    models_base_dir: Option<&Path>,
    endpoint: Option<&str>,
    cancel: Option<&AtomicBool>,
    on_log: F,
) -> Result<(PathBuf, PathBuf)>
where
    F: Fn(&str) + Send + Sync + 'static,
{
    let on_log: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(on_log);
    let log_1 = on_log.clone();
    let model_dir = download_model(model_id, models_base_dir, endpoint, cancel, move |msg| log_1(msg))?;
    let log_2 = on_log.clone();
    let tok_dir = ensure_tokenizer_weights(models_base_dir, endpoint, cancel, move |msg| log_2(msg))?;
    Ok((model_dir, tok_dir))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_download_tool() {
        let tool = detect_download_tool();
        assert!(tool.is_ok(), "應該偵測到系統中的下載工具 (hf 或 python)");
        let tool = tool.unwrap();
        println!("偵測到的下載工具: {:?}", tool);
        assert!(!tool.name().is_empty());
    }

    #[test]
    fn test_hf_hub_cache_dir() {
        let hub = hf_hub_cache_dir();
        assert!(hub.is_some());
        let hub = hub.unwrap();
        assert!(hub.ends_with(Path::new(".cache/huggingface/hub")));
    }

    #[test]
    fn test_is_model_ready_nonexistent() {
        let ready = is_model_ready("NonExistent/Model-999B-Fake", None);
        assert!(!ready);
    }
}
