//! # Qwen3-TTS GUI 應用程式 — egui/eframe 實作
//!
//! 提供圖形化介面進行語音合成，支援：
//! - 文字輸入與參數設定
//! - Python Bridge / Candle Native 後端
//! - 背景執行緒非阻塞合成
//! - 狀態即時更新與音檔播放

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, mpsc};
use std::thread;

use eframe::egui;
use egui::{Color32, Frame, Margin, RichText, ScrollArea, Vec2};

use crate::text_frontend::model_catalog::SUPPORTED_LANGUAGES;
use crate::text_frontend::speaker_presets;
use crate::text_frontend::{PythonBridge, SynthesisOptions, TextFrontend};
use crate::{Decoder12Hz, DecoderConfig};

const CJK_FONT_NAME: &str = "qwen3tts_cjk";

const CJK_FONT_CANDIDATES: &[&str] = &[
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\mingliu.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\NotoSansCJK-Regular.ttc",
    r"C:\Windows\Fonts\NotoSansTC-Regular.otf",
];

/// Install a Windows CJK font fallback so egui can render Chinese UI text.
pub fn install_cjk_fonts(ctx: &egui::Context) -> Option<PathBuf> {
    let font_path = cjk_font_candidates()
        .into_iter()
        .find(|path| path.exists())?;
    let font_bytes = std::fs::read(&font_path).ok()?;
    let mut fonts = egui::FontDefinitions::default();

    fonts.font_data.insert(
        CJK_FONT_NAME.to_owned(),
        egui::FontData::from_owned(font_bytes).into(),
    );

    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, CJK_FONT_NAME.to_owned());
    }

    ctx.set_fonts(fonts);
    Some(font_path)
}

fn cjk_font_candidates() -> Vec<PathBuf> {
    CJK_FONT_CANDIDATES.iter().map(PathBuf::from).collect()
}

// ---------------------------------------------------------------------------
// 後端類型
// ---------------------------------------------------------------------------

/// 文字前端後端
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Python,
    #[cfg(feature = "candle-llm")]
    Candle,
}

impl BackendKind {
    pub const ALL: &'static [Self] = {
        #[cfg(feature = "candle-llm")]
        {
            &[Self::Candle, Self::Python]
        }
        #[cfg(not(feature = "candle-llm"))]
        {
            &[Self::Python]
        }
    };

    pub fn name(&self) -> &'static str {
        match self {
            Self::Python => "Python (bridge)",
            #[cfg(feature = "candle-llm")]
            Self::Candle => "Candle (native Rust)",
        }
    }

    pub fn default() -> Self {
        #[cfg(feature = "candle-llm")]
        {
            Self::Candle
        }
        #[cfg(not(feature = "candle-llm"))]
        {
            Self::Python
        }
    }
}

// ---------------------------------------------------------------------------
// 合成參數
// ---------------------------------------------------------------------------

/// 由 GUI 傳遞給背景執行緒的合成請求
#[derive(Clone)]
pub struct SynthesisParams {
    pub text: String,
    pub model_id: String,
    pub model_dir: Option<PathBuf>,
    pub models_base_dir: PathBuf,
    pub backend: BackendKind,
    pub language: String,
    pub speaker: Option<String>,
    pub instruct: Option<String>,
    pub speed: f64,
    pub output_path: PathBuf,
    pub reference_audio: Option<PathBuf>,
    pub reference_text: Option<String>,
    pub seed: Option<u64>,
    pub max_new_tokens: u32,
    pub auto_download: bool,
    pub hf_mirror: Option<String>,
}

// ---------------------------------------------------------------------------
// 背景事件
// ---------------------------------------------------------------------------

/// 背景執行緒回傳給 GUI 的事件
#[derive(Clone, Debug)]
pub enum WorkerEvent {
    /// 狀態訊息（顯示在 log 區域）
    Status(String),
    /// 進度更新（例如 LLM token 生成進度）
    Progress { current: usize, total: usize },
    /// 合成完成
    Done(Result<SynthesisResult, String>),
    /// 下載完成
    DownloadDone(Result<PathBuf, String>),
}

/// 合成結果
#[derive(Clone, Debug)]
pub struct SynthesisResult {
    pub output_path: PathBuf,
    pub duration_sec: f64,
    pub sample_count: usize,
}

// ---------------------------------------------------------------------------
// 模型 ID 下拉選單候選
// ---------------------------------------------------------------------------

const MODEL_ID_CANDIDATES: &[&str] = &[
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
];

// ---------------------------------------------------------------------------
// 工具函式 — 僅供背景執行緒使用
// ---------------------------------------------------------------------------

#[allow(dead_code)]
mod worker {
    use super::*;
    use std::sync::mpsc::Sender;

    /// 尋找模型 snapshot（優先在指定的 models_base_dir 或 HF 快取中找）
    #[cfg(feature = "candle-llm")]
    pub fn locate_model_snapshot(
        model_id: &str,
        models_base_dir: Option<&Path>,
    ) -> Option<PathBuf> {
        crate::downloader::locate_model_snapshot(model_id, models_base_dir)
    }

    pub fn find_tokenizer_json(
        model_id: &str,
        model_dir: &Path,
        models_base_dir: Option<&Path>,
    ) -> Option<PathBuf> {
        let local = model_dir.join("tokenizer.json");
        if local.exists() {
            return Some(local);
        }
        let snapshots = model_dir.join("snapshots");
        if snapshots.is_dir() {
            if let Ok(entries) = std::fs::read_dir(snapshots) {
                for entry in entries.flatten() {
                    let path = entry.path().join("tokenizer.json");
                    if path.exists() {
                        return Some(path);
                    }
                }
            }
        }
        // 嘗試在自訂 models_base_dir 下找
        if let Some(base) = models_base_dir {
            for candidate in [
                base.join("tokenizer.json"),
                base.join("tokenizer_1.7b.json"),
                base.join("models").join("tokenizer.json"),
            ] {
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
        // 嘗試在執行檔同層與上層目錄尋找
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                for candidate in [
                    exe_dir.join("models").join("tokenizer.json"),
                    exe_dir.join("models").join("tokenizer_1.7b.json"),
                    exe_dir.join("tokenizer.json"),
                    exe_dir.join("..").join("models").join("tokenizer.json"),
                    exe_dir.join("..").join("..").join("models").join("tokenizer.json"),
                ] {
                    if candidate.exists() {
                        return Some(candidate);
                    }
                }
            }
        }
        for fallback in ["models/tokenizer_1.7b.json", "models/tokenizer.json", "tokenizer.json"] {
            let path = PathBuf::from(fallback);
            if path.exists() {
                return Some(path);
            }
        }
        // 嘗試從 Base 模型快取找
        let base_ids: &[&str] = if model_id.contains("0.6B") {
            &[
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            ]
        } else {
            &[
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            ]
        };
        for base_id in base_ids {
            if let Some(base_dir) = crate::downloader::locate_model_snapshot(base_id, models_base_dir) {
                let path = base_dir.join("tokenizer.json");
                if path.exists() {
                    return Some(path);
                }
            }
        }
        // 若找不到但存在 vocab.json，自動嘗試用 build_tokenizer.py 產生
        let vocab_path = model_dir.join("vocab.json");
        if vocab_path.exists() {
            let out_tok = model_dir.join("tokenizer.json");
            let _ = std::process::Command::new("python")
                .arg("tools/build_tokenizer.py")
                .arg("--model-dir")
                .arg(model_dir)
                .arg("--output")
                .arg(&out_tok)
                .output();
            if out_tok.exists() {
                return Some(out_tok);
            }
        }
        None
    }

    pub fn ensure_decoder_weights(
        models_base_dir: Option<&Path>,
        tx: &Sender<WorkerEvent>,
    ) -> Result<PathBuf, String> {
        tx.send(WorkerEvent::Status("檢查 Tokenizer 解碼器權重…".into()))
            .ok();
        if let Some(base) = models_base_dir {
            for sub in ["tokenizer", "tokenizer-q8"] {
                let candidate = base.join(sub);
                if candidate.join("codebook.safetensors").exists()
                    && candidate.join("pre_transformer.safetensors").exists()
                {
                    return Ok(candidate);
                }
            }
        }
        crate::paths::find_existing_tokenizer_weight_dir().map_err(|e| {
            format!(
                "找不到 Tokenizer 解碼器權重：{e}\n\
                 請先執行 convert_tokenizer.exe 或 python tools/convert_weights.py tokenizer"
            )
        })
    }

    pub fn runtime_device() -> candle_core::Device {
        #[cfg(feature = "cuda")]
        {
            match candle_core::Device::new_cuda(0) {
                Ok(device) => {
                    eprintln!("✅ 成功初始化 CUDA Device 0 (NVIDIA GPU 加速已啟用)");
                    device
                }
                Err(err) => {
                    eprintln!("⚠️ 無法初始化 CUDA Device 0: {err}，回退至 CPU");
                    candle_core::Device::Cpu
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle_core::Device::Cpu
        }
    }
}

// ---------------------------------------------------------------------------
// 合成管線（背景執行緒）
// ---------------------------------------------------------------------------

fn run_synthesis(params: SynthesisParams, tx: Sender<WorkerEvent>, cancel: Arc<AtomicBool>) {
    let result = run_synthesis_inner(&params, &tx, &cancel);
    if cancel.load(Ordering::SeqCst) {
        tx.send(WorkerEvent::Status("合成已取消".into())).ok();
        return;
    }
    tx.send(WorkerEvent::Done(result)).ok();
}

fn run_synthesis_inner(
    params: &SynthesisParams,
    tx: &Sender<WorkerEvent>,
    cancel: &AtomicBool,
) -> Result<SynthesisResult, String> {
    // Step 1: 決定裝置
    tx.send(WorkerEvent::Status("初始化裝置…".into()))
        .map_err(|e| e.to_string())?;
    if cancel.load(Ordering::SeqCst) {
        return Err("已取消".into());
    }
    let device = worker::runtime_device();
    let device_name = match device {
        candle_core::Device::Cpu => "🖥 CPU".to_string(),
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(_) => "⚡ CUDA (NVIDIA GPU 硬體加速)".to_string(),
        _ => "Unknown".to_string(),
    };
    tx.send(WorkerEvent::Status(format!("運算裝置：{device_name}")))
        .ok();

    // Step 2: 取得 Token（LLM 或從參數提供的 tokens_path）
    if cancel.load(Ordering::SeqCst) {
        return Err("已取消".into());
    }

    let stream = if params.text.is_empty() {
        return Err("請輸入要合成的文字".into());
    } else {
        tx.send(WorkerEvent::Status("載入 LLM 後端並生成 Token…".into()))
            .ok();
        match params.backend {
            BackendKind::Python => {
                tx.send(WorkerEvent::Status(format!(
                    "使用 Python 橋接 ({})…（首次載入約 1-5 分鐘）",
                    params.model_id
                )))
                .ok();
                let bridge = PythonBridge::new(&params.model_id)
                    .map_err(|e| format!("建立 PythonBridge 失敗：{e}"))?
                    .with_python("python");
                let options = SynthesisOptions {
                    language: params.language.clone(),
                    speaker: params.speaker.clone(),
                    instruct: params.instruct.clone(),
                    reference_audio: params
                        .reference_audio
                        .as_ref()
                        .map(|p| p.to_string_lossy().to_string()),
                    reference_text: params.reference_text.clone(),
                    seed: params.seed,
                    temperature: 0.9,
                    top_k: 50,
                    top_p: 1.0,
                    max_new_tokens: params.max_new_tokens,
                };
                bridge
                    .synthesize(&params.text, &options)
                    .map_err(|e| format!("Python LLM Token 生成失敗：{e}"))?
            }
            #[cfg(feature = "candle-llm")]
            BackendKind::Candle => {
                use crate::text_frontend::CandleLLM;

                let dir = if let Some(d) = &params.model_dir {
                    d.clone()
                } else {
                    match worker::locate_model_snapshot(&params.model_id, Some(&params.models_base_dir)) {
                        Some(d) => d,
                        None => {
                            if params.auto_download {
                                tx.send(WorkerEvent::Status(format!(
                                    "📥 本地找不到模型 {}，自動開始下載至 {}…",
                                    params.model_id,
                                    params.models_base_dir.display()
                                )))
                                .ok();
                                let tx_c = tx.clone();
                                crate::downloader::download_model(
                                    &params.model_id,
                                    Some(&params.models_base_dir),
                                    params.hf_mirror.as_deref(),
                                    Some(cancel),
                                    move |msg| {
                                        tx_c.send(WorkerEvent::Status(msg.to_string())).ok();
                                    },
                                )
                                .map_err(|e| format!("自動下載模型失敗：{e}"))?
                            } else {
                                return Err(format!(
                                    "找不到模型 {}。請勾選「當模型缺失時自動下載」，或點擊「📥 下載模型」按鈕進行下載。",
                                    params.model_id
                                ));
                            }
                        }
                    }
                };

                let sf_path = dir.join("model.safetensors");
                if !sf_path.exists() {
                    return Err(format!("模型 safetensors 不存在：{}", sf_path.display()));
                }

                let tok_path = worker::find_tokenizer_json(
                    &params.model_id,
                    &dir,
                    Some(&params.models_base_dir),
                )
                .ok_or_else(|| {
                    "找不到 tokenizer.json。請先使用 convert_tokenizer.exe 或 tools/build_tokenizer.py 產生。".to_string()
                })?;

                tx.send(WorkerEvent::Status(format!(
                    "載入 Candle LLM ({})…",
                    params.model_id
                )))
                .ok();

                let llm = CandleLLM::from_files(&sf_path, &tok_path, &device)
                    .map_err(|e| format!("載入 CandleLLM 失敗：{e}"))?;

                let options = SynthesisOptions {
                    language: params.language.clone(),
                    speaker: params.speaker.clone(),
                    instruct: params.instruct.clone(),
                    reference_audio: params
                        .reference_audio
                        .as_ref()
                        .map(|p| p.to_string_lossy().to_string()),
                    reference_text: params.reference_text.clone(),
                    seed: params.seed,
                    temperature: 0.9,
                    top_k: 50,
                    top_p: 1.0,
                    max_new_tokens: params.max_new_tokens,
                };

                tx.send(WorkerEvent::Status(format!(
                    "⚡ GPU (CUDA) 正在推理生成語音 Token（依字數上限 {} 幀，約 {:.1} 秒語音）…",
                    params.max_new_tokens,
                    params.max_new_tokens as f64 / 12.0
                )))
                .ok();

                llm.synthesize(&params.text, &options)
                    .map_err(|e| format!("Candle LLM Token 生成失敗：{e}"))?
            }
        }
    };

    let num_frames = stream.num_frames();
    if num_frames == 0 {
        return Err("LLM 未產生任何 Token".into());
    }

    tx.send(WorkerEvent::Status(format!(
        "生成 {num_frames} 幀 Token（語音約 {:.1} 秒）",
        stream.duration_sec()
    )))
    .ok();

    if cancel.load(Ordering::SeqCst) {
        return Err("已取消".into());
    }

    // Step 3: 載入解碼器權重
    tx.send(WorkerEvent::Status("載入 Tokenizer 解碼器權重…".into()))
        .ok();
    let weight_dir = match worker::ensure_decoder_weights(Some(&params.models_base_dir), tx) {
        Ok(dir) => dir,
        Err(err) => {
            if params.auto_download {
                tx.send(WorkerEvent::Status(
                    format!(
                        "📥 本地找不到 Tokenizer 解碼器權重，自動下載至 {} 並轉換…",
                        params.models_base_dir.display()
                    ),
                ))
                .ok();
                let tx_c = tx.clone();
                crate::downloader::ensure_tokenizer_weights(
                    Some(&params.models_base_dir),
                    params.hf_mirror.as_deref(),
                    Some(cancel),
                    move |msg| {
                        tx_c.send(WorkerEvent::Status(msg.to_string())).ok();
                    },
                )
                .map_err(|e| format!("自動下載/轉換 Tokenizer 權重失敗：{e}"))?
            } else {
                return Err(err);
            }
        }
    };

    let target_frames = if params.speed != 1.0 {
        (num_frames as f64 / params.speed).round() as usize
    } else {
        num_frames
    };
    let mut config = DecoderConfig::realtime_with_capacity(num_frames.max(target_frames));
    config.speed = params.speed;

    let mut decoder = Decoder12Hz::from_safetensors(config, &weight_dir, &device)
        .map_err(|e| format!("載入 Tokenizer 權重失敗：{e}"))?;

    if cancel.load(Ordering::SeqCst) {
        return Err("已取消".into());
    }

    // Step 4: 解碼
    tx.send(WorkerEvent::Status(format!(
        "⚡ GPU (CUDA) 正在將 {num_frames} 幀 Codec 解碼為音訊波形…",
    )))
    .ok();
    let sample_rate = 24000u32;
    let all_samples = decoder
        .decode_frames(&stream.frames)
        .map_err(|e| format!("解碼失敗：{e}"))?;

    if cancel.load(Ordering::SeqCst) {
        return Err("已取消".into());
    }

    let duration_sec = all_samples.len() as f64 / sample_rate as f64;
    tx.send(WorkerEvent::Status(format!(
        "產出 {:.2} 秒音頻（{} 個樣本）",
        duration_sec,
        all_samples.len()
    )))
    .ok();

    // Step 5: 寫入 WAV
    tx.send(WorkerEvent::Status(format!(
        "寫入 WAV：{}",
        params.output_path.display()
    )))
    .ok();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&params.output_path, spec)
        .map_err(|e| format!("無法建立 WAV 檔案：{e}"))?;

    for &sample in &all_samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let int_sample = (clamped * i16::MAX as f32) as i16;
        writer
            .write_sample(int_sample)
            .map_err(|e| format!("寫入樣本失敗：{e}"))?;
    }

    writer
        .finalize()
        .map_err(|e| format!("關閉 WAV 檔案失敗：{e}"))?;

    Ok(SynthesisResult {
        output_path: params.output_path.clone(),
        duration_sec,
        sample_count: all_samples.len(),
    })
}

// ---------------------------------------------------------------------------
// 模型目錄提示（保留供未來 UI 資訊面板使用）
// ---------------------------------------------------------------------------

#[allow(dead_code)]
/// 顯示模型 snapshot 目錄的中文描述
fn model_dir_description(model_dir: &Path) -> String {
    for component in model_dir.components().rev() {
        let name = component.as_os_str().to_string_lossy();
        if let Some(cache_name) = name.strip_prefix("models--") {
            return cache_name.replace("--", "/");
        }
    }
    model_dir.to_string_lossy().to_string()
}

// ---------------------------------------------------------------------------
// eframe App
// ---------------------------------------------------------------------------

/// 主 GUI 應用程式狀態
pub struct TtsGuiApp {
    // === 輸入參數 ===
    text: String,
    model_id: String,
    model_dir: String,
    backend: BackendKind,
    language: String,
    language_auto: bool,
    speaker: String,
    speed: f64,
    instruct: String,
    output_path: String,
    max_new_tokens: String,
    seed: String,

    // === Voice Clone 參數 ===
    reference_audio: String,
    reference_text: String,

    // === 模型下載設定 ===
    models_base_dir: String,
    auto_download: bool,
    hf_mirror_index: usize,
    hf_custom_mirror: String,
    is_downloading: bool,

    // === 內部狀態 ===
    status_log: Vec<(String, Color32)>,
    is_processing: bool,
    cancel_flag: Arc<AtomicBool>,

    // === 背景執行緒通訊 ===
    event_rx: Option<mpsc::Receiver<WorkerEvent>>,

    // === 最後合成結果 ===
    last_result: Option<SynthesisResult>,
}

/// 取得 models 預設存放路徑（預設在 qwen3tts-gui.exe 執行檔同層或專案目錄下的 models 資料夾）
pub fn default_models_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            // 若在 target/release 或 target/debug，找專案根目錄的 models
            if (parent.ends_with("release") || parent.ends_with("debug"))
                && parent
                    .parent()
                    .and_then(|p| p.parent())
                    .map(|p| p.join("models").exists())
                    .unwrap_or(false)
            {
                return parent.parent().unwrap().parent().unwrap().join("models");
            }
            return parent.join("models");
        }
    }
    PathBuf::from("models")
}

/// 依據輸入文字長度自動估算適當的音訊 Token 上限（避免跑滿 4096 幀佔用 100% GPU 與爆顯存）
pub fn estimate_max_tokens_for_text(text: &str) -> u32 {
    let zh_count = text
        .chars()
        .filter(|&c| ('\u{4e00}'..='\u{9fff}').contains(&c))
        .count();
    let total_count = text.chars().filter(|c| !c.is_whitespace()).count();
    let other_count = total_count.saturating_sub(zh_count);

    // 中文 1 字約 3.5~4 幀，英文/標點約 1.5 幀，再加上 80 幀安全餘量（約 6 秒的標點停頓與尾音）
    let est = (zh_count * 4 + other_count * 2 + 80) as u32;
    // 下限 150 幀（約 12 秒），上限 2048 幀（約 170 秒）
    est.clamp(150, 2048)
}

impl Default for TtsGuiApp {
    fn default() -> Self {
        Self {
            text: String::new(),
            model_id: MODEL_ID_CANDIDATES[0].to_string(),
            model_dir: String::new(),
            models_base_dir: default_models_dir().to_string_lossy().to_string(),
            backend: BackendKind::default(),
            language: "auto".to_string(),
            language_auto: true,
            speaker: "Vivian".to_string(),
            speed: 1.0,
            instruct: String::new(),
            output_path: "output.wav".to_string(),
            max_new_tokens: "auto".to_string(),
            seed: String::new(),
            reference_audio: String::new(),
            reference_text: String::new(),
            auto_download: true,
            hf_mirror_index: 0,
            hf_custom_mirror: String::new(),
            is_downloading: false,
            status_log: Vec::new(),
            is_processing: false,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            event_rx: None,
            last_result: None,
        }
    }
}

impl TtsGuiApp {
    pub fn new() -> Self {
        let mut app = Self::default();
        app.status_log.push((
            format!(
                "Qwen3-TTS Rust 語音合成 v{} — 輸入文字後按「合成」開始",
                env!("CARGO_PKG_VERSION")
            ),
            Color32::GRAY,
        ));
        app
    }

    /// 取得目前設定的 HuggingFace 下載端點
    fn effective_hf_endpoint(&self) -> Option<String> {
        match self.hf_mirror_index {
            0 => None,
            1 => Some(crate::downloader::HF_MIRROR_ENDPOINT.to_string()),
            _ => {
                let trimmed = self.hf_custom_mirror.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            }
        }
    }

    /// 啟動手動背景下載模型與 Tokenizer 權重
    fn start_model_download(&mut self) {
        if self.is_processing || self.is_downloading {
            return;
        }

        let (tx, rx) = mpsc::channel::<WorkerEvent>();
        let model_id = self.model_id.clone();
        let endpoint = self.effective_hf_endpoint();
        let models_base_dir = if self.models_base_dir.trim().is_empty() {
            default_models_dir()
        } else {
            PathBuf::from(self.models_base_dir.trim())
        };

        self.is_processing = true;
        self.is_downloading = true;
        self.cancel_flag.store(false, Ordering::SeqCst);
        self.event_rx = Some(rx);

        let cancel = self.cancel_flag.clone();
        let download_dir = models_base_dir.clone();

        thread::spawn(move || {
            let tx_c = tx.clone();
            let res = crate::downloader::ensure_synthesis_resources(
                &model_id,
                Some(&download_dir),
                endpoint.as_deref(),
                Some(&cancel),
                move |msg| {
                    tx_c.send(WorkerEvent::Status(msg.to_string())).ok();
                },
            );
            match res {
                Ok((model_dir, _)) => {
                    tx.send(WorkerEvent::DownloadDone(Ok(model_dir))).ok();
                }
                Err(e) => {
                    tx.send(WorkerEvent::DownloadDone(Err(e.to_string()))).ok();
                }
            }
        });

        self.status_log.push((
            format!(
                "▶ 開始下載模型與資源 ({}) 至 {}…",
                self.model_id,
                models_base_dir.display()
            ),
            Color32::WHITE,
        ));
    }

    /// 啟動背景合成
    fn start_synthesis(&mut self) {
        if self.text.trim().is_empty() {
            self.status_log
                .push(("⚠️ 請輸入要合成的文字".into(), Color32::YELLOW));
            return;
        }

        let is_base = self.model_id.contains("-Base");
        let is_voice_design = self.model_id.contains("VoiceDesign");

        if is_base {
            if self.reference_audio.trim().is_empty() {
                self.status_log.push((
                    "⚠️ Base 模型為「聲音複製專用（Voice Clone）」，必須提供參考音檔！\n\
                     💡 提示：若您想直接使用預設說話者發音，請將上方「模型 ID」切換為「Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice」！".into(),
                    Color32::YELLOW,
                ));
                return;
            }
        } else if is_voice_design && self.instruct.trim().is_empty() {
            self.status_log.push((
                "⚠️ VoiceDesign 模型需要提供「語氣指令」（請在進階設定中輸入語氣描述）。".into(),
                Color32::YELLOW,
            ));
            return;
        }

        let effective_speaker = if is_base || is_voice_design {
            None
        } else if self.speaker.trim().is_empty() {
            Some("Vivian".to_string())
        } else {
            Some(self.speaker.trim().to_string())
        };

        let effective_instruct = if is_base {
            None
        } else if self.instruct.trim().is_empty() {
            None
        } else {
            Some(self.instruct.trim().to_string())
        };

        let (tx, rx) = mpsc::channel::<WorkerEvent>();

        let params = SynthesisParams {
            text: self.text.clone(),
            model_id: self.model_id.clone(),
            model_dir: if self.model_dir.trim().is_empty() {
                None
            } else {
                Some(PathBuf::from(self.model_dir.trim()))
            },
            models_base_dir: if self.models_base_dir.trim().is_empty() {
                default_models_dir()
            } else {
                PathBuf::from(self.models_base_dir.trim())
            },
            backend: self.backend,
            language: if self.language_auto {
                "auto".into()
            } else {
                self.language.clone()
            },
            speaker: effective_speaker,
            instruct: effective_instruct,
            speed: self.speed,
            output_path: PathBuf::from(&self.output_path),
            reference_audio: if self.reference_audio.trim().is_empty() {
                None
            } else {
                Some(PathBuf::from(self.reference_audio.trim()))
            },
            reference_text: if self.reference_text.trim().is_empty() {
                None
            } else {
                Some(self.reference_text.trim().to_string())
            },
            seed: if self.seed.trim().is_empty() {
                None
            } else {
                match self.seed.trim().parse() {
                    Ok(s) => Some(s),
                    Err(_) => {
                        self.status_log
                            .push(("⚠️ seed 格式無效，使用隨機種子".into(), Color32::YELLOW));
                        None
                    }
                }
            },
            max_new_tokens: if self.max_new_tokens.trim().is_empty()
                || self.max_new_tokens.trim().eq_ignore_ascii_case("auto")
            {
                estimate_max_tokens_for_text(&self.text)
            } else {
                self.max_new_tokens
                    .trim()
                    .parse()
                    .unwrap_or_else(|_| estimate_max_tokens_for_text(&self.text))
            },
            auto_download: self.auto_download,
            hf_mirror: self.effective_hf_endpoint(),
        };

        self.is_processing = true;
        self.cancel_flag.store(false, Ordering::SeqCst);
        self.event_rx = Some(rx);

        let cancel = self.cancel_flag.clone();

        thread::spawn(move || {
            run_synthesis(params, tx, cancel);
        });

        self.status_log.push(("▶ 開始合成…".into(), Color32::WHITE));
    }

    /// 取消合成
    fn cancel_synthesis(&mut self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
        self.status_log
            .push(("⏹ 正在取消…".into(), Color32::YELLOW));
        self.is_processing = false;
        self.is_downloading = false;
    }

    /// 處理背景執行緒事件
    fn poll_events(&mut self) {
        let Some(rx) = &self.event_rx else { return };
        while let Ok(event) = rx.try_recv() {
            match event {
                WorkerEvent::Status(msg) => {
                    self.status_log.push((msg, Color32::WHITE));
                }
                WorkerEvent::Progress { .. } => {
                    // 可以擴充進度條
                }
                WorkerEvent::DownloadDone(result) => {
                    self.is_processing = false;
                    self.is_downloading = false;
                    match result {
                        Ok(dir) => {
                            self.status_log.push((
                                format!("🎉 模型與解碼器已全部就緒！路徑：{}", dir.display()),
                                Color32::GREEN,
                            ));
                        }
                        Err(err) => {
                            self.status_log.push((
                                format!("❌ 下載失敗：{err}"),
                                Color32::RED,
                            ));
                        }
                    }
                }
                WorkerEvent::Done(result) => {
                    self.is_processing = false;
                    self.is_downloading = false;
                    match result {
                        Ok(res) => {
                            self.last_result = Some(res.clone());
                            self.status_log.push((
                                format!(
                                    "✅ 合成完成！已儲存：{} ({:.2} 秒)",
                                    res.output_path.display(),
                                    res.duration_sec
                                ),
                                Color32::GREEN,
                            ));
                        }
                        Err(err) => {
                            self.status_log
                                .push((format!("❌ 合成失敗：{err}"), Color32::RED));
                        }
                    }
                }
            }
        }
    }

    /// 播放音檔
    fn play_audio(&self) {
        if let Some(result) = &self.last_result {
            let path = &result.output_path;
            if path.exists() {
                match rodio::OutputStream::try_default() {
                    Ok((stream, handle)) => {
                        match rodio::Sink::try_new(&handle) {
                            Ok(sink) => {
                                match std::fs::File::open(path) {
                                    Ok(file) => {
                                        match rodio::Decoder::new(std::io::BufReader::new(file)) {
                                            Ok(decoder) => {
                                                sink.append(decoder);
                                                sink.detach(); // 播放不受 sink 生命週期限制
                                                // Note: `stream` must stay alive for playback
                                                std::mem::forget(stream);
                                            }
                                            Err(e) => {
                                                eprintln!("rodio Decoder 錯誤：{e}");
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("開啟音檔失敗：{e}");
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("rodio Sink 建立失敗：{e}");
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("rodio OutputStream 初始化失敗：{e}");
                    }
                }
            }
        }
    }
}

impl eframe::App for TtsGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 1) 輪詢背景事件
        self.poll_events();

        // 2) 佈局
        egui::TopBottomPanel::top("header")
            .frame(Frame {
                fill: ctx.style().visuals.window_fill(),
                inner_margin: Margin::symmetric(16, 8),
                ..Default::default()
            })
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("🎙 Qwen3-TTS 語音合成")
                            .heading()
                            .color(Color32::from_rgb(100, 180, 255)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                                .size(12.0)
                                .color(Color32::GRAY),
                        );
                    });
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ScrollArea::vertical()
                .id_salt("main_scroll")
                .show(ui, |ui| {
                    ui.add_space(8.0);

                    // === 文字輸入區域 ===
                    Frame::group(ui.style())
                        .inner_margin(Margin::same(8))
                        .show(ui, |ui| {
                            ui.label(RichText::new("📝 合成文字").strong());
                            ui.add_space(4.0);
                            egui::TextEdit::multiline(&mut self.text)
                                .hint_text("請輸入要合成語音的文字…")
                                .desired_rows(8)
                                .desired_width(f32::INFINITY)
                                .frame(true)
                                .show(ui);
                        });

                    ui.add_space(8.0);

                    // === 基本參數 ===
                    Frame::group(ui.style())
                        .inner_margin(Margin::same(8))
                        .show(ui, |ui| {
                            ui.label(RichText::new("⚙ 參數設定").strong());
                            ui.add_space(4.0);
                            egui::Grid::new("params_grid")
                                .num_columns(2)
                                .spacing([16.0, 6.0])
                                .striped(true)
                                .show(ui, |ui| {
                                    // Model ID
                                    ui.label("模型 ID：");
                                    ui.vertical(|ui| {
                                        ui.horizontal(|ui| {
                                            egui::ComboBox::from_id_salt("model_combo")
                                                .width(300.0)
                                                .selected_text(&self.model_id)
                                                .show_ui(ui, |ui| {
                                                    for m in MODEL_ID_CANDIDATES {
                                                        ui.selectable_value(
                                                            &mut self.model_id,
                                                            m.to_string(),
                                                            *m,
                                                        );
                                                    }
                                                });

                                            let base_dir = if self.models_base_dir.trim().is_empty() {
                                                None
                                            } else {
                                                Some(Path::new(self.models_base_dir.trim()))
                                            };
                                            let is_ready = crate::downloader::is_model_ready(
                                                &self.model_id,
                                                base_dir,
                                            );

                                            if is_ready {
                                                ui.label(
                                                    RichText::new("🟢 已就緒")
                                                        .size(12.0)
                                                        .color(Color32::from_rgb(80, 210, 80)),
                                                );
                                            } else {
                                                ui.label(
                                                    RichText::new("⚪ 未下載")
                                                        .size(12.0)
                                                        .color(Color32::from_rgb(220, 150, 80)),
                                                );
                                            }

                                            let dl_btn_text = if is_ready {
                                                "🔄 重新下載"
                                            } else {
                                                "📥 下載模型"
                                            };
                                            let dl_btn = egui::Button::new(
                                                RichText::new(dl_btn_text).size(12.0),
                                            );
                                            if self.is_processing || self.is_downloading {
                                                ui.add_enabled(false, dl_btn);
                                            } else if ui.add(dl_btn).clicked() {
                                                self.start_model_download();
                                            }
                                        });

                                        ui.horizontal(|ui| {
                                            ui.checkbox(
                                                &mut self.auto_download,
                                                "當模型缺失時自動下載",
                                            );
                                            ui.add_space(8.0);
                                            let tok_ready = crate::downloader::is_tokenizer_ready()
                                                || (!self.models_base_dir.trim().is_empty()
                                                    && (Path::new(self.models_base_dir.trim())
                                                        .join("tokenizer")
                                                        .join("codebook.safetensors")
                                                        .exists()
                                                        || Path::new(self.models_base_dir.trim())
                                                            .join("tokenizer-q8")
                                                            .join("codebook.safetensors")
                                                            .exists()));
                                            if tok_ready {
                                                ui.label(
                                                    RichText::new("解碼器: 🟢")
                                                        .size(11.0)
                                                        .color(Color32::from_rgb(80, 210, 80)),
                                                );
                                            } else {
                                                ui.label(
                                                    RichText::new("解碼器: ⚪ 未就緒")
                                                        .size(11.0)
                                                        .color(Color32::from_rgb(220, 150, 80)),
                                                );
                                            }
                                        });
                                    });
                                    ui.end_row();

                                    // Models storage directory
                                    ui.label("模型存放目錄：");
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::TextEdit::singleline(&mut self.models_base_dir)
                                                .desired_width(260.0)
                                                .hint_text("預設：專案或執行檔同層 models 資料夾"),
                                        );
                                        if ui.button("瀏覽…").clicked() {
                                            if let Some(path) = rfd::FileDialog::new()
                                                .set_title("選擇 models 存放目錄")
                                                .pick_folder()
                                            {
                                                self.models_base_dir =
                                                    path.to_string_lossy().to_string();
                                            }
                                        }
                                        if ui.button("重設預設").clicked() {
                                            self.models_base_dir =
                                                default_models_dir().to_string_lossy().to_string();
                                        }
                                    });
                                    ui.end_row();

                                    // Backend
                                    ui.label("後端：");
                                    ui.horizontal(|ui| {
                                        for b in BackendKind::ALL {
                                            ui.selectable_value(&mut self.backend, *b, b.name());
                                        }
                                    });
                                    ui.end_row();

                                    // Device
                                    ui.label("運算裝置：");
                                    ui.horizontal(|ui| {
                                        #[cfg(feature = "cuda")]
                                        {
                                            ui.label(
                                                RichText::new("⚡ CUDA (NVIDIA GPU 硬體加速已啟用)")
                                                    .strong()
                                                    .color(Color32::from_rgb(80, 220, 80)),
                                            );
                                        }
                                        #[cfg(not(feature = "cuda"))]
                                        {
                                            ui.label(
                                                RichText::new("🖥 CPU 運算")
                                                    .color(Color32::from_rgb(220, 180, 80)),
                                            );
                                        }
                                    });
                                    ui.end_row();

                                    // Language
                                    ui.label("語言：");
                                    ui.horizontal(|ui| {
                                        ui.checkbox(&mut self.language_auto, "自動偵測");
                                        if !self.language_auto {
                                            egui::ComboBox::from_id_salt("lang_combo")
                                                .width(200.0)
                                                .selected_text(&self.language)
                                                .show_ui(ui, |ui| {
                                                    for lang in SUPPORTED_LANGUAGES {
                                                        ui.selectable_value(
                                                            &mut self.language,
                                                            lang.to_string(),
                                                            *lang,
                                                        );
                                                    }
                                                });
                                        }
                                    });
                                    ui.end_row();

                                    // Speaker
                                    ui.label("說話者：");
                                    ui.horizontal(|ui| {
                                        let is_base = self.model_id.contains("-Base");
                                        let is_voice_design = self.model_id.contains("VoiceDesign");

                                        if is_base {
                                            ui.label(
                                                RichText::new("（Base 模型專用於聲音複製，不支援預設說話者，請在進階設定提供「參考音訊」）")
                                                    .size(11.5)
                                                    .color(Color32::from_rgb(220, 160, 60)),
                                            );
                                        } else if is_voice_design {
                                            ui.label(
                                                RichText::new("（VoiceDesign 模型使用「語氣指令」設計聲音）")
                                                    .size(11.5)
                                                    .color(Color32::from_rgb(180, 180, 180)),
                                            );
                                        } else {
                                            egui::ComboBox::from_id_salt("speaker_combo")
                                                .width(200.0)
                                                .selected_text(if self.speaker.is_empty() {
                                                    "Vivian".into()
                                                } else {
                                                    self.speaker.clone()
                                                })
                                                .show_ui(ui, |ui| {
                                                    for name in speaker_presets::speaker_names() {
                                                        ui.selectable_value(
                                                            &mut self.speaker,
                                                            name.to_string(),
                                                            name,
                                                        );
                                                    }
                                                });
                                        }
                                    });
                                    ui.end_row();

                                    // Speed
                                    ui.label("語速：");
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::Slider::new(&mut self.speed, 0.5..=2.0)
                                                .step_by(0.05)
                                                .text("x")
                                                .fixed_decimals(2)
                                                .clamping(egui::SliderClamping::Always),
                                        );
                                    });
                                    ui.end_row();
                                });
                        });

                    ui.add_space(8.0);

                    // === 進階設定（可折疊）===
                    egui::CollapsingHeader::new("🔧 進階設定")
                        .default_open(false)
                        .show(ui, |ui| {
                            Frame::group(ui.style())
                                .inner_margin(Margin::same(8))
                                .show(ui, |ui| {
                                    egui::Grid::new("advanced_grid")
                                        .num_columns(2)
                                        .spacing([16.0, 6.0])
                                        .striped(true)
                                        .show(ui, |ui| {
                                            // Instruction
                                            ui.label("語氣指令：");
                                            ui.add(
                                                egui::TextEdit::singleline(&mut self.instruct)
                                                    .hint_text("例如：年輕女性，台灣口語，溫柔親切")
                                                    .desired_width(360.0),
                                            );
                                            ui.end_row();

                                            // Output path
                                            ui.label("輸出路徑：");
                                            ui.horizontal(|ui| {
                                                ui.add(
                                                    egui::TextEdit::singleline(
                                                        &mut self.output_path,
                                                    )
                                                    .desired_width(280.0),
                                                );
                                                if ui.button("瀏覽…").clicked() {
                                                    if let Some(path) = rfd::FileDialog::new()
                                                        .set_title("選擇儲存位置")
                                                        .add_filter("WAV", &["wav"])
                                                        .set_file_name("output.wav")
                                                        .save_file()
                                                    {
                                                        self.output_path =
                                                            path.to_string_lossy().to_string();
                                                    }
                                                }
                                            });
                                            ui.end_row();

                                            // Max new tokens
                                            ui.label("最大 Token 數：");
                                            ui.horizontal(|ui| {
                                                ui.add(
                                                    egui::TextEdit::singleline(
                                                        &mut self.max_new_tokens,
                                                    )
                                                    .desired_width(90.0)
                                                    .hint_text("auto"),
                                                );
                                                let est = estimate_max_tokens_for_text(&self.text);
                                                ui.label(
                                                    RichText::new(format!(
                                                        "（目前預估約 {est} 幀，填 auto 即依字數自適應）"
                                                    ))
                                                    .size(11.5)
                                                    .color(Color32::from_rgb(180, 180, 180)),
                                                );
                                            });
                                            ui.end_row();

                                            // Seed
                                            ui.label("隨機種子：");
                                            ui.horizontal(|ui| {
                                                ui.add(
                                                    egui::TextEdit::singleline(&mut self.seed)
                                                        .desired_width(200.0)
                                                        .hint_text("留空 = 隨機"),
                                                );
                                            });
                                            ui.end_row();

                                            // Model dir (Candle only)
                                            #[cfg(feature = "candle-llm")]
                                            {
                                                ui.label("指定快照目錄 (覆寫)：");
                                                ui.horizontal(|ui| {
                                                    ui.add(
                                                        egui::TextEdit::singleline(
                                                            &mut self.model_dir,
                                                        )
                                                        .desired_width(280.0)
                                                        .hint_text("留空 = 自動從模型存放目錄或快取尋找"),
                                                    );
                                                    if ui.button("選擇…").clicked() {
                                                        if let Some(path) = rfd::FileDialog::new()
                                                            .set_title("選擇模型目錄")
                                                            .pick_folder()
                                                        {
                                                            self.model_dir =
                                                                path.to_string_lossy().to_string();
                                                        }
                                                    }
                                                });
                                                ui.end_row();
                                            }

                                            // HF Mirror endpoint
                                            ui.label("下載鏡像源：");
                                            ui.horizontal(|ui| {
                                                egui::ComboBox::from_id_salt("mirror_combo")
                                                    .width(220.0)
                                                    .selected_text(match self.hf_mirror_index {
                                                        0 => "官方站 (huggingface.co)",
                                                        1 => "國內鏡像 (hf-mirror.com)",
                                                        _ => "自訂鏡像端點…",
                                                    })
                                                    .show_ui(ui, |ui| {
                                                        ui.selectable_value(
                                                            &mut self.hf_mirror_index,
                                                            0,
                                                            "官方站 (huggingface.co)",
                                                        );
                                                        ui.selectable_value(
                                                            &mut self.hf_mirror_index,
                                                            1,
                                                            "國內鏡像 (hf-mirror.com)",
                                                        );
                                                        ui.selectable_value(
                                                            &mut self.hf_mirror_index,
                                                            2,
                                                            "自訂鏡像端點…",
                                                        );
                                                    });

                                                if self.hf_mirror_index == 2 {
                                                    ui.add(
                                                        egui::TextEdit::singleline(
                                                            &mut self.hf_custom_mirror,
                                                        )
                                                        .desired_width(180.0)
                                                        .hint_text("https://..."),
                                                    );
                                                }
                                            });
                                            ui.end_row();

                                            // Reference audio (Voice Clone)
                                            ui.label("參考音訊：");
                                            ui.horizontal(|ui| {
                                                ui.add(
                                                    egui::TextEdit::singleline(
                                                        &mut self.reference_audio,
                                                    )
                                                    .desired_width(280.0)
                                                    .hint_text("Voice Clone 參考音訊路徑"),
                                                );
                                                if ui.button("選擇音檔…").clicked() {
                                                    if let Some(path) = rfd::FileDialog::new()
                                                        .set_title("選擇參考音訊")
                                                        .add_filter("WAV", &["wav"])
                                                        .pick_file()
                                                    {
                                                        self.reference_audio =
                                                            path.to_string_lossy().to_string();
                                                    }
                                                }
                                            });
                                            ui.end_row();

                                            // Reference text
                                            ui.label("參考逐字稿：");
                                            ui.add(
                                                egui::TextEdit::singleline(
                                                    &mut self.reference_text,
                                                )
                                                .desired_width(360.0)
                                                .hint_text(
                                                    "Voice Clone 參考音訊的逐字稿（建議提供）",
                                                ),
                                            );
                                            ui.end_row();
                                        });
                                });
                        });

                    ui.add_space(12.0);

                    // === 操作按鈕 ===
                    ui.horizontal(|ui| {
                        ui.set_min_height(36.0);

                        let synth_btn = egui::Button::new(RichText::new("🎵 合成").size(16.0))
                            .min_size(Vec2::new(120.0, 36.0))
                            .fill(Color32::from_rgb(40, 120, 60));

                        if self.is_processing {
                            ui.add_enabled(false, synth_btn);
                        } else if ui.add(synth_btn).clicked() {
                            self.start_synthesis();
                        }

                        ui.add_space(8.0);

                        let cancel_btn = egui::Button::new(RichText::new("⏹ 取消").size(16.0))
                            .min_size(Vec2::new(80.0, 36.0))
                            .fill(Color32::from_rgb(140, 40, 40));

                        if self.is_processing {
                            if ui.add(cancel_btn).clicked() {
                                self.cancel_synthesis();
                            }
                        } else {
                            ui.add_enabled(false, cancel_btn);
                        }

                        ui.add_space(16.0);

                        // Play button
                        let has_result = self.last_result.is_some();
                        let play_btn = egui::Button::new(RichText::new("▶ 播放").size(16.0))
                            .min_size(Vec2::new(80.0, 36.0));

                        if has_result {
                            if ui.add(play_btn).clicked() {
                                self.play_audio();
                            }
                        } else {
                            ui.add_enabled(false, play_btn);
                        }

                        ui.add_space(8.0);

                        // Save As button
                        let save_btn = egui::Button::new(RichText::new("💾 另存新檔…").size(16.0))
                            .min_size(Vec2::new(100.0, 36.0));

                        if has_result {
                            if ui.add(save_btn).clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .set_title("另存新檔")
                                    .add_filter("WAV", &["wav"])
                                    .set_file_name("output.wav")
                                    .save_file()
                                {
                                    let src =
                                        self.last_result.as_ref().map(|r| r.output_path.clone());
                                    if let Some(src) = src {
                                        let _ = std::fs::copy(&src, &path);
                                        self.status_log.push((
                                            format!("💾 已另存：{}", path.display()),
                                            Color32::WHITE,
                                        ));
                                    }
                                }
                            }
                        } else {
                            ui.add_enabled(false, save_btn);
                        }

                        // Processing spinner
                        if self.is_processing {
                            ui.add_space(12.0);
                            ui.add(egui::Spinner::new().size(20.0));
                            if self.is_downloading {
                                ui.label(
                                    RichText::new("正在下載模型與資源…")
                                        .size(13.0)
                                        .color(Color32::from_rgb(100, 180, 255)),
                                );
                            }
                        }
                    });

                    ui.add_space(12.0);

                    // === 狀態 Log ===
                    Frame::group(ui.style())
                        .inner_margin(Margin::same(8))
                        .show(ui, |ui| {
                            ui.label(RichText::new("📋 執行紀錄").strong());
                            ui.add_space(4.0);
                            ScrollArea::vertical()
                                .id_salt("log_scroll")
                                .max_height(200.0)
                                .stick_to_bottom(true)
                                .show(ui, |ui| {
                                    ui.with_layout(
                                        egui::Layout::top_down_justified(egui::Align::LEFT),
                                        |ui| {
                                            for (msg, color) in &self.status_log {
                                                ui.label(
                                                    RichText::new(msg).size(12.0).color(*color),
                                                );
                                            }
                                        },
                                    );
                                });
                        });

                    ui.add_space(8.0);

                    // === 快速提示 ===
                    ui.label(
                        RichText::new(
                            "💡 提示：合成較長文字時請耐心等候。Candle 後端可選用 CUDA 加速。",
                        )
                        .size(11.0)
                        .color(Color32::GRAY),
                    );
                });
        });

        // 3) 持續更新（若處理中則每幀重繪）
        if self.is_processing {
            ctx.request_repaint();
        }
    }
}
