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
use std::sync::{mpsc, Arc};
use std::thread;

use eframe::egui;
use egui::{Color32, Frame, Margin, RichText, ScrollArea, Vec2};

use crate::text_frontend::model_catalog::SUPPORTED_LANGUAGES;
use crate::text_frontend::speaker_presets;
use crate::text_frontend::{PythonBridge, SynthesisOptions, TextFrontend};
use crate::{Decoder12Hz, DecoderConfig};

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
            &[Self::Python, Self::Candle]
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
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
];

// ---------------------------------------------------------------------------
// 工具函式 — 僅供背景執行緒使用
// ---------------------------------------------------------------------------

#[allow(dead_code)]
mod worker {
    use super::*;
    use std::sync::mpsc::Sender;

    /// 尋找 HuggingFace 快取中的模型 snapshot
    #[cfg(feature = "candle-llm")]
    pub fn locate_model_snapshot(model_id: &str) -> Option<PathBuf> {
        if let Ok(env_dir) = std::env::var("QWEN3_TTS_MODEL_DIR") {
            let p = PathBuf::from(env_dir);
            if p.join("model.safetensors").exists() {
                return Some(p);
            }
        }
        locate_hf_snapshot(model_id)
    }

    fn locate_hf_snapshot(model_id: &str) -> Option<PathBuf> {
        let home = std::env::var("USERPROFILE")
            .or_else(|_| std::env::var("HOME"))
            .ok()
            .map(PathBuf::from)?;
        let hub = home.join(".cache").join("huggingface").join("hub");
        let repo_dir = hub.join(format!("models--{}", model_id.replace('/', "--")));
        let snapshots = repo_dir.join("snapshots");
        let Ok(entries) = std::fs::read_dir(&snapshots) else {
            return None;
        };
        for entry in entries.flatten() {
            let candidate = entry.path().join("model.safetensors");
            if candidate.exists() {
                return Some(entry.path());
            }
        }
        None
    }

    pub fn find_tokenizer_json(model_id: &str, model_dir: &Path) -> Option<PathBuf> {
        let local = model_dir.join("tokenizer.json");
        if local.exists() {
            return Some(local);
        }
        let snapshots = model_dir.join("snapshots");
        if snapshots.is_dir() {
            for entry in std::fs::read_dir(snapshots).ok()?.flatten() {
                let path = entry.path().join("tokenizer.json");
                if path.exists() {
                    return Some(path);
                }
            }
        }
        for fallback in ["models/tokenizer_1.7b.json", "models/tokenizer.json"] {
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
            if let Some(base_dir) = locate_hf_snapshot(base_id) {
                let path = base_dir.join("tokenizer.json");
                if path.exists() {
                    return Some(path);
                }
            }
        }
        None
    }

    pub fn ensure_decoder_weights(tx: &Sender<WorkerEvent>) -> Result<PathBuf, String> {
        tx.send(WorkerEvent::Status("檢查 Tokenizer 解碼器權重…".into()))
            .ok();
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
                Ok(device) => device,
                Err(_) => candle_core::Device::Cpu,
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
        candle_core::Device::Cpu => "CPU".to_string(),
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(_) => "CUDA".to_string(),
        _ => "Unknown".to_string(),
    };
    tx.send(WorkerEvent::Status(format!("裝置：{device_name}")))
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
                    worker::locate_model_snapshot(&params.model_id)
                        .ok_or_else(|| {
                            format!(
                                "找不到模型 {} 的本地 snapshot。請設定 --model-dir 或 QWEN3_TTS_MODEL_DIR",
                                params.model_id
                            )
                        })?
                };

                let sf_path = dir.join("model.safetensors");
                if !sf_path.exists() {
                    return Err(format!("模型 safetensors 不存在：{}", sf_path.display()));
                }

                let tok_path = worker::find_tokenizer_json(&params.model_id, &dir)
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
    let weight_dir = worker::ensure_decoder_weights(tx)?;

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
    tx.send(WorkerEvent::Status("解碼 Token → 音訊…".into()))
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

    // === 內部狀態 ===
    status_log: Vec<(String, Color32)>,
    is_processing: bool,
    cancel_flag: Arc<AtomicBool>,

    // === 背景執行緒通訊 ===
    event_rx: Option<mpsc::Receiver<WorkerEvent>>,

    // === 最後合成結果 ===
    last_result: Option<SynthesisResult>,
}

impl Default for TtsGuiApp {
    fn default() -> Self {
        Self {
            text: String::new(),
            model_id: MODEL_ID_CANDIDATES[0].to_string(),
            model_dir: String::new(),
            backend: BackendKind::default(),
            language: "auto".to_string(),
            language_auto: true,
            speaker: String::new(),
            speed: 1.0,
            instruct: String::new(),
            output_path: "output.wav".to_string(),
            max_new_tokens: "4096".to_string(),
            seed: String::new(),
            reference_audio: String::new(),
            reference_text: String::new(),
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
            "Qwen3-TTS Rust 語音合成 v0.1.16 — 輸入文字後按「合成」開始".into(),
            Color32::GRAY,
        ));
        app
    }

    /// 啟動背景合成
    fn start_synthesis(&mut self) {
        if self.text.trim().is_empty() {
            self.status_log
                .push(("⚠️ 請輸入要合成的文字".into(), Color32::YELLOW));
            return;
        }

        let (tx, rx) = mpsc::channel::<WorkerEvent>();

        let params = SynthesisParams {
            text: self.text.clone(),
            model_id: self.model_id.clone(),
            model_dir: if self.model_dir.trim().is_empty() {
                None
            } else {
                Some(PathBuf::from(self.model_dir.trim()))
            },
            backend: self.backend,
            language: if self.language_auto {
                "auto".into()
            } else {
                self.language.clone()
            },
            speaker: if self.speaker.trim().is_empty() {
                None
            } else {
                Some(self.speaker.trim().to_string())
            },
            instruct: if self.instruct.trim().is_empty() {
                None
            } else {
                Some(self.instruct.trim().to_string())
            },
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
            max_new_tokens: self.max_new_tokens.parse().unwrap_or(4096),
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
                WorkerEvent::Done(result) => {
                    self.is_processing = false;
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
                        ui.label(RichText::new("v0.1.16").size(12.0).color(Color32::GRAY));
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
                                    egui::ComboBox::from_id_salt("model_combo")
                                        .width(360.0)
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
                                    ui.end_row();

                                    // Backend
                                    ui.label("後端：");
                                    ui.horizontal(|ui| {
                                        for b in BackendKind::ALL {
                                            ui.selectable_value(&mut self.backend, *b, b.name());
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
                                        egui::ComboBox::from_id_salt("speaker_combo")
                                            .width(200.0)
                                            .selected_text(if self.speaker.is_empty() {
                                                "（無）".into()
                                            } else {
                                                self.speaker.clone()
                                            })
                                            .show_ui(ui, |ui| {
                                                ui.selectable_value(
                                                    &mut self.speaker,
                                                    String::new(),
                                                    "（無）",
                                                );
                                                for name in speaker_presets::speaker_names() {
                                                    ui.selectable_value(
                                                        &mut self.speaker,
                                                        name.to_string(),
                                                        name,
                                                    );
                                                }
                                            });
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
                                            ui.add(
                                                egui::TextEdit::singleline(
                                                    &mut self.max_new_tokens,
                                                )
                                                .desired_width(120.0)
                                                .hint_text("4096"),
                                            );
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
                                                ui.label("模型目錄：");
                                                ui.horizontal(|ui| {
                                                    ui.add(
                                                        egui::TextEdit::singleline(
                                                            &mut self.model_dir,
                                                        )
                                                        .desired_width(280.0)
                                                        .hint_text("留空 = 自動從 HF 快取找"),
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
