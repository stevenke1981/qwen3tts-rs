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
use crate::text_frontend::{PythonBridge, SynthesisOptions};
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
}

impl Default for BackendKind {
    fn default() -> Self {
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

/// 運算裝置偏好設定
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    /// 自動偵測：若有可用 CUDA GPU 則優先使用，失敗或無 GPU 時自動回退至 CPU
    Auto,
    /// 強制使用 NVIDIA CUDA GPU 加速（無可用 GPU 或未編譯 CUDA 時回傳錯誤）
    Cuda,
    /// 強制使用 CPU 運算
    Cpu,
}

impl DevicePreference {
    pub const ALL: &'static [Self] = &[Self::Auto, Self::Cuda, Self::Cpu];

    pub fn name(&self) -> &'static str {
        match self {
            Self::Auto => "自動偵測 (優先 CUDA GPU)",
            Self::Cuda => "⚡ CUDA (NVIDIA GPU)",
            Self::Cpu => "🖥️ CPU (中央處理器)",
        }
    }
}

impl Default for DevicePreference {
    fn default() -> Self {
        Self::Auto
    }
}

/// Initialize the requested device and return its actual name or failure reason.
///
/// This is blocking; desktop callers must run it on a background thread.
/// Dynamic CUDA DLLs required at process startup must already be installed.
pub fn probe_device(preference: DevicePreference) -> Result<String, String> {
    worker::resolve_device(preference).map(|(_, name)| name)
}

/// Non-blocking device controls shared by the desktop UI and headless tests.
#[derive(Default)]
pub struct DeviceSelection {
    preference: DevicePreference,
    pending: Option<mpsc::Receiver<Result<String, String>>>,
    detected: Option<Result<String, String>>,
    actual: Option<Result<String, String>>,
}

impl DeviceSelection {
    /// Return the preference passed to the next synthesis worker.
    pub fn preference(&self) -> DevicePreference {
        self.preference
    }

    /// Change the preference, invalidating the previous probe result.
    pub fn set_preference(&mut self, preference: DevicePreference) {
        if self.preference != preference {
            self.preference = preference;
            self.detected = None;
            self.actual = None;
            // Dropping this receiver prevents an old probe from replacing new state.
            self.pending = None;
        }
    }

    /// Start device initialization on a background thread, never the UI thread.
    pub fn request_probe(&mut self) {
        self.request_probe_with(probe_device);
    }

    /// Start a probe with an injectable resolver for tests or host integrations.
    pub fn request_probe_with<F>(&mut self, resolver: F)
    where
        F: FnOnce(DevicePreference) -> Result<String, String> + Send + 'static,
    {
        if self.pending.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        let preference = self.preference;
        self.detected = None;
        self.pending = Some(rx);
        if let Err(error) = thread::Builder::new().name("device-probe".into()).spawn(move || {
            let _ = tx.send(resolver(preference));
        }) {
            self.pending = None;
            self.detected = Some(Err(format!("無法啟動裝置偵測：{error}")));
        }
    }

    /// Consume any completed probe without blocking.
    pub fn poll(&mut self) {
        let Some(rx) = &self.pending else { return };
        let result = match rx.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => Err("裝置偵測執行緒意外結束，請重新偵測".into()),
        };
        self.pending = None;
        self.detected = Some(result);
    }

    /// Last completed availability check; it does not certify synthesis success.
    pub fn detected(&self) -> Option<&Result<String, String>> {
        self.detected.as_ref()
    }

    /// Last device resolution reported by the synthesis worker.
    pub fn actual(&self) -> Option<&Result<String, String>> {
        self.actual.as_ref()
    }

    /// Apply typed synthesis events without parsing human-readable log messages.
    pub fn handle_event(&mut self, event: &WorkerEvent) {
        if let WorkerEvent::DeviceResolved(result) = event {
            self.actual = Some(result.clone());
        }
    }

    /// Render device controls; GPU initialization runs only inside the probe worker.
    pub fn show(&mut self, ui: &mut egui::Ui, backend: BackendKind, busy: bool) {
        self.poll();
        ui.vertical(|ui| {
            ui.add_enabled_ui(!busy && self.pending.is_none(), |ui| {
                ui.horizontal(|ui| {
                    let mut preference = self.preference;
                    egui::ComboBox::from_id_salt("device_preference")
                        .selected_text(preference.name())
                        .show_ui(ui, |ui| {
                            for choice in DevicePreference::ALL {
                                ui.selectable_value(&mut preference, *choice, choice.name());
                            }
                        });
                    self.set_preference(preference);
                    if ui.add_enabled(self.pending.is_none(), egui::Button::new("重新偵測")).clicked() {
                        self.request_probe();
                    }
                });
            });
            if self.detected.is_none() && self.pending.is_none() && !busy {
                self.request_probe();
            }
            if self.pending.is_some() {
                ui.label("正在背景偵測裝置…");
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            }
            for (label, result) in [("偵測結果", &self.detected), ("最近合成裝置", &self.actual)] {
                if let Some(result) = result {
                    match result {
                        Ok(name) => { ui.label(format!("{label}：{name}")); }
                        Err(reason) => { ui.colored_label(Color32::RED, format!("{label}：{reason}")); }
                    }
                }
            }
            if !cfg!(feature = "cuda") {
                ui.label("此 CPU 版本未編譯 CUDA 支援；Auto 使用 CPU，指定 CUDA 會回報錯誤。");
            }
            if backend == BackendKind::Python {
                ui.label("此設定控制 Rust 解碼器；Python 文字前端的 GPU 由 Python 環境自行決定。");
            }
        });
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
    pub device_pref: DevicePreference,
    pub language: String,
    pub speaker: Option<String>,
    pub instruct: Option<String>,
    pub speed: f64,
    pub output_path: PathBuf,
    pub reference_audio: Option<PathBuf>,
    pub reference_text: Option<String>,
    pub seed: Option<u64>,
    pub max_new_tokens: u32,
    /// `true` 表示 `max_new_tokens` 為自動估算（分段合成時逐段重新估算）
    pub max_tokens_auto: bool,
    /// 長文自動分段的每段字數上限；`None` 表示不分段（整段一次合成）
    pub chunk_max_chars: Option<usize>,
    pub auto_download: bool,
    pub hf_mirror: Option<String>,
}

// ---------------------------------------------------------------------------
// 背景事件
// ---------------------------------------------------------------------------

/// 背景執行緒回傳給 GUI 的事件
#[derive(Clone, Debug)]
pub enum WorkerEvent {
    /// Device initialization actually performed by the synthesis worker.
    DeviceResolved(Result<String, String>),
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

    pub fn resolve_device(pref: DevicePreference) -> Result<(candle_core::Device, String), String> {
        match pref {
            DevicePreference::Cpu => {
                Ok((candle_core::Device::Cpu, "🖥️ CPU (中央處理器)".to_string()))
            }
            DevicePreference::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    match initialize_cuda() {
                        Ok(device) => {
                            eprintln!("✅ 成功初始化 CUDA Device 0 (NVIDIA GPU 加速已啟用)");
                            Ok((device, "⚡ CUDA (NVIDIA GPU 硬體加速)".to_string()))
                        }
                        Err(err) => {
                            Err(format!(
                                "指定使用 CUDA 但初始化失敗：{err}。\n\
                                 請確認已安裝 NVIDIA 顯卡驅動程式，或切換為「自動偵測」/「CPU」。"
                            ))
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(
                        "目前執行的二進制檔未編譯 CUDA 支援。\n\
                         請執行 build_release_cuda.ps1 編譯 CUDA 版本，或切換為「CPU」。"
                            .to_string(),
                    )
                }
            }
            DevicePreference::Auto => {
                #[cfg(feature = "cuda")]
                {
                    match initialize_cuda() {
                        Ok(device) => {
                            eprintln!("✅ 成功初始化 CUDA Device 0 (NVIDIA GPU 加速已啟用)");
                            Ok((device, "⚡ CUDA (NVIDIA GPU 硬體加速)".to_string()))
                        }
                        Err(err) => {
                            eprintln!("⚠️ 無法初始化 CUDA Device 0: {err}，自動回退至 CPU");
                            Ok((
                                candle_core::Device::Cpu,
                                format!("🖥️ CPU (自動回退：CUDA 初始化失敗：{err})"),
                            ))
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Ok((candle_core::Device::Cpu, "🖥️ CPU (此版本未編譯 CUDA 支援)".to_string()))
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn initialize_cuda() -> Result<candle_core::Device, String> {
        // The CUDA loader can panic if its driver library is absent.
        std::panic::catch_unwind(|| candle_core::Device::new_cuda(0))
            .map_err(|panic| {
                let reason = panic.downcast_ref::<String>().map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("CUDA driver initialization panicked");
                reason.to_string()
            })?
            .map_err(|error| error.to_string())
    }
}

// ---------------------------------------------------------------------------
// 合成管線（背景執行緒）
// ---------------------------------------------------------------------------

/// LLM 後端統一包裝：多段合成時重複使用同一模型實例，確保各段音色一致
enum LlmBackend {
    Python(PythonBridge),
    #[cfg(feature = "candle-llm")]
    Candle(Box<crate::text_frontend::CandleLLM>),
}

impl LlmBackend {
    fn synthesize(
        &self,
        text: &str,
        options: &SynthesisOptions,
    ) -> crate::Result<crate::text_frontend::TokenStream> {
        use crate::text_frontend::TextFrontend;
        match self {
            Self::Python(bridge) => bridge.synthesize(text, options),
            #[cfg(feature = "candle-llm")]
            Self::Candle(llm) => llm.synthesize(text, options),
        }
    }
}

/// 由 GUI 參數建立各分段共用的合成選項（說話者／語氣指令／參考音訊等聲音條件一致）
fn base_synthesis_options(params: &SynthesisParams) -> SynthesisOptions {
    SynthesisOptions {
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
    }
}

/// 單一分段的 Token 上限：使用者明確指定最大 Token 數時沿用，否則依該段字數自動估算
fn per_chunk_token_limit(params: &SynthesisParams, chunk_text: &str) -> u32 {
    if params.max_tokens_auto {
        estimate_max_tokens_for_text(chunk_text)
    } else {
        params.max_new_tokens
    }
}

/// 對分段音頻頭尾施加短線性淡化，避免分段合併接縫出現爆音
fn apply_edge_fades(samples: &mut [f32], fade_samples: usize) {
    let n = samples.len();
    if n < 2 {
        return;
    }
    let fade = fade_samples.min(n / 2);
    for i in 0..fade {
        let gain = i as f32 / fade as f32;
        samples[i] *= gain;
        samples[n - 1 - i] *= gain;
    }
}

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
    // Step 1: 決定運算裝置（依偏好設定自動偵測或手動指定）
    tx.send(WorkerEvent::Status("初始化運算裝置…".into()))
        .map_err(|e| e.to_string())?;
    if cancel.load(Ordering::SeqCst) {
        return Err("已取消".into());
    }
    let resolved = worker::resolve_device(params.device_pref);
    tx.send(WorkerEvent::DeviceResolved(
        resolved.as_ref().map(|(_, name)| name.clone()).map_err(Clone::clone),
    )).ok();
    let (device, device_name) = resolved?;
    tx.send(WorkerEvent::Status(format!("運算裝置：{device_name}")))
        .ok();

    // Step 2: 載入 LLM 後端（僅載入一次；多段合成共用同一模型與聲音條件，確保音色一致）
    if params.text.trim().is_empty() {
        return Err("請輸入要合成的文字".into());
    }
    if cancel.load(Ordering::SeqCst) {
        return Err("已取消".into());
    }

    let backend = match params.backend {
        BackendKind::Python => {
            tx.send(WorkerEvent::Status(format!(
                "使用 Python 橋接 ({})…（首次載入約 1-5 分鐘）",
                params.model_id
            )))
            .ok();
            let bridge = PythonBridge::new(&params.model_id)
                .map_err(|e| format!("建立 PythonBridge 失敗：{e}"))?
                .with_python("python");
            LlmBackend::Python(bridge)
        }
        #[cfg(feature = "candle-llm")]
        BackendKind::Candle => {
            use crate::text_frontend::CandleLLM;

            let dir = if let Some(d) = &params.model_dir {
                d.clone()
            } else {
                match worker::locate_model_snapshot(
                    &params.model_id,
                    Some(&params.models_base_dir),
                ) {
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
            LlmBackend::Candle(Box::new(llm))
        }
    };

    // Step 3: 長文自動分段（句尾邊界優先；逐段生成後合併，避免長文一次推理卡死）
    let chunks = match params.chunk_max_chars {
        Some(max_chars) => split_text_into_chunks(&params.text, max_chars),
        None => vec![params.text.trim().to_string()],
    };
    if chunks.is_empty() {
        return Err("請輸入要合成的文字".into());
    }
    if chunks.len() > 1 {
        tx.send(WorkerEvent::Status(format!(
            "📄 長文自動分成 {} 段合成（每段上限 {} 字），所有分段共用相同模型與聲音設定",
            chunks.len(),
            params.chunk_max_chars.unwrap_or(DEFAULT_CHUNK_CHARS)
        )))
        .ok();
        if params.backend == BackendKind::Python {
            tx.send(WorkerEvent::Status(
                "⚠️ Python 後端每個分段都需重新載入模型，長文建議改用 Candle (native Rust) 後端"
                    .into(),
            ))
            .ok();
        }
    }

    // Step 4: 確保解碼器權重並建立解碼器（權重只載入一次，供所有分段共用）
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

    // 解碼器容量依「最大單段」幀數設定（含語速拉伸），逐段解碼可限制記憶體用量
    let max_chunk_frames = chunks
        .iter()
        .map(|chunk| per_chunk_token_limit(params, chunk) as usize)
        .max()
        .unwrap_or(64);
    let target_frames = if params.speed != 1.0 {
        (max_chunk_frames as f64 / params.speed).round() as usize
    } else {
        max_chunk_frames
    };
    let mut config = DecoderConfig::realtime_with_capacity(max_chunk_frames.max(target_frames));
    config.speed = params.speed;

    let mut decoder = Decoder12Hz::from_safetensors(config, &weight_dir, &device)
        .map_err(|e| format!("載入 Tokenizer 權重失敗：{e}"))?;

    // Step 5: 逐段生成 Token → 逐段解碼 → 合併為單一音頻
    let sample_rate = 24000u32;
    let gap_samples = (sample_rate as f64 * 0.1).round() as usize; // 分段接縫 100ms 停頓
    let fade_samples = 120usize; // 5ms 邊界淡化避免爆音
    let total_chunks = chunks.len();
    let base_options = base_synthesis_options(params);
    let mut all_samples: Vec<f32> = Vec::new();
    let mut total_frames = 0usize;

    for (idx, chunk_text) in chunks.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            return Err("已取消".into());
        }

        let mut options = base_options.clone();
        options.max_new_tokens = per_chunk_token_limit(params, chunk_text);

        // 加強語言判斷：若為 auto，逐句/逐段自動偵測該段文字的語言
        let detected_lang = if options.language.trim().is_empty()
            || options.language.eq_ignore_ascii_case("auto")
        {
            let detected = crate::text_frontend::detect_language_from_text(chunk_text);
            options.language = detected.to_string();
            Some(detected)
        } else {
            None
        };

        let lang_desc = if let Some(d) = detected_lang {
            format!("，語言：{}", crate::text_frontend::language_display_name(d))
        } else {
            String::new()
        };

        if total_chunks > 1 {
            tx.send(WorkerEvent::Status(format!(
                "🧩 [{}/{}] 生成語音 Token（{} 字，上限 {} 幀 ≈ {:.1} 秒語音{}）…",
                idx + 1,
                total_chunks,
                chunk_text.chars().count(),
                options.max_new_tokens,
                options.max_new_tokens as f64 / 12.0,
                lang_desc
            )))
            .ok();
        } else {
            tx.send(WorkerEvent::Status(format!(
                "⚡ 正在推理生成語音 Token（上限 {} 幀，約 {:.1} 秒語音{}）…",
                options.max_new_tokens,
                options.max_new_tokens as f64 / 12.0,
                lang_desc
            )))
            .ok();
        }

        let stream = backend
            .synthesize(chunk_text, &options)
            .map_err(|e| format!("分段 {} 語音 Token 生成失敗：{e}", idx + 1))?;

        let num_frames = stream.num_frames();
        if num_frames == 0 {
            tx.send(WorkerEvent::Status(format!(
                "⚠️ 分段 {} 未產生任何 Token，已跳過",
                idx + 1
            )))
            .ok();
            continue;
        }
        total_frames += num_frames;

        let mut samples = decoder
            .decode_frames(&stream.frames)
            .map_err(|e| format!("分段 {} 音頻解碼失敗：{e}", idx + 1))?;

        if cancel.load(Ordering::SeqCst) {
            return Err("已取消".into());
        }

        if total_chunks > 1 {
            tx.send(WorkerEvent::Status(format!(
                "✅ [{}/{}] 完成：{} 幀、約 {:.1} 秒音頻",
                idx + 1,
                total_chunks,
                num_frames,
                samples.len() as f64 / sample_rate as f64
            )))
            .ok();
            tx.send(WorkerEvent::Progress {
                current: idx + 1,
                total: total_chunks,
            })
            .ok();
        }

        apply_edge_fades(&mut samples, fade_samples);
        if !all_samples.is_empty() {
            all_samples.extend(std::iter::repeat_n(0.0f32, gap_samples));
        }
        all_samples.extend(samples);
    }

    if all_samples.is_empty() {
        return Err("所有分段都未產生任何音頻".into());
    }

    let duration_sec = all_samples.len() as f64 / sample_rate as f64;
    tx.send(WorkerEvent::Status(format!(
        "產出 {:.2} 秒音頻（{} 個樣本、共 {total_frames} 幀、{total_chunks} 段合併）",
        duration_sec,
        all_samples.len()
    )))
    .ok();

    // Step 6: 寫入 WAV
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
    device_selection: DeviceSelection,
    language: String,
    language_auto: bool,
    speaker: String,
    speed: f64,
    instruct: String,
    output_path: String,
    max_new_tokens: String,
    seed: String,

    // === 長文分段設定 ===
    auto_chunk: bool,
    chunk_chars: String,

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
    /// 分段合成進度 (已完成段數, 總段數)
    chunk_progress: Option<(usize, usize)>,

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

/// 每段字數上限預設值（約 120 字 ≈ 480 幀 ≈ 40 秒語音，單段生成可避免長文推理卡死）
pub const DEFAULT_CHUNK_CHARS: usize = 120;

/// 每段字數上限的允許範圍
const MIN_CHUNK_CHARS: usize = 10;
const MAX_CHUNK_CHARS: usize = 2000;

/// 句尾主要標點（一句話拆分優先點）
const SENTENCE_TERMINATORS: &[char] = &['。', '！', '？', '!', '?', '；', ';', '…', '\n', '\r'];

/// 句尾閉合後引號與括號（保留在句尾，不孤立切斷）
const CLOSING_QUOTES: &[char] = &[
    '」', '』', '”', '’', '"', '\'', '）', ')', '】', ']', '》', '＞', '>',
];

/// 子句標點（單句長度超過保護上限時的次要拆分點）
const CLAUSE_PUNCTUATION: &[char] = &['，', ',', '、', '：', ':', '—', '·', '-'];

/// 將長文以「一句話」為基本拆分點（依句尾標點與換行切分），並包含單句超長保護。
///
/// 切分策略（皆保留標點、不丟失任何非空白字元）：
/// 1. 嚴格以「完整句子」為邊界進行切分，每句話獨立為一個分段，以取得最佳語氣與停頓；
/// 2. 句尾的連續標點（如「？！？！」「……」）與後引號/括號（如「！」」「。”」）完整保留在該句尾端；
/// 3. 若單一句話長度超過 `max_chars`（例如整大段無句號），則自動降級在子句標點切分，仍超長時依字數硬切保護。
pub fn split_text_into_chunks(text: &str, max_chars: usize) -> Vec<String> {
    let max_chars = max_chars.max(MIN_CHUNK_CHARS);
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }

    // Step 1: 依句尾標點與換行切分為個別句子（標點與尾隨引號保留在句尾）
    let sentences = split_into_individual_sentences(text);
    if sentences.is_empty() {
        return Vec::new();
    }

    // Step 2: 逐句檢查，若單句超過 max_chars 則啟用子句/硬切保護，否則保持單句獨立
    let mut chunks = Vec::new();
    for sentence in sentences {
        let trimmed = sentence.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.chars().count() <= max_chars {
            chunks.push(trimmed.to_string());
        } else {
            // 單句超長：先試子句標點，仍超長則硬切
            chunks.extend(split_oversized_sentence(trimmed, max_chars));
        }
    }
    chunks
}

/// 依句尾標點切成個別句子（標點與後續緊鄰的閉合引號/括號保留在句尾）
fn split_into_individual_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut sentences = Vec::new();
    let mut start = 0;
    let mut i = 0;

    while i < n {
        let ch = chars[i];
        let is_sentence_end = if ch == '.' {
            // 英文點號：檢查是否為句尾（後接空白、換行、引號或結尾，且非數字小數點）
            let is_decimal = i > 0
                && i + 1 < n
                && chars[i - 1].is_ascii_digit()
                && chars[i + 1].is_ascii_digit();
            let is_next_space_or_end = i + 1 == n
                || chars[i + 1].is_whitespace()
                || CLOSING_QUOTES.contains(&chars[i + 1]);
            !is_decimal && is_next_space_or_end
        } else {
            SENTENCE_TERMINATORS.contains(&ch)
        };

        if is_sentence_end {
            // 往前吞掉所有連續的句尾標點（如「？？？」「！！」「……」）與換行
            while i + 1 < n && (SENTENCE_TERMINATORS.contains(&chars[i + 1]) || chars[i + 1] == '.') {
                i += 1;
            }
            // 再吞掉緊隨其後的閉合引號與括號（如「！」」「。”」）
            while i + 1 < n && CLOSING_QUOTES.contains(&chars[i + 1]) {
                i += 1;
            }
            let s: String = chars[start..=i].iter().collect();
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            i += 1;
            // 跳過接續的空白
            while i < n && (chars[i] == ' ' || chars[i] == '\t' || chars[i] == '\n' || chars[i] == '\r') {
                i += 1;
            }
            start = i;
        } else {
            i += 1;
        }
    }

    if start < n {
        let s: String = chars[start..n].iter().collect();
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            sentences.push(trimmed.to_string());
        }
    }

    sentences
}

/// 超長單句切分：先試子句標點，仍超長則依字數硬切
fn split_oversized_sentence(sentence: &str, max_chars: usize) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    for ch in sentence.chars() {
        current.push(ch);
        if CLAUSE_PUNCTUATION.contains(&ch) {
            pieces.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        pieces.push(current);
    }

    let mut result = Vec::new();
    let mut pending = String::new();
    for piece in pieces {
        if piece.chars().count() > max_chars {
            if !pending.trim().is_empty() {
                result.push(std::mem::take(&mut pending));
            } else {
                pending.clear();
            }
            let mut buf = String::new();
            for ch in piece.chars() {
                buf.push(ch);
                if buf.chars().count() >= max_chars {
                    result.push(std::mem::take(&mut buf));
                }
            }
            if !buf.is_empty() {
                pending = buf;
            }
        } else if pending.chars().count() + piece.chars().count() <= max_chars {
            pending.push_str(&piece);
        } else {
            result.push(std::mem::take(&mut pending));
            pending = piece;
        }
    }
    if !pending.trim().is_empty() {
        result.push(pending);
    }
    result
}

impl Default for TtsGuiApp {
    fn default() -> Self {
        Self {
            text: String::new(),
            model_id: MODEL_ID_CANDIDATES[0].to_string(),
            model_dir: String::new(),
            models_base_dir: default_models_dir().to_string_lossy().to_string(),
            backend: BackendKind::default(),
            device_selection: DeviceSelection::default(),
            language: "auto".to_string(),
            language_auto: true,
            speaker: "Vivian".to_string(),
            speed: 1.0,
            instruct: String::new(),
            output_path: "output.wav".to_string(),
            max_new_tokens: "auto".to_string(),
            seed: String::new(),
            auto_chunk: true,
            chunk_chars: DEFAULT_CHUNK_CHARS.to_string(),
            reference_audio: String::new(),
            reference_text: String::new(),
            auto_download: true,
            hf_mirror_index: 0,
            hf_custom_mirror: String::new(),
            is_downloading: false,
            status_log: Vec::new(),
            is_processing: false,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            chunk_progress: None,
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

        let effective_instruct = if is_base || self.instruct.trim().is_empty() {
            None
        } else {
            Some(self.instruct.trim().to_string())
        };

        // 長文自動分段設定解析
        let chunk_max_chars = if self.auto_chunk {
            match self.chunk_chars.trim().parse::<usize>() {
                Ok(n) => Some(n.clamp(MIN_CHUNK_CHARS, MAX_CHUNK_CHARS)),
                Err(_) => {
                    self.status_log.push((
                        format!("⚠️ 每段字數格式無效，使用預設 {DEFAULT_CHUNK_CHARS} 字"),
                        Color32::YELLOW,
                    ));
                    Some(DEFAULT_CHUNK_CHARS)
                }
            }
        } else {
            None
        };

        // 最大 Token 數：分段合成時自動估算會逐段重新計算，使用者明確指定則作為每段上限
        let trimmed_max = self.max_new_tokens.trim();
        let (max_new_tokens, max_tokens_auto) =
            if trimmed_max.is_empty() || trimmed_max.eq_ignore_ascii_case("auto") {
                (estimate_max_tokens_for_text(&self.text), true)
            } else {
                match trimmed_max.parse::<u32>() {
                    Ok(v) => (v, false),
                    Err(_) => {
                        self.status_log.push((
                            "⚠️ 最大 Token 數格式無效，改用自動估算".into(),
                            Color32::YELLOW,
                        ));
                        (estimate_max_tokens_for_text(&self.text), true)
                    }
                }
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
            device_pref: self.device_selection.preference(),
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
            max_new_tokens,
            max_tokens_auto,
            chunk_max_chars,
            auto_download: self.auto_download,
            hf_mirror: self.effective_hf_endpoint(),
        };

        self.is_processing = true;
        self.chunk_progress = None;
        self.cancel_flag.store(false, Ordering::SeqCst);
        self.event_rx = Some(rx);

        let cancel = self.cancel_flag.clone();

        thread::spawn(move || {
            run_synthesis(params, tx, cancel);
        });

        let chunk_note = match chunk_max_chars {
            Some(max_chars) => {
                let n = split_text_into_chunks(&self.text, max_chars).len();
                if n > 1 {
                    format!("（長文自動分成 {n} 段，每段上限 {max_chars} 字，生成後合併）")
                } else {
                    String::new()
                }
            }
            None => String::new(),
        };
        self.status_log
            .push((format!("▶ 開始合成…{chunk_note}"), Color32::WHITE));
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
                WorkerEvent::DeviceResolved(result) => {
                    self.device_selection.handle_event(&WorkerEvent::DeviceResolved(result));
                }
                WorkerEvent::Status(msg) => {
                    self.status_log.push((msg, Color32::WHITE));
                }
                WorkerEvent::Progress { current, total } => {
                    self.chunk_progress = Some((current, total));
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
                    self.chunk_progress = None;
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
                                    self.device_selection.show(ui, self.backend, self.is_processing);
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

                                    // Long-text auto chunking
                                    ui.label("長文自動分段：");
                                    ui.horizontal(|ui| {
                                        ui.checkbox(&mut self.auto_chunk, "啟用");
                                        if self.auto_chunk {
                                            ui.label("每段上限");
                                            ui.add(
                                                egui::TextEdit::singleline(&mut self.chunk_chars)
                                                    .desired_width(56.0)
                                                    .hint_text(DEFAULT_CHUNK_CHARS.to_string()),
                                            );
                                            ui.label("字");

                                            let max_chars = self
                                                .chunk_chars
                                                .trim()
                                                .parse::<usize>()
                                                .unwrap_or(DEFAULT_CHUNK_CHARS)
                                                .clamp(MIN_CHUNK_CHARS, MAX_CHUNK_CHARS);
                                            let total_chars =
                                                self.text.trim().chars().count();
                                            if total_chars > max_chars {
                                                let n = split_text_into_chunks(
                                                    &self.text,
                                                    max_chars,
                                                )
                                                .len();
                                                ui.label(
                                                    RichText::new(format!(
                                                        "（目前 {total_chars} 字 → 分 {n} 段逐段生成，相同聲音，完成後合併為單一音檔）"
                                                    ))
                                                    .size(11.5)
                                                    .color(Color32::from_rgb(120, 200, 120)),
                                                );
                                            } else {
                                                ui.label(
                                                    RichText::new(
                                                        "（文字未達分段門檻，整段一次合成）",
                                                    )
                                                    .size(11.5)
                                                    .color(Color32::from_rgb(180, 180, 180)),
                                                );
                                            }
                                        } else {
                                            ui.label(
                                                RichText::new(
                                                    "（停用：整段一次生成，長文字可能耗時過久或卡死）",
                                                )
                                                .size(11.5)
                                                .color(Color32::from_rgb(220, 160, 60)),
                                            );
                                        }
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
                                                let hint = if self.auto_chunk {
                                                    format!(
                                                        "（整段預估約 {est} 幀；分段合成時此上限套用於每一段，填 auto 即逐段自適應）"
                                                    )
                                                } else {
                                                    format!(
                                                        "（目前預估約 {est} 幀，填 auto 即依字數自適應）"
                                                    )
                                                };
                                                ui.label(
                                                    RichText::new(hint)
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
                            } else if let Some((current, total)) = self.chunk_progress {
                                if total > 1 {
                                    ui.add(
                                        egui::ProgressBar::new(current as f32 / total as f32)
                                            .desired_width(120.0),
                                    );
                                    ui.label(
                                        RichText::new(format!("已完成 {current}/{total} 段"))
                                            .size(13.0)
                                            .color(Color32::from_rgb(120, 200, 120)),
                                    );
                                }
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
                            "💡 提示：長文會依「長文自動分段」設定逐段生成（聲音條件一致）後合併為單一音檔，避免一次生成卡死。Candle 後端可選用 CUDA 加速。",
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

// ---------------------------------------------------------------------------
// 測試
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_short_text_returns_single_chunk() {
        let chunks = split_text_into_chunks("你好，世界。", 100);
        assert_eq!(chunks, vec!["你好，世界。".to_string()]);
    }

    #[test]
    fn split_empty_text_returns_empty() {
        assert!(split_text_into_chunks("", 100).is_empty());
        assert!(split_text_into_chunks("   \n  ", 100).is_empty());
    }

    #[test]
    fn split_at_sentence_boundaries() {
        let text = "第一句話。第二句話！第三句話？";
        let chunks = split_text_into_chunks(text, 100);
        assert_eq!(chunks.len(), 3, "三句話應切成 3 個分段：{chunks:?}");
        assert_eq!(chunks[0], "第一句話。");
        assert_eq!(chunks[1], "第二句話！");
        assert_eq!(chunks[2], "第三句話？");
    }

    #[test]
    fn split_keeps_closing_quotes_with_sentence() {
        let text = "他說：「你好！」然後轉身離開了。";
        let chunks = split_text_into_chunks(text, 100);
        assert_eq!(chunks.len(), 2, "引號應完整保留在句尾：{chunks:?}");
        assert_eq!(chunks[0], "他說：「你好！」");
        assert_eq!(chunks[1], "然後轉身離開了。");
    }

    #[test]
    fn split_keeps_all_characters() {
        let text = "今天天氣真好，我們去公園散步吧！路上遇到了老朋友。\n聊了很久，才互相道別再見。";
        let chunks = split_text_into_chunks(text, 15);
        let joined: String = chunks.join("");
        assert_eq!(joined, text.replace('\n', "").trim(), "分段合併後必須與原文一致：{chunks:?}");
    }

    #[test]
    fn oversized_sentence_falls_back_to_clause_and_hard_split() {
        // 沒有句尾標點的超長句：先試子句標點
        let text = "這是一個非常長的子句，中間有逗號分隔，最後沒有句號結束";
        let chunks = split_text_into_chunks(text, 12);
        assert!(chunks.len() >= 2, "應切成多段：{chunks:?}");
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 12, "分段超長：{chunk}");
        }

        // 完全沒有標點的超長句：依字數硬切
        let text = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥";
        let chunks = split_text_into_chunks(text, 10);
        assert_eq!(chunks.len(), 3, "24 字上限 10 應切成 3 段：{chunks:?}");
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 10);
        }
        assert_eq!(chunks.join(""), text);
    }

    #[test]
    fn each_sentence_is_individual_chunk() {
        let text = "嗨。好。走。好。";
        let chunks = split_text_into_chunks(text, 100);
        assert_eq!(chunks.len(), 4, "每句話獨立為一個分段：{chunks:?}");
        assert_eq!(chunks, vec!["嗨。", "好。", "走。", "好。"]);
    }

    #[test]
    fn estimate_max_tokens_bounds() {
        assert_eq!(estimate_max_tokens_for_text(""), 150);
        assert_eq!(estimate_max_tokens_for_text(&"字".repeat(1000)), 2048);
        let mid = estimate_max_tokens_for_text(&"字".repeat(120));
        assert!((150..=2048).contains(&mid));
    }

    #[test]
    fn edge_fades_modifies_boundaries_safely() {
        let mut samples = vec![1.0f32; 100];
        apply_edge_fades(&mut samples, 10);
        assert_eq!(samples[0], 0.0);
        assert!(samples[1] > 0.0 && samples[1] < 1.0);
        assert_eq!(samples[99], 0.0);
        assert_eq!(samples[50], 1.0);

        // 短樣本測試
        let mut short_samples = vec![1.0f32; 1];
        apply_edge_fades(&mut short_samples, 10);
        assert_eq!(short_samples, vec![1.0f32]);
    }

    #[test]
    fn english_text_chunking() {
        let text = "Hello world! This is a test sentence. Another sentence goes here; let us continue.";
        let chunks = split_text_into_chunks(text, 100);
        assert_eq!(chunks.len(), 4);
        assert_eq!(
            chunks,
            vec![
                "Hello world!",
                "This is a test sentence.",
                "Another sentence goes here;",
                "let us continue."
            ]
        );
    }
}
