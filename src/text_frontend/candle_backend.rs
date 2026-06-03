//! # Candle 原生 LLM 後端
//!
//! 純 Rust/Candle 的文本前端 LLM 推理，無 Python 依賴。
//!
//! ## 載入流程
//! ```text
//! 1. 讀取 tokenizer.json (HuggingFace tokenizers crate 格式)
//! 2. 讀取 model.safetensors (TalkerWeightLoader)
//! 3. 構建 TalkerForConditionalGeneration + InputBuilder
//! ```
//!
//! ## 推理流程
//! ```text
//! 文字 → BPE tokenize → 對話模板包裝 → InputBuilder::build
//!      → talker.generate → TokenParser::parse → TokenStream
//! ```
//!
//! ## 對話模板（對應 Qwen3TTSModel._build_assistant_text）
//! `<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n`
//!
//! 完整 token 序列 = 3 (role) + N (text) + 5 (tail) = N+8。
//! 這 8 個固定 token 結構由 `InputBuilder::build` 預期。

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::Device;
use tokenizers::Tokenizer;

use crate::talker::sampling::{Sampler, SamplingOptions as TalkerSamplingOptions};
use crate::talker::{
    InputBuilder, TalkerConfig, TalkerForConditionalGeneration, TalkerWeightLoader,
};
use crate::text_frontend::token_parser::TokenParser;
use crate::text_frontend::{SynthesisOptions, TextFrontend, TokenStream};
use crate::{Error, Result};

// ---------------------------------------------------------------------------
// CandleLLM
// ---------------------------------------------------------------------------

/// 純 Rust/Candle LLM 後端
///
/// 直接從 HuggingFace `model.safetensors` 載入 Talker，無需任何 Python 依賴。
/// 權重以 BF16 載入、計算時升級到 F32 保持數值對齊。
///
/// # 範例
/// ```no_run
/// use qwen3tts::text_frontend::{CandleLLM, SynthesisOptions, TextFrontend};
/// use candle_core::Device;
///
/// let device = Device::Cpu;
/// let backend = CandleLLM::from_pretrained_dir("path/to/Qwen3-TTS-12Hz-0.6B-Base", &device)?;
/// let stream = backend.synthesize("你好", &SynthesisOptions::default())?;
/// # Ok::<_, qwen3tts::Error>(())
/// ```
pub struct CandleLLM {
    tokenizer: Arc<Tokenizer>,
    talker: Arc<TalkerForConditionalGeneration>,
    config: TalkerConfig,
    device: Device,
    parser: TokenParser,
}

impl CandleLLM {
    /// 從 HuggingFace 模型目錄載入（需含 `model.safetensors` 與 `tokenizer.json`）
    ///
    /// # 參數
    /// - `model_dir`: 包含 `model.safetensors` 與 `tokenizer.json` 的目錄
    /// - `device`: 計算裝置（CPU/CUDA/Metal）
    pub fn from_pretrained_dir(model_dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // ── 1. 載入 tokenizer.json ──
        let tokenizer_path = find_tokenizer_json(model_dir).ok_or_else(|| {
            Error::Config(format!(
                "找不到 tokenizer.json 於 {model_dir:?}。\
                 請先執行 `python tools/build_tokenizer.py` 產生。"
            ))
        })?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            Error::Config(format!(
                "Failed to load tokenizer from {tokenizer_path:?}: {e}"
            ))
        })?;

        // ── 2. 載入 safetensors 權重 ──
        let safetensors_path = model_dir.join("model.safetensors");
        if !safetensors_path.exists() {
            return Err(Error::Config(format!(
                "找不到 model.safetensors 於 {model_dir:?}"
            )));
        }
        let loader = TalkerWeightLoader::from_safetensors(&safetensors_path, device)?;

        // ── 3. 從權重 shape 推斷 config（支援 0.6B / 1.7B）──
        let config = loader.infer_config()?;
        let talker = loader.build_talker(&config)?;

        log::info!(
            "CandleLLM loaded: hidden={} intermediate={} num_layers={} code_predictor_hidden={}",
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            config.code_predictor.hidden_size
        );

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            talker: Arc::new(talker),
            config,
            device: device.clone(),
            parser: TokenParser::new(24000),
        })
    }

    /// 從單獨的 safetensors 與 tokenizer 檔案載入
    pub fn from_files(
        safetensors_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| Error::Config(format!("Failed to load tokenizer: {e}")))?;
        let loader = TalkerWeightLoader::from_safetensors(safetensors_path, device)?;
        let config = loader.infer_config()?;
        let talker = loader.build_talker(&config)?;
        log::info!(
            "CandleLLM loaded: hidden={} intermediate={} num_layers={} code_predictor_hidden={}",
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            config.code_predictor.hidden_size
        );
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            talker: Arc::new(talker),
            config,
            device: device.clone(),
            parser: TokenParser::new(24000),
        })
    }

    /// 取得底層 tokenizer 的參考（用於測試或進階用途）
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// 取得底層 talker 的參考
    pub fn talker(&self) -> &TalkerForConditionalGeneration {
        &self.talker
    }

    // -----------------------------------------------------------------------
    // 內部：將文字編碼為完整 chat-template token 序列
    // -----------------------------------------------------------------------

    /// 將使用者文字編碼為符合 Qwen3-TTS chat template 的完整 token 序列。
    ///
    /// 格式：`<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n`
    /// 對應 token 數：3 (role) + N (text) + 5 (tail) = N+8
    fn build_prompt_ids(&self, text: &str) -> Result<Vec<u32>> {
        let prompt = format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| Error::Config(format!("Tokenizer encode error: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn build_instruct_ids(&self, instruct: &str) -> Result<Vec<u32>> {
        let prompt = format!("<|im_start|>user\n{instruct}<|im_end|>\n");
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| Error::Config(format!("Tokenizer encode error: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }
}

// ---------------------------------------------------------------------------
// TextFrontend 實作
// ---------------------------------------------------------------------------

impl TextFrontend for CandleLLM {
    fn synthesize(&self, text: &str, options: &SynthesisOptions) -> Result<TokenStream> {
        if text.is_empty() {
            return Err(Error::Config("text cannot be empty".into()));
        }

        // ── 1. 文字 → token 序列 ──
        let prompt_ids = self.build_prompt_ids(text)?;
        let instruct_ids = options
            .instruct
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| self.build_instruct_ids(s))
            .transpose()?;
        log::debug!(
            "CandleLLM::synthesize text={:?} prompt_len={} instruct_len={} tokens={:?}",
            text,
            prompt_ids.len(),
            instruct_ids.as_ref().map_or(0, Vec::len),
            prompt_ids
        );

        // ── 2. 構建 talker 輸入 ──
        let builder = InputBuilder::new(&self.talker, &self.device);
        let (inputs_embeds, attention_mask, trailing_text_hidden, tts_pad_embed) = builder
            .build(
                &prompt_ids,
                instruct_ids.as_deref(),
                &options.language,
                options.speaker.as_deref(),
            )
            .map_err(map_candle_err)?;

        // ── 3. 自迴歸生成 codec tokens ──
        let max_new_tokens = options.max_new_tokens as usize;
        let codes_tensor = if options.temperature <= 0.0 {
            self.talker
                .generate(
                    &inputs_embeds,
                    Some(&attention_mask),
                    Some(&trailing_text_hidden),
                    Some(&tts_pad_embed),
                    max_new_tokens,
                    &self.device,
                )
                .map_err(map_candle_err)?
        } else {
            let seed = options.seed.unwrap_or_else(|| {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                text.hash(&mut hasher);
                options.language.hash(&mut hasher);
                options.speaker.hash(&mut hasher);
                options.instruct.hash(&mut hasher);
                hasher.finish()
            });
            let mut sampler = Sampler::new(seed);
            let sampling = TalkerSamplingOptions {
                temperature: options.temperature,
                top_k: options.top_k as usize,
                top_p: options.top_p,
            };
            self.talker
                .generate_sampled(
                    &inputs_embeds,
                    Some(&attention_mask),
                    Some(&trailing_text_hidden),
                    Some(&tts_pad_embed),
                    max_new_tokens,
                    &self.device,
                    &mut sampler,
                    sampling,
                )
                .map_err(map_candle_err)?
        };

        // ── 4. Tensor → Vec<Vec<u16>> ──
        let (num_frames, _codebooks) = codes_tensor.dims2().map_err(map_candle_err)?;
        let flat: Vec<u32> = codes_tensor
            .flatten_all()
            .map_err(map_candle_err)?
            .to_vec1::<u32>()
            .map_err(map_candle_err)?;
        let codes: Vec<Vec<u16>> = (0..num_frames)
            .map(|i| {
                flat[i * 16..(i + 1) * 16]
                    .iter()
                    .map(|&x| x as u16)
                    .collect()
            })
            .collect();

        log::info!("CandleLLM::synthesize generated {} frames", codes.len());

        // ── 5. 解析為 TokenStream（自動過濾 EOS/PAD）──
        self.parser.parse(&codes, options)
    }
}

impl std::fmt::Debug for CandleLLM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleLLM")
            .field("vocab_size", &self.tokenizer.get_vocab_size(true))
            .field("device", &self.device)
            .field("hidden_size", &self.config.hidden_size)
            .field("num_layers", &self.config.num_hidden_layers)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// 輔助函式
// ---------------------------------------------------------------------------

/// 嘗試在 model_dir 內找到 tokenizer.json（支援直接放置或 snapshot 子目錄）。
fn find_tokenizer_json(model_dir: &Path) -> Option<PathBuf> {
    let direct = model_dir.join("tokenizer.json");
    if direct.exists() {
        return Some(direct);
    }
    // 嘗試在 snapshots/<sha>/ 子目錄找
    let snapshots = model_dir.join("snapshots");
    if snapshots.is_dir() {
        for entry in std::fs::read_dir(snapshots).ok()?.flatten() {
            let p = entry.path().join("tokenizer.json");
            if p.exists() {
                return Some(p);
            }
        }
    }
    // 嘗試在 ../models/tokenizer.json 找（開發模式）
    for fallback in [
        model_dir
            .parent()
            .map(|p| p.join("models/tokenizer_1.7b.json")),
        model_dir.parent().map(|p| p.join("models/tokenizer.json")),
        Some(PathBuf::from("models/tokenizer_1.7b.json")),
        Some(PathBuf::from("models/tokenizer.json")),
    ]
    .into_iter()
    .flatten()
    {
        if fallback.exists() {
            return Some(fallback);
        }
    }

    for base_id in [
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ] {
        if let Some(base_dir) = locate_hf_model_snapshot(base_id) {
            let path = base_dir.join("tokenizer.json");
            if path.exists() {
                return Some(path);
            }
        }
    }
    None
}

fn locate_hf_model_snapshot(model_id: &str) -> Option<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()
        .map(PathBuf::from)?;
    let snapshots = home
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join(format!("models--{}", model_id.replace('/', "--")))
        .join("snapshots");
    let entries = std::fs::read_dir(snapshots).ok()?;
    for entry in entries.flatten() {
        let candidate = entry.path().join("tokenizer.json");
        if candidate.exists() {
            return Some(entry.path());
        }
    }
    None
}

fn map_candle_err(e: candle_core::Error) -> Error {
    Error::Config(format!("Candle error: {e}"))
}
