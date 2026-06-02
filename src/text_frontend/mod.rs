//! # 文本前端模組
//!
//! 將輸入文字轉換為解碼器可消費的多碼本 Token 序列。
//!
//! ## 管線流程
//!
//! ```text
//! 輸入文字
//!   │
//!   ▼
//! ┌─────────────────────────────┐
//! │  TextFrontend 後端           │
//! │  (PythonBridge / CandleLLM) │
//! │  載入 LLM → 產生 talker     │
//! │  codes [seq, 16]            │
//! └──────────┬──────────────────┘
//!            │ talker_codes
//!            ▼
//! ┌─────────────────────────────┐
//! │  TokenParser                │
//! │  移除 EOS/PAD → 分割 frame │
//! └──────────┬──────────────────┘
//!            │ frames: Vec<[u16; 16]>
//!            ▼
//! ┌─────────────────────────────┐
//! │  TtsDecoder::decode_chunk   │
//! │  → PCM f32 片段             │
//! └─────────────────────────────┘
//! ```
//!
//! ## 支援後端
//! - **PythonBridge** (✅ 即時可用): 子行程調用 qwen-tts Python 套件
//! - **CandleLLM** (🔜 規劃中): 純 Rust/Candle + GGUF 權重

mod python_bridge;
mod token_parser;

#[cfg(feature = "candle-llm")]
pub mod candle_backend;

pub use python_bridge::PythonBridge;
pub use token_parser::TokenParser;

use crate::Result;

// ---------------------------------------------------------------------------
// 合成選項
// ---------------------------------------------------------------------------

/// 合成選項
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// 語言 (預設 "auto")
    pub language: String,

    /// 說話者 (None = 預設)
    pub speaker: Option<String>,

    /// LLM 取樣溫度
    pub temperature: f64,

    /// Top-k 取樣
    pub top_k: u32,

    /// Top-p 取樣
    pub top_p: f64,

    /// 最大生成 Token 數
    pub max_new_tokens: u32,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            language: "auto".into(),
            speaker: None,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            max_new_tokens: 4096,
        }
    }
}

// ---------------------------------------------------------------------------
// Token 串流
// ---------------------------------------------------------------------------

/// 從 LLM 回傳的多碼本 Token 串流
///
/// 每幀包含 16 個 u16 token（對應 16 層碼本），
/// 可直接餵入 `TtsDecoder::decode_chunk`。
#[derive(Debug, Clone)]
pub struct TokenStream {
    /// 多碼本 token 序列，每幀 16 個 u16
    pub frames: Vec<[u16; 16]>,

    /// 取樣率 (Hz)，解碼器輸出匹配此值
    pub sample_rate: u32,
}

impl TokenStream {
    /// 建立空的 TokenStream
    pub fn new(sample_rate: u32) -> Self {
        Self {
            frames: Vec::new(),
            sample_rate,
        }
    }

    /// 幀數
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// 音頻長度（秒）
    pub fn duration_sec(&self) -> f64 {
        self.frames.len() as f64 / 12.5 // 12Hz frame rate for 12Hz mode
    }

    /// Serialize frames to the binary token format consumed by `TokenParser`.
    ///
    /// Format: little-endian `u32` frame count followed by `frames * 16` `u16`
    /// codec tokens. This keeps text-front-end output reusable by fully native
    /// Rust decode runs via the CLI `--tokens` path.
    pub fn to_binary(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(4 + self.frames.len() * 16 * 2);
        data.extend_from_slice(&(self.frames.len() as u32).to_le_bytes());
        for frame in &self.frames {
            for token in frame {
                data.extend_from_slice(&token.to_le_bytes());
            }
        }
        data
    }

    /// Write frames to a binary token file compatible with `TokenParser`.
    pub fn write_binary(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, self.to_binary())?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TextFrontend Trait
// ---------------------------------------------------------------------------

/// 文本前端的抽象介面：文字 → 多碼本語義 Token
///
/// # 實作要求
/// - 必須是 `Send + Sync`，可安全跨執行緒使用
/// - 內部 LLM 推理可以是阻塞的，由 `async` 外層處理
pub trait TextFrontend: Send + Sync {
    /// 將文字轉換為解碼器可消費的多碼本 Token 序列
    ///
    /// # 參數
    /// - `text`: 要合成的文字
    /// - `options`: 合成選項（語言、說話者、採樣參數等）
    ///
    /// # 回傳
    /// - `TokenStream`: 包含所有幀的 Token 串流
    fn synthesize(&self, text: &str, options: &SynthesisOptions) -> Result<TokenStream>;
}
