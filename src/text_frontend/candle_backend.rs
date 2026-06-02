//! # Candle/GGUF LLM 後端（骨架）
//!
//! 純 Rust/Candle 的文本前端 LLM 推理。
//! 需 GGUF 權重與 Qwen2 架構實作。
//!
//! ## 狀態
//! 🔜 規劃中 — 等待 `candle-llama` 或自定義 Candle Qwen2 實作整合。

use crate::text_frontend::{SynthesisOptions, TextFrontend, TokenStream};
use crate::Result;

/// 純 Rust/Candle LLM 後端
///
/// 使用 GGUF 格式的 Qwen3-TTS 權重進行純 Rust 推理。
/// 無需 Python 依賴。
pub struct CandleLLM {
    // TODO: model: candle_llama::Pipeline,
    // TODO: tokenizer: tokenizers::Tokenizer,
    // TODO: device: candle_core::Device,
}

impl CandleLLM {
    /// 從 GGUF 檔案載入模型
    pub fn from_gguf(_model_path: &str) -> Result<Self> {
        Err(crate::Error::Config(
            "Candle LLM backend not yet implemented:\n\
             - Wait for candle-llama Qwen2 TTS support\n\
             - Or implement Qwen2 transformer in Candle\n\
             - GGUF weights available at: cstr/qwen3-tts-0.6b-base-GGUF\n\
             - For now, use PythonBridge backend instead"
                .into(),
        ))
    }
}

impl TextFrontend for CandleLLM {
    fn synthesize(&self, _text: &str, _options: &SynthesisOptions) -> Result<TokenStream> {
        Err(crate::Error::Config(
            "Candle LLM backend not yet implemented — use PythonBridge for now".into(),
        ))
    }
}
