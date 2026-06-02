//! # Talker LLM 模組
//!
//! 純 Rust/Candle 的 Qwen3-TTS Talker 模型實作。
//!
//! ## 架構
//!
//! - `config`: Talker 配置（特化 Token ID、語言/說話者對照表）
//! - `primitives`: RMSNorm、RoPE、SwiGLU MLP、KV Cache
//! - `talker_attention`: GQA 注意力（head_dim=128, QK Norm, 3D RoPE）
//! - `decoder_layer`: 單層 Transformer 解碼器
//! - `model`: 28 層 TalkerModel
//! - `code_predictor`: 5 層子碼本預測器
//! - `talker`: TalkerForConditionalGeneration（生成引擎）
//! - `input_builder`: 從文字建構輸入嵌入
//! - `weight_loader`: 從 model.safetensors 載入權重
//!
//! ## 使用方式
//!
//! ```rust,ignore
//! use qwen3tts::talker::{TalkerForConditionalGeneration, TalkerConfig};
//! use qwen3tts::talker::weight_loader::TalkerWeightLoader;
//!
//! let device = candle_core::Device::Cpu;
//! let config = TalkerConfig::default();
//! let loader = TalkerWeightLoader::from_safetensors("model.safetensors", &device)?;
//! let talker = loader.build_talker(&config)?;
//! ```

pub mod code_predictor;
pub mod config;
pub mod decoder_layer;
pub mod input_builder;
pub mod model;
pub mod primitives;
pub mod talker;
pub mod talker_attention;
pub mod weight_loader;

pub use code_predictor::CodePredictor;
pub use config::{CodePredictorConfig, TalkerConfig};
pub use model::TalkerModel;
pub use talker::TalkerForConditionalGeneration;
pub use weight_loader::TalkerWeightLoader;
