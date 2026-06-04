//! # Qwen3-TTS Rust 重寫 — 核心程式庫
//!
//! 將 Qwen3-TTS 的 Codec Decode 模組從 PyTorch 移植到純 Rust/Candle。
//!
//! ## 雙模式解碼器
//! - **12Hz 即時模式**: Causal ConvNet + MTP，首包延遲 ≤97ms
//! - **25Hz 高品質模式**: Block-wise Flow Matching DiT，首包延遲 ≤300ms
//!
//! ## 設計原則
//! - 零運行時動態記憶體分配（熱路徑中）
//! - 所有緩衝區在 `new()` 時預分配
//! - 純同步 `decode_chunk()` 調用，異步僅用於 I/O 邊界

pub mod codec;
pub mod paths;
pub mod quantization;
pub mod speaker_converter;
pub mod talker;
pub mod text_frontend;
pub mod tokenizer;
pub mod tokenizer_converter;
pub mod vocoder;
pub mod weights;

mod decoder_12hz;
mod decoder_25hz;

pub use decoder_12hz::Decoder12Hz;
pub use decoder_25hz::Decoder25Hz;

use std::fmt;

use thiserror::Error;

// ---------------------------------------------------------------------------
// 錯誤類型
// ---------------------------------------------------------------------------

/// Qwen3-TTS 領域錯誤
#[derive(Error)]
pub enum Error {
    /// Candle 張量操作錯誤
    #[error("Candle tensor error: {0}")]
    Candle(#[from] candle_core::Error),

    /// 權重載入錯誤
    #[error("Weight loading error: {0}")]
    Weight(String),

    /// 配置錯誤（非法參數等）
    #[error("Configuration error: {0}")]
    Config(String),

    /// 解碼器執行錯誤
    #[error("Decode error at layer {layer}: {reason}")]
    Decode { layer: usize, reason: String },

    /// 輸入 Token 無效
    #[error("Invalid token {token} at layer {layer}")]
    InvalidToken { token: u16, layer: usize },

    /// 聲碼器錯誤
    #[error("Vocoder error: {0}")]
    Vocoder(String),

    /// I/O 錯誤
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// 序列化錯誤
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 使用 Display 實作作為 Debug 輸出
        write!(f, "{self}")
    }
}

/// 便捷類型別名
pub type Result<T> = std::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// 解碼器配置
// ---------------------------------------------------------------------------

/// 解碼器模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderMode {
    /// 12Hz 即時模式（Causal ConvNet + MTP）
    RealTime,
    /// 25Hz 高品質模式（Flow Matching DiT）
    HighQuality,
}

/// 解碼器配置
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// 解碼器模式
    pub mode: DecoderMode,

    /// 碼本層數（預設 16）
    pub num_codebook_layers: usize,

    /// 每層碼本大小（預設 2048）
    pub codebook_size: usize,

    /// 嵌入維度（預設 512）
    pub embedding_dim: usize,

    /// 取樣率（Hz）
    pub sample_rate: u32,

    /// 因果卷積核大小
    pub kernel_size: usize,

    /// 因果卷積通道數
    pub conv_channels: usize,

    /// 環形緩衝區最大幀數
    pub ring_buffer_capacity: usize,

    /// 語音合成語速 (1.0 為正常速度)
    pub speed: f64,

    // --- 12Hz 完整解碼器參數 ---
    /// pre_conv 輸出通道（預設 1024）
    pub latent_dim: usize,

    /// Transformer 隱藏維度（預設 512）
    pub transformer_dim: usize,

    /// Transformer 注意力頭數
    pub transformer_heads: usize,

    /// Transformer KV 頭數（GQA）
    pub transformer_kv_heads: usize,

    /// Transformer 層數
    pub transformer_layers: usize,

    /// Sliding window 大小
    pub sliding_window: usize,

    /// Upsample 核大小
    pub upsample_kernel: usize,

    /// SnakeBeta 參數
    pub snake_beta: f64,

    // --- 25Hz 參數 ---
    /// DiT 隱藏維度（25Hz 模式使用）
    pub dit_hidden_dim: usize,

    /// DiT 注意力頭數
    pub dit_num_heads: usize,

    /// DiT 區塊數
    pub dit_num_blocks: usize,

    /// Flow Matching ODE 步數
    pub ode_steps: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            mode: DecoderMode::RealTime,
            num_codebook_layers: 16,
            codebook_size: 2048,
            embedding_dim: 512,
            sample_rate: 24000,
            kernel_size: 3,
            conv_channels: 512,
            ring_buffer_capacity: 64,
            speed: 1.0,
            latent_dim: 1024,
            transformer_dim: 512,
            transformer_heads: 16,
            transformer_kv_heads: 16,
            transformer_layers: 8,
            sliding_window: 72,
            upsample_kernel: 8,
            snake_beta: 1.0,
            dit_hidden_dim: 1024,
            dit_num_heads: 16,
            dit_num_blocks: 12,
            ode_steps: 32,
        }
    }
}

impl DecoderConfig {
    /// 建立 12Hz 即時模式配置
    pub fn realtime() -> Self {
        Self {
            mode: DecoderMode::RealTime,
            ..Default::default()
        }
    }

    /// 建立可處理指定幀數的 12Hz 即時模式配置。
    ///
    /// 容量會保留至少預設值，短句仍沿用原本的 64 幀配置。
    pub fn realtime_with_capacity(capacity: usize) -> Self {
        let mut config = Self::realtime();
        config.ring_buffer_capacity = config.ring_buffer_capacity.max(capacity);
        config
    }

    /// 建立 25Hz 高品質模式配置
    pub fn high_quality() -> Self {
        Self {
            mode: DecoderMode::HighQuality,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// 解碼器特徵
// ---------------------------------------------------------------------------

/// Qwen3-TTS 解碼器核心特徵
///
/// 所有實作必須滿足：
/// - `Send + Sync`：可安全地在執行緒間傳遞
/// - 熱路徑零動態分配
/// - `reset_state` 零分配重置
pub trait TtsDecoder: Send + Sync {
    /// 建立新解碼器實例，載入權重
    fn new(config: DecoderConfig) -> Result<Self>
    where
        Self: Sized;

    /// 流式解碼一幀 Token，傳回 PCM f32 片段
    ///
    /// # 參數
    /// - `tokens`: 形狀為 `(num_layers,)` 的 Token 陣列，每層一個 u16 Token
    ///
    /// # 回傳值
    /// - `Ok(Vec<f32>)`: PCM 音訊片段
    ///
    /// # 注意事項
    /// - 此方法為純同步阻塞調用（內部可使用 rayon 並行）
    /// - 熱路徑中不得分配新的 Vec/Buffer（回傳值除外）
    fn decode_chunk(&mut self, tokens: &[u16]) -> Result<Vec<f32>>;

    /// 重置所有內部狀態（用於會話切換）
    ///
    /// 必須零分配重置所有緩衝區
    fn reset_state(&mut self);
}

// ---------------------------------------------------------------------------
// 公用工具函式
// ---------------------------------------------------------------------------

/// 計算兩個張量之間的餘弦相似度
pub fn cosine_sim(a: &candle_core::Tensor, b: &candle_core::Tensor) -> Result<f64> {
    let a_flat = a.flatten_all()?;
    let b_flat = b.flatten_all()?;

    let dot = (&a_flat * &b_flat)?.sum_all()?.to_scalar::<f64>()?;
    let norm_a = a_flat.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt();
    let norm_b = b_flat.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }

    Ok(dot / (norm_a * norm_b))
}

/// 計算均方誤差（MSE）
pub fn mean_squared_error(a: &candle_core::Tensor, b: &candle_core::Tensor) -> Result<f64> {
    let diff = (a - b)?;
    let mse = diff.sqr()?.sum_all()?.to_scalar::<f64>()?;
    let numel = a.flatten_all()?.elem_count() as f64;
    Ok(mse / numel)
}

#[cfg(test)]
mod tests {
    use super::{DecoderConfig, DecoderMode};

    #[test]
    fn realtime_with_capacity_expands_ring_buffer() {
        let config = DecoderConfig::realtime_with_capacity(80);

        assert_eq!(config.mode, DecoderMode::RealTime);
        assert_eq!(config.ring_buffer_capacity, 80);
    }

    #[test]
    fn realtime_with_capacity_keeps_default_for_short_requests() {
        let config = DecoderConfig::realtime_with_capacity(8);

        assert_eq!(
            config.ring_buffer_capacity,
            DecoderConfig::realtime().ring_buffer_capacity
        );
    }
}
