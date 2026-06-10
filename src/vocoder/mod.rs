//! # 聲碼器模組 (FP16 only)
//!
//! 將解碼器的聲學表徵（upsample 輸出）轉換為 PCM 音訊波形。
//!
//! Qwen3-TTS 不使用獨立 HiFi-GAN；聲碼器由下列元件組成：
//!
//! 1. `decoder_start` — CausalConv1d (latent_dim → decoder_dim, k=7)
//! 2. `decoder_blocks` — 4 個 DecoderDecoderBlock（含 SnakeBeta + CausalTransConvNet + ResidualUnits）
//! 3. `final_snake_a/b` — SnakeBeta 激活
//! 4. `final_conv` — CausalConv1d (output_dim → 1, k=7)
//!
//! ## 精確度要求
//! - 永不整數量化，最低精度 FP16
//! - 對精度極度敏感，量化收益極低且風險極高

use candle_core::{Device, Tensor};

use crate::codec::{snake_beta, CausalConv1d, CausalConvConfig, DecoderBlock};
use crate::weights::WeightLoader;
use crate::Result;

/// 聲碼器配置
#[derive(Debug, Clone)]
pub struct VocoderConfig {
    /// 輸入通道數（latent_dim）
    pub in_channels: usize,
    /// 第一個 CausalConv 的輸出維度（decoder_dim）
    pub decoder_dim: usize,
    /// 取樣率
    pub sample_rate: u32,
    /// 內部狀態容量（RingBuffer）
    pub state_capacity: usize,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            in_channels: 1024,
            decoder_dim: 1536,
            sample_rate: 24000,
            state_capacity: 64,
        }
    }
}

/// 聲碼器特徵
pub trait Vocoder: Send + Sync {
    /// 從 WeightLoader 載入權重（使用 `decoder_blocks.safetensors` 中的 `{0..6}.*` 鍵）
    fn from_loader(loader: &WeightLoader, _device: &Device, config: &VocoderConfig) -> Result<Self>
    where
        Self: Sized;

    /// 將聲學表徵解碼為 PCM 波形
    ///
    /// # 參數
    /// - `features`: 聲學表徵張量，形狀 (1, channels, frames)
    ///
    /// # 回傳值
    /// PCM f32 取樣陣列（形狀 (samples,)）
    fn decode(&self, features: &Tensor) -> Result<Vec<f32>>;

    /// 取樣率
    fn sample_rate(&self) -> u32;
}

/// HiFi-GAN 相容聲碼器
///
/// 包裝 Qwen3-TTS 解碼器的最終音訊生成階段。
/// 載入 `decoder_blocks.safetensors` 中的 `0.conv` ~ `6.conv` 權重。
pub struct HifiGanVocoder {
    config: VocoderConfig,
    /// decoder_start: CausalConv1d(latent_dim → decoder_dim, k=7)
    decoder_start: CausalConv1d,
    /// decoder_blocks: 4 個 DecoderDecoderBlock
    decoder_blocks: Vec<DecoderBlock>,
    /// final_snake_a: SnakeBeta alpha
    final_snake_a: Tensor,
    /// final_snake_b: SnakeBeta beta
    final_snake_b: Tensor,
    /// final_conv: CausalConv1d(output_dim → 1, k=7)
    final_conv: CausalConv1d,
}

impl HifiGanVocoder {
    /// 建立聲碼器
    pub fn new(
        config: VocoderConfig,
        decoder_start: CausalConv1d,
        decoder_blocks: Vec<DecoderBlock>,
        final_snake_a: Tensor,
        final_snake_b: Tensor,
        final_conv: CausalConv1d,
    ) -> Self {
        Self {
            config,
            decoder_start,
            decoder_blocks,
            final_snake_a,
            final_snake_b,
            final_conv,
        }
    }
}

impl Vocoder for HifiGanVocoder {
    fn from_loader(
        loader: &WeightLoader,
        _device: &Device,
        config: &VocoderConfig,
    ) -> Result<Self> {
        // 0.conv: decoder_start  — CausalConv(latent_dim → decoder_dim, k=7)
        let (w0, b0) = loader.conv1d_pair("0.conv")?;
        let cfg0 = CausalConvConfig::from_weight(&w0, 1, 1);
        let decoder_start = CausalConv1d::new(w0, b0, cfg0, config.state_capacity)?;

        // 1..=4: decoder_blocks
        let mut decoder_blocks = Vec::with_capacity(4);
        for i in 1..=4 {
            decoder_blocks.push(DecoderBlock::from_loader(loader, &format!("{i}"))?);
        }

        // 5: final SnakeBeta
        let final_snake_a = loader.get("5.alpha")?.clone();
        let final_snake_b = loader.get("5.beta")?.clone();

        // 6.conv: final_conv — CausalConv(output_dim → 1, k=7)
        let (w6, b6) = loader.conv1d_pair("6.conv")?;
        let cfg6 = CausalConvConfig::from_weight(&w6, 1, 1);
        let final_conv = CausalConv1d::new(w6, b6, cfg6, config.state_capacity)?;

        Ok(Self {
            config: config.clone(),
            decoder_start,
            decoder_blocks,
            final_snake_a,
            final_snake_b,
            final_conv,
        })
    }

    fn decode(&self, features: &Tensor) -> Result<Vec<f32>> {
        let h = self.decoder_start.forward(features)?;
        let mut h = h;
        for db in &self.decoder_blocks {
            h = db.forward(&h)?;
        }
        let h = snake_beta(&h, &self.final_snake_a, &self.final_snake_b)?;
        let h = self.final_conv.forward(&h)?;
        let output: Vec<f32> = h.squeeze(0)?.squeeze(0)?.to_vec1()?;
        Ok(output)
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

// ---------------------------------------------------------------------------
// 單元測試
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_vocoder_creation() {
        let device = Device::Cpu;
        let weight_dir = Path::new("weights/tokenizer");
        if !weight_dir.join("decoder_blocks.safetensors").exists() {
            eprintln!("Skipping: decoder blocks not found");
            return;
        }
        let loader = WeightLoader::from_dir(weight_dir, &device).unwrap();
        let config = VocoderConfig::default();
        let vocoder = HifiGanVocoder::from_loader(&loader, &device, &config).unwrap();
        assert_eq!(vocoder.sample_rate(), 24000);
    }

    #[test]
    fn test_vocoder_load_and_decode() {
        let device = Device::Cpu;
        let weight_dir = Path::new("weights/tokenizer");
        if !weight_dir.join("decoder_blocks.safetensors").exists() {
            eprintln!("Skipping: decoder blocks not found");
            return;
        }
        let loader = WeightLoader::from_dir(weight_dir, &device).unwrap();
        let config = VocoderConfig::default();
        let vocoder = HifiGanVocoder::from_loader(&loader, &device, &config).unwrap();

        // 模擬 upsample 輸出：3 個 12Hz 幀經 upsample（×4）→ 12 幀
        // 然後 vocoder：12 → 96 → 480 → 1920 → 5760 samples
        let input = Tensor::zeros((1, 1024, 12), candle_core::DType::F32, &device).unwrap();
        let output = vocoder.decode(&input).unwrap();
        assert_eq!(output.len(), 5760);
    }
}
