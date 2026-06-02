//! # 聲碼器模組 (FP16 only)
//!
//! 將解碼器產生的聲學表徵轉換為 PCM 音訊波形。
//!
//! ## 精確度要求
//! - 永不整數量化，最低精度 FP16
//! - 對精度極度敏感，量化收益極低且風險極高

use candle_core::{Device, Tensor};

use crate::{Error, Result};

/// 聲碼器配置
#[derive(Debug, Clone)]
pub struct VocoderConfig {
    /// 輸入通道數
    pub in_channels: usize,
    /// 上取樣倍數
    pub upsample_factor: usize,
    /// 取樣率
    pub sample_rate: u32,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            in_channels: 512,
            upsample_factor: 256,
            sample_rate: 24000,
        }
    }
}

/// 聲碼器特徵
pub trait Vocoder: Send + Sync {
    /// 載入權重
    fn load(tensors: &std::collections::HashMap<String, Tensor>) -> Result<Self>
    where
        Self: Sized;

    /// 將聲學表徵解碼為 PCM 波形
    ///
    /// # 參數
    /// - `features`: 聲學表徵張量，形狀 (channels, frames)
    ///
    /// # 回傳值
    /// PCM f32 取樣陣列
    fn decode(&self, features: &Tensor) -> Result<Vec<f32>>;

    /// 取樣率
    fn sample_rate(&self) -> u32;
}

/// HiFi-GAN 聲碼器（佔位）
///
/// 完整的聲碼器實作將在 Phase 3 中遷移。
/// 目前為樁代碼。
pub struct HifiGanVocoder {
    config: VocoderConfig,
    #[allow(dead_code)]
    device: Device,
}

impl HifiGanVocoder {
    /// 建立預設聲碼器
    pub fn new(config: VocoderConfig, device: &Device) -> Self {
        Self {
            config,
            device: device.clone(),
        }
    }
}

impl Vocoder for HifiGanVocoder {
    fn load(_tensors: &std::collections::HashMap<String, Tensor>) -> Result<Self> {
        // TODO(Phase 3): 實作 HiFi-GAN 權重載入
        Err(Error::Vocoder(
            "Vocoder loading not yet implemented (Phase 3)".into(),
        ))
    }

    fn decode(&self, _features: &Tensor) -> Result<Vec<f32>> {
        // TODO(Phase 3): 實作 HiFi-GAN 前向傳播
        Err(Error::Vocoder(
            "Vocoder decoding not yet implemented (Phase 3)".into(),
        ))
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}
