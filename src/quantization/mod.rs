//! # 自定義量化校準工具
//!
//! 提供碼本專屬量化、逐層敏感度分析、INT8 校準等工具。
//!
//! ## 設計原則
//! - 禁止使用通用 LLM 量化配置（Q4_K_M / Q5_K_M）
//! - 每層量化校準使用專屬數據集
//! - 所有量化權重通過 cosine ≥ 0.995 方可合入

use candle_core::{Device, Tensor};

/// 參數建構輔助
///
/// 簡化的參數管理，用於模型骨架階段的張量建立。
pub struct VarBuilder {
    device: Device,
}

impl VarBuilder {
    /// 從 Device 建立
    pub fn from_device(device: Device) -> Self {
        Self { device }
    }

    /// 建立 dummy VarBuilder（所有張量用 zeros）
    pub fn dummy() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    /// 取得指定形狀的零張量（骨架階段用）
    pub fn get<S: Into<candle_core::Shape>>(&self, _name: &str, shape: S) -> Tensor {
        Tensor::zeros(shape, candle_core::DType::F32, &self.device).unwrap()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 建立子作用域 VarBuilder
    pub fn sub(&self, _prefix: impl AsRef<str>) -> Self {
        Self {
            device: self.device.clone(),
        }
    }
}

/// 線性層權重初始化（He initialization scaled for transformers）
pub fn linear_init(in_dim: usize, out_dim: usize, device: &Device) -> Tensor {
    let scale = (1.0 / (in_dim as f64).sqrt()) as f32;
    Tensor::rand(-scale, scale, (out_dim, in_dim), device).unwrap()
}

/// 零初始化
pub fn zero_init(device: &Device) -> Tensor {
    Tensor::zeros(0, candle_core::DType::F32, device).unwrap()
}

/// 量化敏感度分析結果
pub struct SensitivityReport {
    /// 每層的量化誤差（cosine distance）
    pub per_layer_cosine: Vec<f64>,
    /// 建議保留 FP16 的錨點層
    pub anchor_layers: Vec<usize>,
}

/// 逐層敏感度分析
///
/// 對碼本 Embedding 或卷積層，計算量化前後的餘弦相似度，
/// 找出對量化敏感的「錨點層」需保留 FP16。
pub fn analyze_sensitivity(
    _original: &Tensor,
    _quantized: &Tensor,
) -> crate::Result<SensitivityReport> {
    // TODO(Phase 4): 實作完整敏感度分析
    Ok(SensitivityReport {
        per_layer_cosine: Vec::new(),
        anchor_layers: Vec::new(),
    })
}

/// INT8 量化校準
///
/// 使用校準數據集對碼本進行 INT8 量化。
/// 返回量化後的權重及縮放因子。
pub fn calibrate_int8(
    _weights: &Tensor,
    _calibration_data: &Tensor,
) -> crate::Result<(Tensor, Tensor)> {
    // TODO(Phase 4): 實作 INT8 校準
    Err(crate::Error::Config(
        "INT8 calibration not yet implemented".into(),
    ))
}

/// 權重轉換: PyTorch safetensors -> Candle 相容格式
pub fn convert_from_pytorch(
    _pytorch_tensors: &std::collections::HashMap<String, Tensor>,
) -> crate::Result<std::collections::HashMap<String, Tensor>> {
    // TODO(Phase 3): 實作權重名稱對映與維度轉換
    Err(crate::Error::Config(
        "Weight conversion not yet implemented".into(),
    ))
}

// ---------------------------------------------------------------------------
// 單元測試
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_init_shape() {
        let device = Device::Cpu;
        let t = linear_init(512, 1024, &device);
        assert_eq!(t.dims(), &[1024, 512]);
    }

    #[test]
    fn test_var_builder_creation() {
        let device = Device::Cpu;
        let vb = VarBuilder::from_device(device.clone());
        // Device doesn't implement Display, just check it can be created
        assert!(vb.device().is_cpu());
    }
}
