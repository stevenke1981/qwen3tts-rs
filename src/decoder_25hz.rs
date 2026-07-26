//! # 25Hz 高品質解碼器 (Flow Matching DiT)
//!
//! 使用 Block-wise Flow Matching Diffusion Transformer 進行高品質音頻解碼。
//!
//! ## 延遲目標
//! - 首包延遲 ≤ 300ms
//!
//! ## 設計要點
//! - ODE 步數與原版嚴格一致
//! - 分塊上下文窗口管理
//! - 單線程 ODE 求解

use candle_core::{Device, Tensor};

use crate::codec::FlowMatchingDecoder;
use crate::{DecoderConfig, Error, Result, TtsDecoder};

/// 25Hz 高品質解碼器
#[allow(dead_code)]
pub struct Decoder25Hz {
    _config: DecoderConfig,
    device: Device,

    /// Flow Matching 解碼器
    flow_matching: FlowMatchingDecoder,

    /// 上下文窗口
    context_window: Vec<Tensor>,

    /// 最大上下文長度
    max_context_len: usize,
}

impl std::fmt::Debug for Decoder25Hz {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decoder25Hz")
            .field("max_context_len", &self.max_context_len)
            .finish_non_exhaustive()
    }
}

impl Decoder25Hz {
    /// 建立 25Hz 解碼器（目前回傳錯誤——此模式暫緩，無可參照的第三方實作）
    ///
    /// # 參數
    /// - `config`: 解碼器配置
    /// - `_device`: 運算裝置
    pub fn new(_config: DecoderConfig, _device: &Device) -> Result<Self> {
        Err(Error::Config(
            "25Hz Flow-Matching decoder 尚未實作，且無可參照的第三方實作，暫緩此模式；如需啟用請參考 spec.md Phase 2".into(),
        ))
    }
}

impl TtsDecoder for Decoder25Hz {
    fn new(_config: DecoderConfig) -> Result<Self> {
        Err(Error::Config(
            "Decoder25Hz::new with default not yet implemented. Use Decoder25Hz::new(...) directly."
                .into(),
        ))
    }

    fn decode_chunk(&mut self, _tokens: &[u16]) -> Result<Vec<f32>> {
        // TODO(Phase 2): 實作 25Hz 解碼管線
        Err(Error::Config(
            "25Hz decoder not yet implemented (Phase 2)".into(),
        ))
    }

    fn reset_state(&mut self) {
        self.context_window.clear();
    }
}

// ---------------------------------------------------------------------------
// 單元測試
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn test_decoder_creation_returns_error() {
        let device = Device::Cpu;
        let config = DecoderConfig::high_quality();
        let err = Decoder25Hz::new(config, &device).expect_err("建構 25Hz decoder 應回傳錯誤");
        let msg = format!("{err}");
        assert!(
            msg.contains("尚未實作"),
            "錯誤訊息應提及「尚未實作」，實際: {msg}"
        );
    }
}
