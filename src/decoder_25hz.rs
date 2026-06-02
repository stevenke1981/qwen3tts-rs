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

use crate::codec::{DiTConfig, FlowMatchingDecoder, OdeSolverConfig, OdeSolverType};
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

impl Decoder25Hz {
    /// 建立 25Hz 解碼器
    ///
    /// # 參數
    /// - `config`: 解碼器配置
    /// - `device`: 運算裝置
    pub fn new(config: DecoderConfig, device: &Device) -> Result<Self> {
        let dit_config = DiTConfig {
            hidden_size: config.dit_hidden_dim,
            num_attention_heads: config.dit_num_heads,
            num_hidden_layers: config.dit_num_blocks,
            cond_dim: config.embedding_dim,
            ..Default::default()
        };

        let solver_config = OdeSolverConfig {
            solver_type: OdeSolverType::Euler,
            num_steps: config.ode_steps,
            ..Default::default()
        };

        let flow_matching = FlowMatchingDecoder::new(dit_config, solver_config, device);

        Ok(Self {
            _config: config,
            device: device.clone(),
            flow_matching,
            context_window: Vec::with_capacity(8),
            max_context_len: 8,
        })
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
    fn test_decoder_creation() {
        let device = Device::Cpu;
        let config = DecoderConfig::high_quality();
        let decoder = Decoder25Hz::new(config, &device).unwrap();
        assert_eq!(decoder.flow_matching.solver().config.num_steps, 32);
    }
}
