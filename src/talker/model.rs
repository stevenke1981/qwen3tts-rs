//! # TalkerModel — 28 層 Transformer 解碼器
//!
//! 對應 `Qwen3TTSTalkerModel` (PyTorch)。
//! 包含 28 層解碼器 + 最終 LayerNorm。

use candle_core::{Result, Tensor};

use super::decoder_layer::TalkerDecoderLayer;
use super::primitives::RMSNorm;

/// Talker 主模型
#[derive(Debug, Clone)]
pub struct TalkerModel {
    pub layers: Vec<TalkerDecoderLayer>,
    pub norm: RMSNorm,
}

impl TalkerModel {
    pub fn new(layers: Vec<TalkerDecoderLayer>, norm: RMSNorm) -> Self {
        Self { layers, norm }
    }

    /// 前向傳播
    ///
    /// # 參數
    /// - `hidden_states`: [batch, seq_len, hidden_size]
    /// - `cos`, `sin`: 3D RoPE 嵌入
    /// - `attention_mask`: 可選的注意力遮罩
    /// - `kv_caches`: 可選的各層 KV cache
    ///
    /// # 回傳
    /// - `(output, past_kv)` 其中 output: [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_caches: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<(Tensor, Vec<Option<(Tensor, Tensor)>>)> {
        let mut h = hidden_states.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            let cache = if i < kv_caches.len() {
                kv_caches[i].as_ref().map(|(k, v)| (k, v))
            } else {
                None
            };
            let (next_h, updated_cache) = layer.forward(&h, cos, sin, attention_mask, cache)?;
            h = next_h;
            if i < kv_caches.len() {
                kv_caches[i] = Some(updated_cache);
            }
        }

        // Final norm
        let h = self.norm.forward(&h)?;

        Ok((h, kv_caches.to_vec()))
    }
}
