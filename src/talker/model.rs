//! # TalkerModel — 28 層 Transformer 解碼器
//!
//! 對應 `Qwen3TTSTalkerModel` (PyTorch)。
//! 包含 28 層解碼器 + 最終 LayerNorm。

use candle_core::{Result, Tensor};

use super::decoder_layer::TalkerDecoderLayer;
use super::primitives::RMSNorm;
use crate::alignment_stage_dump::StageDumpObserver;

/// Talker 主模型
#[derive(Debug, Clone)]
pub struct TalkerModel {
    pub layers: Vec<TalkerDecoderLayer>,
    pub norm: RMSNorm,
}

impl TalkerModel {
    fn validate_kv_caches(
        &self,
        caches: &[Option<(Tensor, Tensor)>],
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<()> {
        let device = hidden_states.device();
        if !cos.device().same_device(device)
            || !sin.device().same_device(device)
            || attention_mask.is_some_and(|m| !m.device().same_device(device))
        {
            return Err(candle_core::Error::Msg(
                "Talker input tensors are on different devices".into(),
            ));
        }
        if caches.len() != self.layers.len() {
            return Err(candle_core::Error::Msg(
                "Talker KV cache count mismatch".into(),
            ));
        }
        let present = caches.first().map(|c| c.is_some()).unwrap_or(false);
        let mut seq = None;
        for (i, cache) in caches.iter().enumerate() {
            if cache.is_some() != present {
                return Err(candle_core::Error::Msg(
                    "inconsistent Talker KV cache presence".into(),
                ));
            }
            if let Some((k, v)) = cache {
                let kd = k.dims4()?;
                let vd = v.dims4()?;
                let a = &self.layers[i].self_attn;
                if !k.device().same_device(device)
                    || !v.device().same_device(device)
                    || kd != vd
                    || kd.0 != 1
                    || kd.1 != a.num_kv_heads
                    || kd.3 != a.head_dim
                    || k.dtype() != v.dtype()
                {
                    return Err(candle_core::Error::Msg(format!(
                        "invalid Talker KV cache at layer {i}"
                    )));
                }
                if let Some(expected) = seq {
                    if kd.2 != expected {
                        return Err(candle_core::Error::Msg(
                            "inconsistent Talker KV cache sequence".into(),
                        ));
                    }
                } else {
                    seq = Some(kd.2);
                }
            }
        }
        Ok(())
    }
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
    ) -> Result<Tensor> {
        self.validate_kv_caches(kv_caches, hidden_states, cos, sin, attention_mask)?;
        let mut h = hidden_states.clone();
        if self.layers.len() > 28 {
            return Err(candle_core::Error::Msg(
                "Talker supports at most 28 cache layers".into(),
            ));
        }
        let mut next_caches: [Option<(Tensor, Tensor)>; 28] = std::array::from_fn(|_| None);

        for (i, layer) in self.layers.iter().enumerate() {
            let cache = if i < self.layers.len() {
                kv_caches[i].as_ref().map(|(k, v)| (k, v))
            } else {
                None
            };
            let (next_h, updated_cache) = layer.forward(&h, cos, sin, attention_mask, cache)?;
            h = next_h;
            if i < self.layers.len() {
                next_caches[i] = Some(updated_cache);
            }
        }

        // Final norm
        let h = self.norm.forward(&h)?;
        for (dst, src) in kv_caches
            .iter_mut()
            .zip(next_caches[..self.layers.len()].iter_mut())
        {
            *dst = src.take();
        }

        Ok(h)
    }

    pub fn forward_with_observer<O: StageDumpObserver>(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_caches: &mut [Option<(Tensor, Tensor)>],
        phase: &str,
        observer: &mut O,
    ) -> Result<Tensor> {
        self.validate_kv_caches(kv_caches, hidden_states, cos, sin, attention_mask)?;
        let mut h = hidden_states.clone();
        if self.layers.len() > 28 {
            return Err(candle_core::Error::Msg(
                "Talker supports at most 28 cache layers".into(),
            ));
        }
        let mut next_caches: [Option<(Tensor, Tensor)>; 28] = std::array::from_fn(|_| None);
        let capture = observer.wants_capture();
        if capture {
            if phase == "prefill" {
                observer.on_stage("talker-input-embed", &h, "BTH")?;
            }
            observer.on_stage(&format!("talker-input-{phase}"), &h, "BTH")?;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let cache = if i < self.layers.len() {
                kv_caches[i].as_ref().map(|(k, v)| (k, v))
            } else {
                None
            };
            let (next_h, updated_cache) = layer.forward_with_observer(
                &h,
                cos,
                sin,
                attention_mask,
                cache,
                i,
                phase,
                observer,
            )?;
            h = next_h;
            if i < self.layers.len() {
                next_caches[i] = Some(updated_cache);
            }
        }
        let h = self.norm.forward(&h)?;
        for (dst, src) in kv_caches
            .iter_mut()
            .zip(next_caches[..self.layers.len()].iter_mut())
        {
            *dst = src.take();
        }
        if capture {
            observer.on_stage(&format!("talker-hidden-{phase}-final"), &h, "BTH")?;
            if phase != "prefill" {
                observer.on_stage(&format!("talker-hidden-{phase}"), &h, "BTH")?;
            }
        }
        Ok(h)
    }
}
