//! # Talker 解碼器層
//!
//! 單個 Transformer 解碼器層：
//! 1. Input LayerNorm
//! 2. Self-Attention (TalkerAttention)
//! 3. Residual + Post-Attention LayerNorm
//! 4. SwiGLU MLP
//! 5. Residual

use candle_core::{Result, Tensor};

use super::primitives::{RMSNorm, SwiGLUMLP};
use super::talker_attention::TalkerAttention;

/// 單個 Talker 解碼器層
#[derive(Debug, Clone)]
pub struct TalkerDecoderLayer {
    pub input_layernorm: RMSNorm,
    pub self_attn: TalkerAttention,
    pub post_attention_layernorm: RMSNorm,
    pub mlp: SwiGLUMLP,
}

impl TalkerDecoderLayer {
    pub fn new(
        input_layernorm: RMSNorm,
        self_attn: TalkerAttention,
        post_attention_layernorm: RMSNorm,
        mlp: SwiGLUMLP,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        // ── Pre-Attention Norm ──
        let residual = hidden_states.clone();
        let normed = self.input_layernorm.forward(hidden_states)?;

        // ── Self Attention ──
        let (attn_out, cache) =
            self.self_attn
                .forward(&normed, cos, sin, attention_mask, kv_cache)?;
        let hidden_states = (residual + attn_out)?;

        // ── Post-Attention Norm + MLP ──
        let residual = hidden_states.clone();
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let hidden_states = (residual + mlp_out)?;

        Ok((hidden_states, cache))
    }
}

/// 標準解碼器層（用於 Code Predictor，標準 1D RoPE 注意力）
#[derive(Debug, Clone)]
pub struct StandardDecoderLayer {
    pub input_layernorm: RMSNorm,
    pub self_attn: super::talker_attention::StandardAttention,
    pub post_attention_layernorm: RMSNorm,
    pub mlp: SwiGLUMLP,
}

impl StandardDecoderLayer {
    pub fn new(
        input_layernorm: RMSNorm,
        self_attn: super::talker_attention::StandardAttention,
        post_attention_layernorm: RMSNorm,
        mlp: SwiGLUMLP,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let residual = hidden_states.clone();
        let normed = self.input_layernorm.forward(hidden_states)?;
        let (attn_out, cache) =
            self.self_attn
                .forward(&normed, cos, sin, attention_mask, kv_cache)?;
        let hidden_states = (residual + attn_out)?;

        let residual = hidden_states.clone();
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let hidden_states = (residual + mlp_out)?;
        Ok((hidden_states, cache))
    }
}
