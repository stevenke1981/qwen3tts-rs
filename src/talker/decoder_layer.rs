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
use crate::alignment_stage_dump::StageDumpObserver;

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

    pub fn forward_with_observer<O: StageDumpObserver>(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<(&Tensor, &Tensor)>,
        layer_index: usize,
        phase: &str,
        observer: &mut O,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let capture = observer.wants_capture();
        let residual = hidden_states.clone();
        let normed = self.input_layernorm.forward(hidden_states)?;
        if capture {
            observer.on_stage(
                &format!("talker-{phase}-l{layer_index}-input-norm"),
                &normed,
                "BTH",
            )?;
        }
        let (attn_out, cache) =
            self.self_attn
                .forward(&normed, cos, sin, attention_mask, kv_cache)?;
        if capture {
            observer.on_stage(
                &format!("talker-{phase}-l{layer_index}-attention-output"),
                &attn_out,
                "BTH",
            )?;
        }
        let hidden_states = (residual + attn_out)?;
        let residual = hidden_states.clone();
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        if capture {
            observer.on_stage(
                &format!("talker-{phase}-l{layer_index}-post-attention-norm"),
                &normed,
                "BTH",
            )?;
        }
        let mlp_out = self.mlp.forward(&normed)?;
        if capture {
            observer.on_stage(
                &format!("talker-{phase}-l{layer_index}-mlp-output"),
                &mlp_out,
                "BTH",
            )?;
        }
        let hidden_states = (residual + mlp_out)?;
        if capture {
            observer.on_stage(
                &format!("talker-hidden-{phase}-l{layer_index}"),
                &hidden_states,
                "BTH",
            )?;
        }
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

    pub fn forward_with_observer<O: StageDumpObserver>(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<(&Tensor, &Tensor)>,
        layer_index: usize,
        phase: &str,
        observer: &mut O,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let capture = observer.wants_capture();
        let residual = hidden_states.clone();
        let normed = self.input_layernorm.forward(hidden_states)?;
        if capture {
            observer.on_stage(
                &format!("code-predictor-{phase}-l{layer_index}-input-norm"),
                &normed,
                "BTH",
            )?;
        }
        let (attn_out, cache) =
            self.self_attn
                .forward(&normed, cos, sin, attention_mask, kv_cache)?;
        if capture {
            observer.on_stage(
                &format!("code-predictor-{phase}-l{layer_index}-attention-output"),
                &attn_out,
                "BTH",
            )?;
        }
        let hidden_states = (residual + attn_out)?;
        let residual = hidden_states.clone();
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        if capture {
            observer.on_stage(
                &format!("code-predictor-{phase}-l{layer_index}-post-attention-norm"),
                &normed,
                "BTH",
            )?;
        }
        let mlp_out = self.mlp.forward(&normed)?;
        if capture {
            observer.on_stage(
                &format!("code-predictor-{phase}-l{layer_index}-mlp-output"),
                &mlp_out,
                "BTH",
            )?;
        }
        let hidden_states = (residual + mlp_out)?;
        if capture {
            observer.on_stage(
                &format!("code-predictor-hidden-{phase}-l{layer_index}"),
                &hidden_states,
                "BTH",
            )?;
        }
        Ok((hidden_states, cache))
    }
}
