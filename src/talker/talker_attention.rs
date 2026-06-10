//! # Talker 注意力機制
//!
//! 實作 `Qwen3TTSTalkerAttention`：
//! - GQA (16 查詢頭, 8 KV 頭)
//! - head_dim = 128 (與 hidden_size/num_heads 解耦)
//! - QK RMSNorm（每頭單獨正規化）
//! - 3D Multimodal RoPE

use candle_core::{Result, Tensor};

use super::primitives::{apply_multimodal_rotary_pos_emb, linear, RMSNorm};

/// Talker 注意力層
#[derive(Debug, Clone)]
pub struct TalkerAttention {
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,

    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_kv_groups: usize,
    pub scaling: f64,
}

impl TalkerAttention {
    pub fn new(
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        o_proj: Tensor,
        q_norm_weight: Tensor,
        k_norm_weight: Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
    ) -> Self {
        let num_kv_groups = num_heads / num_kv_heads;
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: RMSNorm::new(q_norm_weight, rms_norm_eps),
            k_norm: RMSNorm::new(k_norm_weight, rms_norm_eps),
            num_heads,
            num_kv_heads,
            head_dim,
            num_kv_groups,
            scaling: (head_dim as f64).powf(-0.5),
        }
    }

    /// 前向傳播
    ///
    /// # 參數
    /// - `hidden_states`: [batch, seq_len, hidden_size]
    /// - `cos`, `sin`: [batch, 1, seq_len, head_dim] — RoPE 嵌入
    /// - `attention_mask`: [seq_len, seq_len] 或 `None`
    /// - `kv_cache`: 可選的 KV cache（(k, v) 元組）
    ///
    /// # 回傳
    /// - `(output, cache)` 其中 output 形狀為 [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let (b_sz, seq_len, _hidden_size) = hidden_states.dims3()?;

        // ── 投影 ──
        let q = linear(hidden_states, &self.q_proj)?;
        let k = linear(hidden_states, &self.k_proj)?;
        let v = linear(hidden_states, &self.v_proj)?;

        // Reshape: [batch, seq_len, num_heads * head_dim] → [batch, seq_len, num_heads, head_dim]
        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;

        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q.permute((0, 2, 1, 3))?;
        let k = k.permute((0, 2, 1, 3))?;
        let v = v.permute((0, 2, 1, 3))?;

        // ── QK Norm（每頭正規化）──
        // Flatten head dim into batch for norm, then reshape back
        let q = self.apply_qk_norm(&q, &self.q_norm)?;
        let k = self.apply_qk_norm(&k, &self.k_norm)?;

        // ── RoPE ──
        let (q, k) = apply_multimodal_rotary_pos_emb(&q, &k, cos, sin)?;

        // ── KV Cache ──
        let (cache_kv_k, cache_kv_v) = if let Some((cache_k, cache_v)) = kv_cache {
            let k = Tensor::cat(&[cache_k, &k], 2)?;
            let v = Tensor::cat(&[cache_v, &v], 2)?;
            (k, v)
        } else {
            (k, v)
        };

        // ── GQA: repeat KV heads ──
        let k = self.repeat_kv(&cache_kv_k)?;
        let v = self.repeat_kv(&cache_kv_v)?;

        // ── Scaled Dot-Product Attention ──
        let scale = self.scaling;
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * scale)?;

        let attn_weights = if let Some(mask) = attention_mask {
            // mask: [seq_len, seq_len] or broadcastable
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax(&attn_weights, 3)?;
        let attn_output = attn_weights.matmul(&v)?;

        // ── 輸出投影 ──
        // attn_output: [batch, num_heads, seq_len, head_dim]
        // → [batch, seq_len, num_heads * head_dim]
        let attn_output = attn_output.permute((0, 2, 1, 3))?;
        let attn_output = attn_output.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
        let output = linear(&attn_output, &self.o_proj)?;

        Ok((output, (cache_kv_k, cache_kv_v)))
    }

    /// 對 Q 或 K 的每頭做 RMSNorm
    fn apply_qk_norm(&self, x: &Tensor, norm: &RMSNorm) -> Result<Tensor> {
        // x: [batch, num_heads, seq_len, head_dim]
        // Flatten batch*num_heads into single batch dim
        let (b_sz, num_heads, seq_len, head_dim) = x.dims4()?;
        let x = x.reshape((b_sz * num_heads, seq_len, head_dim))?;
        let x = norm.forward(&x)?;
        x.reshape((b_sz, num_heads, seq_len, head_dim))
    }

    /// Repeat KV heads for GQA
    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.num_kv_groups == 1 {
            return Ok(x.clone());
        }
        // x: [batch, num_kv_heads, seq_len, head_dim]
        let (b_sz, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        // Expand
        let x = x.unsqueeze(2)?; // [batch, num_kv_heads, 1, seq_len, head_dim]
        let x = x.expand((b_sz, num_kv_heads, self.num_kv_groups, seq_len, head_dim))?;
        x.reshape((b_sz, num_kv_heads * self.num_kv_groups, seq_len, head_dim))
    }
}

// ---------------------------------------------------------------------------
// 標準注意力（用於 Code Predictor）
// ---------------------------------------------------------------------------

/// 標準 1D RoPE 注意力（無 QK Norm、無 3D RoPE）
#[derive(Debug, Clone)]
pub struct StandardAttention {
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,

    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,

    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_kv_groups: usize,
    pub scaling: f64,
}

impl StandardAttention {
    pub fn new(
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        o_proj: Tensor,
        q_norm_weight: Tensor,
        k_norm_weight: Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
    ) -> Self {
        let num_kv_groups = num_heads / num_kv_heads;
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: RMSNorm::new(q_norm_weight, rms_norm_eps),
            k_norm: RMSNorm::new(k_norm_weight, rms_norm_eps),
            num_heads,
            num_kv_heads,
            head_dim,
            num_kv_groups,
            scaling: (head_dim as f64).powf(-0.5),
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
        let (b_sz, seq_len, _hidden_size) = hidden_states.dims3()?;

        let q = linear(hidden_states, &self.q_proj)?;
        let k = linear(hidden_states, &self.k_proj)?;
        let v = linear(hidden_states, &self.v_proj)?;

        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;

        let q = q.permute((0, 2, 1, 3))?;
        let k = k.permute((0, 2, 1, 3))?;
        let v = v.permute((0, 2, 1, 3))?;

        // QK Norm
        let q = self.apply_qk_norm(&q, &self.q_norm)?;
        let k = self.apply_qk_norm(&k, &self.k_norm)?;

        // 標準 1D RoPE
        let (q, k) = apply_multimodal_rotary_pos_emb(&q, &k, cos, sin)?;

        let (cache_kv_k, cache_kv_v) = if let Some((cache_k, cache_v)) = kv_cache {
            let k = Tensor::cat(&[cache_k, &k], 2)?;
            let v = Tensor::cat(&[cache_v, &v], 2)?;
            (k, v)
        } else {
            (k, v)
        };

        let k = self.repeat_kv(&cache_kv_k)?;
        let v = self.repeat_kv(&cache_kv_v)?;

        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = (attn * self.scaling)?;
        let attn = if let Some(mask) = attention_mask {
            attn.broadcast_add(mask)?
        } else {
            attn
        };
        let attn = candle_nn::ops::softmax(&attn, 3)?;
        let out = attn.matmul(&v)?;

        let out = out.permute((0, 2, 1, 3))?;
        let out = out.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
        let out = linear(&out, &self.o_proj)?;
        Ok((out, (cache_kv_k, cache_kv_v)))
    }

    fn apply_qk_norm(&self, x: &Tensor, norm: &RMSNorm) -> Result<Tensor> {
        let (b_sz, num_heads, seq_len, head_dim) = x.dims4()?;
        let x = x.reshape((b_sz * num_heads, seq_len, head_dim))?;
        let x = norm.forward(&x)?;
        x.reshape((b_sz, num_heads, seq_len, head_dim))
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.num_kv_groups == 1 {
            return Ok(x.clone());
        }
        let (b_sz, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x.unsqueeze(2)?;
        let x = x.expand((b_sz, num_kv_heads, self.num_kv_groups, seq_len, head_dim))?;
        x.reshape((b_sz, num_kv_heads * self.num_kv_groups, seq_len, head_dim))
    }
}
