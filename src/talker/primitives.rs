//! # Talker 基礎原語
//!
//! RMSNorm、標準 RoPE、3D Multimodal RoPE、SwiGLU MLP。

use super::config::TalkerConfig;
use candle_core::{DType, Device, Result, Tensor};

// ---------------------------------------------------------------------------
// Embedding Lookup Helper
// ---------------------------------------------------------------------------

/// 從權重矩陣做 embedding lookup
///
/// weight: [vocab_size, dim]
/// input_ids: [batch, seq_len] 中的值為索引
/// 回傳: [batch, seq_len, dim]
pub fn embedding_lookup(weight: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
    let dim = weight.dim(1)?;
    let shape = input_ids.shape().dims().to_vec();
    let flat_ids = input_ids.flatten_all()?;
    let result = weight.index_select(&flat_ids, 0)?;
    let mut output_shape = shape;
    output_shape.push(dim);
    result.reshape(output_shape)
}

/// PyTorch `F.linear(x, weight, None)` equivalent for rank >= 2 tensors.
///
/// `x`: `[..., in_features]`
/// `weight`: `[out_features, in_features]`
/// returns: `[..., out_features]`
pub fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let rank = x.rank();
    let in_features = x.dim(rank - 1)?;
    let out_features = weight.dim(0)?;
    let batch = x.elem_count() / in_features;
    let x2 = x.reshape((batch, in_features))?;
    let y = x2.matmul(&weight.transpose(0, 1)?)?;
    let mut shape = x.shape().dims()[..rank - 1].to_vec();
    shape.push(out_features);
    y.reshape(shape)
}

/// PyTorch `F.linear(x, weight, bias)` equivalent for rank >= 2 tensors.
pub fn linear_with_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    linear(x, weight)?.broadcast_add(bias)
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// RMS Normalization（與 T5LayerNorm 等價）
#[derive(Debug, Clone)]
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl RMSNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let ndim = x.rank();
        let variance = x_f32.sqr()?.mean_keepdim(ndim - 1)?;
        let normalized = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        normalized
            .to_dtype(input_dtype)?
            .broadcast_mul(&self.weight)
    }
}

// ---------------------------------------------------------------------------
// 標準 1D RoPE（用於 Code Predictor）
// ---------------------------------------------------------------------------

/// 標準 1D Rotary Position Embedding
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RotaryEmbedding {
    inv_freq: Tensor,
    max_seq_len: usize,
    theta: f64,
}

impl RotaryEmbedding {
    pub fn new(max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let dim = 64;
        let inv_freq: Vec<f32> = (0..dim)
            .map(|i| 1.0 / theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_slice(&inv_freq, (1, 1, dim), device)?;
        Ok(Self {
            inv_freq,
            max_seq_len,
            theta,
        })
    }

    pub fn forward(&self, x: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_batch, _seq_len) = position_ids.dims2()?;
        let head_dim = x.dim(2)?;
        let half = head_dim / 2;

        let pos = position_ids.unsqueeze(1)?.to_dtype(DType::F32)?;
        let inv_freq = if half <= 64 {
            self.inv_freq.narrow(2, 0, half)?
        } else {
            let new_inv: Vec<f32> = (0..half)
                .map(|i| 1.0 / self.theta.powf(i as f64 / half as f64) as f32)
                .collect();
            Tensor::from_slice(&new_inv, (1, 1, half), x.device())?
        };

        let freqs = pos.matmul(&inv_freq)?;
        let freqs = freqs.unsqueeze(3)?;
        let emb = Tensor::cat(&[freqs.clone(), freqs], 3)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Ok((cos, sin))
    }
}

/// 應用標準 1D RoPE 到 Q/K
pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let q_embed = rotate_half_and_apply(q, cos, sin)?;
    let k_embed = rotate_half_and_apply(k, cos, sin)?;
    Ok((q_embed, k_embed))
}

// ---------------------------------------------------------------------------
// 3D Multimodal RoPE（用於 Talker 主模型）
// ---------------------------------------------------------------------------

/// 3D Multimodal Rotary Position Embedding
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MultimodalRotaryEmbedding {
    inv_freq: Tensor,
    max_seq_len: usize,
    theta: f64,
    mrope_section: Vec<usize>,
    rope_interleaved: bool,
}

impl MultimodalRotaryEmbedding {
    pub fn new(config: &TalkerConfig, device: &Device) -> Result<Self> {
        let dim = config.head_dim / 2;
        let inv_freq: Vec<f32> = (0..dim)
            .map(|i| 1.0 / config.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_slice(&inv_freq, dim, device)?;
        Ok(Self {
            inv_freq,
            max_seq_len: config.max_position_embeddings,
            theta: config.rope_theta,
            mrope_section: config.mrope_section.clone(),
            rope_interleaved: config.rope_interleaved,
        })
    }

    /// 計算 3D cos/sin
    /// position_ids: [3, batch, seq_len]
    /// 回傳 cos/sin: [batch, 1, seq_len, head_dim]
    pub fn forward(&self, _x: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let ids = position_ids.to_vec3::<u32>()?;
        let axes = ids.len();
        let batch = ids[0].len();
        let seq_len = ids[0][0].len();
        debug_assert_eq!(axes, 3);

        let inv_freq = self.inv_freq.to_vec1::<f32>()?;
        let half_dim = inv_freq.len();
        let head_dim = half_dim * 2;
        let mut cos = vec![0.0f32; batch * seq_len * head_dim];
        let mut sin = vec![0.0f32; batch * seq_len * head_dim];

        for b in 0..batch {
            for t in 0..seq_len {
                for d in 0..head_dim {
                    let axis = self.mrope_axis_for_dim(d, half_dim);
                    let freq_idx = d % half_dim;
                    let angle = ids[axis][b][t] as f32 * inv_freq[freq_idx];
                    let out_idx = (b * seq_len + t) * head_dim + d;
                    cos[out_idx] = angle.cos();
                    sin[out_idx] = angle.sin();
                }
            }
        }

        let cos = Tensor::from_slice(&cos, (batch, 1, seq_len, head_dim), position_ids.device())?;
        let sin = Tensor::from_slice(&sin, (batch, 1, seq_len, head_dim), position_ids.device())?;
        Ok((cos, sin))
    }

    fn mrope_axis_for_dim(&self, dim: usize, half_dim: usize) -> usize {
        let half_dim_idx = dim % half_dim;
        if self.rope_interleaved {
            let modality_num = self.mrope_section.len();
            for axis in 1..modality_num {
                let end = self.mrope_section[axis] * modality_num;
                if half_dim_idx >= axis
                    && half_dim_idx < end
                    && (half_dim_idx - axis) % modality_num == 0
                {
                    return axis;
                }
            }
            0
        } else {
            let mut offset = 0usize;
            for (axis, section) in self.mrope_section.iter().map(|s| s * 2).enumerate() {
                if dim >= offset && dim < offset + section {
                    return axis;
                }
                offset += section;
            }
            0
        }
    }
}

/// 應用 3D Multimodal RoPE 到 Q/K
pub fn apply_multimodal_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let q_embed = rotate_half_and_apply(q, cos, sin)?;
    let k_embed = rotate_half_and_apply(k, cos, sin)?;
    Ok((q_embed, k_embed))
}

// ---------------------------------------------------------------------------
// 內部輔助函式
// ---------------------------------------------------------------------------

fn rotate_half_and_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: [batch, num_heads, seq_len, head_dim]
    let head_dim = x.dim(3)?;
    let half = head_dim / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let rotated = Tensor::cat(&[x2.neg()?, x1], 3)?;
    x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?
}

// ---------------------------------------------------------------------------
// SwiGLU MLP
// ---------------------------------------------------------------------------

/// SwiGLU 多層感知器
#[derive(Debug, Clone)]
pub struct SwiGLUMLP {
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
}

impl SwiGLUMLP {
    pub fn new(gate_proj: Tensor, up_proj: Tensor, down_proj: Tensor) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = linear(x, &self.gate_proj)?;
        let gate = gate.silu()?;
        let up = linear(x, &self.up_proj)?;
        let hidden = (gate * up)?;
        linear(&hidden, &self.down_proj)
    }
}

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

/// 簡易 KV Cache
#[derive(Debug, Clone)]
pub struct KVCache {
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn update(&mut self, key: &Tensor, value: &Tensor, layer_idx: usize) {
        if layer_idx >= self.k.len() {
            self.k
                .resize(layer_idx + 1, Tensor::new(&[0.0f32], &Device::Cpu).unwrap());
            self.v
                .resize(layer_idx + 1, Tensor::new(&[0.0f32], &Device::Cpu).unwrap());
        }
        if self.k[layer_idx].elem_count() == 0 || self.k[layer_idx].dims().len() < 3 {
            self.k[layer_idx] = key.clone();
            self.v[layer_idx] = value.clone();
        } else {
            self.k[layer_idx] = Tensor::cat(&[&self.k[layer_idx], key], 2).unwrap();
            self.v[layer_idx] = Tensor::cat(&[&self.v[layer_idx], value], 2).unwrap();
        }
    }

    pub fn get(&self, layer_idx: usize) -> Option<(&Tensor, &Tensor)> {
        if layer_idx < self.k.len() && self.k[layer_idx].elem_count() > 1 {
            Some((&self.k[layer_idx], &self.v[layer_idx]))
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.k.len()
    }

    pub fn seq_len(&self, layer_idx: usize) -> usize {
        if layer_idx < self.k.len() && self.k[layer_idx].dims().len() >= 3 {
            self.k[layer_idx].dim(2).unwrap_or(0)
        } else {
            0
        }
    }
}

// ---------------------------------------------------------------------------
// 因果注意力遮罩
// ---------------------------------------------------------------------------

/// 建立因果注意力遮罩（上三角 -inf）
pub fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0f32 } else { f32::NEG_INFINITY }))
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len), device)
}
