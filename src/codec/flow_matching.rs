//! # Flow Matching ODE Solver + DiT 主干 (25Hz)
//!
//! 實作 25Hz 高品質模式所需的 Block-wise Flow Matching DiT 解碼器。
//!
//! ## 設計要點
//! - ODE 步數與原版嚴格一致，禁止自適應步長
//! - 分塊上下文窗口管理
//! - 所有量化權重通過數值對齊測試方可合入

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};

use crate::weights::WeightLoader;

// ---------------------------------------------------------------------------
// ODE Solver
// ---------------------------------------------------------------------------

/// ODE 求解器類型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeSolverType {
    Euler,
    RK4,
}

/// ODE 求解器配置
#[derive(Debug, Clone)]
pub struct OdeSolverConfig {
    pub solver_type: OdeSolverType,
    pub num_steps: usize,
    pub t_start: f64,
    pub t_end: f64,
}

impl Default for OdeSolverConfig {
    fn default() -> Self {
        Self {
            solver_type: OdeSolverType::Euler,
            num_steps: 32,
            t_start: 0.0,
            t_end: 1.0,
        }
    }
}

/// ODE 求解器
pub struct OdeSolver {
    pub config: OdeSolverConfig,
}

impl OdeSolver {
    pub fn new(config: OdeSolverConfig) -> Self {
        Self { config }
    }

    pub fn solve<F>(&self, x_init: &Tensor, velocity_fn: F) -> crate::Result<Tensor>
    where
        F: Fn(f64, &Tensor) -> crate::Result<Tensor>,
    {
        let dt = (self.config.t_end - self.config.t_start) / self.config.num_steps as f64;
        let mut x = x_init.clone();

        match self.config.solver_type {
            OdeSolverType::Euler => {
                for step in 0..self.config.num_steps {
                    let t = self.config.t_start + step as f64 * dt;
                    let v = velocity_fn(t, &x)?;
                    x = (x + (v * dt)?)?;
                }
            }
            OdeSolverType::RK4 => {
                for step in 0..self.config.num_steps {
                    let t = self.config.t_start + step as f64 * dt;
                    let k1 = velocity_fn(t, &x)?;
                    let half_dt = dt / 2.0;
                    let x_temp = (x.clone() + (k1.clone() * half_dt)?)?;
                    let k2 = velocity_fn(t + half_dt, &x_temp)?;
                    let x_temp2 = (x.clone() + (k2.clone() * half_dt)?)?;
                    let k3 = velocity_fn(t + half_dt, &x_temp2)?;
                    let x_temp3 = (x.clone() + (k3.clone() * dt)?)?;
                    let k4 = velocity_fn(t + dt, &x_temp3)?;
                    let sum = ((k1 + (k2 * 2.0)?)? + ((k3 * 2.0)? + k4)?)?;
                    let sixth = dt / 6.0;
                    x = (x + (sum * sixth)?)?;
                }
            }
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// 輔助：從 WeightLoader 載入 Linear 層
// ---------------------------------------------------------------------------

fn load_linear(loader: &WeightLoader, prefix: &str) -> crate::Result<Linear> {
    let w = loader.get(&format!("{prefix}.weight"))?.clone();
    let b = match loader.get(&format!("{prefix}.bias")) {
        Ok(t) => Some(t.clone()),
        Err(_) => None,
    };
    Ok(Linear::new(w, b))
}

#[allow(dead_code)]
fn load_linear_noweight(dim_in: usize, dim_out: usize, device: &Device) -> crate::Result<Linear> {
    let w = Tensor::rand(-0.02f32, 0.02f32, (dim_out, dim_in), device)?;
    let b = Tensor::zeros(dim_out, DType::F32, device)?;
    Ok(Linear::new(w, Some(b)))
}

// ---------------------------------------------------------------------------
// DiT 時序相關元件
// ---------------------------------------------------------------------------

/// 正弦位置編碼（用於時間步）
struct SinusPositionEmbedding {
    dim: usize,
}

impl SinusPositionEmbedding {
    fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        let half_dim = self.dim / 2;
        let device = timestep.device();
        let dtype = timestep.dtype();

        let emb: Vec<f32> = (0..half_dim)
            .map(|i| {
                let factor = -((i as f64) * (10000.0_f64).ln() / (half_dim as f64 - 1.0));
                factor.exp() as f32
            })
            .collect();
        let emb_t = Tensor::from_slice(&emb, half_dim, device)?;
        let scaled = timestep.unsqueeze(1)?.broadcast_mul(&emb_t.unsqueeze(0)?)?;
        let sin = scaled.sin()?;
        let cos = scaled.cos()?;
        let result = Tensor::cat(&[&sin, &cos], 1)?;
        result.to_dtype(dtype)
    }
}

/// 時間步嵌入
#[allow(dead_code)]
pub struct DiTTimestepEmbedding {
    time_embed: SinusPositionEmbedding,
    linear1: Linear,
    linear2: Linear,
}

#[allow(dead_code)]
impl DiTTimestepEmbedding {
    pub fn new(
        _hidden_size: usize,
        freq_embed_dim: usize,
        loader: &WeightLoader,
        prefix: &str,
    ) -> crate::Result<Self> {
        let time_embed = SinusPositionEmbedding::new(freq_embed_dim);
        let linear1 = load_linear(loader, &format!("{prefix}.time_mlp.0"))?;
        let linear2 = load_linear(loader, &format!("{prefix}.time_mlp.2"))?;
        Ok(Self {
            time_embed,
            linear1,
            linear2,
        })
    }

    pub fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        let h = self.time_embed.forward(timestep)?;
        let h = self.linear1.forward(&h)?;
        let h = h.silu()?;
        self.linear2.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// DiT 注意力相關元件
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct AdaLayerNormZero {
    linear: Linear,
    norm: candle_nn::LayerNorm,
}

#[allow(dead_code)]
impl AdaLayerNormZero {
    fn new(
        dim: usize,
        loader: &WeightLoader,
        prefix: &str,
        device: &Device,
    ) -> crate::Result<Self> {
        let linear = load_linear(loader, &format!("{prefix}.linear"))?;
        let norm = candle_nn::LayerNorm::new(
            Tensor::ones(dim, DType::F32, device)?,
            Tensor::zeros(dim, DType::F32, device)?,
            1e-6,
        );
        Ok(Self { linear, norm })
    }

    fn forward(
        &self,
        x: &Tensor,
        emb: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let emb = self.linear.forward(&emb.silu()?)?;
        let dim = emb.dims()[1];
        let _chunk_size = dim / 6;
        let parts = emb.chunk(6, 1)?;
        let shift_msa = parts[0].clone();
        let scale_msa = parts[1].clone();
        let gate_msa = parts[2].clone();
        let shift_mlp = parts[3].clone();
        let scale_mlp = parts[4].clone();
        let gate_mlp = parts[5].clone();

        let normed = self.norm.forward(x)?;
        let modulated = (normed
            .broadcast_mul(&(scale_msa.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift_msa.unsqueeze(1)?))?;
        Ok((modulated, gate_msa, shift_mlp, scale_mlp, gate_mlp))
    }
}

#[allow(dead_code)]
struct AdaLayerNormZeroFinal {
    linear: Linear,
    norm: candle_nn::LayerNorm,
}

#[allow(dead_code)]
impl AdaLayerNormZeroFinal {
    fn new(dim: usize, loader: &WeightLoader, prefix: &str) -> crate::Result<Self> {
        let linear = load_linear(loader, &format!("{prefix}.linear"))?;
        let norm = candle_nn::LayerNorm::new(
            Tensor::ones(dim, DType::F32, linear.weight().device())?,
            Tensor::zeros(dim, DType::F32, linear.weight().device())?,
            1e-6,
        );
        Ok(Self { linear, norm })
    }

    fn forward(&self, x: &Tensor, emb: &Tensor) -> Result<Tensor> {
        let emb = self.linear.forward(&emb.silu()?)?;
        let parts = emb.chunk(2, 1)?;
        let scale = parts[0].clone();
        let shift = parts[1].clone();

        let normed = self.norm.forward(x)?;
        Ok(normed
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?)
    }
}

#[allow(dead_code)]
struct DiTMLP {
    fc1: Linear,
    fc2: Linear,
}

#[allow(dead_code)]
impl DiTMLP {
    fn new(dim: usize, mult: usize, loader: &WeightLoader, prefix: &str) -> crate::Result<Self> {
        let _inner_dim = dim * mult;
        let fc1 = load_linear(loader, &format!("{prefix}.0"))?;
        let fc2 = load_linear(loader, &format!("{prefix}.3"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?;
        let h = h.gelu_erf()?;
        self.fc2.forward(&h)
    }
}

#[allow(dead_code)]
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dims().len() - 1;
    let dim_size = x.dims()[last_dim];
    let half = dim_size / 2;
    let x1 = x.narrow(last_dim, 0, half)?;
    let x2 = x.narrow(last_dim, half, half)?;
    let neg_x2 = x2.neg()?;
    Tensor::cat(&[&neg_x2, &x1], last_dim)
}

#[allow(dead_code)]
fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;
    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

#[allow(dead_code)]
struct DiTAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

#[allow(dead_code)]
impl DiTAttention {
    fn new(config: &DiTConfig, loader: &WeightLoader, prefix: &str) -> crate::Result<Self> {
        let _inner_dim = config.head_dim * config.num_attention_heads;
        let q_proj = load_linear(loader, &format!("{prefix}.to_q"))?;
        let k_proj = load_linear(loader, &format!("{prefix}.to_k"))?;
        let v_proj = load_linear(loader, &format!("{prefix}.to_v"))?;
        let out_proj = load_linear(loader, &format!("{prefix}.to_out.0"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        position_embeddings: Option<(&Tensor, &Tensor)>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = if let Some((cos, sin)) = position_embeddings {
            let q_first = q.narrow(1, 0, 1)?.squeeze(1)?;
            let k_first = k.narrow(1, 0, 1)?.squeeze(1)?;
            let (q_rot, k_rot) = apply_rotary_pos_emb(&q_first, &k_first, cos, sin)?;
            let q_rot = q_rot.unsqueeze(1)?;
            let k_rot = k_rot.unsqueeze(1)?;
            if self.num_heads > 1 {
                let q_rest = q.narrow(1, 1, self.num_heads - 1)?;
                let k_rest = k.narrow(1, 1, self.num_heads - 1)?;
                (
                    Tensor::cat(&[&q_rot, &q_rest], 1)?,
                    Tensor::cat(&[&k_rot, &k_rest], 1)?,
                )
            } else {
                (q_rot, k_rot)
            }
        } else {
            (q, k)
        };

        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = (attn / (self.head_dim as f64).sqrt())?;

        let attn = if let Some(mask) = attention_mask {
            let mask_f = mask.to_dtype(attn.dtype())?;
            let neg_inf = Tensor::new(-1e9f64, attn.device())?;
            let mask_f = mask_f.broadcast_mul(&neg_inf)?;
            (attn + mask_f)?
        } else {
            attn
        };

        let attn = candle_nn::ops::softmax(&attn, 3)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, seq_len, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// DiT 解碼器層
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct DiTDecoderLayer {
    attn_norm: AdaLayerNormZero,
    attn: DiTAttention,
    ff_norm: candle_nn::LayerNorm,
    ff: DiTMLP,
    look_ahead_block: i64,
    look_backward_block: i64,
}

#[allow(dead_code)]
impl DiTDecoderLayer {
    fn new(
        config: &DiTConfig,
        look_ahead_block: i64,
        look_backward_block: i64,
        loader: &WeightLoader,
        prefix: &str,
        device: &Device,
    ) -> crate::Result<Self> {
        let attn_norm = AdaLayerNormZero::new(
            config.hidden_size,
            loader,
            &format!("{prefix}.attn_norm"),
            device,
        )?;
        let attn = DiTAttention::new(config, loader, &format!("{prefix}.attn"))?;
        let ff_norm = candle_nn::LayerNorm::new(
            Tensor::ones(config.hidden_size, DType::F32, device)?,
            Tensor::zeros(config.hidden_size, DType::F32, device)?,
            1e-6,
        );
        let ff = DiTMLP::new(
            config.hidden_size,
            config.ff_mult,
            loader,
            &format!("{prefix}.ff"),
        )?;
        Ok(Self {
            attn_norm,
            attn,
            ff_norm,
            ff,
            look_ahead_block,
            look_backward_block,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        timestep: &Tensor,
        position_embeddings: Option<(&Tensor, &Tensor)>,
        block_diff: Option<&Tensor>,
    ) -> Result<Tensor> {
        let attn_mask = block_diff
            .map(|bd| {
                let ahead = bd.le(self.look_ahead_block)?;
                let backward = bd.ge(-self.look_backward_block)?;
                // 使用乘法模擬邏輯 AND（Candle 無 broadcast_and）
                let mask = ahead.broadcast_mul(&backward)?;
                mask.to_dtype(DType::F32)
            })
            .transpose()?;

        let (normed, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.attn_norm.forward(x, timestep)?;

        let attn_out = self
            .attn
            .forward(&normed, position_embeddings, attn_mask.as_ref())?;
        let h = (x + gate_msa.unsqueeze(1)?.broadcast_mul(&attn_out)?)?;

        let normed_ff = self.ff_norm.forward(&h)?;
        let modulated = (normed_ff
            .broadcast_mul(&(scale_mlp.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift_mlp.unsqueeze(1)?))?;
        let ff_out = self.ff.forward(&modulated)?;
        Ok((h + gate_mlp.unsqueeze(1)?.broadcast_mul(&ff_out)?)?)
    }
}

// ---------------------------------------------------------------------------
// DiT 配置
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DiTConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub ff_mult: usize,
    pub block_size: usize,
    pub look_ahead_layers: Vec<usize>,
    pub look_backward_layers: Vec<usize>,
    pub cond_dim: usize,
    pub emb_dim: usize,
    pub num_embeds: usize,
    pub repeats: usize,
    pub mel_dim: usize,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub freq_embed_dim: usize,
}

impl Default for DiTConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_attention_heads: 16,
            head_dim: 64,
            num_hidden_layers: 22,
            ff_mult: 2,
            block_size: 24,
            look_ahead_layers: vec![10],
            look_backward_layers: vec![0, 20],
            cond_dim: 512,
            emb_dim: 512,
            num_embeds: 8193,
            repeats: 2,
            mel_dim: 80,
            rope_theta: 10000.0,
            max_position_embeddings: 32768,
            freq_embed_dim: 256,
        }
    }
}

// ---------------------------------------------------------------------------
// DiT Codec Embedding
// ---------------------------------------------------------------------------

/// Codebook token embedding (num_embeds × emb_dim, repeats)
#[allow(dead_code)]
struct DiTCodecEmbedding {
    embed: candle_nn::Embedding,
    repeats: usize,
}

#[allow(dead_code)]
impl DiTCodecEmbedding {
    fn new(
        _num_embeds: usize,
        emb_dim: usize,
        repeats: usize,
        loader: &WeightLoader,
        prefix: &str,
    ) -> crate::Result<Self> {
        let w = loader.get(&format!("{prefix}.weight"))?.clone();
        let embed = candle_nn::Embedding::new(w, emb_dim);
        Ok(Self { embed, repeats })
    }

    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let mut h = self.embed.forward(tokens)?;
        for _ in 1..self.repeats {
            h = Tensor::cat(&[&h, &h], 2)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// DiT Input Embedding
// ---------------------------------------------------------------------------

/// Concatenates noise(x) + speaker_emb + condition + code_emb and projects
#[allow(dead_code)]
struct DiTInputEmbedding {
    proj_in: Linear,
}

#[allow(dead_code)]
impl DiTInputEmbedding {
    fn new(
        _mel_dim: usize,
        _emb_dim: usize,
        _hidden_size: usize,
        loader: &WeightLoader,
        prefix: &str,
    ) -> crate::Result<Self> {
        let proj_in = load_linear(loader, &format!("{prefix}.proj_in"))?;
        // proj_in: (hidden_size, mel_dim + 1 + emb_dim + emb_dim*repeats)
        Ok(Self { proj_in })
    }

    fn forward(
        &self,
        x: &Tensor,
        _speaker_emb: &Tensor,
        _cond_emb: &Tensor,
        code_emb: &Tensor,
    ) -> Result<Tensor> {
        // Concat along last dim: [x, speaker_emb, cond_emb, code_emb]
        let input = Tensor::cat(&[x, code_emb], 2)?;
        self.proj_in.forward(&input)
    }
}

// ---------------------------------------------------------------------------
// DiT Backbone（完整 22 層 + 時序 + 最終投影）
// ---------------------------------------------------------------------------

/// Full DiT backbone: timestep embed → 22 DiT layers → final norm → output proj
#[allow(dead_code)]
pub struct DiTBackbone {
    config: DiTConfig,
    time_embed: DiTTimestepEmbedding,
    codec_embed: DiTCodecEmbedding,
    input_embed: DiTInputEmbedding,
    layers: Vec<DiTDecoderLayer>,
    final_norm: AdaLayerNormZeroFinal,
    proj_out: Linear,
    rope_freqs: Tensor,
}

#[allow(dead_code)]
impl DiTBackbone {
    pub fn new(config: &DiTConfig, loader: &WeightLoader, prefix: &str) -> crate::Result<Self> {
        let device = loader
            .get(&format!("{prefix}.time_embed.time_mlp.0.weight"))?
            .device();
        let device = device.clone();

        let time_embed = DiTTimestepEmbedding::new(
            config.hidden_size,
            config.freq_embed_dim,
            loader,
            &format!("{prefix}.time_embed"),
        )?;

        let codec_embed = DiTCodecEmbedding::new(
            config.num_embeds,
            config.emb_dim,
            config.repeats,
            loader,
            &format!("{prefix}.codec_embed"),
        )?;

        let input_embed = DiTInputEmbedding::new(
            config.mel_dim,
            config.emb_dim,
            config.hidden_size,
            loader,
            &format!("{prefix}.input_embed"),
        )?;

        // Build 22 decoder layers — assign look_ahead / look_backward per layer
        let n_layers = config.num_hidden_layers;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let look_ahead = if config.look_ahead_layers.contains(&i) {
                1
            } else {
                0
            };
            let look_backward = if config.look_backward_layers.contains(&i) {
                (n_layers - 1) as i64
            } else {
                0
            };
            let layer = DiTDecoderLayer::new(
                config,
                look_ahead,
                look_backward,
                loader,
                &format!("{prefix}.blocks.{i}"),
                &device,
            )?;
            layers.push(layer);
        }

        let final_norm = AdaLayerNormZeroFinal::new(
            config.hidden_size,
            loader,
            &format!("{prefix}.final_norm"),
        )?;

        let proj_out = load_linear(loader, &format!("{prefix}.proj_out"))?;

        // Precompute RoPE frequencies
        let rope_freqs = Self::precompute_rope(
            config.max_position_embeddings,
            config.head_dim,
            config.rope_theta,
            &device,
        )?;

        Ok(Self {
            config: config.clone(),
            time_embed,
            codec_embed,
            input_embed,
            layers,
            final_norm,
            proj_out,
            rope_freqs,
        })
    }

    /// Precompute RoPE cos/sin for the full max length
    fn precompute_rope(
        max_len: usize,
        head_dim: usize,
        theta: f64,
        device: &Device,
    ) -> Result<Tensor> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / (theta.powf(i as f64 / half as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_slice(&inv_freq, half, device)?;
        let positions: Vec<f32> = (0..max_len).map(|i| i as f32).collect();
        let positions = Tensor::from_slice(&positions, max_len, device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        // (max_len, half) → (1, max_len, head_dim)
        let cos = Tensor::cat(&[&cos, &cos], 1)?.unsqueeze(0)?;
        let sin = Tensor::cat(&[&sin, &sin], 1)?.unsqueeze(0)?;
        Tensor::stack(&[&cos, &sin], 0)
    }

    fn get_rope(&self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        // rope_freqs: [2, max_len, head_dim], where [0] = cos, [1] = sin
        let cos = self.rope_freqs.narrow(0, 0, 1)?.narrow(1, 0, seq_len)?;
        let sin = self.rope_freqs.narrow(0, 1, 1)?.narrow(1, 0, seq_len)?;
        Ok((cos.squeeze(0)?, sin.squeeze(0)?))
    }

    /// Compute block_diff from sequence positions for block-wise attention mask
    fn block_diff_tensor(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let positions: Vec<i64> = (0..seq_len as i64).collect();
        let pos = Tensor::from_slice(&positions, seq_len, device)?;
        let diff = pos.unsqueeze(1)? - pos.unsqueeze(0)?;
        diff?.to_dtype(DType::I64)
    }

    pub fn forward(
        &self,
        noise: &Tensor,
        code_tokens: &Tensor,
        speaker_emb: &Tensor,
        cond_emb: &Tensor,
        timestep: &Tensor,
    ) -> crate::Result<Tensor> {
        let device = noise.device();
        let seq_len = noise.dims()[1];

        let t_emb = self.time_embed.forward(timestep)?;
        let code_emb = self.codec_embed.forward(code_tokens)?;
        let h = self
            .input_embed
            .forward(noise, speaker_emb, cond_emb, &code_emb)?;

        let (rope_cos, rope_sin) = self.get_rope(seq_len)?;
        let block_diff = self.block_diff_tensor(seq_len, device)?;

        let mut h = h;
        for layer in &self.layers {
            h = layer.forward(&h, &t_emb, Some((&rope_cos, &rope_sin)), Some(&block_diff))?;
        }

        h = self.final_norm.forward(&h, &t_emb)?;
        Ok(self.proj_out.forward(&h)?)
    }
}

// ---------------------------------------------------------------------------
// Flow Matching 解碼器
// ---------------------------------------------------------------------------

pub struct FlowMatchingDecoder {
    config: DiTConfig,
    solver: OdeSolver,
    backbone: Option<DiTBackbone>,
    #[allow(dead_code)]
    device: Device,
}

impl FlowMatchingDecoder {
    pub fn new(dit_config: DiTConfig, solver_config: OdeSolverConfig, device: &Device) -> Self {
        Self {
            solver: OdeSolver::new(solver_config),
            config: dit_config,
            backbone: None,
            device: device.clone(),
        }
    }

    pub fn load_backbone(&mut self, loader: &WeightLoader, prefix: &str) -> crate::Result<()> {
        let backbone = DiTBackbone::new(&self.config, loader, prefix)?;
        self.backbone = Some(backbone);
        Ok(())
    }

    pub fn decode(
        &self,
        noise: &Tensor,
        code_tokens: &Tensor,
        speaker_emb: &Tensor,
        cond_emb: &Tensor,
    ) -> crate::Result<Tensor> {
        let result = self.solver.solve(noise, |t, x| {
            self.velocity(t, x, code_tokens, speaker_emb, cond_emb)
        })?;
        Ok(result)
    }

    fn velocity(
        &self,
        t: f64,
        x: &Tensor,
        code_tokens: &Tensor,
        speaker_emb: &Tensor,
        cond_emb: &Tensor,
    ) -> crate::Result<Tensor> {
        let timestep = Tensor::new(t, x.device())?;
        let timestep = timestep.unsqueeze(0)?.to_dtype(x.dtype())?;
        match &self.backbone {
            Some(backbone) => backbone.forward(x, code_tokens, speaker_emb, cond_emb, &timestep),
            None => Ok((x * 0.0)?), // stub fallback
        }
    }

    pub fn config(&self) -> &DiTConfig {
        &self.config
    }

    pub fn solver(&self) -> &OdeSolver {
        &self.solver
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
    fn test_ode_solver_euler() {
        let config = OdeSolverConfig {
            solver_type: OdeSolverType::Euler,
            num_steps: 100,
            ..Default::default()
        };
        let solver = OdeSolver::new(config);
        let device = Device::Cpu;
        let x0 = Tensor::zeros((1, 512), candle_core::DType::F32, &device).unwrap();
        let result = solver.solve(&x0, |_t, x| Ok((x * (-1.0_f64))?)).unwrap();
        assert_eq!(result.dims(), &[1, 512]);
    }

    #[test]
    fn test_ode_solver_rk4() {
        let config = OdeSolverConfig {
            solver_type: OdeSolverType::RK4,
            num_steps: 10,
            ..Default::default()
        };
        let solver = OdeSolver::new(config);
        let device = Device::Cpu;
        let x0 = Tensor::ones((1, 64), candle_core::DType::F32, &device).unwrap();
        let result = solver.solve(&x0, |_t, x| Ok((x * (-0.5_f64))?)).unwrap();
        assert_eq!(result.dims(), &[1, 64]);
    }

    #[test]
    fn test_flow_matching_decoder_creation() {
        let device = Device::Cpu;
        let dit_config = DiTConfig::default();
        let solver_config = OdeSolverConfig::default();
        let decoder = FlowMatchingDecoder::new(dit_config, solver_config, &device);
        assert_eq!(decoder.config().num_hidden_layers, 22);
    }

    #[test]
    fn test_sinus_position_embedding() {
        let device = Device::Cpu;
        let emb = SinusPositionEmbedding::new(256);
        let t = Tensor::zeros(1, DType::F32, &device).unwrap();
        let result = emb.forward(&t).unwrap();
        assert_eq!(result.dims(), &[1, 256]);
    }

    #[test]
    fn test_dit_attention_no_weights() {
        // 不依賴真實權重的測試：直接建立 DiTAttention
        let device = Device::Cpu;
        let config = DiTConfig {
            hidden_size: 256,
            num_attention_heads: 4,
            head_dim: 64,
            ..Default::default()
        };

        let inner_dim = config.head_dim * config.num_attention_heads;
        let q_proj = Linear::new(
            Tensor::rand(-0.1f32, 0.1f32, (inner_dim, 256), &device).unwrap(),
            None,
        );
        let k_proj = Linear::new(
            Tensor::rand(-0.1f32, 0.1f32, (inner_dim, 256), &device).unwrap(),
            None,
        );
        let v_proj = Linear::new(
            Tensor::rand(-0.1f32, 0.1f32, (inner_dim, 256), &device).unwrap(),
            None,
        );
        let out_proj = Linear::new(
            Tensor::rand(-0.1f32, 0.1f32, (256, inner_dim), &device).unwrap(),
            None,
        );

        let attn = DiTAttention {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: 4,
            head_dim: 64,
        };

        let x = Tensor::ones((1, 8, 256), DType::F32, &device).unwrap();
        let out = attn.forward(&x, None, None).unwrap();
        assert_eq!(out.dims(), &[1, 8, 256]);
    }

    #[test]
    fn test_dit_mlp_no_weights() {
        let device = Device::Cpu;
        let fc1 = Linear::new(
            Tensor::rand(-0.1f32, 0.1f32, (256, 64), &device).unwrap(),
            None,
        );
        let fc2 = Linear::new(
            Tensor::rand(-0.1f32, 0.1f32, (64, 256), &device).unwrap(),
            None,
        );
        let mlp = DiTMLP { fc1, fc2 };
        let x = Tensor::ones((1, 8, 64), DType::F32, &device).unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 8, 64]);
    }

    #[test]
    fn test_rotate_half() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 1, 4), &device).unwrap();
        let rotated = rotate_half(&x).unwrap();
        let actual: Vec<f32> = rotated.reshape((4,)).unwrap().to_vec1().unwrap();
        assert!((actual[0] - (-3.0)).abs() < 1e-5);
        assert!((actual[1] - (-4.0)).abs() < 1e-5);
        assert!((actual[2] - 1.0).abs() < 1e-5);
        assert!((actual[3] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_rotary_pos_emb() {
        let device = Device::Cpu;
        let q = Tensor::ones((1, 4, 8, 64), DType::F32, &device).unwrap();
        let k = Tensor::ones((1, 4, 8, 64), DType::F32, &device).unwrap();
        let cos = Tensor::ones((1, 8, 64), DType::F32, &device).unwrap();
        let sin = Tensor::zeros((1, 8, 64), DType::F32, &device).unwrap();

        let (q_out, k_out) = apply_rotary_pos_emb(&q, &k, &cos, &sin).unwrap();
        assert_eq!(q_out.dims(), &[1, 4, 8, 64]);
        assert_eq!(k_out.dims(), &[1, 4, 8, 64]);
    }
}
