use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};

use crate::weights::WeightLoader;

pub struct PreTransformerConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub sliding_window: usize,
    pub ffn_hidden_mult: usize,
    pub max_seq_len: usize,
    pub rope_theta: f64,
    pub eps: f64,
}

impl Default for PreTransformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 1024,
            hidden_dim: 512,
            num_heads: 16,
            num_kv_heads: 4,
            num_layers: 8,
            sliding_window: 72,
            ffn_hidden_mult: 4,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            eps: 1e-6,
        }
    }
}

pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let rms = (x_f32.sqr()?.mean_keepdim(2)? + self.eps)?
            .sqrt()?
            .recip()?;
        let x_norm = x_f32.broadcast_mul(&rms)?;
        let w = self
            .weight
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let x_norm = x_norm.broadcast_mul(&w)?;
        x_norm.to_dtype(x.dtype())
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    /// 預先快取的 sliding window mask，形狀 (1, 1, max_seq_len, max_seq_len)
    sliding_mask: Tensor,
}

/// 在建構時預先建立 sliding window attention mask
///
/// 回傳形狀 (1, 1, max_seq_len, max_seq_len) 的張量，
/// 可見位置為 0.0，遮罩位置為 -1e9。
fn build_sliding_mask(max_seq_len: usize, window: usize, device: &Device) -> Result<Tensor> {
    let n = max_seq_len;
    let mut mask = Vec::with_capacity(n * n);
    for q in 0..n {
        let ws = if window == 0 {
            0_usize
        } else {
            (q + 1).saturating_sub(window)
        };
        for k in 0..n {
            let visible = k <= q && k >= ws;
            mask.push(if visible { 0.0_f32 } else { -1.0e9_f32 });
        }
    }
    Tensor::from_slice(&mask, (1, 1, n, n), device)
}

impl Attention {
    fn get_bias(w: &WeightLoader, name: &str) -> crate::Result<Option<Tensor>> {
        Ok(if w.has(name) {
            Some(w.get(name)?.clone())
        } else {
            None
        })
    }

    fn from_loader(
        w: &WeightLoader,
        prefix: &str,
        num_heads: usize,
        num_kv_heads: usize,
        sliding_window: usize,
        max_seq_len: usize,
        rope_theta: f64,
        device: &Device,
    ) -> crate::Result<Self> {
        let qw = w.get(&format!("{prefix}.q_proj.weight"))?.clone();
        let head_dim = qw.dims()[0] / num_heads;
        let qb = Self::get_bias(w, &format!("{prefix}.q_proj.bias"))?;
        let kw = w.get(&format!("{prefix}.k_proj.weight"))?.clone();
        let kb = Self::get_bias(w, &format!("{prefix}.k_proj.bias"))?;
        let vw = w.get(&format!("{prefix}.v_proj.weight"))?.clone();
        let vb = Self::get_bias(w, &format!("{prefix}.v_proj.bias"))?;
        let ow = w.get(&format!("{prefix}.o_proj.weight"))?.clone();
        let ob = Self::get_bias(w, &format!("{prefix}.o_proj.bias"))?;

        let (cos, sin) = precompute_rope(max_seq_len, head_dim, rope_theta, device)?;
        let sliding_mask = build_sliding_mask(max_seq_len, sliding_window, device)?;

        Ok(Self {
            q_proj: Linear::new(qw, qb),
            k_proj: Linear::new(kw, kb),
            v_proj: Linear::new(vw, vb),
            o_proj: Linear::new(ow, ob),
            num_heads,
            num_kv_heads,
            head_dim,
            cos,
            sin,
            sliding_mask,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;

        let q_r = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let k_r = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        let q_rot = apply_rope_half(&q_r, &cos, &sin)?;
        let k_rot = apply_rope_half(&k_r, &cos, &sin)?;

        let n_repeat = self.num_heads / self.num_kv_heads;

        let (k_e, v_e) = if n_repeat > 1 {
            (repeat_kv(&k_rot, n_repeat)?, repeat_kv(&v, n_repeat)?)
        } else {
            (k_rot, v)
        };

        let scale = (self.head_dim as f64).sqrt().recip();
        let k_t = k_e.transpose(2, 3)?.contiguous()?;
        let attn = (q_rot.matmul(&k_t)? * scale)?;
        // 使用預先快取的 sliding mask，僅 narrow 到當前 seq_len，零分配
        let mask = self
            .sliding_mask
            .narrow(2, 0, seq_len)?
            .narrow(3, 0, seq_len)?;
        let attn = attn.broadcast_add(&mask)?;
        let attn = candle_nn::ops::softmax(&attn, 3)?;
        let attn = attn.matmul(&v_e)?;

        let attn =
            attn.permute((0, 2, 1, 3))?
                .reshape((b, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&attn)
    }
}

fn repeat_kv(x: &Tensor, n: usize) -> Result<Tensor> {
    if n == 1 {
        return Ok(x.clone());
    }
    let (b, kv, s, d) = x.dims4()?;
    let x = x.unsqueeze(2)?.expand((b, kv, n, s, d))?;
    x.reshape((b, kv * n, s, d))
}

fn apply_rope_half(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let head_dim = x.dim(3)?;
    let half = head_dim / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    let x1_cos = x1.broadcast_mul(&cos)?;
    let x2_sin = x2.broadcast_mul(&sin)?;
    let first = (&x1_cos - &x2_sin)?;

    let x2_cos = x2.broadcast_mul(&cos)?;
    let x1_sin = x1.broadcast_mul(&sin)?;
    let second = (&x2_cos + &x1_sin)?;

    Tensor::cat(&[&first, &second], 3)
}

fn precompute_rope(
    max_seq_len: usize,
    head_dim: usize,
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| (1.0 / theta.powf(i as f64 / half as f64)) as f32)
        .collect();
    let inv = Tensor::from_slice(&inv, (half,), device)?.unsqueeze(0)?;

    let pos: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let pos = Tensor::from_slice(&pos, (max_seq_len,), device)?.unsqueeze(1)?;

    let f = pos.matmul(&inv)?; // (T, H/2)
    let cos = f.cos()?; // (T, H/2) — Candle rope internally handles half-dim duplication
    let sin = f.sin()?; // (T, H/2)
    Ok((cos, sin))
}

struct Ffn {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Ffn {
    fn from_loader(w: &WeightLoader, prefix: &str) -> crate::Result<Self> {
        let gw = w.get(&format!("{prefix}.gate_proj.weight"))?.clone();
        let gb = Attention::get_bias(w, &format!("{prefix}.gate_proj.bias"))?;
        let uw = w.get(&format!("{prefix}.up_proj.weight"))?.clone();
        let ub = Attention::get_bias(w, &format!("{prefix}.up_proj.bias"))?;
        let dw = w.get(&format!("{prefix}.down_proj.weight"))?.clone();
        let db = Attention::get_bias(w, &format!("{prefix}.down_proj.bias"))?;
        Ok(Self {
            gate: Linear::new(gw, gb),
            up: Linear::new(uw, ub),
            down: Linear::new(dw, db),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = candle_nn::Activation::Silu.forward(&self.gate.forward(x)?)?;
        let u = self.up.forward(x)?;
        self.down.forward(&(g * u)?)
    }
}

pub struct TransformerBlock {
    input_norm: RMSNorm,
    attn: Attention,
    attn_scale: Tensor,
    post_norm: RMSNorm,
    ffn: Ffn,
    mlp_scale: Tensor,
}

impl TransformerBlock {
    pub fn from_loader(
        w: &WeightLoader,
        prefix: &str,
        cfg: &PreTransformerConfig,
        device: &Device,
    ) -> crate::Result<Self> {
        let inw = w.get(&format!("{prefix}.input_layernorm.weight"))?.clone();
        let pnw = w
            .get(&format!("{prefix}.post_attention_layernorm.weight"))?
            .clone();
        let as_ = w
            .get(&format!("{prefix}.self_attn_layer_scale.scale"))?
            .clone();
        let ms_ = w.get(&format!("{prefix}.mlp_layer_scale.scale"))?.clone();
        Ok(Self {
            input_norm: RMSNorm::new(inw, cfg.eps),
            attn: Attention::from_loader(
                w,
                &format!("{prefix}.self_attn"),
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.sliding_window,
                cfg.max_seq_len,
                cfg.rope_theta,
                device,
            )?,
            attn_scale: as_,
            post_norm: RMSNorm::new(pnw, cfg.eps),
            ffn: Ffn::from_loader(w, &format!("{prefix}.mlp"))?,
            mlp_scale: ms_,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let r = x;
        let xn = self.input_norm.forward(x)?;
        let attn_out = self.attn.forward(&xn)?;
        let ascale = self.attn_scale.unsqueeze(0)?.unsqueeze(0)?;
        let x = r.broadcast_add(&ascale.broadcast_mul(&attn_out)?)?;
        let r = &x;
        let xn = self.post_norm.forward(&x)?;
        let ffn_out = self.ffn.forward(&xn)?;
        let mscale = self.mlp_scale.unsqueeze(0)?.unsqueeze(0)?;
        let x = r.broadcast_add(&mscale.broadcast_mul(&ffn_out)?)?;
        Ok(x)
    }
}

pub struct PreTransformer {
    input_proj: Linear,
    layers: Vec<TransformerBlock>,
    norm: RMSNorm,
    output_proj: Linear,
}

impl PreTransformer {
    pub fn from_loader(
        w: &WeightLoader,
        cfg: &PreTransformerConfig,
        device: &Device,
    ) -> crate::Result<Self> {
        let iw = w.get("pre_transformer.input_proj.weight")?.clone();
        let ib = Attention::get_bias(w, "pre_transformer.input_proj.bias")?;
        let ow = w.get("pre_transformer.output_proj.weight")?.clone();
        let ob = Attention::get_bias(w, "pre_transformer.output_proj.bias")?;
        let nw = w.get("pre_transformer.norm.weight")?.clone();

        let mut layers = Vec::new();
        for i in 0..cfg.num_layers {
            layers.push(TransformerBlock::from_loader(
                w,
                &format!("pre_transformer.layers.{i}"),
                cfg,
                device,
            )?);
        }

        Ok(Self {
            input_proj: Linear::new(iw, ib),
            layers,
            norm: RMSNorm::new(nw, cfg.eps),
            output_proj: Linear::new(ow, ob),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?.contiguous()?;
        let mut h = self.input_proj.forward(&x)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        let h = self.norm.forward(&h)?;
        let h = self.output_proj.forward(&h)?;
        h.transpose(1, 2)?.contiguous()
    }
}
