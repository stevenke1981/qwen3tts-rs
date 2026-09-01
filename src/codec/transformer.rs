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

    /// 單幀流式注意力（O(window) per step）
    ///
    /// 與 `forward()` 不同，此方法只處理一幀輸入，
    /// 使用 `DeviceKvCache` 設備端快取管理歷史 K,V 狀態。
    ///
    /// # 參數
    /// - `x`: 當前幀，形狀 `(1, 1, hidden_dim)`
    /// - `position`: 當前位置索引（用於 RoPE 查表）
    /// - `kv_cache`: 此層的設備端 KV 快取
    pub fn step(&self, x: &Tensor, position: usize, kv_cache: &mut DeviceKvCache) -> crate::Result<Tensor> {
        // QKV projections
        let q = self.q_proj.forward(x)?; // (1, 1, num_heads * head_dim)
        let k = self.k_proj.forward(x)?; // (1, 1, num_kv_heads * head_dim)
        let v = self.v_proj.forward(x)?; // (1, 1, num_kv_heads * head_dim)

        let b = 1_usize;
        let seq_len = 1_usize;

        // Reshape to (batch, heads, seq, head_dim)
        let q_r = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let k_r = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v_r = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        // RoPE (apply to current position)
        let cos = self.cos.narrow(0, position, 1)?;
        let sin = self.sin.narrow(0, position, 1)?;
        let q_rot = apply_rope_half(&q_r, &cos, &sin)?;
        let k_rot = apply_rope_half(&k_r, &cos, &sin)?;

        // Update on-device KV cache (returns full window: (1, num_kv_heads, window_len, head_dim))
        let (k_cache, v_cache) = kv_cache.step(&k_rot, &v_r)?;

        // GQA repeat
        let n_repeat = self.num_heads / self.num_kv_heads;
        let (k_e, v_e) = if n_repeat > 1 {
            (
                repeat_kv(&k_cache, n_repeat)?,
                repeat_kv(&v_cache, n_repeat)?,
            )
        } else {
            (k_cache, v_cache)
        };
        // k_e: (1, num_heads, window, head_dim)

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt().recip();
        let attn = (q_rot.matmul(&k_e.transpose(2, 3)?)? * scale)?;
        // No mask needed: sliding window is enforced by cache truncation, causal is automatic
        let attn = candle_nn::ops::softmax(&attn, 3)?;
        let attn = attn.matmul(&v_e)?; // (1, num_heads, 1, head_dim)

        let attn = attn
            .permute((0, 2, 1, 3))?
            .reshape((1, 1, self.num_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&attn)?)
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

// ---------------------------------------------------------------------------
// 設備端 KV 快取（純 Device 張量流式快取）
// ---------------------------------------------------------------------------

/// 純設備端 KV 快取（Device-Resident KV Cache）
///
/// 維護每層滑動窗口（72）之 K, V 張量，全程在 Device 上執行，
/// 零 Host-Device 跨設備複製與記憶體分配。
#[derive(Debug, Clone)]
pub struct DeviceKvCache {
    k_cache: Option<Tensor>, // (1, num_kv_heads, seq_len, head_dim)
    v_cache: Option<Tensor>, // (1, num_kv_heads, seq_len, head_dim)
    sliding_window: usize,
}

impl DeviceKvCache {
    /// 建立新設備端 KV 快取
    pub fn new(sliding_window: usize) -> Self {
        Self {
            k_cache: None,
            v_cache: None,
            sliding_window,
        }
    }

    /// 步進寫入當前幀 K,V 並取得滑動窗口內的所有歷史張量
    pub fn step(&mut self, k: &Tensor, v: &Tensor) -> crate::Result<(Tensor, Tensor)> {
        let (new_k, new_v) = if let (Some(past_k), Some(past_v)) = (&self.k_cache, &self.v_cache) {
            let k_cat = Tensor::cat(&[past_k, k], 2)?;
            let v_cat = Tensor::cat(&[past_v, v], 2)?;
            let cur_len = k_cat.dim(2)?;
            if cur_len > self.sliding_window {
                let offset = cur_len - self.sliding_window;
                (
                    k_cat.narrow(2, offset, self.sliding_window)?.contiguous()?,
                    v_cat.narrow(2, offset, self.sliding_window)?.contiguous()?,
                )
            } else {
                (k_cat, v_cat)
            }
        } else {
            (k.clone(), v.clone())
        };

        self.k_cache = Some(new_k.clone());
        self.v_cache = Some(new_v.clone());
        Ok((new_k, new_v))
    }

    /// 重置快取（零分配）
    pub fn reset(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
    }

    /// 當前快取的長度
    pub fn len(&self) -> usize {
        self.k_cache.as_ref().and_then(|k| k.dim(2).ok()).unwrap_or(0)
    }

    /// 快取是否為空
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 容量（滑動窗口大小）
    pub fn capacity(&self) -> usize {
        self.sliding_window
    }
}

/// 相容個別模組引用別名
pub type KvRing = DeviceKvCache;

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

    /// 單幀流式 Transformer 層
    ///
    /// 輸入形狀 `(1, 1, hidden_dim)`，回傳相同形狀。
    /// 使用 `kv_cache` 管理注意力 K/V 快取。
    pub fn step(&self, x: &Tensor, position: usize, kv_cache: &mut DeviceKvCache) -> crate::Result<Tensor> {
        let r = x;
        let xn = self.input_norm.forward(x)?;
        let attn_out = self.attn.step(&xn, position, kv_cache)?;
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

    /// 建立每層的設備端 KV 快取
    pub fn new_kv_caches(cfg: &PreTransformerConfig) -> Vec<DeviceKvCache> {
        (0..cfg.num_layers)
            .map(|_| DeviceKvCache::new(cfg.sliding_window))
            .collect()
    }

    /// 建立每層的 KV 快取（相容舊名）
    pub fn new_kv_rings(cfg: &PreTransformerConfig) -> Vec<DeviceKvCache> {
        Self::new_kv_caches(cfg)
    }

    /// 單幀流式推理解碼
    ///
    /// 處理一幀輸入，使用 `kv_caches` 管理跨層 K/V 快取。
    /// 輸入形狀 `(1, input_dim, 1)`，回傳 `(1, hidden_dim, 1)`。
    ///
    /// # 參數
    /// - `x`: 輸入幀，形狀 `(1, input_dim, 1)`
    /// - `kv_caches`: 由 `new_kv_caches()` 建立，長度為 `num_layers`
    /// - `position`: 當前位置索引（用於 RoPE）
    pub fn step(
        &self,
        x: &Tensor,
        kv_caches: &mut [DeviceKvCache],
        position: usize,
    ) -> crate::Result<Tensor> {
        let mut h = x.transpose(1, 2)?.contiguous()?; // (1, 1, input_dim)
        h = self.input_proj.forward(&h)?; // (1, 1, hidden_dim)
        for (layer, kv_cache) in self.layers.iter().zip(kv_caches.iter_mut()) {
            h = layer.step(&h, position, kv_cache)?;
        }
        h = self.norm.forward(&h)?;
        h = self.output_proj.forward(&h)?; // (1, 1, output_dim)
        Ok(h.transpose(1, 2)?.contiguous()?) // (1, output_dim, 1)
    }

    /// 重置所有設備端 KV 快取（零分配）
    pub fn reset_kv_caches(kv_caches: &mut [DeviceKvCache]) {
        for cache in kv_caches.iter_mut() {
            cache.reset();
        }
    }

    /// 重置所有 KV 快取（相容舊名）
    pub fn reset_kv_rings(kv_rings: &mut [DeviceKvCache]) {
        Self::reset_kv_caches(kv_rings);
    }
}

// ---------------------------------------------------------------------------
// 單元測試
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::*;

    fn test_device() -> Device {
        Device::Cpu
    }

    // ----------------------------------------------------------------
    // DeviceKvCache 單元測試
    // ----------------------------------------------------------------

    #[test]
    fn test_device_kv_cache_basic_step() {
        let device = test_device();
        let mut cache = DeviceKvCache::new(8); // sliding_window=8
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 8);

        // Step one position: K and V both shape (1, 2, 1, 4)
        let k = Tensor::ones((1, 2, 1, 4), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 1, 4), DType::F32, &device).unwrap();
        let (k_cached, v_cached) = cache.step(&k, &v).unwrap();

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
        assert_eq!(k_cached.shape().dims(), &[1, 2, 1, 4]);
        assert_eq!(v_cached.shape().dims(), &[1, 2, 1, 4]);

        let k_flat: Vec<f32> = k_cached.flatten_all().unwrap().to_vec1().unwrap();
        assert!(k_flat.iter().all(|&x| (x - 1.0).abs() < 1e-6));
        let v_flat: Vec<f32> = v_cached.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v_flat.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn test_device_kv_cache_wraparound() {
        let device = test_device();
        let mut cache = DeviceKvCache::new(4); // sliding_window=4
        assert_eq!(cache.capacity(), 4);

        // Write 6 positions (window=4, so last 4 survive)
        let mut last_k = None;
        let mut last_v = None;
        for i in 0..6 {
            let k = Tensor::from_slice(&[i as f32 * 10.0; 2], (1, 1, 1, 2), &device).unwrap();
            let v = Tensor::from_slice(&[i as f32; 2], (1, 1, 1, 2), &device).unwrap();
            let (k_out, v_out) = cache.step(&k, &v).unwrap();
            last_k = Some(k_out);
            last_v = Some(v_out);
        }
        assert_eq!(cache.len(), 4);

        let k_gar = last_k.unwrap();
        let v_gar = last_v.unwrap();
        assert_eq!(k_gar.shape().dims(), &[1, 1, 4, 2]);
        let k_flat: Vec<f32> = k_gar.flatten_all().unwrap().to_vec1().unwrap();
        // Expected positions 2, 3, 4, 5: [20, 20, 30, 30, 40, 40, 50, 50]
        assert_eq!(k_flat[0..2], [20.0, 20.0], "oldest should be pos 2");
        assert_eq!(k_flat[2..4], [30.0, 30.0], "pos 3");
        assert_eq!(k_flat[4..6], [40.0, 40.0], "pos 4");
        assert_eq!(k_flat[6..8], [50.0, 50.0], "newest should be pos 5");

        let v_flat: Vec<f32> = v_gar.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v_flat[0..2], [2.0, 2.0]);
        assert_eq!(v_flat[6..8], [5.0, 5.0]);
    }

    #[test]
    fn test_device_kv_cache_reset() {
        let device = test_device();
        let mut cache = DeviceKvCache::new(4);
        let k = Tensor::ones((1, 2, 1, 3), DType::F32, &device).unwrap();
        let v = Tensor::ones((1, 2, 1, 3), DType::F32, &device).unwrap();
        let _ = cache.step(&k, &v).unwrap();
        assert_eq!(cache.len(), 1);
        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // ----------------------------------------------------------------
    // PreTransformer step API 整合測試（合成權重）
    // ----------------------------------------------------------------

    /// 建立小型合成 PreTransformer 權重
    fn create_small_pretransformer(device: &Device) -> (PreTransformer, PreTransformerConfig) {
        let cfg = PreTransformerConfig {
            input_dim: 8,
            hidden_dim: 16,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            sliding_window: 8,
            ffn_hidden_mult: 2,
            max_seq_len: 64,
            rope_theta: 10000.0,
            eps: 1e-6,
        };
        let head_dim = cfg.hidden_dim / cfg.num_heads; // 4
        let _kv_head_dim = cfg.hidden_dim / cfg.num_heads; // 4 (same for GQA)

        // Build weights as HashMap
        use std::collections::HashMap;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        // input_proj: (hidden_dim, input_dim)
        tensors.insert(
            "pre_transformer.input_proj.weight".to_string(),
            Tensor::rand(
                -0.1f32 as f64,
                0.1f32 as f64,
                (cfg.hidden_dim, cfg.input_dim),
                device,
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap(),
        );
        // output_proj: (input_dim, hidden_dim)
        tensors.insert(
            "pre_transformer.output_proj.weight".to_string(),
            Tensor::rand(
                -0.1f32 as f64,
                0.1f32 as f64,
                (cfg.input_dim, cfg.hidden_dim),
                device,
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap(),
        );
        // norm
        tensors.insert(
            "pre_transformer.norm.weight".to_string(),
            Tensor::ones((cfg.hidden_dim,), DType::F32, device).unwrap(),
        );

        let mk_rand = |shape| {
            Tensor::rand(-0.1f32 as f64, 0.1f32 as f64, shape, device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
        };

        for i in 0..cfg.num_layers {
            let prefix = format!("pre_transformer.layers.{i}");

            // Input layernorm
            tensors.insert(
                format!("{prefix}.input_layernorm.weight"),
                Tensor::ones((cfg.hidden_dim,), DType::F32, device).unwrap(),
            );
            // Post attention layernorm
            tensors.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                Tensor::ones((cfg.hidden_dim,), DType::F32, device).unwrap(),
            );

            // QKV projections
            let qw = mk_rand((cfg.hidden_dim, cfg.hidden_dim));
            let kw = mk_rand((cfg.num_kv_heads * head_dim, cfg.hidden_dim));
            let vw = mk_rand((cfg.num_kv_heads * head_dim, cfg.hidden_dim));
            let ow = mk_rand((cfg.hidden_dim, cfg.hidden_dim));
            tensors.insert(format!("{prefix}.self_attn.q_proj.weight"), qw);
            tensors.insert(format!("{prefix}.self_attn.k_proj.weight"), kw);
            tensors.insert(format!("{prefix}.self_attn.v_proj.weight"), vw);
            tensors.insert(format!("{prefix}.self_attn.o_proj.weight"), ow);

            // Layer scales
            tensors.insert(
                format!("{prefix}.self_attn_layer_scale.scale"),
                Tensor::ones((cfg.hidden_dim,), DType::F32, device).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.mlp_layer_scale.scale"),
                Tensor::ones((cfg.hidden_dim,), DType::F32, device).unwrap(),
            );

            // MLP (FFN)
            let ffn_hidden = cfg.hidden_dim * cfg.ffn_hidden_mult;
            tensors.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                mk_rand((ffn_hidden, cfg.hidden_dim)),
            );
            tensors.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                mk_rand((ffn_hidden, cfg.hidden_dim)),
            );
            tensors.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                mk_rand((cfg.hidden_dim, ffn_hidden)),
            );
        }

        let loader = crate::weights::WeightLoader::from_tensors(tensors, device);
        let pt = PreTransformer::from_loader(&loader, &cfg, device).unwrap();
        (pt, cfg)
    }

    #[test]
    fn test_pretransformer_step_matches_forward() {
        let device = test_device();
        let (pt, cfg) = create_small_pretransformer(&device);
        let num_frames = 5;

        // Create synthetic input: (1, input_dim, num_frames)
        let input_data: Vec<f32> = (0..cfg.input_dim * num_frames)
            .map(|i| (i % 7) as f32 * 0.1)
            .collect();
        let input =
            Tensor::from_slice(&input_data, (1, cfg.input_dim, num_frames), &device).unwrap();

        // === Batch forward ===
        let batch_out = pt.forward(&input).unwrap();
        let batch_shape = batch_out.shape().dims().to_vec();
        assert_eq!(batch_shape, [1, cfg.input_dim, num_frames]);

        // === Streaming step ===
        let mut kv_rings = PreTransformer::new_kv_rings(&cfg);
        assert_eq!(kv_rings.len(), cfg.num_layers);
        let mut step_outputs: Vec<Vec<f32>> = Vec::new();

        for pos in 0..num_frames {
            // Extract single frame: (1, input_dim, 1)
            let frame = input.narrow(2, pos, 1).unwrap();
            let out = pt.step(&frame, &mut kv_rings, pos).unwrap();
            assert_eq!(out.shape().dims(), &[1, cfg.input_dim, 1]);
            step_outputs.push(out.flatten_all().unwrap().to_vec1().unwrap());
        }

        // Compare batch vs step at each position
        for pos in 0..num_frames {
            let batch_frame = batch_out.narrow(2, pos, 1).unwrap();
            let batch_flat: Vec<f32> = batch_frame.flatten_all().unwrap().to_vec1().unwrap();
            let step_flat = &step_outputs[pos];

            // Compute cosine similarity
            let dot: f32 = batch_flat
                .iter()
                .zip(step_flat.iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm_b: f32 = batch_flat.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_s: f32 = step_flat.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = dot / (norm_b * norm_s + 1e-10);

            assert!(
                cos > 0.999,
                "Position {pos}: cosine sim = {cos:.8} < 0.999 (batch_len={}, step_len={})",
                batch_flat.len(),
                step_flat.len(),
            );

            let mse: f32 = batch_flat
                .iter()
                .zip(step_flat.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                / batch_flat.len() as f32;
            assert!(mse < 1e-4, "Position {pos}: MSE = {mse:.8} >= 1e-4");
        }
    }

    #[test]
    fn test_pretransformer_step_isolated_positions() {
        // Verify that step() with reset between frames gives same result
        // as forward() on each individual frame (no cross-frame dependency)
        let device = test_device();
        let (pt, cfg) = create_small_pretransformer(&device);

        for pos in 0..3 {
            let frame_data: Vec<f32> = (0..cfg.input_dim)
                .map(|i| ((pos * 7 + i) % 13) as f32 * 0.05)
                .collect();
            let input = Tensor::from_slice(&frame_data, (1, cfg.input_dim, 1), &device).unwrap();

            // Batch forward on single frame
            let batch_out = pt.forward(&input).unwrap();
            let batch_flat: Vec<f32> = batch_out.flatten_all().unwrap().to_vec1().unwrap();

            // Step with fresh KV rings (position=0 since it's the first/only frame)
            let mut kv_rings = PreTransformer::new_kv_rings(&cfg);
            let step_out = pt.step(&input, &mut kv_rings, 0).unwrap();
            let step_flat: Vec<f32> = step_out.flatten_all().unwrap().to_vec1().unwrap();

            let dot: f32 = batch_flat
                .iter()
                .zip(step_flat.iter())
                .map(|(a, b)| a * b)
                .sum();
            let nb = batch_flat.iter().map(|x| x * x).sum::<f32>().sqrt();
            let ns = step_flat.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = dot / (nb * ns + 1e-10);
            assert!(cos > 0.999, "Isolated pos {pos}: cosine = {cos:.8}");
        }
    }

    #[test]
    fn test_pretransformer_reset_kv_rings() {
        let device = test_device();
        let (pt, cfg) = create_small_pretransformer(&device);
        let mut kv_rings = PreTransformer::new_kv_rings(&cfg);

        // Use the rings
        let frame = Tensor::zeros((1, cfg.input_dim, 1), DType::F32, &device).unwrap();
        let _ = pt.step(&frame, &mut kv_rings, 0).unwrap();
        assert!(kv_rings.iter().any(|r| r.len() > 0));

        // Reset
        PreTransformer::reset_kv_rings(&mut kv_rings);
        assert!(kv_rings.iter().all(|r| r.is_empty()));
    }

    #[test]
    fn test_kv_cache_debug() {
        let cache = DeviceKvCache::new(4);
        let debug_str = format!("{cache:?}");
        assert!(debug_str.contains("DeviceKvCache"));
        assert!(debug_str.contains("sliding_window: 4"));
    }
}
