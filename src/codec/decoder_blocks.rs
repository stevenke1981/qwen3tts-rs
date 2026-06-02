use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, Module};

use crate::weights::WeightLoader;

pub fn snake_beta(x: &Tensor, alpha: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let alpha = alpha.reshape((1, alpha.elem_count(), 1))?;
    let beta = beta.reshape((1, beta.elem_count(), 1))?;
    let beta_x = x.broadcast_mul(&beta)?;
    let sin2 = beta_x.sin()?.sqr()?;
    let one = Tensor::new(1.0f32, x.device())?;
    let sin2_scaled = sin2.broadcast_mul(&one.broadcast_div(&beta)?)?;
    Ok(x.broadcast_add(&alpha.broadcast_mul(&sin2_scaled)?)?)
}

pub struct ConvNeXtBlock {
    pub gamma: Tensor,
    pub dwconv: Conv1d,
    pub norm: candle_nn::LayerNorm,
    pub pwconv1: Linear,
    pub pwconv2: Linear,
}

impl ConvNeXtBlock {
    pub fn from_loader(w: &WeightLoader, prefix: &str) -> crate::Result<Self> {
        let gamma = w.get(&format!("{prefix}.gamma"))?.clone();
        let dw_w = w.get(&format!("{prefix}.dwconv.conv.weight"))?.clone();
        let dw_b = w.get(&format!("{prefix}.dwconv.conv.bias"))?.clone();
        let ch = dw_w.dims()[0];
        let cfg = Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: ch,
            cudnn_fwd_algo: None,
        };
        let nw = w.get(&format!("{prefix}.norm.weight"))?.clone();
        let nb = w.get(&format!("{prefix}.norm.bias"))?.clone();
        let p1w = w.get(&format!("{prefix}.pwconv1.weight"))?.clone();
        let p1b = w.get(&format!("{prefix}.pwconv1.bias"))?.clone();
        let p2w = w.get(&format!("{prefix}.pwconv2.weight"))?.clone();
        let p2b = w.get(&format!("{prefix}.pwconv2.bias"))?.clone();
        Ok(Self {
            gamma,
            dwconv: Conv1d::new(dw_w, Some(dw_b), cfg),
            norm: candle_nn::LayerNorm::new(nw, nb, 1e-6),
            pwconv1: Linear::new(p1w, Some(p1b)),
            pwconv2: Linear::new(p2w, Some(p2b)),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let r = x;
        let h = self.dwconv.forward(x)?;
        let h = h.transpose(1, 2)?;
        let h = self.norm.forward(&h)?;
        let h = candle_nn::Activation::Gelu.forward(&self.pwconv1.forward(&h)?)?;
        let h = self.pwconv2.forward(&h)?;
        let h = h.transpose(1, 2)?;
        let g = self.gamma.reshape((1, self.gamma.elem_count(), 1))?;
        Ok(g.broadcast_mul(&h)?.broadcast_add(r)?)
    }
}

pub struct UpsampleBlock {
    pub ct: ConvTranspose1d,
    pub cn: ConvNeXtBlock,
}

impl UpsampleBlock {
    pub fn from_loader(w: &WeightLoader, prefix: &str) -> crate::Result<Self> {
        let ct_w = w.get(&format!("{prefix}.0.conv.weight"))?.clone();
        let ct_b = w.get(&format!("{prefix}.0.conv.bias"))?.clone();
        Ok(Self {
            ct: ConvTranspose1d::new(
                ct_w,
                Some(ct_b),
                ConvTranspose1dConfig {
                    stride: 2,
                    padding: 0,
                    output_padding: 0,
                    dilation: 1,
                    groups: 1,
                },
            ),
            cn: ConvNeXtBlock::from_loader(w, &format!("{prefix}.1"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.cn.forward(&self.ct.forward(x)?)
    }
}

pub struct ResidualUnit {
    c1: Conv1d,
    c2: Conv1d,
    a1: Tensor,
    b1: Tensor,
    a2: Tensor,
    b2: Tensor,
}

impl ResidualUnit {
    pub fn from_loader(w: &WeightLoader, prefix: &str) -> crate::Result<Self> {
        let k1 = w.get(&format!("{prefix}.conv1.conv.weight"))?.dims()[2];
        let k2 = w.get(&format!("{prefix}.conv2.conv.weight"))?.dims()[2];
        Ok(Self {
            c1: Conv1d::new(
                w.get(&format!("{prefix}.conv1.conv.weight"))?.clone(),
                Some(w.get(&format!("{prefix}.conv1.conv.bias"))?.clone()),
                Conv1dConfig {
                    padding: k1 / 2,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            ),
            c2: Conv1d::new(
                w.get(&format!("{prefix}.conv2.conv.weight"))?.clone(),
                Some(w.get(&format!("{prefix}.conv2.conv.bias"))?.clone()),
                Conv1dConfig {
                    padding: k2 / 2,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            ),
            a1: w.get(&format!("{prefix}.act1.alpha"))?.clone(),
            b1: w.get(&format!("{prefix}.act1.beta"))?.clone(),
            a2: w.get(&format!("{prefix}.act2.alpha"))?.clone(),
            b2: w.get(&format!("{prefix}.act2.beta"))?.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = snake_beta(x, &self.a1, &self.b1)?;
        let h = self.c1.forward(&h)?;
        let h = snake_beta(&h, &self.a2, &self.b2)?;
        Ok((self.c2.forward(&h)? + x)?)
    }
}

pub struct DecoderBlock {
    sb_a: Tensor,
    sb_b: Tensor,
    ct: ConvTranspose1d,
    rus: Vec<ResidualUnit>,
}

impl DecoderBlock {
    pub fn from_loader(w: &WeightLoader, prefix: &str) -> crate::Result<Self> {
        let ct_w = w.get(&format!("{prefix}.block.1.conv.weight"))?.clone();
        let ct_b = w.get(&format!("{prefix}.block.1.conv.bias"))?.clone();
        let k = ct_w.dims()[2];
        let s = k / 2;
        let p = (k - s + 1) / 2;
        let op = (k - s) % 2;
        let mut rus = Vec::new();
        for i in 2..=4 {
            rus.push(ResidualUnit::from_loader(
                w,
                &format!("{prefix}.block.{i}"),
            )?);
        }
        Ok(Self {
            sb_a: w.get(&format!("{prefix}.block.0.alpha"))?.clone(),
            sb_b: w.get(&format!("{prefix}.block.0.beta"))?.clone(),
            ct: ConvTranspose1d::new(
                ct_w,
                Some(ct_b),
                ConvTranspose1dConfig {
                    stride: s,
                    padding: p,
                    output_padding: op,
                    dilation: 1,
                    groups: 1,
                },
            ),
            rus,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = snake_beta(x, &self.sb_a, &self.sb_b)?;
        let h = self.ct.forward(&h)?;
        let mut h = h;
        for ru in &self.rus {
            h = ru.forward(&h)?;
        }
        Ok(h)
    }
}
