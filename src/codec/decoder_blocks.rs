use candle_core::Tensor;
use candle_nn::{ConvTranspose1d, ConvTranspose1dConfig, Linear, Module};

use crate::codec::{CausalConv1d, CausalConvConfig};
use crate::weights::WeightLoader;
use crate::Result;

pub fn snake_beta(x: &Tensor, alpha: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let alpha = alpha.reshape((1, alpha.elem_count(), 1))?.exp()?;
    let beta = beta.reshape((1, beta.elem_count(), 1))?.exp()?;
    let alpha_x = x.broadcast_mul(&alpha)?;
    let sin2 = alpha_x.sin()?.sqr()?;
    let denom = (beta + 1e-9f64)?;
    Ok(x.broadcast_add(&sin2.broadcast_div(&denom)?)?)
}

pub struct CausalTransConvNet {
    conv: ConvTranspose1d,
    right_crop: usize,
}

impl CausalTransConvNet {
    pub fn new(weight: Tensor, bias: Option<Tensor>, stride: usize) -> Self {
        let kernel_size = weight.dims()[2];
        Self {
            conv: ConvTranspose1d::new(
                weight,
                bias,
                ConvTranspose1dConfig {
                    stride,
                    padding: 0,
                    output_padding: 0,
                    dilation: 1,
                    groups: 1,
                },
            ),
            right_crop: kernel_size.saturating_sub(stride),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(x)?;
        if self.right_crop == 0 {
            return Ok(y);
        }
        let len = y.dim(2)?;
        Ok(y.narrow(2, 0, len.saturating_sub(self.right_crop))?)
    }
}

pub struct ConvNeXtBlock {
    pub gamma: Tensor,
    pub dwconv: CausalConv1d,
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
        let dw_cfg = CausalConvConfig::from_weight(&dw_w, 1, ch);
        let nw = w.get(&format!("{prefix}.norm.weight"))?.clone();
        let nb = w.get(&format!("{prefix}.norm.bias"))?.clone();
        let p1w = w.get(&format!("{prefix}.pwconv1.weight"))?.clone();
        let p1b = w.get(&format!("{prefix}.pwconv1.bias"))?.clone();
        let p2w = w.get(&format!("{prefix}.pwconv2.weight"))?.clone();
        let p2b = w.get(&format!("{prefix}.pwconv2.bias"))?.clone();
        Ok(Self {
            gamma,
            dwconv: CausalConv1d::new(dw_w, Some(dw_b), dw_cfg, 32)?,
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
    c1: CausalConv1d,
    c2: CausalConv1d,
    a1: Tensor,
    b1: Tensor,
    a2: Tensor,
    b2: Tensor,
}

impl ResidualUnit {
    pub fn from_loader(w: &WeightLoader, prefix: &str, dilation: usize) -> crate::Result<Self> {
        let _k1 = w.get(&format!("{prefix}.conv1.conv.weight"))?.dims()[2];
        let _k2 = w.get(&format!("{prefix}.conv2.conv.weight"))?.dims()[2];
        let c1_w = w.get(&format!("{prefix}.conv1.conv.weight"))?.clone();
        let c1_b = w.get(&format!("{prefix}.conv1.conv.bias"))?.clone();
        let c2_w = w.get(&format!("{prefix}.conv2.conv.weight"))?.clone();
        let c2_b = w.get(&format!("{prefix}.conv2.conv.bias"))?.clone();
        let c1_cfg = CausalConvConfig::from_weight(&c1_w, dilation, 1);
        let c2_cfg = CausalConvConfig::from_weight(&c2_w, 1, 1);
        Ok(Self {
            c1: CausalConv1d::new(c1_w, Some(c1_b), c1_cfg, 32)?,
            c2: CausalConv1d::new(c2_w, Some(c2_b), c2_cfg, 32)?,
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
    ct: CausalTransConvNet,
    rus: Vec<ResidualUnit>,
}

impl DecoderBlock {
    pub fn from_loader(w: &WeightLoader, prefix: &str) -> crate::Result<Self> {
        let ct_w = w.get(&format!("{prefix}.block.1.conv.weight"))?.clone();
        let ct_b = w.get(&format!("{prefix}.block.1.conv.bias"))?.clone();
        let k = ct_w.dims()[2];
        let s = k / 2;
        let mut rus = Vec::new();
        for (i, dilation) in [(2, 1), (3, 3), (4, 9)] {
            rus.push(ResidualUnit::from_loader(
                w,
                &format!("{prefix}.block.{i}"),
                dilation,
            )?);
        }
        Ok(Self {
            sb_a: w.get(&format!("{prefix}.block.0.alpha"))?.clone(),
            sb_b: w.get(&format!("{prefix}.block.0.beta"))?.clone(),
            ct: CausalTransConvNet::new(ct_w, Some(ct_b), s),
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
