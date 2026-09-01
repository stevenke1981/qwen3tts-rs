use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};

use crate::Result;
use crate::codec::{CausalConv1d, CausalConvConfig};
use crate::weights::WeightLoader;

pub fn snake_beta(x: &Tensor, alpha: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let alpha = alpha.reshape((1, alpha.elem_count(), 1))?.exp()?;
    let beta = beta.reshape((1, beta.elem_count(), 1))?.exp()?;
    let alpha_x = x.broadcast_mul(&alpha)?;
    let sin2 = alpha_x.sin()?.sqr()?;
    let denom = (beta + 1e-9f64)?;
    Ok(x.broadcast_add(&sin2.broadcast_div(&denom)?)?)
}

pub struct CausalTransConvNet {
    weight: Tensor,
    bias_3d: Option<Tensor>,
    stride: usize,
    right_crop: usize,
    device: Device,
    /// 設備端重疊加法緩衝區: (1, out_channels, right_crop)
    overlap_buf: Tensor,
}

impl CausalTransConvNet {
    pub fn new(weight: Tensor, bias: Option<Tensor>, stride: usize) -> crate::Result<Self> {
        let kernel_size = weight.dims()[2];
        let out_channels = weight.dims()[1];
        let right_crop = kernel_size.saturating_sub(stride);
        let device = weight.device().clone();
        let overlap_buf = Tensor::zeros(
            (1, out_channels, right_crop),
            weight.dtype(),
            &device,
        )?;
        let bias_3d = if let Some(ref b) = bias {
            Some(b.reshape((1, b.elem_count(), 1))?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias_3d,
            stride,
            right_crop,
            device,
            overlap_buf,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let raw = x.conv_transpose1d(&self.weight, 0, 0, self.stride, 1, 1)?;
        let mut y = if self.right_crop == 0 {
            raw
        } else {
            let len = raw.dim(2)?;
            raw.narrow(2, 0, len.saturating_sub(self.right_crop))?
        };
        if let Some(ref b3d) = self.bias_3d {
            y = y.broadcast_add(b3d)?;
        }
        Ok(y)
    }

    /// 純設備端轉置卷積串流步進 (Overlap-Add)
    pub fn step(&mut self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, t_in) = x.dims3()?;
        let raw = x.conv_transpose1d(&self.weight, 0, 0, self.stride, 1, 1)?;

        let full = if self.right_crop > 0 {
            let raw_head = (&raw.narrow(2, 0, self.right_crop)? + &self.overlap_buf)?;
            let raw_tail_len = raw.dim(2)? - self.right_crop;
            let raw_tail = raw.narrow(2, self.right_crop, raw_tail_len)?;
            Tensor::cat(&[&raw_head, &raw_tail], 2)?
        } else {
            raw
        };

        let t_out = t_in * self.stride;
        let mut emitted = full.narrow(2, 0, t_out)?;
        if let Some(ref b3d) = self.bias_3d {
            emitted = emitted.broadcast_add(b3d)?;
        }

        if self.right_crop > 0 {
            self.overlap_buf = full.narrow(2, t_out, self.right_crop)?.contiguous()?;
        }

        Ok(emitted)
    }

    pub fn reset_state(&mut self) {
        if self.right_crop > 0 {
            let out_channels = self.weight.dims()[1];
            if let Ok(zeros) = Tensor::zeros(
                (1, out_channels, self.right_crop),
                self.weight.dtype(),
                &self.device,
            ) {
                self.overlap_buf = zeros;
            }
        }
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

    pub fn step(&mut self, x: &Tensor) -> Result<Tensor> {
        let r = x;
        let h = self.dwconv.step_tensor(x)?;
        let h = h.transpose(1, 2)?;
        let h = self.norm.forward(&h)?;
        let h = candle_nn::Activation::Gelu.forward(&self.pwconv1.forward(&h)?)?;
        let h = self.pwconv2.forward(&h)?;
        let h = h.transpose(1, 2)?;
        let g = self.gamma.reshape((1, self.gamma.elem_count(), 1))?;
        Ok(g.broadcast_mul(&h)?.broadcast_add(r)?)
    }

    pub fn reset_state(&mut self) {
        self.dwconv.reset_state();
    }
}

pub struct UpsampleBlock {
    pub ct: CausalTransConvNet,
    pub cn: ConvNeXtBlock,
}

impl UpsampleBlock {
    pub fn from_loader(w: &WeightLoader, prefix: &str) -> crate::Result<Self> {
        let ct_w = w.get(&format!("{prefix}.0.conv.weight"))?.clone();
        let ct_b = w.get(&format!("{prefix}.0.conv.bias"))?.clone();
        Ok(Self {
            ct: CausalTransConvNet::new(ct_w, Some(ct_b), 2)?,
            cn: ConvNeXtBlock::from_loader(w, &format!("{prefix}.1"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.cn.forward(&self.ct.forward(x)?)
    }

    pub fn step(&mut self, x: &Tensor) -> Result<Tensor> {
        let h = self.ct.step(x)?;
        self.cn.step(&h)
    }

    pub fn reset_state(&mut self) {
        self.ct.reset_state();
        self.cn.reset_state();
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

    pub fn step(&mut self, x: &Tensor) -> Result<Tensor> {
        let h = snake_beta(x, &self.a1, &self.b1)?;
        let h = self.c1.step_tensor(&h)?;
        let h = snake_beta(&h, &self.a2, &self.b2)?;
        let h = self.c2.step_tensor(&h)?;
        Ok((h + x)?)
    }

    pub fn reset_state(&mut self) {
        self.c1.reset_state();
        self.c2.reset_state();
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
            ct: CausalTransConvNet::new(ct_w, Some(ct_b), s)?,
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

    pub fn step(&mut self, x: &Tensor) -> Result<Tensor> {
        let h = snake_beta(x, &self.sb_a, &self.sb_b)?;
        let mut h = self.ct.step(&h)?;
        for ru in &mut self.rus {
            h = ru.step(&h)?;
        }
        Ok(h)
    }

    pub fn reset_state(&mut self) {
        self.ct.reset_state();
        for ru in &mut self.rus {
            ru.reset_state();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trans_conv_step_matches_forward() {
        let device = Device::Cpu;
        let in_c = 4;
        let out_c = 8;
        let stride = 2;
        let kernel_size = 4;
        let weight = Tensor::randn(0.0f32, 1.0f32, (in_c, out_c, kernel_size), &device).unwrap();
        let bias = Some(Tensor::randn(0.0f32, 1.0f32, (out_c,), &device).unwrap());

        let mut ct = CausalTransConvNet::new(weight, bias, stride).unwrap();

        let num_frames = 6;
        let input_data: Vec<f32> = (0..in_c * num_frames)
            .map(|i| ((i * 13) % 29) as f32 * 0.1)
            .collect();
        let full_input = Tensor::from_slice(&input_data, (1, in_c, num_frames), &device).unwrap();

        // Batch forward
        let batch_out = ct.forward(&full_input).unwrap();
        let batch_flat: Vec<f32> = batch_out.flatten_all().unwrap().to_vec1().unwrap();

        // Step by step
        let mut stream_outputs = Vec::new();
        for t in 0..num_frames {
            let frame = full_input.narrow(2, t, 1).unwrap();
            let out_t = ct.step(&frame).unwrap();
            assert_eq!(out_t.shape().dims(), &[1, out_c, stride]);
            stream_outputs.push(out_t);
        }

        let stream_full = Tensor::cat(&stream_outputs.iter().collect::<Vec<_>>(), 2).unwrap();
        let stream_flat: Vec<f32> = stream_full.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(batch_flat.len(), stream_flat.len());
        let max_diff: f32 = batch_flat
            .iter()
            .zip(stream_flat.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "CausalTransConvNet step must match forward: max_diff = {max_diff}"
        );
    }
}

