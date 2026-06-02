//! # MTP (Multi-Token Prediction) 模組 (12Hz)
//!
//! 多 Token 預測模組，從碼本嵌入產生聲學碼本 Token 序列。

use candle_core::{Device, Module, Tensor};
use candle_nn as nn;

use crate::quantization::VarBuilder;
use crate::Error;

/// MTP 解碼器輸出
pub struct MtpOutput {
    /// 預測的 Token logits，形狀: (num_layers, codebook_size)
    pub logits: Tensor,
    /// 取樣後的 Token ID
    pub tokens: Vec<u16>,
}

/// MTP 解碼器配置
#[derive(Debug, Clone)]
pub struct MtpConfig {
    pub num_layers: usize,
    pub codebook_size: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub num_blocks: usize,
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            num_layers: 16,
            codebook_size: 2048,
            embedding_dim: 512,
            hidden_dim: 1024,
            num_blocks: 4,
        }
    }
}

/// 簡易 Transformer 區塊（LayerNorm + FFN）
struct TransformerBlock {
    norm: nn::LayerNorm,
    ffn: nn::Linear,
    ffn_out: nn::Linear,
}

impl TransformerBlock {
    fn new(hidden_dim: usize, vb: &VarBuilder) -> Self {
        let weight = vb.get("norm_weight", hidden_dim);
        let bias = vb.get("norm_bias", hidden_dim);
        let norm = nn::LayerNorm::new(weight, bias, 1e-5);

        let w = vb.get("ffn_weight", (hidden_dim * 4, hidden_dim));
        let b = vb.get("ffn_bias", hidden_dim * 4);
        let ffn = nn::Linear::new(w, Some(b));

        let w2 = vb.get("ffn_out_weight", (hidden_dim, hidden_dim * 4));
        let b2 = vb.get("ffn_out_bias", hidden_dim);
        let ffn_out = nn::Linear::new(w2, Some(b2));

        Self { norm, ffn, ffn_out }
    }

    fn forward(&self, x: &Tensor) -> crate::Result<Tensor> {
        let residual = x.clone();
        let x = self.norm.forward(x)?;
        let x = self.ffn.forward(&x)?;
        let x = x.gelu()?;
        let x = self.ffn_out.forward(&x)?;
        Ok((x + residual)?)
    }
}

/// MTP 解碼器
pub struct MtpDecoder {
    config: MtpConfig,
    input_proj: nn::Linear,
    blocks: Vec<TransformerBlock>,
    output_projs: Vec<nn::Linear>,
}

impl MtpDecoder {
    /// 建立 MTP 解碼器
    pub fn new(config: MtpConfig, _device: &Device) -> Self {
        let vb = VarBuilder::dummy();

        let w = vb.get(
            "mtp_input_weight",
            (config.hidden_dim, config.embedding_dim),
        );
        let b = vb.get("mtp_input_bias", config.hidden_dim);
        let input_proj = nn::Linear::new(w, Some(b));

        let blocks = (0..config.num_blocks)
            .map(|i| TransformerBlock::new(config.hidden_dim, &vb.sub(format!("mtp_block_{i}"))))
            .collect();

        let output_projs = (0..config.num_layers)
            .map(|l| {
                let w = vb.get(
                    &format!("mtp_output_{l}_weight"),
                    (config.codebook_size, config.hidden_dim),
                );
                let b = vb.get(&format!("mtp_output_{l}_bias"), config.codebook_size);
                nn::Linear::new(w, Some(b))
            })
            .collect();

        Self {
            config,
            input_proj,
            blocks,
            output_projs,
        }
    }

    /// 前向傳播
    pub fn forward(&self, embeddings: &Tensor) -> crate::Result<Tensor> {
        let mut x = self.input_proj.forward(embeddings)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        let mut outputs = Vec::with_capacity(self.config.num_layers);
        for (l, proj) in self.output_projs.iter().enumerate() {
            let layer_x = x.narrow(0, l, 1)?;
            let logits = proj.forward(&layer_x)?;
            outputs.push(logits);
        }

        let stacked = Tensor::stack(&outputs, 0)?;
        stacked.squeeze(1).map_err(Into::into)
    }

    /// 解碼一步：輸入嵌入，產生取樣後的 Token
    pub fn decode_step(&self, embeddings: &Tensor, temperature: f64) -> crate::Result<MtpOutput> {
        let logits = self.forward(embeddings)?;
        let tokens = self.sample(&logits, temperature)?;
        Ok(MtpOutput { logits, tokens })
    }

    /// 從 logits 取樣 Token
    fn sample(&self, logits: &Tensor, temperature: f64) -> crate::Result<Vec<u16>> {
        let mut tokens = Vec::with_capacity(self.config.num_layers);

        for l in 0..self.config.num_layers {
            let layer_logits = logits.narrow(0, l, 1)?;
            let scaled = if (temperature - 1.0).abs() > 1e-6 {
                (layer_logits / temperature)?
            } else {
                layer_logits
            };
            let probs = candle_nn::ops::softmax(&scaled, 1)?;
            let token = sample_argmax(&probs)?;
            tokens.push(token as u16);
        }

        Ok(tokens)
    }
}

/// 從機率分佈取 argmax（簡化取樣）
fn sample_argmax(probs: &Tensor) -> crate::Result<u32> {
    let argmax = probs.argmax(1)?.to_scalar::<u32>().map_err(Error::Candle)?;
    Ok(argmax)
}

// ---------------------------------------------------------------------------
// 單元測試
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_mtp_creation() {
        let device = Device::Cpu;
        let config = MtpConfig::default();
        let mtp = MtpDecoder::new(config, &device);
        assert_eq!(mtp.blocks.len(), 4);
        assert_eq!(mtp.output_projs.len(), 16);
    }

    #[test]
    fn test_mtp_forward_shape() {
        let device = Device::Cpu;
        let config = MtpConfig::default();
        let mtp = MtpDecoder::new(config, &device);
        let embeddings = Tensor::zeros((16, 512), candle_core::DType::F32, &device).unwrap();
        let logits = mtp.forward(&embeddings).unwrap();
        assert_eq!(logits.dims(), &[16, 2048]);
    }
}
