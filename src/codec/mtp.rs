//! # MTP (Multi-Token Prediction) 模組 (12Hz)
//!
//! 多 Token 預測模組，從碼本嵌入產生聲學碼本 Token 序列。
//!
//! ## 初始化方式
//!
//! | 方法 | 用途 | 權重來源 |
//! |------|------|----------|
//! | [`MtpDecoder::new()`] | 單元測試（全零張量） | 無，輸出為零 |
//! | [`MtpDecoder::from_loader()`] | 生產推理 | safetensors 真實權重 |
//!
//! **警告**: `new()` 使用 `VarBuilder::dummy()`，所有權重為零，
//! `forward()` 永遠輸出全零 logits。僅用於形狀驗證測試。
//!
//! ## 權重命名約定（對應 `from_loader`）
//!
//! ```text
//! mtp.input_proj.weight          → (hidden_dim, embedding_dim)
//! mtp.input_proj.bias            → (hidden_dim,)
//! mtp.blocks.{i}.norm.weight     → (hidden_dim,)
//! mtp.blocks.{i}.norm.bias       → (hidden_dim,)
//! mtp.blocks.{i}.ffn.weight      → (hidden_dim * 4, hidden_dim)
//! mtp.blocks.{i}.ffn.bias        → (hidden_dim * 4,)
//! mtp.blocks.{i}.ffn_out.weight  → (hidden_dim, hidden_dim * 4)
//! mtp.blocks.{i}.ffn_out.bias    → (hidden_dim,)
//! mtp.output.{l}.weight          → (codebook_size, hidden_dim)
//! mtp.output.{l}.bias            → (codebook_size,)
//! ```

use candle_core::{Device, Module, Tensor};
use candle_nn as nn;

use crate::quantization::VarBuilder;
use crate::weights::WeightLoader;
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
    /// 從 VarBuilder 建立（全零張量，僅測試用）
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

    /// 從 safetensors 載入真實權重
    ///
    /// # 參數
    /// - `loader`: 權重載入器
    /// - `prefix`: 命名空間前綴，例如 `"mtp.blocks.0"`
    /// - `hidden_dim`: 隱藏維度
    fn from_loader(loader: &WeightLoader, prefix: &str, _hidden_dim: usize) -> crate::Result<Self> {
        let nw = loader.get(&format!("{prefix}.norm.weight"))?.clone();
        let nb = loader.get(&format!("{prefix}.norm.bias"))?.clone();
        let norm = nn::LayerNorm::new(nw, nb, 1e-5);

        let fw = loader.get(&format!("{prefix}.ffn.weight"))?.clone();
        let fb = loader.get(&format!("{prefix}.ffn.bias"))?.clone();
        let ffn = nn::Linear::new(fw, Some(fb));

        let fow = loader.get(&format!("{prefix}.ffn_out.weight"))?.clone();
        let fob = loader.get(&format!("{prefix}.ffn_out.bias"))?.clone();
        let ffn_out = nn::Linear::new(fow, Some(fob));

        Ok(Self { norm, ffn, ffn_out })
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
    /// 建立 MTP 解碼器（測試用，全零權重）
    ///
    /// **警告**: 所有權重為零，`forward()` 輸出全零 logits。
    /// 生產環境請使用 [`MtpDecoder::from_loader()`]。
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

    /// 從 safetensors 載入真實權重建立 MTP 解碼器
    ///
    /// # 參數
    /// - `loader`: 已載入權重的 [`WeightLoader`]
    /// - `config`: MTP 解碼器配置
    ///
    /// # 權重命名約定
    ///
    /// 見模組級文檔。
    ///
    /// # 錯誤
    /// - 若缺少必備張量，回傳 `Error::Weight`
    pub fn from_loader(loader: &WeightLoader, config: MtpConfig) -> crate::Result<Self> {
        let iw = loader.get("mtp.input_proj.weight")?.clone();
        let ib = loader.get("mtp.input_proj.bias").ok().cloned();
        let input_proj = nn::Linear::new(iw, ib);

        let mut blocks = Vec::with_capacity(config.num_blocks);
        for i in 0..config.num_blocks {
            let prefix = format!("mtp.blocks.{i}");
            let block = TransformerBlock::from_loader(loader, &prefix, config.hidden_dim)?;
            blocks.push(block);
        }

        let mut output_projs = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            let w = loader.get(&format!("mtp.output.{l}.weight"))?.clone();
            let b = loader.get(&format!("mtp.output.{l}.bias"))?.clone();
            output_projs.push(nn::Linear::new(w, Some(b)));
        }

        Ok(Self {
            config,
            input_proj,
            blocks,
            output_projs,
        })
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

    /// 輔助測試：f32 張量視圖（所有值為 linspace 0, 0.01, 0.02, ...）
    struct TestTensor {
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl safetensors::tensor::View for TestTensor {
        fn dtype(&self) -> safetensors::tensor::Dtype {
            safetensors::tensor::Dtype::F32
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> std::borrow::Cow<'_, [u8]> {
            std::borrow::Cow::Borrowed(&self.data)
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    fn make_test_tensor(name: &str, shape: Vec<usize>) -> (String, TestTensor) {
        let n: usize = shape.iter().product();
        let mut data = Vec::with_capacity(n * 4);
        for i in 0..n {
            data.extend_from_slice(&((i as f32 * 0.01).to_le_bytes()));
        }
        (name.to_string(), TestTensor { shape, data })
    }

    /// 測試 from_loader：使用暫存 safetensors 載入真實權重
    #[test]
    fn test_mtp_from_loader_round_trip() {
        let device = Device::Cpu;
        let tmp = std::env::temp_dir().join(format!(
            "qwen3tts-mtp-loader-test-{}.safetensors",
            std::process::id()
        ));

        let config = MtpConfig {
            num_layers: 2,
            codebook_size: 8,
            embedding_dim: 4,
            hidden_dim: 8,
            num_blocks: 1,
        };

        let tensors = vec![
            make_test_tensor(
                "mtp.input_proj.weight",
                vec![config.hidden_dim, config.embedding_dim],
            ),
            make_test_tensor("mtp.input_proj.bias", vec![config.hidden_dim]),
            make_test_tensor("mtp.blocks.0.norm.weight", vec![config.hidden_dim]),
            make_test_tensor("mtp.blocks.0.norm.bias", vec![config.hidden_dim]),
            make_test_tensor(
                "mtp.blocks.0.ffn.weight",
                vec![config.hidden_dim * 4, config.hidden_dim],
            ),
            make_test_tensor("mtp.blocks.0.ffn.bias", vec![config.hidden_dim * 4]),
            make_test_tensor(
                "mtp.blocks.0.ffn_out.weight",
                vec![config.hidden_dim, config.hidden_dim * 4],
            ),
            make_test_tensor("mtp.blocks.0.ffn_out.bias", vec![config.hidden_dim]),
            make_test_tensor(
                "mtp.output.0.weight",
                vec![config.codebook_size, config.hidden_dim],
            ),
            make_test_tensor("mtp.output.0.bias", vec![config.codebook_size]),
            make_test_tensor(
                "mtp.output.1.weight",
                vec![config.codebook_size, config.hidden_dim],
            ),
            make_test_tensor("mtp.output.1.bias", vec![config.codebook_size]),
        ];

        safetensors::serialize_to_file(tensors, None, &tmp).unwrap();
        let loader = crate::weights::WeightLoader::from_file(&tmp, &device).unwrap();
        let mtp = MtpDecoder::from_loader(&loader, config.clone()).unwrap();

        assert_eq!(mtp.blocks.len(), 1);
        assert_eq!(mtp.output_projs.len(), 2);

        // forward 應回傳非零 logits（權重非零）
        let embeddings =
            Tensor::ones((2, config.embedding_dim), candle_core::DType::F32, &device).unwrap();
        let logits = mtp.forward(&embeddings).unwrap();
        assert_eq!(logits.dims(), &[2, config.codebook_size]);

        let _ = std::fs::remove_file(&tmp);
    }

    /// 測試 from_loader：缺少張量時應回傳明確錯誤
    #[test]
    fn test_mtp_from_loader_missing_tensor() {
        let device = Device::Cpu;
        let tmp = std::env::temp_dir().join(format!(
            "qwen3tts-mtp-loader-missing-test-{}.safetensors",
            std::process::id()
        ));

        let tensors = vec![
            make_test_tensor("mtp.input_proj.weight", vec![8, 4]),
            make_test_tensor("mtp.input_proj.bias", vec![8]),
        ];
        safetensors::serialize_to_file(tensors, None, &tmp).unwrap();
        let loader = crate::weights::WeightLoader::from_file(&tmp, &device).unwrap();

        let config = MtpConfig {
            num_layers: 2,
            codebook_size: 8,
            embedding_dim: 4,
            hidden_dim: 8,
            num_blocks: 1,
        };
        let result = MtpDecoder::from_loader(&loader, config);
        match result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("mtp.blocks.0"),
                    "Expected missing block error, got: {msg}"
                );
            }
            Ok(_) => panic!("Expected Weight error for missing tensor"),
        }

        let _ = std::fs::remove_file(&tmp);
    }

    /// 測試 from_loader：形狀不匹配在 forward 時應報錯
    #[test]
    fn test_mtp_from_loader_wrong_shape() {
        let device = Device::Cpu;
        let tmp = std::env::temp_dir().join(format!(
            "qwen3tts-mtp-loader-shape-test-{}.safetensors",
            std::process::id()
        ));

        // 所有張量名稱齊全，但 input_proj.weight 形狀 [8,8] 而非預期 [8,4]
        let tensors = vec![
            make_test_tensor("mtp.input_proj.weight", vec![8, 8]),
            make_test_tensor("mtp.input_proj.bias", vec![8]),
            make_test_tensor("mtp.blocks.0.norm.weight", vec![8]),
            make_test_tensor("mtp.blocks.0.norm.bias", vec![8]),
            make_test_tensor("mtp.blocks.0.ffn.weight", vec![32, 8]),
            make_test_tensor("mtp.blocks.0.ffn.bias", vec![32]),
            make_test_tensor("mtp.blocks.0.ffn_out.weight", vec![8, 32]),
            make_test_tensor("mtp.blocks.0.ffn_out.bias", vec![8]),
            make_test_tensor("mtp.output.0.weight", vec![8, 8]),
            make_test_tensor("mtp.output.0.bias", vec![8]),
        ];
        safetensors::serialize_to_file(tensors, None, &tmp).unwrap();
        let loader = crate::weights::WeightLoader::from_file(&tmp, &device).unwrap();

        let config = MtpConfig {
            num_layers: 1,
            codebook_size: 8,
            embedding_dim: 4,
            hidden_dim: 8,
            num_blocks: 1,
        };
        // 建構成功（Candle Linear::new 不檢查形狀）
        let mtp = MtpDecoder::from_loader(&loader, config).unwrap();
        // forward 時 input_proj 的 weight shape [8,8] vs embedding [2,4] 應報錯
        let embeddings = Tensor::ones((2, 4), candle_core::DType::F32, &device).unwrap();
        let result = mtp.forward(&embeddings);
        assert!(result.is_err(), "Shape mismatch should cause forward error");

        let _ = std::fs::remove_file(&tmp);
    }
}
