//! # safetensors 權重載入器
//!
//! 從 `tools/convert_weights.py` 轉換的 safetensors 檔案中載入權重，
//! 提供型別化的存取子與解碼器元件建構方法。
//!
//! ## 命名慣例
//!
//! 與 `convert_weights.py` 的輸出對應：
//!
//! | Python 鍵 | Rust 存取子 | 形狀 |
//! |-----------|------------|------|
//! | `codebook_weights` | `codebook_weights()` | (16, 2048, 512) |
//! | `pre_conv.weight` | `conv1d_weight("pre_conv")` | (1024, 512, 3) |
//! | `pre_transformer.layers.0.self_attn.q_proj.weight` | `linear_weight("pre_transformer.layers.0.self_attn.q_proj")` | (1024, 512) |

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

use crate::codec::{CausalConv1d, CausalConvConfig, CodebookLookup, ParallelCodebook};
use crate::quantization::{has_quantized_tensor, load_quantized_f32};
use crate::{Error, Result};

/// safetensors 權重載入器
///
/// 封裝一組從 safetensors 檔案載入的張量，提供型別化的存取方法。
pub struct WeightLoader {
    /// 所有張量名稱 → Tensor 的對照表
    tensors: HashMap<String, Tensor>,
    /// 運算裝置
    device: Device,
}

impl WeightLoader {
    /// 從單個 safetensors 檔案載入
    pub fn from_file(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let data = std::fs::read(path.as_ref())?;
        Self::from_bytes(&data, device)
    }

    /// 從記憶體中的 safetensors 位元組載入
    pub fn from_bytes(data: &[u8], device: &Device) -> Result<Self> {
        let sf = SafeTensors::deserialize(data)
            .map_err(|e| Error::Weight(format!("Failed to deserialize safetensors: {e}")))?;

        let mut tensors = HashMap::new();
        insert_safetensors_tensors(&sf, device, &mut tensors)?;

        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// 從目錄載入所有 `.safetensors` 檔案
    ///
    /// 合併多個檔案中的張量為一個載入器。
    /// 檔名衝突時，後載入的覆蓋前者。
    pub fn from_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let mut tensors = HashMap::new();
        let dir = dir.as_ref();

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "safetensors") {
                let data = std::fs::read(&path)?;
                let sf = SafeTensors::deserialize(&data).map_err(|e| {
                    Error::Weight(format!("Failed to deserialize {}: {e}", path.display()))
                })?;
                insert_safetensors_tensors(&sf, device, &mut tensors)?;
                let n = sf.tensors().len();
                log::info!("Loaded {n} tensors from {}", path.display());
            }
        }

        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// 載入「輕量級」權重集合 (codebook + pre_conv)
    ///
    /// 從 `lightweight.safetensors` 載入：
    /// - `codebook_weights`: (16, 2048, 512)
    /// - `pre_conv.weight`: (1024, 512, 3)
    /// - `pre_conv.bias`: (1024)
    pub fn from_lightweight(dir_or_file: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = dir_or_file.as_ref();
        if path.is_dir() {
            Self::from_dir(path, device)
        } else {
            Self::from_file(path, device)
        }
    }

    // ------------------------------------------------------------------
    // 基本存取
    // ------------------------------------------------------------------

    /// 依名稱取得張量
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Weight(format!("Tensor '{name}' not found")))
    }

    /// 檢查張量是否存在
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// 列出所有張量名稱
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.tensors.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// 張量總數
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// 是否為空
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// 返回裝置參考
    pub fn device(&self) -> &Device {
        &self.device
    }

    // ------------------------------------------------------------------
    // 型別化存取子
    // ------------------------------------------------------------------

    /// 取得 16 層碼本嵌入權重，形狀 (16, 2048, 512)
    pub fn codebook_weights(&self) -> Result<Tensor> {
        Ok(self.get("codebook_weights")?.clone())
    }

    /// 建立 ParallelCodebook (含 CodebookLookup)
    pub fn build_codebook(&self) -> Result<ParallelCodebook> {
        let cw = self.codebook_weights()?;
        let lookup = CodebookLookup::new(cw)?;
        Ok(ParallelCodebook::new(lookup))
    }

    /// 取得 Conv1d 權重
    ///
    /// # 命名慣例
    /// - 檔案中鍵為 `{name}.weight`
    pub fn conv1d_weight(&self, name: &str) -> Result<Tensor> {
        let key = format!("{name}.weight");
        Ok(self.get(&key)?.clone())
    }

    /// 取得 Conv1d 偏置（若存在）
    ///
    /// # 命名慣例
    /// - 檔案中鍵為 `{name}.bias`
    pub fn conv1d_bias(&self, name: &str) -> Option<Result<Tensor>> {
        let key = format!("{name}.bias");
        if self.has(&key) {
            Some(self.get(&key).map(|t| t.clone()))
        } else {
            None
        }
    }

    /// 取得 Linear 層權重
    ///
    /// # 命名慣例
    /// - 檔案中鍵為 `{name}.weight`
    pub fn linear_weight(&self, name: &str) -> Result<Tensor> {
        let key = format!("{name}.weight");
        Ok(self.get(&key)?.clone())
    }

    /// 取得 Linear 層偏置（若存在）
    pub fn linear_bias(&self, name: &str) -> Option<Result<Tensor>> {
        let key = format!("{name}.bias");
        if self.has(&key) {
            Some(self.get(&key).map(|t| t.clone()))
        } else {
            None
        }
    }

    /// 取得 Conv1d 層的 (weight, Option<bias>) 元組
    pub fn conv1d_pair(&self, name: &str) -> Result<(Tensor, Option<Tensor>)> {
        let w = self.conv1d_weight(name)?;
        let b = self.conv1d_bias(name).transpose()?;
        Ok((w, b))
    }

    // ------------------------------------------------------------------
    // 解碼器元件建構
    // ------------------------------------------------------------------

    /// 建立 CausalConv1d 從命名權重
    ///
    /// # 參數
    /// - `name`: 命名空間（例如 `"pre_conv"`）
    /// - `config`: 卷積配置
    /// - `capacity`: 環形緩衝區容量
    pub fn build_causal_conv(
        &self,
        name: &str,
        config: CausalConvConfig,
        capacity: usize,
    ) -> Result<CausalConv1d> {
        let (weight, bias) = self.conv1d_pair(name)?;
        CausalConv1d::new(weight, bias, config, capacity)
    }

    /// 從輕量級集合建立 Decoder12Hz 所需的全部卷積層
    ///
    /// 目前僅返回 pre_conv 一層。
    /// 當完整 decoder 實作後，會返回所有 conv 層。
    pub fn build_decoder_convs(&self, config: &crate::DecoderConfig) -> Result<Vec<CausalConv1d>> {
        let conv_cfg = CausalConvConfig {
            in_channels: config.embedding_dim,
            out_channels: config.embedding_dim,
            kernel_size: config.kernel_size,
            dilation: 1,
            groups: 1,
        };

        // pre_conv: (1024, 512, 3) — 在完整 pipeline 中為 codebook_dim→latent_dim
        // 目前簡化為一層 conv
        let conv = self.build_causal_conv("pre_conv", conv_cfg, config.ring_buffer_capacity)?;
        Ok(vec![conv])
    }
}

// ---------------------------------------------------------------------------
// safetensors -> Candle Tensor 轉換
// ---------------------------------------------------------------------------

fn insert_safetensors_tensors(
    sf: &SafeTensors<'_>,
    device: &Device,
    tensors: &mut HashMap<String, Tensor>,
) -> Result<()> {
    for (name, view) in sf.tensors() {
        if let Some(base) = name.strip_suffix(".meta") {
            if has_quantized_tensor(sf, base) {
                let restored = load_quantized_f32(sf, base)?;
                let tensor = Tensor::from_slice(&restored.values, &*restored.shape, device)
                    .map_err(Error::from)?;
                tensors.insert(base.to_string(), tensor);
            }
            continue;
        }
        if name.ends_with(".qweight") || name.ends_with(".scales") {
            continue;
        }
        let tensor = tensor_from_view(&view, device)?;
        tensors.insert(name.to_string(), tensor);
    }
    Ok(())
}

/// 將 safetensors TensorView 轉換為 Candle Tensor
fn tensor_from_view(view: &TensorView, device: &Device) -> Result<Tensor> {
    let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
    let dtype = safetensors_dtype_to_candle(view.dtype())?;
    let data = view.data().to_vec();

    // Keep native inference tensors in F32 for the existing decoder/talker code.
    match dtype {
        candle_core::DType::F32 => {
            let floats = f32_values_from_le_bytes(&data)?;
            Tensor::from_slice(&floats, &*shape, device).map_err(Into::into)
        }
        candle_core::DType::BF16 => {
            let floats = bf16_values_to_f32(&data)?;
            Tensor::from_slice(&floats, &*shape, device).map_err(Into::into)
        }
        candle_core::DType::F16 => {
            let floats = f16_values_to_f32(&data)?;
            Tensor::from_slice(&floats, &*shape, device).map_err(Into::into)
        }
        other => Err(Error::Weight(format!(
            "Unsupported dtype for tensor conversion: {other:?}"
        ))),
    }
}

fn f32_values_from_le_bytes(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() % 4 != 0 {
        return Err(Error::Weight(format!(
            "Invalid F32 tensor byte length {}",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn bf16_values_to_f32(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() % 2 != 0 {
        return Err(Error::Weight(format!(
            "Invalid BF16 tensor byte length {}",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
            f32::from_bits(bits << 16)
        })
        .collect())
}

fn f16_values_to_f32(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() % 2 != 0 {
        return Err(Error::Weight(format!(
            "Invalid F16 tensor byte length {}",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(2)
        .map(|chunk| f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect())
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x03ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut frac_norm = frac;
            let mut exp_norm = -14;
            while (frac_norm & 0x0400) == 0 {
                frac_norm <<= 1;
                exp_norm -= 1;
            }
            frac_norm &= 0x03ff;
            let exp_bits = ((exp_norm + 127) as u32) << 23;
            sign | exp_bits | (frac_norm << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        let exp_bits = ((exp - 15 + 127) as u32) << 23;
        sign | exp_bits | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

/// 從 safetensors dtype 映射到 Candle dtype
fn safetensors_dtype_to_candle(dtype: Dtype) -> Result<candle_core::DType> {
    match dtype {
        Dtype::F32 => Ok(candle_core::DType::F32),
        Dtype::F16 => Ok(candle_core::DType::F16),
        Dtype::BF16 => Ok(candle_core::DType::BF16),
        _ => Err(Error::Weight(format!(
            "safetensors dtype {dtype:?} not supported, only F32/F16/BF16"
        ))),
    }
}

// ---------------------------------------------------------------------------
// 單元測試
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_from_file_not_found() {
        let device = test_device();
        let result = WeightLoader::from_file("nonexistent.safetensors", &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_file_converts_bf16_tensor_to_f32() {
        #[derive(Debug)]
        struct TestTensor {
            dtype: Dtype,
            shape: Vec<usize>,
            data: Vec<u8>,
        }
        impl safetensors::tensor::View for TestTensor {
            fn dtype(&self) -> Dtype {
                self.dtype
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

        let tmp = std::env::temp_dir().join(format!(
            "qwen3tts-bf16-loader-test-{}.safetensors",
            std::process::id()
        ));
        let tensor = TestTensor {
            dtype: Dtype::BF16,
            shape: vec![2],
            data: vec![0x80, 0x3f, 0x00, 0x40], // 1.0, 2.0 in BF16 little-endian.
        };
        safetensors::serialize_to_file(vec![("bf16_weight".to_string(), tensor)], None, &tmp)
            .unwrap();

        let loader = WeightLoader::from_file(&tmp, &test_device()).unwrap();
        let values = loader.get("bf16_weight").unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(values, vec![1.0, 2.0]);
        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn test_keys_empty_loader() {
        let device = test_device();
        let path = std::path::Path::new("weights/lightweight/lightweight.safetensors");
        if path.exists() {
            let loader = WeightLoader::from_file(path, &device).unwrap();
            let keys = loader.keys();
            assert!(!keys.is_empty());
            assert!(keys.contains(&"codebook_weights".to_string()));
        }
    }

    #[test]
    fn test_from_dir_invalid_dir() {
        let device = test_device();
        let result = WeightLoader::from_dir("weights/nonexistent", &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv1d_pair_missing_bias() {
        let device = test_device();
        let path = std::path::Path::new("weights/lightweight/lightweight.safetensors");
        if path.exists() {
            let loader = WeightLoader::from_file(path, &device).unwrap();
            // pre_conv has both weight and bias
            let (w, b) = loader.conv1d_pair("pre_conv").unwrap();
            assert_eq!(w.dims(), &[1024, 512, 3]);
            assert!(b.is_some());
        }
    }

    #[test]
    fn test_from_lightweight_file() {
        let device = test_device();
        // 輕量級權重檔案應存在於 ./weights/lightweight/
        let path = std::path::Path::new("weights/lightweight/lightweight.safetensors");
        if path.exists() {
            let loader = WeightLoader::from_lightweight(path, &device).unwrap();
            assert!(loader.has("codebook_weights"));
            assert!(loader.has("pre_conv.weight"));
            assert!(loader.has("pre_conv.bias"));
            let cw = loader.codebook_weights().unwrap();
            assert_eq!(cw.dims(), &[16, 2048, 512]);
        } else {
            println!(
                "Skipping lightweight test: weights file not found at {:?}",
                path
            );
        }
    }

    #[test]
    fn test_from_file_dequantizes_q8_tensor() {
        let device = test_device();
        let values = vec![-1.0, -0.25, 0.25, 1.0, 0.5, -0.5];
        let quantized = crate::quantization::quantize_f32_values(
            &values,
            &[2, 3],
            crate::quantization::QuantizationFormat::Q8_0,
            3,
        )
        .unwrap();
        let tmp = std::env::temp_dir().join(format!(
            "qwen3tts-weight-loader-quant-test-{}.safetensors",
            std::process::id()
        ));
        crate::quantization::save_quantized_safetensors(
            &tmp,
            vec![("linear.weight".to_string(), quantized)],
        )
        .unwrap();

        let loader = WeightLoader::from_file(&tmp, &device).unwrap();
        let tensor = loader.get("linear.weight").unwrap();

        assert_eq!(tensor.dims(), &[2, 3]);
        let restored = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(crate::quantization::cosine_f32(&values, &restored).unwrap() > 0.999);
        let _ = std::fs::remove_file(tmp);
    }
}
