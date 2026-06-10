//! # 自定義量化校準工具
//!
//! 提供碼本專屬量化、逐層敏感度分析、INT8 校準等工具。
//!
//! ## 設計原則
//! - 禁止使用通用 LLM 量化配置（Q4_K_M / Q5_K_M）
//! - 每層量化校準使用專屬數據集
//! - 所有量化權重通過 cosine ≥ 0.995 方可合入

use std::borrow::Cow;
use std::path::Path;
use std::str::FromStr;

use candle_core::{Device, Tensor};
use safetensors::tensor::{Dtype, View};
use safetensors::SafeTensors;
use serde::Serialize;

/// Supported safetensors quantization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationFormat {
    /// Symmetric per-group int8, stored as signed bytes in a U8 tensor.
    Q8_0,
    /// Symmetric per-group int4, packed two signed nibbles per byte.
    Q4_0,
}

impl QuantizationFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    fn id(self) -> u32 {
        match self {
            Self::Q8_0 => 8,
            Self::Q4_0 => 4,
        }
    }

    fn qmax(self) -> f32 {
        match self {
            Self::Q8_0 => 127.0,
            Self::Q4_0 => 7.0,
        }
    }
}

impl TryFrom<u32> for QuantizationFormat {
    type Error = crate::Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            8 => Ok(Self::Q8_0),
            4 => Ok(Self::Q4_0),
            other => Err(crate::Error::Weight(format!(
                "unsupported quantized tensor format id: {other}"
            ))),
        }
    }
}

impl FromStr for QuantizationFormat {
    type Err = crate::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "q8_0" | "q8" | "int8" => Ok(Self::Q8_0),
            "q4_0" | "q4" | "int4" => Ok(Self::Q4_0),
            other => Err(crate::Error::Config(format!(
                "unsupported quantization format: {other}"
            ))),
        }
    }
}

/// A quantized tensor payload plus enough metadata to restore F32 values.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub format: QuantizationFormat,
    pub shape: Vec<usize>,
    pub group_size: usize,
    pub qweight: Vec<u8>,
    pub scales: Vec<f32>,
    num_elements: usize,
}

impl QuantizedTensor {
    /// Dequantize into an F32 vector in the original tensor order.
    pub fn dequantize_to_vec(&self) -> Vec<f32> {
        match self.format {
            QuantizationFormat::Q8_0 => self.dequantize_q8(),
            QuantizationFormat::Q4_0 => self.dequantize_q4(),
        }
    }

    fn dequantize_q8(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.num_elements);
        for (idx, &byte) in self.qweight.iter().take(self.num_elements).enumerate() {
            let group = idx / self.group_size;
            let q = i8::from_le_bytes([byte]) as f32;
            out.push(q * self.scales[group]);
        }
        out
    }

    fn dequantize_q4(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.num_elements);
        for idx in 0..self.num_elements {
            let byte = self.qweight[idx / 2];
            let nibble = if idx % 2 == 0 {
                byte & 0x0f
            } else {
                (byte >> 4) & 0x0f
            };
            let group = idx / self.group_size;
            let q = (nibble as i8 - 8) as f32;
            out.push(q * self.scales[group]);
        }
        out
    }
}

/// F32 tensor restored from a quantized safetensors payload.
#[derive(Debug, Clone)]
pub struct DequantizedTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// Options for quantizing one safetensors file.
#[derive(Debug, Clone)]
pub struct QuantizeFileOptions {
    pub format: QuantizationFormat,
    pub group_size: usize,
    pub min_elements: usize,
    pub min_cosine: Option<f64>,
    pub preserve_low_cosine: bool,
}

/// Per-tensor quantization report.
#[derive(Debug, Clone, Serialize)]
pub struct QuantizedTensorReport {
    pub name: String,
    pub quantized: bool,
    pub format: String,
    pub elements: usize,
    pub original_bytes: usize,
    pub stored_bytes: usize,
    pub cosine: Option<f64>,
    pub max_abs_error: Option<f32>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
struct OwnedBytesTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl OwnedBytesTensor {
    fn from_view(view: &safetensors::tensor::TensorView<'_>) -> Self {
        Self {
            dtype: view.dtype(),
            shape: view.shape().to_vec(),
            data: view.data().to_vec(),
        }
    }

    fn u8(shape: Vec<usize>, data: Vec<u8>) -> Self {
        Self {
            dtype: Dtype::U8,
            shape,
            data,
        }
    }

    fn f32(shape: Vec<usize>, values: Vec<f32>) -> Self {
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        Self {
            dtype: Dtype::F32,
            shape,
            data,
        }
    }

    fn u32(shape: Vec<usize>, values: Vec<u32>) -> Self {
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        Self {
            dtype: Dtype::U32,
            shape,
            data,
        }
    }
}

impl View for OwnedBytesTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Quantize an F32 slice using symmetric per-group quantization.
pub fn quantize_f32_values(
    values: &[f32],
    shape: &[usize],
    format: QuantizationFormat,
    group_size: usize,
) -> crate::Result<QuantizedTensor> {
    if group_size == 0 {
        return Err(crate::Error::Config(
            "group_size must be greater than 0".into(),
        ));
    }
    let expected: usize = shape.iter().product();
    if expected != values.len() {
        return Err(crate::Error::Config(format!(
            "shape {:?} has {expected} elements but values length is {}",
            shape,
            values.len()
        )));
    }

    let num_groups = values.len().div_ceil(group_size);
    let mut scales = Vec::with_capacity(num_groups);
    let mut q_signed = Vec::with_capacity(values.len());
    for group in values.chunks(group_size) {
        let max_abs = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs <= f32::EPSILON {
            1.0
        } else {
            max_abs / format.qmax()
        };
        scales.push(scale);
        for &value in group {
            let q = (value / scale).round();
            let q = match format {
                QuantizationFormat::Q8_0 => q.clamp(-127.0, 127.0) as i8,
                QuantizationFormat::Q4_0 => q.clamp(-8.0, 7.0) as i8,
            };
            q_signed.push(q);
        }
    }

    let qweight = match format {
        QuantizationFormat::Q8_0 => q_signed
            .into_iter()
            .map(|q| q.to_le_bytes()[0])
            .collect::<Vec<_>>(),
        QuantizationFormat::Q4_0 => {
            let mut packed = Vec::with_capacity(q_signed.len().div_ceil(2));
            for pair in q_signed.chunks(2) {
                let lo = ((pair[0] + 8) as u8) & 0x0f;
                let hi = if pair.len() > 1 {
                    (((pair[1] + 8) as u8) & 0x0f) << 4
                } else {
                    0
                };
                packed.push(lo | hi);
            }
            packed
        }
    };

    Ok(QuantizedTensor {
        format,
        shape: shape.to_vec(),
        group_size,
        qweight,
        scales,
        num_elements: values.len(),
    })
}

/// Save quantized tensors into a safetensors file.
pub fn save_quantized_safetensors(
    path: impl AsRef<Path>,
    tensors: Vec<(String, QuantizedTensor)>,
) -> crate::Result<()> {
    let mut entries = Vec::with_capacity(tensors.len() * 3);
    for (name, tensor) in tensors {
        let mut meta = vec![
            tensor.format.id(),
            tensor.group_size as u32,
            tensor.num_elements as u32,
            tensor.shape.len() as u32,
        ];
        for &dim in &tensor.shape {
            meta.push(dim as u32);
        }

        entries.push((
            format!("{name}.qweight"),
            OwnedBytesTensor::u8(vec![tensor.qweight.len()], tensor.qweight),
        ));
        entries.push((
            format!("{name}.scales"),
            OwnedBytesTensor::f32(vec![tensor.scales.len()], tensor.scales),
        ));
        entries.push((
            format!("{name}.meta"),
            OwnedBytesTensor::u32(vec![meta.len()], meta),
        ));
    }

    safetensors::serialize_to_file(entries, None, path.as_ref()).map_err(|err| {
        crate::Error::Weight(format!(
            "failed to write quantized safetensors {}: {err}",
            path.as_ref().display()
        ))
    })
}

/// Quantize one safetensors file with the conservative codec policy.
pub fn quantize_safetensors_file(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    options: &QuantizeFileOptions,
) -> crate::Result<Vec<QuantizedTensorReport>> {
    let data = std::fs::read(input.as_ref())?;
    let safetensors = SafeTensors::deserialize(&data).map_err(|err| {
        crate::Error::Weight(format!(
            "failed to deserialize {}: {err}",
            input.as_ref().display()
        ))
    })?;

    let mut entries = Vec::new();
    let mut report = Vec::new();
    for (name, view) in safetensors.tensors() {
        let elements = view.shape().iter().product::<usize>();
        let original_bytes = view.data().len();
        let skip_reason = skip_quantization_reason(&name, view.dtype(), elements, options);
        if let Some(reason) = skip_reason {
            entries.push((name.clone(), OwnedBytesTensor::from_view(&view)));
            report.push(QuantizedTensorReport {
                name,
                quantized: false,
                format: "f32".to_string(),
                elements,
                original_bytes,
                stored_bytes: original_bytes,
                cosine: None,
                max_abs_error: None,
                reason: Some(reason),
            });
            continue;
        }

        let values = f32_values(view.data());
        let quantized =
            quantize_f32_values(&values, view.shape(), options.format, options.group_size)?;
        let restored = quantized.dequantize_to_vec();
        let cosine = cosine_f32(&values, &restored)?;
        let max_abs_error = values
            .iter()
            .zip(&restored)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if options.preserve_low_cosine
            && options
                .min_cosine
                .is_some_and(|min_cosine| cosine < min_cosine)
        {
            entries.push((name.clone(), OwnedBytesTensor::from_view(&view)));
            report.push(QuantizedTensorReport {
                name,
                quantized: false,
                format: "f32".to_string(),
                elements,
                original_bytes,
                stored_bytes: original_bytes,
                cosine: Some(cosine),
                max_abs_error: Some(max_abs_error),
                reason: Some(format!(
                    "cosine {:.6} < min_cosine {:.6}; preserved as anchor",
                    cosine,
                    options.min_cosine.unwrap()
                )),
            });
            continue;
        }
        let stored_bytes =
            quantized.qweight.len() + quantized.scales.len() * 4 + (4 + quantized.shape.len()) * 4;
        let base_name = name.clone();
        let mut meta = vec![
            quantized.format.id(),
            quantized.group_size as u32,
            quantized.num_elements as u32,
            quantized.shape.len() as u32,
        ];
        for &dim in &quantized.shape {
            meta.push(dim as u32);
        }
        entries.push((
            format!("{base_name}.qweight"),
            OwnedBytesTensor::u8(vec![quantized.qweight.len()], quantized.qweight),
        ));
        entries.push((
            format!("{base_name}.scales"),
            OwnedBytesTensor::f32(vec![quantized.scales.len()], quantized.scales),
        ));
        entries.push((
            format!("{base_name}.meta"),
            OwnedBytesTensor::u32(vec![meta.len()], meta),
        ));
        report.push(QuantizedTensorReport {
            name,
            quantized: true,
            format: options.format.as_str().to_string(),
            elements,
            original_bytes,
            stored_bytes,
            cosine: Some(cosine),
            max_abs_error: Some(max_abs_error),
            reason: None,
        });
    }

    if let Some(parent) = output.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    safetensors::serialize_to_file(entries, None, output.as_ref()).map_err(|err| {
        crate::Error::Weight(format!(
            "failed to write quantized safetensors {}: {err}",
            output.as_ref().display()
        ))
    })?;
    Ok(report)
}

/// Load and dequantize one tensor from a quantized safetensors file.
pub fn load_quantized_f32(
    safetensors: &SafeTensors<'_>,
    name: &str,
) -> crate::Result<DequantizedTensor> {
    let meta_view = safetensors
        .tensor(&format!("{name}.meta"))
        .map_err(|err| crate::Error::Weight(format!("missing {name}.meta: {err}")))?;
    if meta_view.dtype() != Dtype::U32 {
        return Err(crate::Error::Weight(format!(
            "{name}.meta must be U32, got {:?}",
            meta_view.dtype()
        )));
    }
    let meta = u32_values(meta_view.data());
    if meta.len() < 4 {
        return Err(crate::Error::Weight(format!(
            "{name}.meta is too short: {} entries",
            meta.len()
        )));
    }
    let format = QuantizationFormat::try_from(meta[0])?;
    let group_size = meta[1] as usize;
    let num_elements = meta[2] as usize;
    let shape_len = meta[3] as usize;
    if meta.len() != 4 + shape_len {
        return Err(crate::Error::Weight(format!(
            "{name}.meta shape length mismatch: header says {shape_len}, meta has {} dims",
            meta.len().saturating_sub(4)
        )));
    }
    let shape: Vec<usize> = meta[4..].iter().map(|&dim| dim as usize).collect();

    let qview = safetensors
        .tensor(&format!("{name}.qweight"))
        .map_err(|err| crate::Error::Weight(format!("missing {name}.qweight: {err}")))?;
    if qview.dtype() != Dtype::U8 {
        return Err(crate::Error::Weight(format!(
            "{name}.qweight must be U8, got {:?}",
            qview.dtype()
        )));
    }
    let scales_view = safetensors
        .tensor(&format!("{name}.scales"))
        .map_err(|err| crate::Error::Weight(format!("missing {name}.scales: {err}")))?;
    if scales_view.dtype() != Dtype::F32 {
        return Err(crate::Error::Weight(format!(
            "{name}.scales must be F32, got {:?}",
            scales_view.dtype()
        )));
    }
    let scales = f32_values(scales_view.data());

    let quantized = QuantizedTensor {
        format,
        shape: shape.clone(),
        group_size,
        qweight: qview.data().to_vec(),
        scales,
        num_elements,
    };
    Ok(DequantizedTensor {
        shape,
        values: quantized.dequantize_to_vec(),
    })
}

/// Return true when `name` is stored as a quantized tensor in the file.
pub fn has_quantized_tensor(safetensors: &SafeTensors<'_>, name: &str) -> bool {
    safetensors.tensor(&format!("{name}.qweight")).is_ok()
        && safetensors.tensor(&format!("{name}.scales")).is_ok()
        && safetensors.tensor(&format!("{name}.meta")).is_ok()
}

/// Cosine similarity for F32 slices.
pub fn cosine_f32(a: &[f32], b: &[f32]) -> crate::Result<f64> {
    if a.len() != b.len() {
        return Err(crate::Error::Config(format!(
            "cosine length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }
    Ok(dot / (norm_a.sqrt() * norm_b.sqrt()))
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn skip_quantization_reason(
    name: &str,
    dtype: Dtype,
    elements: usize,
    options: &QuantizeFileOptions,
) -> Option<String> {
    if dtype != Dtype::F32 {
        return Some(format!("dtype {dtype:?} is preserved"));
    }
    if elements < options.min_elements {
        return Some(format!(
            "elements {elements} < min_elements {}",
            options.min_elements
        ));
    }
    let lower = name.to_ascii_lowercase();
    for marker in [
        ".bias", "bias", "norm", "alpha", "beta", "snake", "vocoder", "hifigan",
    ] {
        if lower.ends_with(marker) || lower.contains(&format!("{marker}.")) {
            return Some(format!("sensitive tensor matched '{marker}'"));
        }
    }
    None
}

fn u32_values(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// 參數建構輔助
///
/// 簡化的參數管理，用於模型骨架階段的張量建立。
pub struct VarBuilder {
    device: Device,
}

impl VarBuilder {
    /// 從 Device 建立
    pub fn from_device(device: Device) -> Self {
        Self { device }
    }

    /// 建立 dummy VarBuilder（所有張量用 zeros）
    pub fn dummy() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    /// 取得指定形狀的零張量（骨架階段用）
    pub fn get<S: Into<candle_core::Shape>>(&self, _name: &str, shape: S) -> Tensor {
        Tensor::zeros(shape, candle_core::DType::F32, &self.device).unwrap()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 建立子作用域 VarBuilder
    pub fn sub(&self, _prefix: impl AsRef<str>) -> Self {
        Self {
            device: self.device.clone(),
        }
    }
}

/// 線性層權重初始化（He initialization scaled for transformers）
pub fn linear_init(in_dim: usize, out_dim: usize, device: &Device) -> Tensor {
    let scale = (1.0 / (in_dim as f64).sqrt()) as f32;
    Tensor::rand(-scale, scale, (out_dim, in_dim), device).unwrap()
}

/// 零初始化
pub fn zero_init(device: &Device) -> Tensor {
    Tensor::zeros(0, candle_core::DType::F32, device).unwrap()
}

/// 量化敏感度分析結果
pub struct SensitivityReport {
    /// 每層的量化誤差（cosine distance）
    pub per_layer_cosine: Vec<f64>,
    /// 建議保留 FP16 的錨點層
    pub anchor_layers: Vec<usize>,
}

/// 逐層敏感度分析
///
/// 對碼本 Embedding 或卷積層，計算量化前後的餘弦相似度，
/// 找出對量化敏感的「錨點層」需保留 FP16。
pub fn analyze_sensitivity(
    _original: &Tensor,
    _quantized: &Tensor,
) -> crate::Result<SensitivityReport> {
    // TODO(Phase 4): 實作完整敏感度分析
    Ok(SensitivityReport {
        per_layer_cosine: Vec::new(),
        anchor_layers: Vec::new(),
    })
}

/// INT8 量化校準
///
/// 使用校準數據集對碼本進行 INT8 量化。
/// 返回量化後的權重及縮放因子。
pub fn calibrate_int8(
    _weights: &Tensor,
    _calibration_data: &Tensor,
) -> crate::Result<(Tensor, Tensor)> {
    // TODO(Phase 4): 實作 INT8 校準
    Err(crate::Error::Config(
        "INT8 calibration not yet implemented".into(),
    ))
}

/// 權重轉換: PyTorch safetensors -> Candle 相容格式
pub fn convert_from_pytorch(
    _pytorch_tensors: &std::collections::HashMap<String, Tensor>,
) -> crate::Result<std::collections::HashMap<String, Tensor>> {
    // TODO(Phase 3): 實作權重名稱對映與維度轉換
    Err(crate::Error::Config(
        "Weight conversion not yet implemented".into(),
    ))
}

// ---------------------------------------------------------------------------
// 單元測試
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_init_shape() {
        let device = Device::Cpu;
        let t = linear_init(512, 1024, &device);
        assert_eq!(t.dims(), &[1024, 512]);
    }

    #[test]
    fn test_var_builder_creation() {
        let device = Device::Cpu;
        let vb = VarBuilder::from_device(device.clone());
        // Device doesn't implement Display, just check it can be created
        assert!(vb.device().is_cpu());
    }

    #[test]
    fn q8_round_trip_keeps_high_cosine() {
        let values = vec![-1.0, -0.5, -0.125, 0.0, 0.2, 0.5, 0.9, 1.0];
        let quantized = quantize_f32_values(&values, &[2, 4], QuantizationFormat::Q8_0, 4).unwrap();
        let restored = quantized.dequantize_to_vec();

        assert_eq!(quantized.format, QuantizationFormat::Q8_0);
        assert_eq!(quantized.shape, vec![2, 4]);
        assert!(cosine_f32(&values, &restored).unwrap() > 0.999);
    }

    #[test]
    fn q4_packs_two_signed_values_per_byte() {
        let values = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let quantized = quantize_f32_values(&values, &[5], QuantizationFormat::Q4_0, 5).unwrap();

        assert_eq!(quantized.format, QuantizationFormat::Q4_0);
        assert_eq!(quantized.qweight.len(), 3);
        assert_eq!(quantized.scales.len(), 1);
        assert_eq!(quantized.dequantize_to_vec().len(), values.len());
    }

    #[test]
    fn quantized_safetensors_round_trip_restores_shape_and_values() {
        let values = vec![-1.0, -0.25, 0.25, 1.0, 0.5, -0.5];
        let quantized = quantize_f32_values(&values, &[2, 3], QuantizationFormat::Q8_0, 3).unwrap();
        let tmp = std::env::temp_dir().join(format!(
            "qwen3tts-quant-roundtrip-{}.safetensors",
            std::process::id()
        ));

        save_quantized_safetensors(&tmp, vec![("linear.weight".to_string(), quantized)]).unwrap();
        let raw = std::fs::read(&tmp).unwrap();
        let loaded = safetensors::SafeTensors::deserialize(&raw).unwrap();
        let restored = load_quantized_f32(&loaded, "linear.weight").unwrap();

        assert_eq!(restored.shape, vec![2, 3]);
        assert!(cosine_f32(&values, &restored.values).unwrap() > 0.999);
        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn quantize_safetensors_file_keeps_bias_and_quantizes_large_weight() {
        let input = std::env::temp_dir().join(format!(
            "qwen3tts-quant-input-{}.safetensors",
            std::process::id()
        ));
        let output = std::env::temp_dir().join(format!(
            "qwen3tts-quant-output-{}.safetensors",
            std::process::id()
        ));
        let weight_values = vec![-1.0, -0.5, 0.25, 1.0, 0.75, -0.25, 0.5, -0.75];
        let bias_values = vec![0.1, -0.1];
        let tensors = vec![
            (
                "linear.weight".to_string(),
                OwnedBytesTensor::f32(vec![2, 4], weight_values.clone()),
            ),
            (
                "linear.bias".to_string(),
                OwnedBytesTensor::f32(vec![2], bias_values.clone()),
            ),
        ];
        safetensors::serialize_to_file(tensors, None, &input).unwrap();

        let report = quantize_safetensors_file(
            &input,
            &output,
            &QuantizeFileOptions {
                format: QuantizationFormat::Q8_0,
                group_size: 4,
                min_elements: 4,
                min_cosine: None,
                preserve_low_cosine: false,
            },
        )
        .unwrap();
        let raw = std::fs::read(&output).unwrap();
        let loaded = SafeTensors::deserialize(&raw).unwrap();

        assert!(has_quantized_tensor(&loaded, "linear.weight"));
        assert!(loaded.tensor("linear.bias").is_ok());
        assert_eq!(report.iter().filter(|item| item.quantized).count(), 1);
        let restored = load_quantized_f32(&loaded, "linear.weight").unwrap();
        assert!(cosine_f32(&weight_values, &restored.values).unwrap() > 0.999);

        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }

    #[test]
    fn quantize_safetensors_file_preserves_low_cosine_anchor() {
        let input = std::env::temp_dir().join(format!(
            "qwen3tts-quant-anchor-input-{}.safetensors",
            std::process::id()
        ));
        let output = std::env::temp_dir().join(format!(
            "qwen3tts-quant-anchor-output-{}.safetensors",
            std::process::id()
        ));
        let values = vec![-1.0, -0.01, 0.02, 1.0, 0.03, -0.04, 0.05, -0.06];
        let tensors = vec![(
            "sensitive.weight".to_string(),
            OwnedBytesTensor::f32(vec![2, 4], values.clone()),
        )];
        safetensors::serialize_to_file(tensors, None, &input).unwrap();

        let report = quantize_safetensors_file(
            &input,
            &output,
            &QuantizeFileOptions {
                format: QuantizationFormat::Q4_0,
                group_size: 8,
                min_elements: 4,
                min_cosine: Some(0.999999),
                preserve_low_cosine: true,
            },
        )
        .unwrap();
        let raw = std::fs::read(&output).unwrap();
        let loaded = SafeTensors::deserialize(&raw).unwrap();

        assert!(!has_quantized_tensor(&loaded, "sensitive.weight"));
        assert_eq!(report[0].quantized, false);
        assert!(report[0].reason.as_deref().unwrap().contains("cosine"));

        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }
}
