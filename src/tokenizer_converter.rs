//! Rust converter for Qwen3-TTS 12Hz tokenizer decoder weights.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use safetensors::tensor::{Dtype, SafeTensors, TensorView, View};

use crate::{Error, Result};

const TOKENIZER_REPO_CACHE: &str = "models--Qwen--Qwen3-TTS-Tokenizer-12Hz";

#[derive(Debug, Clone)]
pub struct ConverterOptions {
    pub input_dir: Option<PathBuf>,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ConvertedTokenizerWeights {
    pub output_dir: PathBuf,
    pub files: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
struct OwnedTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl OwnedTensor {
    fn f32(name: impl Into<String>, shape: Vec<usize>, values: Vec<f32>) -> (String, Self) {
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        (
            name.into(),
            Self {
                dtype: Dtype::F32,
                shape,
                data,
            },
        )
    }

    fn from_view(view: &TensorView<'_>) -> Self {
        Self {
            dtype: view.dtype(),
            shape: view.shape().to_vec(),
            data: view.data().to_vec(),
        }
    }
}

impl View for OwnedTensor {
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

/// Converts Qwen3-TTS tokenizer decoder weights into Rust/Candle safetensors.
pub fn convert_tokenizer_weights(options: &ConverterOptions) -> Result<ConvertedTokenizerWeights> {
    let input_dir = if let Some(input_dir) = &options.input_dir {
        input_dir.clone()
    } else {
        locate_hf_tokenizer_snapshot()?
    };
    let model_path = input_dir.join("model.safetensors");
    if !model_path.exists() {
        return Err(Error::Config(format!(
            "Tokenizer source model.safetensors not found at {}",
            model_path.display()
        )));
    }

    std::fs::create_dir_all(&options.output_dir)?;
    let data = std::fs::read(&model_path)?;
    let tensors = SafeTensors::deserialize(&data).map_err(|err| {
        Error::Weight(format!(
            "Failed to deserialize {}: {err}",
            model_path.display()
        ))
    })?;

    let mut written = Vec::new();

    save_tensors(
        &options.output_dir.join("codebook.safetensors"),
        vec![extract_codebook_weights(&tensors)?],
        &mut written,
    )?;
    save_tensors(
        &options.output_dir.join("lightweight.safetensors"),
        extract_lightweight_weights(&tensors)?,
        &mut written,
    )?;
    save_tensors(
        &options.output_dir.join("pre_transformer.safetensors"),
        extract_prefixed(&tensors, "decoder.pre_transformer", "pre_transformer.")?,
        &mut written,
    )?;
    save_tensors(
        &options.output_dir.join("upsample.safetensors"),
        extract_prefixed(&tensors, "decoder.upsample", "upsample.")?,
        &mut written,
    )?;
    save_tensors(
        &options.output_dir.join("decoder_blocks.safetensors"),
        extract_prefixed(&tensors, "decoder.decoder", "decoder.")?,
        &mut written,
    )?;

    let config_src = input_dir.join("config.json");
    if config_src.exists() {
        std::fs::copy(&config_src, options.output_dir.join("config.json"))?;
        written.push(options.output_dir.join("config.json"));
    }

    Ok(ConvertedTokenizerWeights {
        output_dir: options.output_dir.clone(),
        files: written,
    })
}

fn locate_hf_tokenizer_snapshot() -> Result<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map_err(|_| Error::Config("Cannot locate user home directory".into()))?;
    let snapshots = PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join(TOKENIZER_REPO_CACHE)
        .join("snapshots");
    let mut candidates = Vec::new();
    if snapshots.exists() {
        for entry in std::fs::read_dir(&snapshots)? {
            let path = entry?.path();
            if path.join("model.safetensors").exists() {
                candidates.push(path);
            }
        }
    }
    candidates.sort();
    candidates.pop().ok_or_else(|| {
        Error::Config(format!(
            "Qwen3-TTS tokenizer snapshot not found under {}. Download Qwen/Qwen3-TTS-Tokenizer-12Hz or pass --input.",
            snapshots.display()
        ))
    })
}

fn extract_codebook_weights(tensors: &SafeTensors<'_>) -> Result<(String, OwnedTensor)> {
    let first_usage = tensor_f32(
        tensors,
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage",
    )?;
    let first_sum = tensor_f32(
        tensors,
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
    )?;
    let first_out = tensor_f32(tensors, "decoder.quantizer.rvq_first.output_proj.weight")?;
    let rest_out = tensor_f32(tensors, "decoder.quantizer.rvq_rest.output_proj.weight")?;

    let mut all = Vec::with_capacity(16 * 2048 * 512);
    project_codebook(&first_usage, &first_sum, &first_out, &mut all)?;
    for i in 0..15 {
        let usage = tensor_f32(
            tensors,
            &format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"),
        )?;
        let sum = tensor_f32(
            tensors,
            &format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"),
        )?;
        project_codebook(&usage, &sum, &rest_out, &mut all)?;
    }

    Ok(OwnedTensor::f32(
        "codebook_weights",
        vec![16, 2048, 512],
        all,
    ))
}

fn project_codebook(
    usage: &TensorF32,
    embedding_sum: &TensorF32,
    output_proj: &TensorF32,
    out: &mut Vec<f32>,
) -> Result<()> {
    expect_shape("cluster_usage", usage, &[2048])?;
    expect_shape("embedding_sum", embedding_sum, &[2048, 256])?;
    expect_shape("output_proj", output_proj, &[512, 256, 1])?;

    let mut projected = vec![0.0f32; 2048 * 512];
    projected
        .par_chunks_mut(512)
        .enumerate()
        .for_each(|(token, row)| {
            let denom = usage.values[token].max(1e-5);
            let sum_base = token * 256;
            for (out_dim, value) in row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                let proj_base = out_dim * 256;
                for in_dim in 0..256 {
                    let embed = embedding_sum.values[sum_base + in_dim] / denom;
                    acc += embed * output_proj.values[proj_base + in_dim];
                }
                *value = acc;
            }
        });
    out.extend(projected);
    Ok(())
}

fn extract_lightweight_weights(tensors: &SafeTensors<'_>) -> Result<Vec<(String, OwnedTensor)>> {
    Ok(vec![
        copy_tensor(tensors, "decoder.pre_conv.conv.weight", "pre_conv.weight")?,
        copy_tensor(tensors, "decoder.pre_conv.conv.bias", "pre_conv.bias")?,
    ])
}

fn extract_prefixed(
    tensors: &SafeTensors<'_>,
    source_prefix: &str,
    target_prefix: &str,
) -> Result<Vec<(String, OwnedTensor)>> {
    let mut out = BTreeMap::new();
    for (name, view) in tensors.tensors() {
        if name.starts_with(source_prefix) {
            let rust_key = name.replace("decoder.", "");
            if rust_key.starts_with(target_prefix) || source_prefix == "decoder.decoder" {
                out.insert(rust_key, OwnedTensor::from_view(&view));
            }
        }
    }
    if out.is_empty() {
        return Err(Error::Weight(format!(
            "No tensors found with prefix {source_prefix}"
        )));
    }
    Ok(out.into_iter().collect())
}

fn copy_tensor(
    tensors: &SafeTensors<'_>,
    source: &str,
    target: &str,
) -> Result<(String, OwnedTensor)> {
    let view = tensors
        .tensor(source)
        .map_err(|err| Error::Weight(format!("Missing tensor {source}: {err}")))?;
    Ok((target.to_string(), OwnedTensor::from_view(&view)))
}

fn save_tensors(
    path: &Path,
    tensors: Vec<(String, OwnedTensor)>,
    written: &mut Vec<PathBuf>,
) -> Result<()> {
    safetensors::serialize_to_file(tensors, None, path)
        .map_err(|err| Error::Weight(format!("Failed to write {}: {err}", path.display())))?;
    written.push(path.to_path_buf());
    Ok(())
}

#[derive(Debug, Clone)]
struct TensorF32 {
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn tensor_f32(tensors: &SafeTensors<'_>, name: &str) -> Result<TensorF32> {
    let view = tensors
        .tensor(name)
        .map_err(|err| Error::Weight(format!("Missing tensor {name}: {err}")))?;
    if view.dtype() != Dtype::F32 {
        return Err(Error::Weight(format!(
            "Tensor {name} has dtype {:?}; Rust converter currently expects F32",
            view.dtype()
        )));
    }
    let values = view
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(TensorF32 {
        shape: view.shape().to_vec(),
        values,
    })
}

fn expect_shape(name: &str, tensor: &TensorF32, expected: &[usize]) -> Result<()> {
    if tensor.shape == expected {
        Ok(())
    } else {
        Err(Error::Weight(format!(
            "{name} shape mismatch: got {:?}, expected {:?}",
            tensor.shape, expected
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_tensor_view_reports_shape_dtype_and_len() {
        let (_, tensor) = OwnedTensor::f32("x", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tensor.dtype(), Dtype::F32);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data_len(), 16);
    }

    #[test]
    fn safetensors_round_trip_owned_tensor() {
        let tmp = std::env::temp_dir().join(format!(
            "qwen3tts-converter-test-{}.safetensors",
            std::process::id()
        ));
        let tensor = OwnedTensor::f32("x", vec![2], vec![1.0, 2.0]);
        safetensors::serialize_to_file(vec![tensor], None, &tmp).unwrap();
        let raw = std::fs::read(&tmp).unwrap();
        let loaded = SafeTensors::deserialize(&raw).unwrap();
        let view = loaded.tensor("x").unwrap();
        assert_eq!(view.dtype(), Dtype::F32);
        assert_eq!(view.shape(), &[2]);
        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn project_codebook_matches_manual_projection() {
        let usage = TensorF32 {
            shape: vec![2048],
            values: vec![1.0; 2048],
        };
        let mut sums = vec![0.0; 2048 * 256];
        sums[0] = 2.0;
        sums[1] = 3.0;
        let embedding_sum = TensorF32 {
            shape: vec![2048, 256],
            values: sums,
        };
        let mut proj = vec![0.0; 512 * 256];
        proj[0] = 5.0;
        proj[1] = 7.0;
        let output_proj = TensorF32 {
            shape: vec![512, 256, 1],
            values: proj,
        };
        let mut out = Vec::new();
        project_codebook(&usage, &embedding_sum, &output_proj, &mut out).unwrap();
        assert_eq!(out.len(), 2048 * 512);
        assert_eq!(out[0], 31.0);
    }
}
