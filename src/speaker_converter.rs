//! Rust converter for Qwen3-TTS Base-model speaker encoder weights.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::PathBuf;

use safetensors::tensor::{Dtype, SafeTensors, TensorView, View};

use crate::{Error, Result};

const BASE_MODEL_CACHE_CANDIDATES: &[&str] = &[
    "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base",
    "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base",
];

#[derive(Debug, Clone)]
pub struct SpeakerConverterOptions {
    pub input_dir: Option<PathBuf>,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ConvertedSpeakerWeights {
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

/// Convert `speaker_encoder.*` tensors from a Base model snapshot.
pub fn convert_speaker_encoder_weights(
    options: &SpeakerConverterOptions,
) -> Result<ConvertedSpeakerWeights> {
    let input_dir = if let Some(input_dir) = &options.input_dir {
        input_dir.clone()
    } else {
        locate_hf_base_snapshot()?
    };
    let model_path = input_dir.join("model.safetensors");
    if !model_path.exists() {
        return Err(Error::Config(format!(
            "Base model.safetensors not found at {}",
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

    let speaker_tensors = extract_speaker_encoder_tensors(&tensors)?;
    let output_path = options.output_dir.join("speaker_encoder.safetensors");
    safetensors::serialize_to_file(speaker_tensors, None, &output_path).map_err(|err| {
        Error::Weight(format!("Failed to write {}: {err}", output_path.display()))
    })?;

    Ok(ConvertedSpeakerWeights {
        output_dir: options.output_dir.clone(),
        files: vec![output_path],
    })
}

fn locate_hf_base_snapshot() -> Result<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map_err(|_| Error::Config("Cannot locate user home directory".into()))?;
    let hub = PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub");

    let mut candidates = Vec::new();
    for repo in BASE_MODEL_CACHE_CANDIDATES {
        let snapshots = hub.join(repo).join("snapshots");
        if snapshots.exists() {
            for entry in std::fs::read_dir(&snapshots)? {
                let path = entry?.path();
                if path.join("model.safetensors").exists() {
                    candidates.push(path);
                }
            }
        }
    }
    candidates.sort();
    candidates.pop().ok_or_else(|| {
        Error::Config(
            "Qwen3-TTS Base model snapshot not found. Download a Base model or pass --input."
                .into(),
        )
    })
}

fn extract_speaker_encoder_tensors(
    tensors: &SafeTensors<'_>,
) -> Result<Vec<(String, OwnedTensor)>> {
    let mut out = BTreeMap::new();
    for (name, view) in tensors.tensors() {
        if name.starts_with("speaker_encoder.") {
            out.insert(name.to_string(), OwnedTensor::from_view(&view));
        }
    }
    if out.is_empty() {
        return Err(Error::Weight(
            "No tensors found with prefix speaker_encoder.".into(),
        ));
    }
    Ok(out.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_only_speaker_encoder_tensors() {
        let tmp = std::env::temp_dir().join(format!(
            "qwen3tts-speaker-converter-test-{}.safetensors",
            std::process::id()
        ));
        let speaker = test_tensor("speaker_encoder.linear.weight", vec![1.0, 2.0]);
        let other = test_tensor("talker.weight", vec![3.0]);
        safetensors::serialize_to_file(vec![speaker, other], None, &tmp).unwrap();

        let raw = std::fs::read(&tmp).unwrap();
        let loaded = SafeTensors::deserialize(&raw).unwrap();
        let extracted = extract_speaker_encoder_tensors(&loaded).unwrap();

        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].0, "speaker_encoder.linear.weight");

        let _ = std::fs::remove_file(tmp);
    }

    fn test_tensor(name: &str, values: Vec<f32>) -> (String, OwnedTensor) {
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        (
            name.to_string(),
            OwnedTensor {
                dtype: Dtype::F32,
                shape: vec![data.len() / 4],
                data,
            },
        )
    }
}
