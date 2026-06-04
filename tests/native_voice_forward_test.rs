use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use qwen3tts::text_frontend::voice_clone::speaker_encoder::{
    NativeSpeakerEncoder, upstream_mel_spectrogram,
};
use qwen3tts::text_frontend::voice_clone::speech_tokenizer::NativeSpeechTokenizerEncoder;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct VoiceCloneFixture {
    waveform_shape: Vec<usize>,
    waveform: Vec<f32>,
    mel_shape: Vec<usize>,
    mel: Vec<f32>,
    speaker_shape: Vec<usize>,
    speaker: Vec<f32>,
    codes: Vec<Vec<u16>>,
}

fn local_appdata_path(parts: &[&str]) -> Option<PathBuf> {
    let mut path = PathBuf::from(std::env::var("LOCALAPPDATA").ok()?);
    path.push("qwen3tts-rs");
    for part in parts {
        path.push(part);
    }
    Some(path)
}

#[test]
fn native_speaker_encoder_forwards_mels_with_real_weights_when_available() {
    let Some(path) = local_appdata_path(&["speaker-encoder", "speaker_encoder.safetensors"]) else {
        eprintln!("Skipping: LOCALAPPDATA is not set");
        return;
    };
    if !path.exists() {
        eprintln!("Skipping: speaker weights not found at {}", path.display());
        return;
    }

    let device = Device::Cpu;
    let encoder = NativeSpeakerEncoder::from_safetensors(&path, &device).unwrap();
    let mels = Tensor::zeros((1, 32, 128), DType::F32, &device).unwrap();
    let embedding = encoder.forward_mels(&mels).unwrap();

    assert_eq!(embedding.dim(0).unwrap(), 1);
    assert!(embedding.dim(1).unwrap() >= 1024);
}

#[test]
fn native_speech_tokenizer_forwards_waveform_to_16_codebooks_when_available() {
    let Some(dir) = local_appdata_path(&["tokenizer-12hz"]) else {
        eprintln!("Skipping: LOCALAPPDATA is not set");
        return;
    };
    if !dir.join("encoder.safetensors").exists() {
        eprintln!(
            "Skipping: tokenizer encoder weights not found at {}",
            dir.display()
        );
        return;
    }

    let device = Device::Cpu;
    let encoder = NativeSpeechTokenizerEncoder::from_dir(&dir, &device).unwrap();
    let waveform = Tensor::zeros((1, 24_000), DType::F32, &device).unwrap();
    let codes = encoder.encode_waveform(&waveform).unwrap();

    assert!(codes.num_frames() > 0);
    assert_eq!(codes.frames()[0].len(), 16);
}

#[test]
fn native_mel_extraction_matches_pytorch_fixture() {
    let fixture = load_fixture();
    let device = Device::Cpu;
    let waveform = Tensor::from_slice(
        &fixture.waveform,
        (fixture.waveform_shape[0], fixture.waveform_shape[1]),
        &device,
    )
    .unwrap();

    let mels = upstream_mel_spectrogram(&waveform, &device).unwrap();
    assert_eq!(mels.dims(), fixture.mel_shape.as_slice());
    let rust = mels.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cosine = cosine_sim(&rust, &fixture.mel);
    let max_abs = max_abs_diff(&rust, &fixture.mel);
    println!("native mel fixture cosine={cosine:.8} max_abs={max_abs:.8}");
    assert!(cosine >= 0.99999, "mel cosine {cosine:.8} < 0.99999");
    assert!(max_abs <= 2e-3, "mel max_abs {max_abs:.8} > 2e-3");
}

#[test]
#[ignore = "loads real speaker/tokenizer weights; run explicitly for PyTorch voice-clone alignment"]
fn native_speaker_embedding_matches_pytorch_fixture() {
    let fixture = load_fixture();
    let Some(path) = local_appdata_path(&["speaker-encoder", "speaker_encoder.safetensors"]) else {
        eprintln!("Skipping: LOCALAPPDATA is not set");
        return;
    };
    if !path.exists() {
        eprintln!("Skipping: speaker weights not found at {}", path.display());
        return;
    }

    let device = Device::Cpu;
    let encoder = NativeSpeakerEncoder::from_safetensors(&path, &device).unwrap();
    let mels = Tensor::from_slice(
        &fixture.mel,
        (
            fixture.mel_shape[0],
            fixture.mel_shape[1],
            fixture.mel_shape[2],
        ),
        &device,
    )
    .unwrap();

    let embedding = encoder.forward_mels(&mels).unwrap();
    let embedding = if embedding.dims().len() == 2 && embedding.dim(0).unwrap() == 1 {
        embedding.squeeze(0).unwrap()
    } else {
        embedding
    };
    assert_eq!(embedding.dims(), fixture.speaker_shape.as_slice());
    let rust = embedding.to_vec1::<f32>().unwrap();
    let cosine = cosine_sim(&rust, &fixture.speaker);
    let max_abs = max_abs_diff(&rust, &fixture.speaker);
    println!("native speaker fixture cosine={cosine:.8} max_abs={max_abs:.8}");
    assert!(cosine >= 0.999, "speaker cosine {cosine:.8} < 0.999");
    assert!(max_abs <= 1e-3, "speaker max_abs {max_abs:.8} > 1e-3");
}

#[test]
#[ignore = "loads real tokenizer weights; run explicitly for PyTorch voice-clone alignment"]
fn native_ref_codes_match_pytorch_fixture() {
    let fixture = load_fixture();
    let Some(dir) = local_appdata_path(&["tokenizer-12hz"]) else {
        eprintln!("Skipping: LOCALAPPDATA is not set");
        return;
    };
    if !dir.join("encoder.safetensors").exists() {
        eprintln!(
            "Skipping: tokenizer encoder weights not found at {}",
            dir.display()
        );
        return;
    }

    let device = Device::Cpu;
    let encoder = NativeSpeechTokenizerEncoder::from_dir(&dir, &device).unwrap();
    let waveform = Tensor::from_slice(
        &fixture.waveform,
        (fixture.waveform_shape[0], fixture.waveform_shape[1]),
        &device,
    )
    .unwrap();
    let codes = encoder.encode_waveform(&waveform).unwrap();
    let rust: Vec<Vec<u16>> = codes.frames().iter().map(|frame| frame.to_vec()).collect();

    assert_eq!(rust, fixture.codes);
}

fn load_fixture() -> VoiceCloneFixture {
    let data = std::fs::read_to_string("tests/fixtures/voice_clone_native.json")
        .expect("read voice clone fixture");
    serde_json::from_str(&data).expect("parse voice clone fixture")
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-12)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}
