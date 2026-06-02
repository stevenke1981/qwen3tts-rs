use std::path::Path;

use qwen3tts::{Decoder12Hz, DecoderConfig, TtsDecoder};

#[test]
fn test_decoder_12hz_with_real_weights() {
    let weight_dir = Path::new("weights/tokenizer");
    if !weight_dir.join("codebook.safetensors").exists() {
        eprintln!("Skipping: real weights not found at {weight_dir:?}");
        return;
    }

    let config = DecoderConfig::realtime();
    let device = candle_core::Device::Cpu;

    let mut decoder = Decoder12Hz::from_safetensors(config, weight_dir, &device).unwrap();

    let tokens: Vec<u16> = vec![
        42, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
    ];

    let output = decoder.decode_chunk(&tokens).unwrap();
    let out_len = output.len();
    println!(
        "Frame 1: output length={out_len}, first 5 samples={:.4?}",
        &output[..5.min(out_len)]
    );

    assert_eq!(out_len, 1920, "Expected 1920 samples per frame");

    let output2 = decoder.decode_chunk(&tokens).unwrap();
    let out2_len = output2.len();
    println!(
        "Frame 2: output length={out2_len}, first 5 samples={:.4?}",
        &output2[..5.min(out2_len)]
    );
    assert_eq!(out2_len, 1920, "Expected 1920 samples per frame");

    let diff: f32 = output
        .iter()
        .zip(output2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / out_len as f32;
    println!("Mean absolute difference between frames: {diff:.6}");
    assert!(diff > 0.0, "Frames should differ (causal conv has state)");

    decoder.reset_state();
    let output3 = decoder.decode_chunk(&tokens).unwrap();
    let diff2: f32 = output
        .iter()
        .zip(output3.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / out_len as f32;
    println!("After reset, mean diff from frame 1: {diff2:.6}");
    assert!(
        diff2.abs() < 1e-4,
        "After reset, output should match first frame"
    );
}
