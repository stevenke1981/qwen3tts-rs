//! Debug: Compare Decoder12Hz output with different token inputs
//!
//! Run: cargo test --test debug_decoder -- --nocapture

use qwen3tts::{Decoder12Hz, DecoderConfig, TtsDecoder};
use std::path::Path;

#[test]
fn debug_compare_decoder_outputs() {
    let weight_dir = Path::new("weights/tokenizer");
    if !weight_dir.join("codebook.safetensors").exists() {
        eprintln!("Skipping: no weights");
        return;
    }

    let config = DecoderConfig::realtime();
    let device = candle_core::Device::Cpu;
    let mut decoder = Decoder12Hz::from_safetensors(config, weight_dir, &device).unwrap();

    // Test 1: All zeros (no meaningful tokens)
    let tokens_zero: Vec<u16> = vec![0; 16];
    let out_zero = decoder.decode_chunk(&tokens_zero).unwrap();
    println!("=== ALL ZEROS ===");
    println!(
        "  len={}, min={:.6}, max={:.6}, mean={:.6}",
        out_zero.len(),
        out_zero.iter().cloned().fold(f32::MAX, f32::min),
        out_zero.iter().cloned().fold(f32::MIN, f32::max),
        out_zero.iter().sum::<f32>() / out_zero.len() as f32
    );
    print_first_n("  first 20", &out_zero, 20);

    // Test 2: All-42 (like the integration test)
    decoder.reset_state();
    let tokens_42: Vec<u16> = vec![42; 16];
    let out_42 = decoder.decode_chunk(&tokens_42).unwrap();
    println!("=== ALL 42 ===");
    println!(
        "  len={}, min={:.6}, max={:.6}, mean={:.6}",
        out_42.len(),
        out_42.iter().cloned().fold(f32::MAX, f32::min),
        out_42.iter().cloned().fold(f32::MIN, f32::max),
        out_42.iter().sum::<f32>() / out_42.len() as f32
    );
    print_first_n("  first 20", &out_42, 20);

    // Test 3: Real LLM tokens (from earlier generate)
    decoder.reset_state();
    let tokens_real: [u16; 16] = [
        1221, 1052, 1114, 1364, 1468, 1760, 974, 1318, 746, 391, 161, 1013, 663, 837, 216, 1929,
    ];
    let out_real = decoder.decode_chunk(&tokens_real).unwrap();
    println!("=== REAL LLM TOKENS ===");
    println!(
        "  len={}, min={:.6}, max={:.6}, mean={:.6}",
        out_real.len(),
        out_real.iter().cloned().fold(f32::MAX, f32::min),
        out_real.iter().cloned().fold(f32::MIN, f32::max),
        out_real.iter().sum::<f32>() / out_real.len() as f32
    );
    print_first_n("  first 20", &out_real, 20);

    // Compare: are they all similar?
    println!();
    println!("=== COMPARISON ===");
    let diff_zero_42: f32 = out_zero
        .iter()
        .zip(out_42.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / out_zero.len() as f32;
    let diff_zero_real: f32 = out_zero
        .iter()
        .zip(out_real.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / out_zero.len() as f32;
    let diff_42_real: f32 = out_42
        .iter()
        .zip(out_real.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / out_42.len() as f32;
    println!("  Mean diff zero vs 42:   {:.10}", diff_zero_42);
    println!("  Mean diff zero vs real: {:.10}", diff_zero_real);
    println!("  Mean diff 42 vs real:   {:.10}", diff_42_real);

    // Test 4: Multiple frames like real usage
    decoder.reset_state();
    let mut all_out = Vec::new();
    let test_frames: Vec<[u16; 16]> = vec![
        [
            1221, 1052, 1114, 1364, 1468, 1760, 974, 1318, 746, 391, 161, 1013, 663, 837, 216, 1929,
        ],
        [
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
        ],
        [
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        ],
    ];
    for frame in &test_frames {
        let pcm = decoder.decode_chunk(frame).unwrap();
        all_out.push(pcm);
    }
    println!();
    println!("=== MULTI-FRAME TEST ===");
    for (i, out) in all_out.iter().enumerate() {
        let peak = out.iter().cloned().map(f32::abs).fold(f32::MIN, f32::max);
        print!("  Frame {}: peak={:.6}, ", i, peak);
        print_first_n_short("first 5", out, 5);
    }
}

fn print_first_n(label: &str, data: &[f32], n: usize) {
    print!("  {}: [", label);
    for (i, &v) in data.iter().enumerate().take(n) {
        if i > 0 {
            print!(", ");
        }
        print!("{:.6}", v);
    }
    println!(", ...]");
}

fn print_first_n_short(label: &str, data: &[f32], n: usize) {
    print!("{}: [", label);
    for (i, &v) in data.iter().enumerate().take(n) {
        if i > 0 {
            print!(", ");
        }
        print!("{:.4}", v);
    }
    println!("]");
}
