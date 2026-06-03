use std::path::Path;

use qwen3tts::{
    Decoder12Hz, DecoderConfig, TtsDecoder,
    codec::{CausalConvNet, DecoderBlock, snake_beta},
    weights::WeightLoader,
};

fn read_npy_f32(path: &str) -> Vec<f32> {
    let data = std::fs::read(path).expect("read npy");
    assert_eq!(&data[..6], b"\x93NUMPY", "bad npy magic");
    let major = data[6];
    let (header_len_size, header_len) = if major == 1 {
        (2, u16::from_le_bytes([data[8], data[9]]) as usize)
    } else {
        (
            4,
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize,
        )
    };
    let data_start = 8 + header_len_size + header_len;
    data[data_start..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let dot: f64 = a[..n]
        .iter()
        .zip(&b[..n])
        .map(|(x, y)| *x as f64 * *y as f64)
        .sum();
    let na = a[..n]
        .iter()
        .map(|x| *x as f64 * *x as f64)
        .sum::<f64>()
        .sqrt();
    let nb = b[..n]
        .iter()
        .map(|x| *x as f64 * *x as f64)
        .sum::<f64>()
        .sqrt();
    dot / (na * nb + 1e-12)
}

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

#[test]
fn test_decode_frames_matches_pytorch_reference() {
    let weight_dir = Path::new("weights/tokenizer");
    let reference = Path::new("weights/pt_decoder_out.npy");
    if !weight_dir.join("codebook.safetensors").exists() || !reference.exists() {
        eprintln!("Skipping: real weights or PyTorch reference not found");
        return;
    }

    let config = DecoderConfig::realtime();
    let device = candle_core::Device::Cpu;
    let mut decoder = Decoder12Hz::from_safetensors(config, weight_dir, &device).unwrap();

    let frames: Vec<[u16; 16]> = vec![
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

    let output = decoder.decode_frames(&frames).unwrap();
    let reference = read_npy_f32("weights/pt_decoder_out.npy");
    let cos = cosine_sim(&output, &reference);
    let peak = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

    println!(
        "decode_frames: len={}, peak={peak:.6}, cosine_vs_pt={cos:.8}",
        output.len()
    );

    assert_eq!(output.len(), reference.len());
    assert!(peak > 0.1, "decoded audio peak is too low: {peak}");
    assert!(cos > 0.99, "decode_frames cosine {cos:.8} <= 0.99");
}

#[test]
fn test_decode_frames_speed_adjustment() {
    let weight_dir = Path::new("weights/tokenizer");
    if !weight_dir.join("codebook.safetensors").exists() {
        eprintln!("Skipping: real weights not found");
        return;
    }

    let device = candle_core::Device::Cpu;
    let frames: Vec<[u16; 16]> = vec![
        [1221, 1052, 1114, 1364, 1468, 1760, 974, 1318, 746, 391, 161, 1013, 663, 837, 216, 1929],
        [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600],
        [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
    ];

    // Speed 1.0 (Normal)
    let config_normal = DecoderConfig::realtime();
    let mut decoder_normal = Decoder12Hz::from_safetensors(config_normal, weight_dir, &device).unwrap();
    let output_normal = decoder_normal.decode_frames(&frames).unwrap();

    // Speed 2.0 (Double Speed - half the frames/samples)
    let mut config_fast = DecoderConfig::realtime_with_capacity(frames.len());
    config_fast.speed = 2.0;
    let mut decoder_fast = Decoder12Hz::from_safetensors(config_fast, weight_dir, &device).unwrap();
    let output_fast = decoder_fast.decode_frames(&frames).unwrap();

    // Speed 0.5 (Half Speed - double the frames/samples)
    // 4 frames at 0.5 speed = 8 frames -> capacity 8
    let mut config_slow = DecoderConfig::realtime_with_capacity(8);
    config_slow.speed = 0.5;
    let mut decoder_slow = Decoder12Hz::from_safetensors(config_slow, weight_dir, &device).unwrap();
    let output_slow = decoder_slow.decode_frames(&frames).unwrap();

    println!("output_normal len = {}", output_normal.len());
    println!("output_fast len = {}", output_fast.len());
    println!("output_slow len = {}", output_slow.len());

    assert!(output_fast.len() < output_normal.len());
    assert!(output_slow.len() > output_normal.len());
    assert_eq!(output_fast.len(), 3840);
    assert_eq!(output_normal.len(), 7680);
    assert_eq!(output_slow.len(), 15360);
}

#[test]
fn test_decoder_start_conv_matches_pytorch_reference() {
    let weight_dir = Path::new("weights/tokenizer");
    if !weight_dir.join("decoder_blocks.safetensors").exists()
        || !Path::new("weights/pt_upsample_out.npy").exists()
        || !Path::new("weights/pt_decoder_block.0.npy").exists()
    {
        eprintln!("Skipping: decoder reference fixtures not found");
        return;
    }

    let device = candle_core::Device::Cpu;
    let loader = WeightLoader::from_dir(weight_dir, &device).unwrap();
    let (w, b) = loader.conv1d_pair("0.conv").unwrap();
    let conv = CausalConvNet::new(w, b, 1, 1, 1);

    let input = read_npy_f32("weights/pt_upsample_out.npy");
    let input = candle_core::Tensor::from_slice(&input, (1, 1024, 12), &device).unwrap();
    let output = conv.forward(&input).unwrap();
    let output = output
        .squeeze(0)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();
    let reference = read_npy_f32("weights/pt_decoder_block.0.npy");
    let cos = cosine_sim(&output, &reference);

    println!("decoder_start conv cosine_vs_pt={cos:.8}");
    assert!(cos > 0.999, "decoder_start cosine {cos:.8} <= 0.999");
}

#[test]
fn test_decoder_blocks_match_pytorch_reference() {
    let weight_dir = Path::new("weights/tokenizer");
    if !weight_dir.join("decoder_blocks.safetensors").exists()
        || !Path::new("weights/pt_decoder_block.0.npy").exists()
        || !Path::new("weights/pt_decoder_block.4.npy").exists()
    {
        eprintln!("Skipping: decoder block fixtures not found");
        return;
    }

    let device = candle_core::Device::Cpu;
    let loader = WeightLoader::from_dir(weight_dir, &device).unwrap();
    let shapes = [(1, 1536, 12), (1, 768, 96), (1, 384, 480), (1, 192, 1920)];

    for block_idx in 1..=4 {
        let input = read_npy_f32(&format!("weights/pt_decoder_block.{}.npy", block_idx - 1));
        let input =
            candle_core::Tensor::from_slice(&input, shapes[block_idx - 1], &device).unwrap();
        let block = DecoderBlock::from_loader(&loader, &format!("{block_idx}")).unwrap();
        let output = block.forward(&input).unwrap();
        let output = output
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>();
        let reference = read_npy_f32(&format!("weights/pt_decoder_block.{block_idx}.npy"));
        let cos = cosine_sim(&output, &reference);
        println!("decoder block {block_idx} cosine_vs_pt={cos:.8}");
        assert!(
            cos > 0.999,
            "decoder block {block_idx} cosine {cos:.8} <= 0.999"
        );
    }
}

#[test]
fn test_decoder_final_stage_matches_pytorch_reference() {
    let weight_dir = Path::new("weights/tokenizer");
    if !weight_dir.join("decoder_blocks.safetensors").exists()
        || !Path::new("weights/pt_decoder_block.4.npy").exists()
        || !Path::new("weights/pt_decoder_block.6.npy").exists()
    {
        eprintln!("Skipping: decoder final fixtures not found");
        return;
    }

    let device = candle_core::Device::Cpu;
    let loader = WeightLoader::from_dir(weight_dir, &device).unwrap();
    let input = read_npy_f32("weights/pt_decoder_block.4.npy");
    let input = candle_core::Tensor::from_slice(&input, (1, 96, 5760), &device).unwrap();
    let h = snake_beta(
        &input,
        loader.get("5.alpha").unwrap(),
        loader.get("5.beta").unwrap(),
    )
    .unwrap();
    let snake = h
        .squeeze(0)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();
    let snake_ref = read_npy_f32("weights/pt_decoder_block.5.npy");
    let snake_cos = cosine_sim(&snake, &snake_ref);
    println!("decoder final snake cosine_vs_pt={snake_cos:.8}");

    let (w, b) = loader.conv1d_pair("6.conv").unwrap();
    let conv = CausalConvNet::new(w, b, 1, 1, 1);
    let output = conv.forward(&h).unwrap();
    let output = output
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let reference = read_npy_f32("weights/pt_decoder_block.6.npy");
    let cos = cosine_sim(&output, &reference);
    println!("decoder final conv cosine_vs_pt={cos:.8}");

    assert!(
        snake_cos > 0.999,
        "decoder final snake cosine {snake_cos:.8} <= 0.999"
    );
    assert!(cos > 0.999, "decoder final conv cosine {cos:.8} <= 0.999");
}
