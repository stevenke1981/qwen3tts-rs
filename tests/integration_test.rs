use std::path::Path;
#[cfg(feature = "candle-llm")]
use std::path::PathBuf;

#[cfg(feature = "candle-llm")]
use qwen3tts::text_frontend::CandleLLM;
#[cfg(feature = "candle-llm")]
use qwen3tts::text_frontend::{SynthesisOptions, TextFrontend};
use qwen3tts::{
    codec::{snake_beta, CausalConv1d, CausalConvConfig, DecoderBlock},
    talker::weight_loader::TalkerWeightLoader,
    weights::WeightLoader,
    Decoder12Hz, DecoderConfig, TtsDecoder,
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

/// Verify that streaming decode_chunk (called N times) produces the same
/// output as batch decode_frames for the same N frames.
/// This validates the streaming buffer + transpose logic.
#[test]
fn test_streaming_decode_matches_batch_decode() {
    let weight_dir = Path::new("weights/tokenizer");
    if !weight_dir.join("codebook.safetensors").exists() {
        eprintln!("Skipping: real weights not found at {weight_dir:?}");
        return;
    }

    let config = DecoderConfig::realtime();
    let device = candle_core::Device::Cpu;

    // Build 3 frames of realistic-looking tokens
    let frames: Vec<[u16; 16]> = vec![
        [
            42, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
        ],
        [
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 42,
        ],
        [
            200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 42, 100,
        ],
    ];

    // Batch decode (reference)
    let mut batch_decoder =
        Decoder12Hz::from_safetensors(config.clone(), weight_dir, &device).unwrap();
    let batch_output = batch_decoder.decode_frames(&frames).unwrap();

    // Streaming decode (decode_chunk per frame)
    let mut streaming_decoder = Decoder12Hz::from_safetensors(config, weight_dir, &device).unwrap();
    let mut streamed: Vec<f32> = Vec::new();
    for frame in &frames {
        let chunk = streaming_decoder
            .decode_chunk(frame.as_slice())
            .expect("decode_chunk should succeed");
        streamed.extend(chunk);
    }

    // Compare: total length must match
    assert_eq!(
        batch_output.len(),
        streamed.len(),
        "batch and streaming should produce same total samples"
    );

    // Compare: per-sample differences must be very small
    let max_diff: f32 = batch_output
        .iter()
        .zip(streamed.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mse: f32 = batch_output
        .iter()
        .zip(streamed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / batch_output.len() as f32;

    println!(
        "Streaming vs batch: max_diff={:.10}, mse={:.10}",
        max_diff, mse
    );
    println!(
        "Batch len={}, Streamed len={}",
        batch_output.len(),
        streamed.len()
    );

    // With correct data layout, the outputs should be near-identical
    // (tiny differences from floating-point accumulation order)
    assert!(
        max_diff < 1e-4,
        "Streaming must match batch decode (max_diff={max_diff})"
    );
    assert!(mse < 1e-8, "Streaming must match batch decode (mse={mse})");
}

/// Verify streaming matches batch for 13 frames with varied tokens.
/// This catches a layout bug that only manifests with >3 frames.
#[test]
fn test_streaming_13_frames_matches_batch() {
    let weight_dir = Path::new("weights/tokenizer");
    if !weight_dir.join("codebook.safetensors").exists() {
        eprintln!("Skipping: real weights not found at {weight_dir:?}");
        return;
    }
    let config = DecoderConfig::realtime();
    let device = candle_core::Device::Cpu;

    // Generate 13 frames of deterministic tokens
    let mut frames: Vec<[u16; 16]> = Vec::with_capacity(13);
    for i in 0..13u16 {
        let mut f = [0u16; 16];
        for j in 0..16u16 {
            f[j as usize] = ((i * 42 + j * 137 + 100) % 2048) as u16;
        }
        frames.push(f);
    }

    let mut batch_decoder =
        Decoder12Hz::from_safetensors(config.clone(), weight_dir, &device).unwrap();
    let batch_output = batch_decoder.decode_frames(&frames).unwrap();

    let mut streaming_decoder = Decoder12Hz::from_safetensors(config, weight_dir, &device).unwrap();
    let mut streamed: Vec<f32> = Vec::new();
    for frame in &frames {
        let chunk = streaming_decoder
            .decode_chunk(frame.as_slice())
            .expect("decode_chunk");
        streamed.extend(chunk);
    }

    assert_eq!(
        batch_output.len(),
        streamed.len(),
        "total samples must match"
    );
    let max_diff: f32 = batch_output
        .iter()
        .zip(streamed.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mse: f32 = batch_output
        .iter()
        .zip(streamed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / batch_output.len() as f32;
    println!(
        "13-frame streaming vs batch: max_diff={:.10}, mse={:.10}",
        max_diff, mse
    );
    assert!(
        max_diff < 1e-4,
        "13-frame streaming must match batch (max_diff={max_diff})"
    );
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
fn test_decoder_12hz_with_quantized_weights_smoke() {
    let base_dir = Path::new("weights/tokenizer");
    if !base_dir.join("codebook.safetensors").exists() {
        eprintln!("Skipping: real weights not found at {base_dir:?}");
        return;
    }

    let quantized_dirs = [
        Path::new("weights/tokenizer-q8"),
        Path::new("weights/tokenizer-q4"),
    ];
    if quantized_dirs
        .iter()
        .all(|dir| !dir.join("codebook.safetensors").exists())
    {
        eprintln!("Skipping: quantized tokenizer dirs not found");
        return;
    }

    let device = candle_core::Device::Cpu;
    let tokens: Vec<u16> = vec![
        42, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
    ];

    let mut base_decoder =
        Decoder12Hz::from_safetensors(DecoderConfig::realtime(), base_dir, &device).unwrap();
    let base_output = base_decoder.decode_chunk(&tokens).unwrap();
    let base_peak = base_output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        base_peak > 0.01,
        "base decoded audio peak is too low: {base_peak}"
    );

    for quantized_dir in quantized_dirs {
        if !quantized_dir.join("codebook.safetensors").exists() {
            continue;
        }
        let mut decoder =
            Decoder12Hz::from_safetensors(DecoderConfig::realtime(), quantized_dir, &device)
                .unwrap();
        let output = decoder.decode_chunk(&tokens).unwrap();
        let peak = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let cos = cosine_sim(&base_output, &output);
        println!(
            "quantized decoder {}: len={} peak={peak:.6} cosine_vs_base={cos:.8}",
            quantized_dir.display(),
            output.len()
        );

        assert_eq!(output.len(), base_output.len());
        assert!(
            peak > 0.01,
            "quantized decoded audio peak is too low: {peak}"
        );
        assert!(
            cos > 0.90,
            "quantized decoded audio cosine {cos:.8} <= 0.90 for {}",
            quantized_dir.display()
        );
    }
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
        [
            1221, 1052, 1114, 1364, 1468, 1760, 974, 1318, 746, 391, 161, 1013, 663, 837, 216, 1929,
        ],
        [
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
        ],
        [
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        ],
        [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    ];

    // Speed 1.0 (Normal)
    let config_normal = DecoderConfig::realtime();
    let mut decoder_normal =
        Decoder12Hz::from_safetensors(config_normal, weight_dir, &device).unwrap();
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
    let cfg = CausalConvConfig::from_weight(&w, 1, 1);
    let conv = CausalConv1d::new(w, b, cfg, 32).unwrap();

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
    let cfg = CausalConvConfig::from_weight(&w, 1, 1);
    let conv = CausalConv1d::new(w, b, cfg, 32).unwrap();
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

// ---------------------------------------------------------------------------
// Talker 整合測試
// ---------------------------------------------------------------------------

/// Try to find `model.safetensors` in common locations.
fn find_model_safetensors() -> Option<std::path::PathBuf> {
    // 1. Local weights directory
    let local = Path::new("weights/tokenizer/model.safetensors");
    if local.exists() {
        return Some(local.to_path_buf());
    }

    // 2. HuggingFace cache
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    let cache_dir = Path::new(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots");
    if let Ok(entries) = std::fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let candidate = entry.path().join("model.safetensors");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    None
}

#[test]
fn test_talker_weight_loader_infer_config() {
    let path = match find_model_safetensors() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: model.safetensors not found");
            return;
        }
    };

    let device = candle_core::Device::Cpu;
    let loader = TalkerWeightLoader::from_safetensors(&path, &device)
        .expect("should load model.safetensors");
    let cfg = loader.infer_config().expect("should infer config");

    // Verify against known 0.6B-Base values
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.intermediate_size, 3072);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.num_hidden_layers, 28);
    assert_eq!(cfg.text_hidden_size, 2048);
    assert_eq!(cfg.vocab_size, 3072);
    assert_eq!(cfg.code_predictor.hidden_size, 1024);
    assert_eq!(cfg.code_predictor.intermediate_size, 3072);
    assert_eq!(cfg.code_predictor.num_hidden_layers, 5);

    println!("TalkerConfig: {cfg:?}");
}

#[test]
fn test_talker_build_and_forward_smoke() {
    let path = match find_model_safetensors() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: model.safetensors not found");
            return;
        }
    };

    let device = candle_core::Device::Cpu;
    let loader = TalkerWeightLoader::from_safetensors(&path, &device)
        .expect("should load model.safetensors");
    let cfg = loader.infer_config().expect("should infer config");

    // Build the full talker
    let talker = loader
        .build_talker(&cfg)
        .expect("should build TalkerForConditionalGeneration");

    // Verify shapes by running a tiny prefill
    // Input: [batch=1, seq_len=1] with a single BOS token
    let input_ids = candle_core::Tensor::new(&[cfg.tts_bos_token_id as u32], &device)
        .expect("bos tensor")
        .unsqueeze(0)
        .expect("add batch");
    let attention_mask =
        candle_core::Tensor::ones(&[1, 1], candle_core::DType::I64, &device).expect("mask");

    // Text embedding + projection
    let text_embeds = talker.embed_text(&input_ids).expect("text embedding");
    assert_eq!(
        text_embeds.dims(),
        &[1, 1, cfg.hidden_size],
        "text_embeds shape must match hidden_size"
    );

    // Compute position IDs + RoPE
    let (position_ids, _) = talker
        .compute_position_ids(&attention_mask)
        .expect("position ids");
    let (cos, sin) = talker
        .rope
        .forward(&text_embeds, &position_ids)
        .expect("rope");

    // Run through 28-layer model
    let mut kv_caches = vec![None; cfg.num_hidden_layers];
    let causal_mask =
        qwen3tts::talker::primitives::create_causal_mask(1, &device).expect("causal mask");
    let hidden = talker
        .model
        .forward(&text_embeds, &cos, &sin, Some(&causal_mask), &mut kv_caches)
        .expect("talker model forward");
    assert_eq!(
        hidden.dims(),
        &[1, 1, cfg.hidden_size],
        "model output shape"
    );

    // Verify text embedding + projection gives non-zero results
    let text_flat = text_embeds.flatten_all().expect("flatten");
    let text_sum: f64 = text_flat
        .to_vec1::<f32>()
        .expect("to vec")
        .iter()
        .map(|&x| x as f64)
        .sum();
    assert!(
        text_sum.abs() > 0.001,
        "text embeddings should not be all zero (sum={text_sum})"
    );

    let embed_dims = text_embeds.dims().to_vec();
    let hidden_dims = hidden.dims().to_vec();
    println!("Talker smoke test passed: embeddings={embed_dims:?}, hidden={hidden_dims:?}");
}

// ---------------------------------------------------------------------------
// 端到端測試：文字 → 語音
// ---------------------------------------------------------------------------

/// Find the HuggingFace cache directory containing model.safetensors + tokenizer.json
#[cfg(feature = "candle-llm")]
fn find_model_dir() -> Option<PathBuf> {
    // Try environment variable first
    if let Ok(dir) = std::env::var("QWEN3_TTS_MODEL_DIR") {
        let p = PathBuf::from(dir);
        if p.join("model.safetensors").exists() {
            return Some(p);
        }
    }

    // HuggingFace cache
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    let cache_dir = Path::new(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots");
    if let Ok(entries) = std::fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let candidate = entry.path();
            if candidate.join("model.safetensors").exists() {
                return Some(candidate);
            }
        }
    }
    None
}

#[test]
#[cfg(feature = "candle-llm")]
fn test_text_to_speech_end_to_end() {
    let weight_dir = Path::new("weights/tokenizer");
    if !weight_dir.join("codebook.safetensors").exists() {
        eprintln!("Skipping: codec weights not found at {weight_dir:?}");
        return;
    }

    let model_dir = match find_model_dir() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: Qwen3-TTS model dir not found");
            return;
        }
    };

    // Check tokenizer.json exists
    if !model_dir.join("tokenizer.json").exists() {
        eprintln!("Skipping: tokenizer.json not found in {model_dir:?}");
        eprintln!("Run: python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('{}').save_pretrained('{}')\"", model_dir.display(), model_dir.display());
        return;
    }

    let device = candle_core::Device::Cpu;

    // ── 1. Load talker (text → codec tokens) ──
    eprintln!("Loading CandleLLM from {model_dir:?}...");
    let llm = CandleLLM::from_pretrained_dir(&model_dir, &device).expect("should load CandleLLM");

    // ── 2. Load codec decoder + vocoder (tokens → PCM) ──
    eprintln!("Loading Decoder12Hz from {weight_dir:?}...");
    let config = DecoderConfig::realtime();
    let mut decoder = Decoder12Hz::from_safetensors(config.clone(), weight_dir, &device)
        .expect("should load Decoder12Hz");

    // ── 3. Synthesize text → codec tokens ──
    // 用極短文字減少生成時間（CPU 上每幀約需 1-2 秒）
    let text = "你好。";
    let options = SynthesisOptions {
        language: "Chinese".into(),
        max_new_tokens: 64, // 短文本 8-12 幀足夠
        ..Default::default()
    };
    eprintln!(
        "Synthesizing: {text:?} (max_new_tokens={})...",
        options.max_new_tokens
    );
    let stream = llm.synthesize(text, &options).expect("synthesize");
    let num_frames = stream.num_frames();
    eprintln!(
        "Generated {num_frames} frames (~{:.1}s audio)",
        num_frames as f64 / 12.5
    );
    assert!(num_frames > 0, "should generate at least 1 frame");

    // Debug: dump first 3 frames of tokens
    for (i, frame) in stream.frames.iter().take(3).enumerate() {
        let valid = frame.iter().filter(|&&t| t < 2048).count();
        let min_t = frame.iter().min().copied().unwrap_or(0);
        let max_t = frame.iter().max().copied().unwrap_or(0);
        eprintln!(
            "  frame[{i}]: tokens=[{},{},{},{},...] range=[{min_t},{max_t}] valid_in_range={valid}/16",
            frame[0], frame[1], frame[2], frame[3]
        );
    }
    eprintln!("  ... first 3 frames shown");

    // ── 4. Decode all frames → PCM (batch decode) ──
    // Using batch decode_frames (O(n)) for this test since streaming
    // decode_chunk re-processes all accumulated frames per call (O(n²)).
    let all_pcm = decoder
        .decode_frames(&stream.frames)
        .expect("batch decode frames");

    // ── 5. Verify PCM is valid ──
    let sample_rate = 24000;
    let expected_len = (num_frames as f64 / 12.5 * sample_rate as f64) as usize;
    eprintln!(
        "PCM: {} samples (expected ~{}), {:.1}s @ {}Hz",
        all_pcm.len(),
        expected_len,
        all_pcm.len() as f64 / sample_rate as f64,
        sample_rate
    );
    assert!(all_pcm.len() > 100, "PCM should have more than 100 samples");

    // Check PCM is non-zero (audio, not silence)
    let max_abs = all_pcm.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    eprintln!("PCM max amplitude: {max_abs}");
    assert!(
        max_abs > 0.001,
        "PCM should not be silence (max_abs={max_abs})"
    );

    // ── 6. Save WAV file ──
    let wav_path = Path::new("target/test_output_e2e.wav");
    if let Some(parent) = wav_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Write WAV: f32 samples → i16
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&wav_path, spec).expect("create wav");
    for &sample in &all_pcm {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(i16_sample).expect("write sample");
    }
    writer.finalize().expect("finalize wav");
    eprintln!("WAV saved to {wav_path:?}");

    println!(
        "✅ End-to-end test passed: text → {num_frames} frames → {} samples → WAV",
        all_pcm.len()
    );
}
