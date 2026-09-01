//! Streaming alignment and O(1) performance tests for Decoder12Hz.
//!
//! Validates:
//! 1. Numerical alignment: Sequential `decode_chunk` vs full-batch `decode_frames` (Cosine Sim >= 0.999, MSE < 1e-4).
//! 2. O(1) Constant execution time: Frame latency remains constant across multiple frames (no O(N) or O(N^2) growth).
//! 3. Zero state leakage: `reset_state()` fully resets internal state to identical fresh baseline.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use qwen3tts::weights::WeightLoader;
use qwen3tts::{Decoder12Hz, DecoderConfig, TtsDecoder};

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let dot: f64 = a[..n]
        .iter()
        .zip(&b[..n])
        .map(|(x, y)| *x as f64 * *y as f64)
        .sum();
    let na: f64 = a[..n]
        .iter()
        .map(|x| *x as f64 * *x as f64)
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b[..n]
        .iter()
        .map(|x| *x as f64 * *x as f64)
        .sum::<f64>()
        .sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb + 1e-12)
}

fn mse(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    a[..n]
        .iter()
        .zip(&b[..n])
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum::<f64>()
        / n as f64
}

fn generate_deterministic_frames(count: usize, seed: u16) -> Vec<[u16; 16]> {
    let mut frames = Vec::with_capacity(count);
    for i in 0..count {
        let mut frame = [0u16; 16];
        for (j, item) in frame.iter_mut().enumerate() {
            *item = ((i as u32 * 43 + j as u32 * 179 + seed as u32 * 13) % 2048) as u16;
        }
        frames.push(frame);
    }
    frames
}

/// 建立完整 Decoder12Hz 所需的合成權重載入器
fn create_synthetic_decoder_loader(config: &DecoderConfig, device: &Device) -> WeightLoader {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    let rand_t = |shape: &[usize]| -> Tensor {
        Tensor::rand(-0.02f64, 0.02f64, shape, device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
    };
    let ones_t = |shape: &[usize]| -> Tensor {
        Tensor::ones(shape, DType::F32, device).unwrap()
    };
    let zeros_t = |shape: &[usize]| -> Tensor {
        Tensor::zeros(shape, DType::F32, device).unwrap()
    };

    let emb_dim = config.embedding_dim; // 512
    let lat_dim = config.latent_dim; // 1024
    let tf_dim = config.transformer_dim; // 512
    let num_heads = config.transformer_heads; // 16
    let num_kv_heads = config.transformer_kv_heads; // 16
    let head_dim = tf_dim / num_heads; // 32
    let kv_dim = num_kv_heads * head_dim; // 512

    // 1. Codebook: (16, 2048, 512)
    tensors.insert("codebook_weights".to_string(), rand_t(&[config.num_codebook_layers, config.codebook_size, emb_dim]));

    // 2. Pre-Conv (1024, 512, 3)
    tensors.insert("pre_conv.weight".to_string(), rand_t(&[lat_dim, emb_dim, 3]));
    tensors.insert("pre_conv.bias".to_string(), zeros_t(&[lat_dim]));

    // 3. PreTransformer
    tensors.insert("pre_transformer.input_proj.weight".to_string(), rand_t(&[tf_dim, lat_dim]));
    tensors.insert("pre_transformer.input_proj.bias".to_string(), zeros_t(&[tf_dim]));
    tensors.insert("pre_transformer.output_proj.weight".to_string(), rand_t(&[lat_dim, tf_dim]));
    tensors.insert("pre_transformer.output_proj.bias".to_string(), zeros_t(&[lat_dim]));
    tensors.insert("pre_transformer.norm.weight".to_string(), ones_t(&[tf_dim]));

    for i in 0..config.transformer_layers {
        let prefix = format!("pre_transformer.layers.{i}");
        tensors.insert(format!("{prefix}.input_layernorm.weight"), ones_t(&[tf_dim]));
        tensors.insert(format!("{prefix}.post_attention_layernorm.weight"), ones_t(&[tf_dim]));
        tensors.insert(format!("{prefix}.self_attn.q_proj.weight"), rand_t(&[tf_dim, tf_dim]));
        tensors.insert(format!("{prefix}.self_attn.q_proj.bias"), zeros_t(&[tf_dim]));
        tensors.insert(format!("{prefix}.self_attn.k_proj.weight"), rand_t(&[kv_dim, tf_dim]));
        tensors.insert(format!("{prefix}.self_attn.k_proj.bias"), zeros_t(&[kv_dim]));
        tensors.insert(format!("{prefix}.self_attn.v_proj.weight"), rand_t(&[kv_dim, tf_dim]));
        tensors.insert(format!("{prefix}.self_attn.v_proj.bias"), zeros_t(&[kv_dim]));
        tensors.insert(format!("{prefix}.self_attn.o_proj.weight"), rand_t(&[tf_dim, tf_dim]));
        tensors.insert(format!("{prefix}.self_attn.o_proj.bias"), zeros_t(&[tf_dim]));
        tensors.insert(format!("{prefix}.self_attn_layer_scale.scale"), ones_t(&[tf_dim]));
        tensors.insert(format!("{prefix}.mlp_layer_scale.scale"), ones_t(&[tf_dim]));
        tensors.insert(format!("{prefix}.mlp.gate_proj.weight"), rand_t(&[tf_dim * 4, tf_dim]));
        tensors.insert(format!("{prefix}.mlp.gate_proj.bias"), zeros_t(&[tf_dim * 4]));
        tensors.insert(format!("{prefix}.mlp.up_proj.weight"), rand_t(&[tf_dim * 4, tf_dim]));
        tensors.insert(format!("{prefix}.mlp.up_proj.bias"), zeros_t(&[tf_dim * 4]));
        tensors.insert(format!("{prefix}.mlp.down_proj.weight"), rand_t(&[tf_dim, tf_dim * 4]));
        tensors.insert(format!("{prefix}.mlp.down_proj.bias"), zeros_t(&[tf_dim]));
    }

    // 4. Upsample Blocks
    for i in 0..2 {
        let prefix = format!("upsample.{i}");
        tensors.insert(format!("{prefix}.0.conv.weight"), rand_t(&[lat_dim, lat_dim, 4]));
        tensors.insert(format!("{prefix}.0.conv.bias"), zeros_t(&[lat_dim]));
        tensors.insert(format!("{prefix}.1.gamma"), ones_t(&[lat_dim]));
        tensors.insert(format!("{prefix}.1.dwconv.conv.weight"), rand_t(&[lat_dim, 1, 7]));
        tensors.insert(format!("{prefix}.1.dwconv.conv.bias"), zeros_t(&[lat_dim]));
        tensors.insert(format!("{prefix}.1.norm.weight"), ones_t(&[lat_dim]));
        tensors.insert(format!("{prefix}.1.norm.bias"), zeros_t(&[lat_dim]));
        tensors.insert(format!("{prefix}.1.pwconv1.weight"), rand_t(&[lat_dim * 4, lat_dim]));
        tensors.insert(format!("{prefix}.1.pwconv1.bias"), zeros_t(&[lat_dim * 4]));
        tensors.insert(format!("{prefix}.1.pwconv2.weight"), rand_t(&[lat_dim, lat_dim * 4]));
        tensors.insert(format!("{prefix}.1.pwconv2.bias"), zeros_t(&[lat_dim]));
    }

    // 5. Decoder Start (1536, 1024, 7)
    tensors.insert("0.conv.weight".to_string(), rand_t(&[1536, lat_dim, 7]));
    tensors.insert("0.conv.bias".to_string(), zeros_t(&[1536]));

    // 6. Decoder Blocks
    let block_configs = [
        (1, 1536, 768, 16),
        (2, 768, 384, 10),
        (3, 384, 192, 8),
        (4, 192, 96, 6),
    ];
    for (idx, in_ch, out_ch, k_trans) in block_configs {
        let prefix = format!("{idx}.block");
        tensors.insert(format!("{prefix}.0.alpha"), ones_t(&[in_ch]));
        tensors.insert(format!("{prefix}.0.beta"), ones_t(&[in_ch]));
        tensors.insert(format!("{prefix}.1.conv.weight"), rand_t(&[in_ch, out_ch, k_trans]));
        tensors.insert(format!("{prefix}.1.conv.bias"), zeros_t(&[out_ch]));

        for ru in 2..=4 {
            let ru_prefix = format!("{prefix}.{ru}");
            tensors.insert(format!("{ru_prefix}.act1.alpha"), ones_t(&[out_ch]));
            tensors.insert(format!("{ru_prefix}.act1.beta"), ones_t(&[out_ch]));
            tensors.insert(format!("{ru_prefix}.conv1.conv.weight"), rand_t(&[out_ch, out_ch, 7]));
            tensors.insert(format!("{ru_prefix}.conv1.conv.bias"), zeros_t(&[out_ch]));
            tensors.insert(format!("{ru_prefix}.act2.alpha"), ones_t(&[out_ch]));
            tensors.insert(format!("{ru_prefix}.act2.beta"), ones_t(&[out_ch]));
            tensors.insert(format!("{ru_prefix}.conv2.conv.weight"), rand_t(&[out_ch, out_ch, 1]));
            tensors.insert(format!("{ru_prefix}.conv2.conv.bias"), zeros_t(&[out_ch]));
        }
    }

    // 7. Final SnakeBeta & Final Conv
    tensors.insert("5.alpha".to_string(), ones_t(&[96]));
    tensors.insert("5.beta".to_string(), ones_t(&[96]));
    tensors.insert("6.conv.weight".to_string(), rand_t(&[1, 96, 7]));
    tensors.insert("6.conv.bias".to_string(), zeros_t(&[1]));

    WeightLoader::from_tensors(tensors, device)
}

fn create_test_decoder(config: DecoderConfig, device: &Device) -> (Decoder12Hz, WeightLoader) {
    let weight_dir = Path::new("weights/tokenizer");
    if weight_dir.join("codebook.safetensors").exists() {
        let loader = WeightLoader::from_dir(weight_dir, device).expect("load real weights");
        let dec = Decoder12Hz::from_loader(config, &loader, device).expect("create decoder");
        (dec, loader)
    } else {
        let loader = create_synthetic_decoder_loader(&config, device);
        let dec = Decoder12Hz::from_loader(config, &loader, device).expect("create synthetic decoder");
        (dec, loader)
    }
}

#[test]
fn test_streaming_decode_matches_batch_decode_cosine_999() {
    let config = DecoderConfig::realtime();
    let device = Device::Cpu;

    // Test across 5 frames
    let num_frames = 5;
    let frames = generate_deterministic_frames(num_frames, 42);

    let (mut batch_decoder, loader) = create_test_decoder(config.clone(), &device);
    let batch_output = batch_decoder
        .decode_frames(&frames)
        .expect("batch decode_frames failed");

    let mut stream_decoder =
        Decoder12Hz::from_loader(config, &loader, &device).expect("load stream decoder");
    let mut streamed_output = Vec::with_capacity(num_frames * 1920);

    for (i, frame) in frames.iter().enumerate() {
        let chunk = stream_decoder
            .decode_chunk(frame.as_slice())
            .unwrap_or_else(|e| panic!("decode_chunk failed on frame {i}: {e}"));
        assert_eq!(
            chunk.len(),
            1920,
            "Frame {i}: Expected exactly 1920 PCM samples per frame, got {}",
            chunk.len()
        );
        streamed_output.extend(chunk);
    }

    assert_eq!(
        batch_output.len(),
        streamed_output.len(),
        "Batch and streamed total PCM sample lengths must match exactly"
    );
    assert_eq!(streamed_output.len(), num_frames * 1920);

    let cos = cosine_sim(&batch_output, &streamed_output);
    let mse_val = mse(&batch_output, &streamed_output);
    let max_diff: f32 = batch_output
        .iter()
        .zip(streamed_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "Sequential streaming vs batch decode ({} frames):\n  cosine_sim = {:.8}\n  mse = {:.10}\n  max_diff = {:.8}",
        num_frames, cos, mse_val, max_diff
    );

    assert!(
        cos >= 0.999,
        "Cosine similarity must be >= 0.999, got {cos:.8}"
    );
    assert!(
        mse_val < 1e-4,
        "MSE must be < 1e-4, got {mse_val:.10}"
    );
    assert!(
        max_diff < 1e-3,
        "Max difference must be < 1e-3, got {max_diff:.8}"
    );
}

#[test]
fn test_streaming_constant_time_o1_benchmark() {
    let config = DecoderConfig::realtime_with_capacity(30);
    let device = Device::Cpu;

    let num_frames = 15;
    let frames = generate_deterministic_frames(num_frames, 77);

    let (mut decoder, _) = create_test_decoder(config, &device);

    // Warm-up 1 frame
    let _ = decoder.decode_chunk(&frames[0]).expect("warm-up frame");
    decoder.reset_state();

    let mut latencies_us = Vec::with_capacity(num_frames);

    for (i, frame) in frames.iter().enumerate() {
        let t0 = Instant::now();
        let chunk = decoder
            .decode_chunk(frame.as_slice())
            .unwrap_or_else(|e| panic!("frame {i} failed: {e}"));
        let elapsed = t0.elapsed().as_micros() as f64;
        assert_eq!(chunk.len(), 1920);
        latencies_us.push(elapsed);
    }

    // Compute early (frames 1..4), middle (frames 5..9), and late (frames 10..14) averages
    let early_avg: f64 = latencies_us[1..5].iter().sum::<f64>() / 4.0;
    let mid_avg: f64 = latencies_us[5..10].iter().sum::<f64>() / 5.0;
    let late_avg: f64 = latencies_us[10..15].iter().sum::<f64>() / 5.0;

    println!("=== Decoder12Hz Streaming O(1) Latency Profile (15 frames) ===");
    println!("  Early average  (frames 1..4):   {early_avg:.1} µs ({:.2} ms)", early_avg / 1000.0);
    println!("  Mid average    (frames 5..9):   {mid_avg:.1} µs ({:.2} ms)", mid_avg / 1000.0);
    println!("  Late average   (frames 10..14): {late_avg:.1} µs ({:.2} ms)", late_avg / 1000.0);

    let growth_ratio = late_avg / (early_avg.max(1.0));
    println!("  Late/Early growth ratio: {growth_ratio:.3}x (must be < 2.5 for O(1))");

    assert!(
        growth_ratio < 2.5,
        "Execution time grew significantly ({growth_ratio:.2}x), indicating non-O(1) complexity!"
    );
}

#[test]
fn test_streaming_reset_state_zero_leakage() {
    let config = DecoderConfig::realtime();
    let device = Device::Cpu;

    let frames_a = generate_deterministic_frames(3, 101);
    let frames_b = generate_deterministic_frames(3, 202);

    let (mut decoder, _) = create_test_decoder(config, &device);

    // Pass 1: Decode sequence A on fresh state
    let mut pcm_a1 = Vec::new();
    for frame in &frames_a {
        pcm_a1.extend(decoder.decode_chunk(frame).unwrap());
    }

    // Interleave: Decode sequence B (polluting all internal buffers)
    for frame in &frames_b {
        let _ = decoder.decode_chunk(frame).unwrap();
    }

    // Reset state completely
    decoder.reset_state();

    // Pass 2: Decode sequence A again
    let mut pcm_a2 = Vec::new();
    for frame in &frames_a {
        pcm_a2.extend(decoder.decode_chunk(frame).unwrap());
    }

    assert_eq!(pcm_a1.len(), pcm_a2.len());

    let max_diff: f32 = pcm_a1
        .iter()
        .zip(pcm_a2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cos = cosine_sim(&pcm_a1, &pcm_a2);

    println!(
        "State Reset Zero-Leakage Check:\n  cosine = {:.10}\n  max_diff = {:.10}",
        cos, max_diff
    );

    assert_eq!(
        max_diff, 0.0,
        "reset_state() must produce exact identical output with zero leakage (max_diff={max_diff})"
    );
    assert!(
        (cos - 1.0).abs() < 1e-6,
        "Cosine similarity after reset must be 1.0, got {cos}"
    );
}
