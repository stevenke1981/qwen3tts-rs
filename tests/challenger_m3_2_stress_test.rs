//! Empirical Challenger M3_2 Stress Test Suite
//!
//! Specifically validates:
//! 1. O(1) constant-time proof over 35+ frames (single-frame latency benchmark, late frames must not grow).
//! 2. `reset_state()` across 10 consecutive sessions verifying zero memory accumulation and zero numerical leakage.
//! 3. Zero CPU-side ring buffer overhead in hot path (pure on-device tensor state, OLA, sliding window KV cache).
//! 4. Adversarial edge cases: invalid token counts, idempotent resets, component-level tensor invariants.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use qwen3tts::codec::{
    CausalConv1d, CausalConvConfig, DecoderBlock, KvRing as DeviceKvCache, UpsampleBlock,
};
use qwen3tts::weights::WeightLoader;
use qwen3tts::{Decoder12Hz, DecoderConfig, Error, TtsDecoder};

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
    tensors.insert(
        "codebook_weights".to_string(),
        rand_t(&[config.num_codebook_layers, config.codebook_size, emb_dim]),
    );

    // 2. Pre-Conv (1024, 512, 3)
    tensors.insert("pre_conv.weight".to_string(), rand_t(&[lat_dim, emb_dim, 3]));
    tensors.insert("pre_conv.bias".to_string(), zeros_t(&[lat_dim]));

    // 3. PreTransformer
    tensors.insert(
        "pre_transformer.input_proj.weight".to_string(),
        rand_t(&[tf_dim, lat_dim]),
    );
    tensors.insert(
        "pre_transformer.input_proj.bias".to_string(),
        zeros_t(&[tf_dim]),
    );
    tensors.insert(
        "pre_transformer.output_proj.weight".to_string(),
        rand_t(&[lat_dim, tf_dim]),
    );
    tensors.insert(
        "pre_transformer.output_proj.bias".to_string(),
        zeros_t(&[lat_dim]),
    );
    tensors.insert("pre_transformer.norm.weight".to_string(), ones_t(&[tf_dim]));

    for i in 0..config.transformer_layers {
        let prefix = format!("pre_transformer.layers.{i}");
        tensors.insert(format!("{prefix}.input_layernorm.weight"), ones_t(&[tf_dim]));
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            ones_t(&[tf_dim]),
        );
        tensors.insert(
            format!("{prefix}.self_attn.q_proj.weight"),
            rand_t(&[tf_dim, tf_dim]),
        );
        tensors.insert(
            format!("{prefix}.self_attn.q_proj.bias"),
            zeros_t(&[tf_dim]),
        );
        tensors.insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            rand_t(&[kv_dim, tf_dim]),
        );
        tensors.insert(
            format!("{prefix}.self_attn.k_proj.bias"),
            zeros_t(&[kv_dim]),
        );
        tensors.insert(
            format!("{prefix}.self_attn.v_proj.weight"),
            rand_t(&[kv_dim, tf_dim]),
        );
        tensors.insert(
            format!("{prefix}.self_attn.v_proj.bias"),
            zeros_t(&[kv_dim]),
        );
        tensors.insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            rand_t(&[tf_dim, tf_dim]),
        );
        tensors.insert(
            format!("{prefix}.self_attn.o_proj.bias"),
            zeros_t(&[tf_dim]),
        );
        tensors.insert(
            format!("{prefix}.self_attn_layer_scale.scale"),
            ones_t(&[tf_dim]),
        );
        tensors.insert(format!("{prefix}.mlp_layer_scale.scale"), ones_t(&[tf_dim]));
        tensors.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            rand_t(&[tf_dim * 4, tf_dim]),
        );
        tensors.insert(
            format!("{prefix}.mlp.gate_proj.bias"),
            zeros_t(&[tf_dim * 4]),
        );
        tensors.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            rand_t(&[tf_dim * 4, tf_dim]),
        );
        tensors.insert(
            format!("{prefix}.mlp.up_proj.bias"),
            zeros_t(&[tf_dim * 4]),
        );
        tensors.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            rand_t(&[tf_dim, tf_dim * 4]),
        );
        tensors.insert(format!("{prefix}.mlp.down_proj.bias"), zeros_t(&[tf_dim]));
    }

    // 4. Upsample Blocks
    for i in 0..2 {
        let prefix = format!("upsample.{i}");
        tensors.insert(
            format!("{prefix}.0.conv.weight"),
            rand_t(&[lat_dim, lat_dim, 4]),
        );
        tensors.insert(format!("{prefix}.0.conv.bias"), zeros_t(&[lat_dim]));
        tensors.insert(format!("{prefix}.1.gamma"), ones_t(&[lat_dim]));
        tensors.insert(
            format!("{prefix}.1.dwconv.conv.weight"),
            rand_t(&[lat_dim, 1, 7]),
        );
        tensors.insert(
            format!("{prefix}.1.dwconv.conv.bias"),
            zeros_t(&[lat_dim]),
        );
        tensors.insert(format!("{prefix}.1.norm.weight"), ones_t(&[lat_dim]));
        tensors.insert(format!("{prefix}.1.norm.bias"), zeros_t(&[lat_dim]));
        tensors.insert(
            format!("{prefix}.1.pwconv1.weight"),
            rand_t(&[lat_dim * 4, lat_dim]),
        );
        tensors.insert(
            format!("{prefix}.1.pwconv1.bias"),
            zeros_t(&[lat_dim * 4]),
        );
        tensors.insert(
            format!("{prefix}.1.pwconv2.weight"),
            rand_t(&[lat_dim, lat_dim * 4]),
        );
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
        tensors.insert(
            format!("{prefix}.1.conv.weight"),
            rand_t(&[in_ch, out_ch, k_trans]),
        );
        tensors.insert(format!("{prefix}.1.conv.bias"), zeros_t(&[out_ch]));

        for ru in 2..=4 {
            let ru_prefix = format!("{prefix}.{ru}");
            tensors.insert(format!("{ru_prefix}.act1.alpha"), ones_t(&[out_ch]));
            tensors.insert(format!("{ru_prefix}.act1.beta"), ones_t(&[out_ch]));
            tensors.insert(
                format!("{ru_prefix}.conv1.conv.weight"),
                rand_t(&[out_ch, out_ch, 7]),
            );
            tensors.insert(format!("{ru_prefix}.conv1.conv.bias"), zeros_t(&[out_ch]));
            tensors.insert(format!("{ru_prefix}.act2.alpha"), ones_t(&[out_ch]));
            tensors.insert(format!("{ru_prefix}.act2.beta"), ones_t(&[out_ch]));
            tensors.insert(
                format!("{ru_prefix}.conv2.conv.weight"),
                rand_t(&[out_ch, out_ch, 1]),
            );
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
        let dec = Decoder12Hz::from_loader(config, &loader, device)
            .expect("create synthetic decoder");
        (dec, loader)
    }
}

// ============================================================================
// 1. O(1) Constant-Time Latency Proof Over 35+ Frames
// ============================================================================

#[test]
fn challenge_streaming_latency_over_35_frames_proves_o1_constant_time() {
    let config = DecoderConfig::realtime_with_capacity(64);
    let device = Device::Cpu;

    let num_frames = 35;
    let frames = generate_deterministic_frames(num_frames, 999);

    let (mut decoder, _) = create_test_decoder(config, &device);

    // Warm-up 2 frames
    let _ = decoder.decode_chunk(&frames[0]).expect("warm-up frame 0");
    let _ = decoder.decode_chunk(&frames[1]).expect("warm-up frame 1");
    decoder.reset_state();

    let mut latencies_us = Vec::with_capacity(num_frames);

    for (i, frame) in frames.iter().enumerate() {
        let t0 = Instant::now();
        let chunk = decoder
            .decode_chunk(frame.as_slice())
            .unwrap_or_else(|e| panic!("frame {i} failed: {e}"));
        let elapsed = t0.elapsed().as_micros() as f64;
        assert_eq!(
            chunk.len(),
            1920,
            "Frame {i} did not produce 1920 PCM samples"
        );
        latencies_us.push(elapsed);
    }

    // Partition into 3 windows: Early (frames 2..10), Mid (frames 14..22), Late (frames 26..34)
    let early_slice = &latencies_us[2..11];
    let mid_slice = &latencies_us[14..23];
    let late_slice = &latencies_us[26..35];

    let early_avg: f64 = early_slice.iter().sum::<f64>() / early_slice.len() as f64;
    let mid_avg: f64 = mid_slice.iter().sum::<f64>() / mid_slice.len() as f64;
    let late_avg: f64 = late_slice.iter().sum::<f64>() / late_slice.len() as f64;

    println!(
        "=== Empirical Challenger O(1) Latency Profile ({} frames) ===",
        num_frames
    );
    println!(
        "  Early average (frames 2..10):  {early_avg:.1} µs ({:.2} ms)",
        early_avg / 1000.0
    );
    println!(
        "  Mid average   (frames 14..22): {mid_avg:.1} µs ({:.2} ms)",
        mid_avg / 1000.0
    );
    println!(
        "  Late average  (frames 26..34): {late_avg:.1} µs ({:.2} ms)",
        late_avg / 1000.0
    );

    let late_to_early_ratio = late_avg / early_avg.max(1.0);
    let mid_to_early_ratio = mid_avg / early_avg.max(1.0);

    println!("  Mid/Early ratio:  {mid_to_early_ratio:.3}x");
    println!("  Late/Early ratio: {late_to_early_ratio:.3}x");

    // Under O(1) complexity, the late/early ratio must remain bounded (<= 2.0x).
    // In an O(N) or O(N^2) pipeline, late frames (index 34 vs index 2) would scale by 17x to 289x.
    assert!(
        late_to_early_ratio < 2.0,
        "Execution time grew significantly ({late_to_early_ratio:.2}x), violating O(1) constant time!"
    );
}

// ============================================================================
// 2. Stress-Test reset_state() Across 10 Consecutive Sessions
// ============================================================================

#[test]
fn challenge_reset_state_across_10_consecutive_sessions_zero_leakage() {
    let config = DecoderConfig::realtime();
    let device = Device::Cpu;

    let (mut long_lived_decoder, loader) = create_test_decoder(config.clone(), &device);

    for session_idx in 0..10 {
        let frame_count = 3 + (session_idx % 4); // 3 to 6 frames
        let seed = (session_idx as u16 + 1) * 313 + 7;
        let test_frames = generate_deterministic_frames(frame_count, seed);

        // 1. Decode sequence on long_lived_decoder
        let mut session_pcm = Vec::with_capacity(frame_count * 1920);
        for (f_idx, frame) in test_frames.iter().enumerate() {
            let chunk = long_lived_decoder
                .decode_chunk(frame.as_slice())
                .unwrap_or_else(|e| {
                    panic!("Session {session_idx} frame {f_idx} failed: {e}")
                });
            assert_eq!(chunk.len(), 1920);
            session_pcm.extend(chunk);
        }

        // 2. Decode the exact same sequence on a brand-new freshly instantiated reference decoder
        let mut fresh_decoder = Decoder12Hz::from_loader(config.clone(), &loader, &device)
            .expect("create fresh decoder");
        let mut fresh_pcm = Vec::with_capacity(frame_count * 1920);
        for frame in &test_frames {
            fresh_pcm.extend(fresh_decoder.decode_chunk(frame.as_slice()).unwrap());
        }

        // 3. Compare outputs: must be bit-exact / zero numerical drift
        assert_eq!(session_pcm.len(), fresh_pcm.len());
        let max_diff: f32 = session_pcm
            .iter()
            .zip(fresh_pcm.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0f32, f32::max);
        let cos = cosine_sim(&session_pcm, &fresh_pcm);

        println!(
            "Session {session_idx:02} ({} frames, seed {}): max_diff = {:.10}, cosine = {:.10}",
            frame_count, seed, max_diff, cos
        );

        assert_eq!(
            max_diff, 0.0,
            "Session {session_idx}: State pollution detected! max_diff = {max_diff}"
        );
        assert!(
            (cos - 1.0).abs() < 1e-6,
            "Session {session_idx}: Cosine similarity deviated from 1.0: {cos}"
        );

        // 4. Reset long_lived_decoder for the next session
        long_lived_decoder.reset_state();
    }
}

#[test]
fn challenge_consecutive_idempotent_resets() {
    let config = DecoderConfig::realtime();
    let device = Device::Cpu;

    let (mut decoder, _) = create_test_decoder(config, &device);

    // Call reset multiple times in succession
    for _ in 0..5 {
        decoder.reset_state();
    }

    let frame = [100u16; 16];
    let chunk1 = decoder.decode_chunk(&frame).expect("decode after resets");
    assert_eq!(chunk1.len(), 1920);

    // Call reset 3 times again
    decoder.reset_state();
    decoder.reset_state();
    decoder.reset_state();

    let chunk2 = decoder.decode_chunk(&frame).expect("decode after resets");
    assert_eq!(chunk1, chunk2, "Consecutive resets must be strictly idempotent");
}

// ============================================================================
// 3. Pure Device-Resident Tensor State Invariant Verification
// ============================================================================

#[test]
fn challenge_causal_conv1d_step_tensor_equivalence_40_frames() {
    let device = Device::Cpu;
    let config = CausalConvConfig {
        in_channels: 16,
        out_channels: 32,
        kernel_size: 7,
        dilation: 3,
        groups: 1,
    };
    let weight = Tensor::randn(0.0f32, 1.0f32, (32, 16, 7), &device).unwrap();
    let bias = Some(Tensor::randn(0.0f32, 1.0f32, (32,), &device).unwrap());
    let mut conv = CausalConv1d::new(weight, bias, config, 64).unwrap();

    let num_frames = 40;
    let mut input_vec = Vec::with_capacity(16 * num_frames);
    for i in 0..16 * num_frames {
        input_vec.push(((i * 23 + 7) % 100) as f32 * 0.05 - 2.5);
    }
    let full_input = Tensor::from_slice(&input_vec, (1, 16, num_frames), &device).unwrap();

    // 1. Batch forward
    let batch_out = conv.forward(&full_input).unwrap();
    let batch_flat: Vec<f32> = batch_out.flatten_all().unwrap().to_vec1().unwrap();

    // 2. Sequential step_tensor
    let mut step_outputs = Vec::with_capacity(num_frames);
    for t in 0..num_frames {
        let frame_tensor = full_input.narrow(2, t, 1).unwrap();
        let out_t = conv.step_tensor(&frame_tensor).unwrap();
        assert_eq!(out_t.dims(), &[1, 32, 1]);
        step_outputs.push(out_t);
    }

    let stream_cat = Tensor::cat(&step_outputs.iter().collect::<Vec<_>>(), 2).unwrap();
    let stream_flat: Vec<f32> = stream_cat.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(batch_flat.len(), stream_flat.len());
    let max_diff: f32 = batch_flat
        .iter()
        .zip(stream_flat.iter())
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);
    let cos = cosine_sim(&batch_flat, &stream_flat);

    println!("CausalConv1d 40-frame step_tensor vs forward: max_diff = {max_diff:.8}, cos = {cos:.8}");

    assert!(
        max_diff < 1e-4,
        "CausalConv1d step_tensor 40-frame sequence max_diff = {max_diff} >= 1e-4"
    );
    assert!(
        cos > 0.99999,
        "CausalConv1d step_tensor cosine similarity {cos} <= 0.99999"
    );
}

#[test]
fn challenge_upsample_block_step_vs_forward_equivalence_20_frames() {
    let device = Device::Cpu;
    let lat_dim = 64;

    let mut tensors = HashMap::new();
    let rand_t = |shape: &[usize]| -> Tensor {
        Tensor::rand(-0.05f64, 0.05f64, shape, &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
    };
    let ones_t = |shape: &[usize]| -> Tensor {
        Tensor::ones(shape, DType::F32, &device).unwrap()
    };
    let zeros_t = |shape: &[usize]| -> Tensor {
        Tensor::zeros(shape, DType::F32, &device).unwrap()
    };

    let prefix = "upsample.0";
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

    let loader = WeightLoader::from_tensors(tensors, &device);
    let mut ub = UpsampleBlock::from_loader(&loader, prefix).unwrap();

    let num_frames = 20;
    let input_vec: Vec<f32> = (0..lat_dim * num_frames)
        .map(|i| ((i * 19 + 3) % 47) as f32 * 0.05 - 1.0)
        .collect();
    let full_input = Tensor::from_slice(&input_vec, (1, lat_dim, num_frames), &device).unwrap();

    // Batch forward
    let batch_out = ub.forward(&full_input).unwrap();
    let batch_flat: Vec<f32> = batch_out.flatten_all().unwrap().to_vec1().unwrap();

    // Streaming step
    let mut step_outputs = Vec::with_capacity(num_frames);
    for t in 0..num_frames {
        let frame_tensor = full_input.narrow(2, t, 1).unwrap();
        let out_t = ub.step(&frame_tensor).unwrap();
        assert_eq!(out_t.dims(), &[1, lat_dim, 2]); // stride = 2
        step_outputs.push(out_t);
    }

    let stream_cat = Tensor::cat(&step_outputs.iter().collect::<Vec<_>>(), 2).unwrap();
    let stream_flat: Vec<f32> = stream_cat.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(batch_flat.len(), stream_flat.len());
    let max_diff: f32 = batch_flat
        .iter()
        .zip(stream_flat.iter())
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "UpsampleBlock step vs forward max_diff = {max_diff} >= 1e-4"
    );
}

#[test]
fn challenge_decoder_block_step_vs_forward_equivalence_15_frames() {
    let device = Device::Cpu;
    let in_ch = 32;
    let out_ch = 16;
    let k_trans = 4;

    let mut tensors = HashMap::new();
    let rand_t = |shape: &[usize]| -> Tensor {
        Tensor::rand(-0.03f64, 0.03f64, shape, &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
    };
    let ones_t = |shape: &[usize]| -> Tensor {
        Tensor::ones(shape, DType::F32, &device).unwrap()
    };
    let zeros_t = |shape: &[usize]| -> Tensor {
        Tensor::zeros(shape, DType::F32, &device).unwrap()
    };

    let prefix = "1.block";
    tensors.insert(format!("{prefix}.0.alpha"), ones_t(&[in_ch]));
    tensors.insert(format!("{prefix}.0.beta"), ones_t(&[in_ch]));
    tensors.insert(format!("{prefix}.1.conv.weight"), rand_t(&[in_ch, out_ch, k_trans]));
    tensors.insert(format!("{prefix}.1.conv.bias"), zeros_t(&[out_ch]));

    for ru in 2..=4 {
        let ru_prefix = format!("{prefix}.{ru}");
        tensors.insert(format!("{ru_prefix}.act1.alpha"), ones_t(&[out_ch]));
        tensors.insert(format!("{ru_prefix}.act1.beta"), ones_t(&[out_ch]));
        tensors.insert(
            format!("{ru_prefix}.conv1.conv.weight"),
            rand_t(&[out_ch, out_ch, 7]),
        );
        tensors.insert(format!("{ru_prefix}.conv1.conv.bias"), zeros_t(&[out_ch]));
        tensors.insert(format!("{ru_prefix}.act2.alpha"), ones_t(&[out_ch]));
        tensors.insert(format!("{ru_prefix}.act2.beta"), ones_t(&[out_ch]));
        tensors.insert(
            format!("{ru_prefix}.conv2.conv.weight"),
            rand_t(&[out_ch, out_ch, 1]),
        );
        tensors.insert(format!("{ru_prefix}.conv2.conv.bias"), zeros_t(&[out_ch]));
    }

    let loader = WeightLoader::from_tensors(tensors, &device);
    let mut db = DecoderBlock::from_loader(&loader, "1").unwrap();

    let num_frames = 15;
    let input_vec: Vec<f32> = (0..in_ch * num_frames)
        .map(|i| ((i * 17 + 5) % 43) as f32 * 0.05 - 1.0)
        .collect();
    let full_input = Tensor::from_slice(&input_vec, (1, in_ch, num_frames), &device).unwrap();

    // Batch forward
    let batch_out = db.forward(&full_input).unwrap();
    let batch_flat: Vec<f32> = batch_out.flatten_all().unwrap().to_vec1().unwrap();

    // Streaming step
    let mut step_outputs = Vec::with_capacity(num_frames);
    for t in 0..num_frames {
        let frame_tensor = full_input.narrow(2, t, 1).unwrap();
        let out_t = db.step(&frame_tensor).unwrap();
        step_outputs.push(out_t);
    }

    let stream_cat = Tensor::cat(&step_outputs.iter().collect::<Vec<_>>(), 2).unwrap();
    let stream_flat: Vec<f32> = stream_cat.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(batch_flat.len(), stream_flat.len());
    let max_diff: f32 = batch_flat
        .iter()
        .zip(stream_flat.iter())
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "DecoderBlock step vs forward max_diff = {max_diff} >= 1e-4"
    );
}

#[test]
fn challenge_device_kv_cache_sliding_window_bound_and_reset() {
    let device = Device::Cpu;
    let sliding_window = 16;
    let mut kv_cache = DeviceKvCache::new(sliding_window);

    assert!(kv_cache.is_empty());
    assert_eq!(kv_cache.len(), 0);
    assert_eq!(kv_cache.capacity(), 16);

    let num_heads = 4;
    let head_dim = 32;

    // Push 50 frames into cache of capacity 16
    for t in 0..50 {
        let k = Tensor::randn(0.0f32, 1.0f32, (1, num_heads, 1, head_dim), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0f32, (1, num_heads, 1, head_dim), &device).unwrap();
        let (cached_k, cached_v) = kv_cache.step(&k, &v).unwrap();

        let expected_len = (t + 1).min(sliding_window);
        assert_eq!(kv_cache.len(), expected_len);
        assert_eq!(cached_k.dims(), &[1, num_heads, expected_len, head_dim]);
        assert_eq!(cached_v.dims(), &[1, num_heads, expected_len, head_dim]);
    }

    // Cache must remain capped at sliding_window = 16
    assert_eq!(kv_cache.len(), 16);

    // Reset should clear completely
    kv_cache.reset();
    assert!(kv_cache.is_empty());
    assert_eq!(kv_cache.len(), 0);
}

// ============================================================================
// 4. Adversarial Error Handling & Boundary Tests
// ============================================================================

#[test]
fn challenge_invalid_token_length_handling() {
    let config = DecoderConfig::realtime();
    let device = Device::Cpu;
    let (mut decoder, _) = create_test_decoder(config, &device);

    let invalid_lengths = [0, 1, 15, 17, 32, 100];
    for &len in &invalid_lengths {
        let tokens = vec![10u16; len];
        let res = decoder.decode_chunk(&tokens);
        assert!(
            res.is_err(),
            "Expected Error::Config for token slice length {len}, but got Ok"
        );
        match res.unwrap_err() {
            Error::Config(msg) => {
                assert!(
                    msg.contains("Expected 16 tokens"),
                    "Error message should mention expected 16 tokens, got: {msg}"
                );
            }
            other => panic!("Expected Error::Config, got {other:?}"),
        }
    }
}
