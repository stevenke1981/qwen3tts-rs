//! Empirical Challenger M3 Stress Test Suite
//!
//! Rigorously challenges:
//! 1. Multi-length numerical equivalence: Sequential `decode_chunk` vs batch `decode_frames`
//!    across sequence lengths 1, 2, 3, 5, 10, 20, 50, 75, 80 frames (below and above sliding window 72).
//! 2. Cosine similarity >= 0.999 (target 0.9999+), MSE < 1e-4, max difference bounded.
//! 3. Edge token IDs: All 0s, All 2047s, alternating, single-hot codebooks.
//! 4. Invalid token IDs (>2047) and wrong slice lengths (0, 15, 17 tokens).
//! 5. Multi-frame bursts and variable pacing streaming.
//! 6. State reset zero-leakage and session isolation across multi-frame sequences.
//! 7. Sliding window 72 boundary and wrap-around behavior up to 80 frames.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
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

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    a[..n]
        .iter()
        .zip(&b[..n])
        .map(|(x, y)| (*x - *y).abs())
        .fold(0.0f32, f32::max)
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

// ============================================================================
// 1. Multi-Length Numerical Equivalence Testing (1, 2, 3, 5, 10, 20, 50, 75, 80)
// ============================================================================

#[test]
fn challenge_multi_length_streaming_vs_batch_equivalence() {
    let device = Device::Cpu;
    let sequence_lengths = [1, 2, 3, 5, 10, 20, 50, 75, 80];

    for &num_frames in &sequence_lengths {
        let config = DecoderConfig::realtime_with_capacity(num_frames.max(100));
        let (mut batch_decoder, loader) = create_test_decoder(config.clone(), &device);
        let mut stream_decoder =
            Decoder12Hz::from_loader(config, &loader, &device).expect("create stream decoder");

        let frames = generate_deterministic_frames(num_frames, (num_frames * 17 + 101) as u16);

        let batch_output = batch_decoder
            .decode_frames(&frames)
            .unwrap_or_else(|e| panic!("Batch decode failed for length {num_frames}: {e}"));

        let mut streamed_output = Vec::with_capacity(num_frames * 1920);
        for (i, frame) in frames.iter().enumerate() {
            let chunk = stream_decoder
                .decode_chunk(frame.as_slice())
                .unwrap_or_else(|e| panic!("Stream decode failed for length {num_frames}, frame {i}: {e}"));
            assert_eq!(
                chunk.len(),
                1920,
                "Length {num_frames}, frame {i}: expected 1920 samples, got {}",
                chunk.len()
            );
            streamed_output.extend(chunk);
        }

        assert_eq!(
            batch_output.len(),
            streamed_output.len(),
            "Length mismatch at sequence length {num_frames}"
        );
        assert_eq!(streamed_output.len(), num_frames * 1920);

        let cos = cosine_sim(&batch_output, &streamed_output);
        let mse_val = mse(&batch_output, &streamed_output);
        let diff = max_diff(&batch_output, &streamed_output);

        println!(
            "[Sequence length = {:2} frames ({:6} audio samples)] cosine_sim = {:.8}, mse = {:.10}, max_diff = {:.8}",
            num_frames,
            num_frames * 1920,
            cos,
            mse_val,
            diff
        );

        assert!(
            cos >= 0.999,
            "Cosine similarity failed for length {num_frames}: got {cos:.8} < 0.999"
        );
        assert!(
            mse_val < 1e-4,
            "MSE failed for length {num_frames}: got {mse_val:.10} >= 1e-4"
        );
        assert!(
            diff < 1e-3,
            "Max diff failed for length {num_frames}: got {diff:.8} >= 1e-3"
        );
    }
}

// ============================================================================
// 2. Boundary & Edge Token IDs (0, 2047, alternating, single-hot)
// ============================================================================

#[test]
fn challenge_edge_tokens_all_zeros() {
    let device = Device::Cpu;
    let config = DecoderConfig::realtime();
    let num_frames = 10;

    let frames = vec![[0u16; 16]; num_frames];

    let (mut batch_decoder, loader) = create_test_decoder(config.clone(), &device);
    let mut stream_decoder =
        Decoder12Hz::from_loader(config, &loader, &device).expect("create stream decoder");

    let batch_output = batch_decoder.decode_frames(&frames).unwrap();
    let mut streamed_output = Vec::with_capacity(num_frames * 1920);
    for frame in &frames {
        streamed_output.extend(stream_decoder.decode_chunk(frame).unwrap());
    }

    let cos = cosine_sim(&batch_output, &streamed_output);
    let mse_val = mse(&batch_output, &streamed_output);
    let diff = max_diff(&batch_output, &streamed_output);

    println!(
        "[Edge Tokens: ALL ZEROS (10 frames)] cosine_sim = {:.8}, mse = {:.10}, max_diff = {:.8}",
        cos, mse_val, diff
    );

    assert!(cos >= 0.999, "All zeros cosine sim failed: {cos:.8}");
    assert!(mse_val < 1e-4, "All zeros MSE failed: {mse_val:.10}");
}

#[test]
fn challenge_edge_tokens_all_max_2047() {
    let device = Device::Cpu;
    let config = DecoderConfig::realtime();
    let num_frames = 10;

    let frames = vec![[2047u16; 16]; num_frames];

    let (mut batch_decoder, loader) = create_test_decoder(config.clone(), &device);
    let mut stream_decoder =
        Decoder12Hz::from_loader(config, &loader, &device).expect("create stream decoder");

    let batch_output = batch_decoder.decode_frames(&frames).unwrap();
    let mut streamed_output = Vec::with_capacity(num_frames * 1920);
    for frame in &frames {
        streamed_output.extend(stream_decoder.decode_chunk(frame).unwrap());
    }

    let cos = cosine_sim(&batch_output, &streamed_output);
    let mse_val = mse(&batch_output, &streamed_output);
    let diff = max_diff(&batch_output, &streamed_output);

    println!(
        "[Edge Tokens: ALL MAX 2047 (10 frames)] cosine_sim = {:.8}, mse = {:.10}, max_diff = {:.8}",
        cos, mse_val, diff
    );

    assert!(cos >= 0.999, "All max 2047 cosine sim failed: {cos:.8}");
    assert!(mse_val < 1e-4, "All max 2047 MSE failed: {mse_val:.10}");
}

#[test]
fn challenge_edge_tokens_alternating_min_max() {
    let device = Device::Cpu;
    let config = DecoderConfig::realtime();
    let num_frames = 10;

    let mut frames = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let mut frame = [0u16; 16];
        for (j, item) in frame.iter_mut().enumerate() {
            *item = if (i + j) % 2 == 0 { 0 } else { 2047 };
        }
        frames.push(frame);
    }

    let (mut batch_decoder, loader) = create_test_decoder(config.clone(), &device);
    let mut stream_decoder =
        Decoder12Hz::from_loader(config, &loader, &device).expect("create stream decoder");

    let batch_output = batch_decoder.decode_frames(&frames).unwrap();
    let mut streamed_output = Vec::with_capacity(num_frames * 1920);
    for frame in &frames {
        streamed_output.extend(stream_decoder.decode_chunk(frame).unwrap());
    }

    let cos = cosine_sim(&batch_output, &streamed_output);
    let mse_val = mse(&batch_output, &streamed_output);
    let diff = max_diff(&batch_output, &streamed_output);

    println!(
        "[Edge Tokens: ALTERNATING 0/2047 (10 frames)] cosine_sim = {:.8}, mse = {:.10}, max_diff = {:.8}",
        cos, mse_val, diff
    );

    assert!(cos >= 0.999, "Alternating min/max cosine sim failed: {cos:.8}");
    assert!(mse_val < 1e-4, "Alternating min/max MSE failed: {mse_val:.10}");
}

#[test]
fn challenge_edge_tokens_single_hot_codebook() {
    let device = Device::Cpu;
    let config = DecoderConfig::realtime();
    let num_frames = 16;

    // Frame i has non-zero token ONLY in codebook layer (i % 16)
    let mut frames = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let mut frame = [0u16; 16];
        frame[i % 16] = 1337;
        frames.push(frame);
    }

    let (mut batch_decoder, loader) = create_test_decoder(config.clone(), &device);
    let mut stream_decoder =
        Decoder12Hz::from_loader(config, &loader, &device).expect("create stream decoder");

    let batch_output = batch_decoder.decode_frames(&frames).unwrap();
    let mut streamed_output = Vec::with_capacity(num_frames * 1920);
    for frame in &frames {
        streamed_output.extend(stream_decoder.decode_chunk(frame).unwrap());
    }

    let cos = cosine_sim(&batch_output, &streamed_output);
    let mse_val = mse(&batch_output, &streamed_output);

    println!(
        "[Edge Tokens: SINGLE-HOT CODEBOOK (16 frames)] cosine_sim = {:.8}, mse = {:.10}",
        cos, mse_val
    );

    assert!(cos >= 0.999, "Single-hot codebook cosine sim failed: {cos:.8}");
    assert!(mse_val < 1e-4, "Single-hot codebook MSE failed: {mse_val:.10}");
}

// ============================================================================
// 3. Invalid Tokens & Input Shape Violations (Robustness Challenge)
// ============================================================================

#[test]
fn challenge_invalid_token_slice_lengths() {
    let device = Device::Cpu;
    let config = DecoderConfig::realtime();
    let (mut decoder, _) = create_test_decoder(config, &device);

    // 0 tokens
    let res0 = decoder.decode_chunk(&[]);
    assert!(res0.is_err(), "Expected error on empty slice");
    match res0.unwrap_err() {
        Error::Config(msg) => assert!(msg.contains("Expected 16 tokens")),
        e => panic!("Expected Error::Config, got {e:?}"),
    }

    // 15 tokens (1 short)
    let res15 = decoder.decode_chunk(&[0u16; 15]);
    assert!(res15.is_err(), "Expected error on 15 tokens");
    match res15.unwrap_err() {
        Error::Config(msg) => assert!(msg.contains("Expected 16 tokens")),
        e => panic!("Expected Error::Config, got {e:?}"),
    }

    // 17 tokens (1 too many)
    let res17 = decoder.decode_chunk(&[0u16; 17]);
    assert!(res17.is_err(), "Expected error on 17 tokens");
    match res17.unwrap_err() {
        Error::Config(msg) => assert!(msg.contains("Expected 16 tokens")),
        e => panic!("Expected Error::Config, got {e:?}"),
    }
}

#[test]
fn challenge_out_of_bounds_token_id_fallback() {
    let device = Device::Cpu;
    let config = DecoderConfig::realtime();
    let (mut decoder, _) = create_test_decoder(config, &device);

    // Token 2048 and 65535 are out-of-bounds (codebook size = 2048)
    // ParallelCodebook provides fault-tolerant mean embedding fallback
    let mut invalid_frame = [0u16; 16];
    invalid_frame[0] = 2048; // OOB
    invalid_frame[5] = 3000; // OOB
    invalid_frame[15] = 65535; // OOB

    let res = decoder.decode_chunk(&invalid_frame);
    assert!(
        res.is_ok(),
        "Decoder should gracefully degrade to mean embedding on out-of-bounds tokens without crashing"
    );
    let pcm = res.unwrap();
    assert_eq!(pcm.len(), 1920);
}

// ============================================================================
// 4. Multi-Frame Bursts & Variable Pacing Streaming
// ============================================================================

#[test]
fn challenge_variable_pacing_and_burst_chunks() {
    let device = Device::Cpu;
    let config = DecoderConfig::realtime_with_capacity(50);

    let total_frames = 20;
    let frames = generate_deterministic_frames(total_frames, 999);

    let (mut batch_decoder, loader) = create_test_decoder(config.clone(), &device);
    let mut stream_decoder =
        Decoder12Hz::from_loader(config, &loader, &device).expect("create stream decoder");

    let batch_output = batch_decoder.decode_frames(&frames).unwrap();

    // Stream with burst groups: 1 frame, then 3 frames, then 5 frames, then 1, then 10 frames
    let burst_pattern = [1, 3, 5, 1, 10];
    let mut streamed_output = Vec::with_capacity(total_frames * 1920);
    let mut frame_idx = 0;

    for &burst_len in &burst_pattern {
        for _ in 0..burst_len {
            let chunk = stream_decoder.decode_chunk(&frames[frame_idx]).unwrap();
            assert_eq!(chunk.len(), 1920);
            streamed_output.extend(chunk);
            frame_idx += 1;
        }
    }
    assert_eq!(frame_idx, total_frames);

    let cos = cosine_sim(&batch_output, &streamed_output);
    let mse_val = mse(&batch_output, &streamed_output);
    let diff = max_diff(&batch_output, &streamed_output);

    println!(
        "[Variable Pacing Bursts (20 frames)] cosine_sim = {:.8}, mse = {:.10}, max_diff = {:.8}",
        cos, mse_val, diff
    );

    assert!(cos >= 0.999, "Burst streaming cosine failed: {cos:.8}");
    assert!(mse_val < 1e-4, "Burst streaming MSE failed: {mse_val:.10}");
}

// ============================================================================
// 5. Multi-Session Isolation & Long Sequence State Reset Zero-Leakage
// ============================================================================

#[test]
fn challenge_long_sequence_multi_session_isolation() {
    let device = Device::Cpu;
    let config = DecoderConfig::realtime_with_capacity(100);
    let (mut decoder, _) = create_test_decoder(config, &device);

    let seq_len = 25;
    let frames_session_a = generate_deterministic_frames(seq_len, 1111);
    let frames_session_b = generate_deterministic_frames(seq_len, 2222);
    let frames_session_c = generate_deterministic_frames(seq_len, 3333);

    // Session A - Run 1 (fresh)
    let mut pcm_a1 = Vec::with_capacity(seq_len * 1920);
    for frame in &frames_session_a {
        pcm_a1.extend(decoder.decode_chunk(frame).unwrap());
    }

    // Session B - Pollute state heavily
    decoder.reset_state();
    for frame in &frames_session_b {
        let _ = decoder.decode_chunk(frame).unwrap();
    }

    // Session C - Pollute state further
    decoder.reset_state();
    for frame in &frames_session_c {
        let _ = decoder.decode_chunk(frame).unwrap();
    }

    // Session A - Run 2 (after reset)
    decoder.reset_state();
    let mut pcm_a2 = Vec::with_capacity(seq_len * 1920);
    for frame in &frames_session_a {
        pcm_a2.extend(decoder.decode_chunk(frame).unwrap());
    }

    assert_eq!(pcm_a1.len(), pcm_a2.len());
    let diff = max_diff(&pcm_a1, &pcm_a2);
    let cos = cosine_sim(&pcm_a1, &pcm_a2);

    println!(
        "[Multi-Session Isolation (25 frames)] cosine_sim = {:.10}, max_diff = {:.10}",
        cos, diff
    );

    assert_eq!(
        diff, 0.0,
        "Multi-session reset must guarantee zero leakage across 25 frames, got max_diff={diff}"
    );
    assert!(
        (cos - 1.0).abs() < 1e-6,
        "Cosine similarity after multi-session reset must be exactly 1.0, got {cos}"
    );
}

// ============================================================================
// 6. Sliding Window Boundary (72) & Wrap-Around Stress Test
// ============================================================================

#[test]
fn challenge_sliding_window_wrap_around_equivalence() {
    let device = Device::Cpu;
    // Sliding window is 72. Test with 80 frames (frames 73..80 will trigger wrap-around).
    let num_frames = 80;
    let config = DecoderConfig::realtime_with_capacity(120);

    let (mut batch_decoder, loader) = create_test_decoder(config.clone(), &device);
    let mut stream_decoder =
        Decoder12Hz::from_loader(config, &loader, &device).expect("create stream decoder");

    let frames = generate_deterministic_frames(num_frames, 7272);

    let batch_output = batch_decoder
        .decode_frames(&frames)
        .expect("batch decode 80 frames");

    let mut streamed_output = Vec::with_capacity(num_frames * 1920);
    for (i, frame) in frames.iter().enumerate() {
        let chunk = stream_decoder.decode_chunk(frame).unwrap();
        assert_eq!(chunk.len(), 1920, "Frame {i} sample length");
        streamed_output.extend(chunk);
    }

    assert_eq!(batch_output.len(), streamed_output.len());
    assert_eq!(streamed_output.len(), 80 * 1920);

    let cos = cosine_sim(&batch_output, &streamed_output);
    let mse_val = mse(&batch_output, &streamed_output);
    let diff = max_diff(&batch_output, &streamed_output);

    println!(
        "[Sliding Window 72 Boundary: 80 Frames] cosine_sim = {:.8}, mse = {:.10}, max_diff = {:.8}",
        cos, mse_val, diff
    );

    assert!(
        cos >= 0.999,
        "80-frame sliding window wrap-around cosine failed: {cos:.8} < 0.999"
    );
    assert!(
        mse_val < 1e-4,
        "80-frame sliding window wrap-around MSE failed: {mse_val:.10} >= 1e-4"
    );
    assert!(
        diff < 1e-3,
        "80-frame sliding window wrap-around max diff failed: {diff:.8} >= 1e-3"
    );
}
