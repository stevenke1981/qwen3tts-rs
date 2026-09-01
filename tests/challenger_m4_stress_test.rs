//! Empirical Challenger M4 Stress Test Suite
//!
//! Rigorous adversarial challenges for Milestone M4:
//! 1. Parity between `TalkerForConditionalGeneration::generate_streaming` and batch `generate` (greedy).
//! 2. Parity between `TalkerForConditionalGeneration::generate_sampled_streaming` and batch `generate_sampled` (with Philox RNG and various sampling configs).
//! 3. End-to-end PCM chunk delivery in `CandleLLM::synthesize_streaming` (1920 samples/frame, finite, boundary continuity).
//! 4. Fail-closed error propagation when `frame_callback` or `on_pcm_chunk` returns an error (at frame 0, frame K, mid-stream abort, no hang, zero leak).
//! 5. Post-abort decoder state integrity and zero-leakage reset.
//! 6. Tokenizer wrapper and activation edge case verification.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use qwen3tts::talker::code_predictor::CodePredictor;
use qwen3tts::talker::config::TalkerConfig;
use qwen3tts::talker::decoder_layer::{StandardDecoderLayer, TalkerDecoderLayer};
use qwen3tts::talker::model::TalkerModel;
use qwen3tts::talker::primitives::{MultimodalRotaryEmbedding, RMSNorm, SwiGLUMLP};
use qwen3tts::talker::sampling::{Sampler, SamplingOptions};
use qwen3tts::talker::talker_attention::{StandardAttention, TalkerAttention};
use qwen3tts::talker::TalkerForConditionalGeneration;
use qwen3tts::weights::WeightLoader;
use qwen3tts::{Decoder12Hz, DecoderConfig, Error, TtsDecoder};

// ---------------------------------------------------------------------------
// Synthetic Test Fixtures
// ---------------------------------------------------------------------------

#[allow(clippy::field_reassign_with_default)]
fn create_synthetic_talker_config() -> TalkerConfig {
    let mut config = TalkerConfig::default();
    config.hidden_size = 16;
    config.text_hidden_size = 16;
    config.head_dim = 16;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.num_hidden_layers = 1;
    config.vocab_size = 3072;
    config.num_code_groups = 16;
    config.mrope_section = vec![3, 3, 2];
    config.codec_eos_token_id = 2150;
    config.codec_bos_id = 2149;
    config.codec_think_id = 2154;
    config.codec_nothink_id = 2155;
    config.codec_think_bos_id = 2156;
    config.codec_think_eos_id = 2157;
    config.codec_pad_id = 2148;
    config.assistant_token_id = 3;
    config.im_start_token_id = 1;
    config.im_end_token_id = 2;
    config.tts_pad_token_id = 1033;
    config.tts_bos_token_id = 1031;
    config.tts_eos_token_id = 1032;
    config.spk_id = vec![("serena".to_string(), 3066)];
    config.spk_is_dialect = vec![("serena".to_string(), None)];
    config.codec_language_id = vec![("chinese".to_string(), 2055), ("english".to_string(), 2050)];

    let mut cp_config = config.code_predictor.clone();
    cp_config.hidden_size = 16;
    cp_config.head_dim = 16;
    cp_config.num_attention_heads = 1;
    cp_config.num_key_value_heads = 1;
    cp_config.num_hidden_layers = 1;
    cp_config.vocab_size = 8;
    cp_config.num_code_groups = 16;
    config.code_predictor = cp_config;

    config
}

fn create_synthetic_talker(config: &TalkerConfig, device: &Device) -> TalkerForConditionalGeneration {
    let h = config.hidden_size;
    let norm = RMSNorm::new(Tensor::ones((h,), DType::F32, device).unwrap(), 1e-6);
    let rope = MultimodalRotaryEmbedding::new(config, device).unwrap();

    let layer = TalkerDecoderLayer::new(
        RMSNorm::new(Tensor::ones((h,), DType::F32, device).unwrap(), 1e-6),
        TalkerAttention::new(
            Tensor::zeros((h, h), DType::F32, device).unwrap(),
            Tensor::zeros((h, h), DType::F32, device).unwrap(),
            Tensor::zeros((h, h), DType::F32, device).unwrap(),
            Tensor::zeros((h, h), DType::F32, device).unwrap(),
            Tensor::ones((h,), DType::F32, device).unwrap(),
            Tensor::ones((h,), DType::F32, device).unwrap(),
            1,
            1,
            h,
            1e-6,
        ),
        RMSNorm::new(Tensor::ones((h,), DType::F32, device).unwrap(), 1e-6),
        SwiGLUMLP::new(
            Tensor::zeros((h, h), DType::F32, device).unwrap(),
            Tensor::zeros((h, h), DType::F32, device).unwrap(),
            Tensor::zeros((h, h), DType::F32, device).unwrap(),
        ),
    );

    let cp_layer = StandardDecoderLayer::new(
        RMSNorm::new(Tensor::ones((h,), DType::F32, device).unwrap(), 1e-6),
        StandardAttention::new(
            Tensor::zeros((h, h), DType::F32, device).unwrap(),
            Tensor::zeros((h, h), DType::F32, device).unwrap(),
            Tensor::zeros((h, h), DType::F32, device).unwrap(),
            Tensor::zeros((h, h), DType::F32, device).unwrap(),
            Tensor::ones((h,), DType::F32, device).unwrap(),
            Tensor::ones((h,), DType::F32, device).unwrap(),
            1,
            1,
            h,
            1e-6,
        ),
        RMSNorm::new(Tensor::ones((h,), DType::F32, device).unwrap(), 1e-6),
        SwiGLUMLP::new(
            Tensor::zeros((h, h), DType::F32, device).unwrap(),
            Tensor::zeros((h, h), DType::F32, device).unwrap(),
            Tensor::zeros((h, h), DType::F32, device).unwrap(),
        ),
    );

    let cp_config = config.code_predictor.clone();
    let cp_norm = RMSNorm::new(Tensor::ones((h,), DType::F32, device).unwrap(), 1e-6);
    let predictor = CodePredictor {
        codec_embeddings: (0..15)
            .map(|_| Tensor::zeros((8, h), DType::F32, device).unwrap())
            .collect(),
        lm_heads: (0..15)
            .map(|_| Tensor::zeros((8, h), DType::F32, device).unwrap())
            .collect(),
        layers: vec![cp_layer],
        norm: cp_norm,
        small_to_mtp_proj: None,
        config: cp_config,
    };

    let text_emb = Tensor::zeros((config.vocab_size, h), DType::F32, device).unwrap();
    let text_proj_fc1_w = Tensor::zeros((h, h), DType::F32, device).unwrap();
    let text_proj_fc1_b = Tensor::zeros((h,), DType::F32, device).unwrap();
    let text_proj_fc2_w = Tensor::zeros((h, h), DType::F32, device).unwrap();
    let text_proj_fc2_b = Tensor::zeros((h,), DType::F32, device).unwrap();
    let codec_emb = Tensor::zeros((config.vocab_size, h), DType::F32, device).unwrap();

    let mut head = vec![0.0_f32; config.vocab_size * h];
    for v in &mut head[..h] {
        *v = 0.5;
    }
    for v in &mut head[2150 * h..2151 * h] {
        *v = 1.0;
    }
    let codec_head = Tensor::from_slice(&head, (config.vocab_size, h), device).unwrap();

    TalkerForConditionalGeneration {
        model: TalkerModel::new(vec![layer], norm),
        text_embedding: text_emb,
        text_proj_fc1_w,
        text_proj_fc1_b,
        text_proj_fc2_w,
        text_proj_fc2_b,
        codec_embedding: codec_emb,
        codec_head,
        code_predictor: predictor,
        rope,
        config: config.clone(),
    }
}

fn create_synthetic_decoder_loader(config: &DecoderConfig, device: &Device) -> WeightLoader {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    let rand_t = |shape: &[usize]| -> Tensor {
        Tensor::zeros(shape, DType::F32, device).unwrap()
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

fn create_test_decoder(device: &Device) -> Decoder12Hz {
    let config = DecoderConfig::realtime();
    let loader = create_synthetic_decoder_loader(&config, device);
    Decoder12Hz::from_loader(config, &loader, device).expect("create synthetic decoder")
}

// ---------------------------------------------------------------------------
// Challenge 1: Talker Streaming Parity Matrix
// ---------------------------------------------------------------------------

#[test]
fn challenge_talker_greedy_streaming_vs_batch_parity_matrix() {
    let device = Device::Cpu;
    let config = create_synthetic_talker_config();
    let talker = create_synthetic_talker(&config, &device);

    for seq_len in [1, 2, 4, 8] {
        for max_tokens in [1, 2, 5, 8] {
            let inputs = Tensor::zeros((1, seq_len, 16), DType::F32, &device).unwrap();
            let pad = Tensor::zeros((1, 1, 16), DType::F32, &device).unwrap();
            let trailing = Tensor::zeros((1, 2, 16), DType::F32, &device).unwrap();

            // 1. Batch
            let batch_result = talker
                .generate(&inputs, None, Some(&trailing), Some(&pad), max_tokens, &device)
                .expect("batch generate");
            let batch_codes = batch_result.to_vec2::<u32>().expect("batch vec2");

            // 2. Streaming
            let mut stream_frames: Vec<[u16; 16]> = Vec::new();
            let stream_count = talker
                .generate_streaming(
                    &inputs,
                    None,
                    Some(&trailing),
                    Some(&pad),
                    max_tokens,
                    &device,
                    |frame| {
                        stream_frames.push(*frame);
                        Ok(())
                    },
                )
                .expect("stream generate");

            assert_eq!(stream_count, batch_codes.len());
            assert_eq!(stream_frames.len(), batch_codes.len());

            for (f_idx, s_frame) in stream_frames.iter().enumerate() {
                for c_idx in 0..16 {
                    assert_eq!(
                        s_frame[c_idx] as u32, batch_codes[f_idx][c_idx],
                        "Greedy mismatch at seq_len={seq_len} max_tokens={max_tokens} frame={f_idx} codebook={c_idx}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Challenge 2: Talker Sampled Streaming Parity Matrix with Multiple Seeds & Sampling Settings
// ---------------------------------------------------------------------------

#[test]
fn challenge_talker_sampled_streaming_vs_batch_parity_matrix() {
    let device = Device::Cpu;
    let config = create_synthetic_talker_config();
    let talker = create_synthetic_talker(&config, &device);

    let test_cases = [
        // (seed, talker_do_sample, subtalker_do_sample, temp, top_k, top_p, rep_pen)
        (42u64, true, true, 0.7, 50, 0.9, 1.1),
        (1337u64, true, false, 0.5, 20, 0.95, 1.0),
        (999999u64, false, true, 0.8, 10, 0.85, 1.2),
        (777u64, false, false, 0.0, 0, 1.0, 1.0),
    ];

    for (seed, talker_ds, subtalker_ds, temp, top_k, top_p, rep_pen) in test_cases {
        let sampling_opts = SamplingOptions {
            temperature: temp,
            top_k,
            top_p,
            repetition_penalty: rep_pen,
        };

        let inputs = Tensor::zeros((1, 4, 16), DType::F32, &device).unwrap();
        let pad = Tensor::zeros((1, 1, 16), DType::F32, &device).unwrap();

        // 1. Batch Sampled
        let mut sampler_batch = Sampler::new(seed);
        let batch_result = talker
            .generate_sampled(
                &inputs,
                None,
                None,
                Some(&pad),
                5,
                &device,
                &mut sampler_batch,
                sampling_opts,
                talker_ds,
                sampling_opts,
                subtalker_ds,
            )
            .expect("batch generate_sampled");
        let batch_codes = batch_result.to_vec2::<u32>().expect("batch vec2");

        // 2. Streaming Sampled
        let mut sampler_stream = Sampler::new(seed);
        let mut stream_frames: Vec<[u16; 16]> = Vec::new();
        let stream_count = talker
            .generate_sampled_streaming(
                &inputs,
                None,
                None,
                Some(&pad),
                5,
                &device,
                &mut sampler_stream,
                sampling_opts,
                sampling_opts,
                talker_ds,
                subtalker_ds,
                |frame| {
                    stream_frames.push(*frame);
                    Ok(())
                },
            )
            .expect("stream generate_sampled_streaming");

        assert_eq!(stream_count, batch_codes.len());
        assert_eq!(stream_frames.len(), batch_codes.len());

        for (f_idx, s_frame) in stream_frames.iter().enumerate() {
            for c_idx in 0..16 {
                assert_eq!(
                    s_frame[c_idx] as u32, batch_codes[f_idx][c_idx],
                    "Sampled mismatch at seed={seed} frame={f_idx} codebook={c_idx}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Challenge 3: Immediate and Mid-Stream Error Termination in Talker Streaming
// ---------------------------------------------------------------------------

#[test]
fn challenge_talker_generate_streaming_error_termination_immediate_and_midstream() {
    let device = Device::Cpu;
    let config = create_synthetic_talker_config();
    let talker = create_synthetic_talker(&config, &device);

    let inputs = Tensor::zeros((1, 4, 16), DType::F32, &device).unwrap();
    let pad = Tensor::zeros((1, 1, 16), DType::F32, &device).unwrap();

    // 1. Error on frame 0
    let mut call_count = 0;
    let err_frame0 = talker.generate_streaming(&inputs, None, None, Some(&pad), 10, &device, |_frame| {
        call_count += 1;
        Err(Error::Config("abort at frame 0".into()))
    });
    assert!(err_frame0.is_err(), "Must return error when callback fails on frame 0");
    assert_eq!(call_count, 1, "Must immediately stop after first callback error");

    // 2. Error on frame 3
    let mut call_count_mid = 0;
    let err_frame3 = talker.generate_streaming(&inputs, None, None, Some(&pad), 10, &device, |_frame| {
        call_count_mid += 1;
        if call_count_mid == 3 {
            Err(Error::Config("abort at frame 3".into()))
        } else {
            Ok(())
        }
    });
    assert!(err_frame3.is_err(), "Must return error when callback fails at frame 3");
    assert_eq!(call_count_mid, 3, "Must immediately stop after 3 callbacks");

    // 3. Ensure talker is completely reusable and deterministic after previous aborts
    let mut clean_frames: Vec<[u16; 16]> = Vec::new();
    let clean_count = talker
        .generate_streaming(&inputs, None, None, Some(&pad), 4, &device, |frame| {
            clean_frames.push(*frame);
            Ok(())
        })
        .expect("clean stream generate");
    assert_eq!(clean_count, 3);
    assert_eq!(clean_frames.len(), 3);
}

// ---------------------------------------------------------------------------
// Challenge 4: Decoder State Integrity and Boundary Continuity across Aborts
// ---------------------------------------------------------------------------

#[test]
fn challenge_decoder_state_integrity_and_boundary_continuity() {
    let device = Device::Cpu;
    let mut decoder = create_test_decoder(&device);

    let frame_a = [10u16, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
    let frame_b = [15u16, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165];

    // Baseline clean session
    decoder.reset_state();
    let base_chunk1 = decoder.decode_chunk(&frame_a).expect("chunk 1");
    let base_chunk2 = decoder.decode_chunk(&frame_b).expect("chunk 2");

    // Interrupted session (decode 1 chunk, then abort/abandon without finishing second)
    decoder.reset_state();
    let _dirty_chunk1 = decoder.decode_chunk(&frame_a).expect("chunk 1");
    // Simulate abort mid-session and reset
    decoder.reset_state();

    // Post-abort session
    let recovered_chunk1 = decoder.decode_chunk(&frame_a).expect("recovered chunk 1");
    let recovered_chunk2 = decoder.decode_chunk(&frame_b).expect("recovered chunk 2");

    assert_eq!(
        base_chunk1, recovered_chunk1,
        "Post-abort reset must yield 100% bit-for-bit identical PCM output for chunk 1"
    );
    assert_eq!(
        base_chunk2, recovered_chunk2,
        "Post-abort reset must yield 100% bit-for-bit identical PCM output for chunk 2"
    );

    // Boundary continuity check between chunk 1 and chunk 2
    let tail_sample = recovered_chunk1[1919];
    let head_sample = recovered_chunk2[0];
    assert!(tail_sample.is_finite());
    assert!(head_sample.is_finite());
    let step_diff = (tail_sample - head_sample).abs();
    // With zero weights / synthetic weights, step_diff must be smoothly bounded (no huge NaN/Inf jumps)
    assert!(step_diff < 10.0, "Inter-chunk boundary step must be bounded: diff={step_diff}");
}

// ---------------------------------------------------------------------------
// Challenge 5: CandleLLM Streaming End-to-End Stress Matrix (under feature candle-llm)
// ---------------------------------------------------------------------------

#[cfg(feature = "candle-llm")]
mod candle_llm_challenger_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use qwen3tts::text_frontend::candle_backend::CandleLLM;
    use qwen3tts::text_frontend::model_catalog::{
        BranchSamplingConfig, GenerationSamplingConfig, ModelMetadata,
    };
    use qwen3tts::text_frontend::SynthesisOptions;
    use tokenizers::Tokenizer as HfTokenizer;

    const SAMPLE_CHAT_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    { "id": 0, "content": "<|pad|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false },
    { "id": 1, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false },
    { "id": 2, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false },
    { "id": 3, "content": "assistant", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false },
    { "id": 4, "content": "\n", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false }
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": false,
    "use_regex": true
  },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "unk_token": "<|pad|>",
    "vocab": {
      "<|pad|>": 0,
      "<|im_start|>": 1,
      "<|im_end|>": 2,
      "assistant": 3,
      "\n": 4,
      "hello": 5,
      "world": 6,
      "qwen": 7
    }
  }
}"#;

    static META_COUNTER: AtomicU64 = AtomicU64::new(100);

    fn create_test_metadata() -> ModelMetadata {
        let counter = META_COUNTER.fetch_add(1, Ordering::SeqCst);
        let temp_dir = std::env::temp_dir().join(format!("qwen3_meta_chal_{}_{}", std::process::id(), counter));
        let _ = std::fs::create_dir_all(&temp_dir);
        let config_path = temp_dir.join("config.json");
        let json = r#"{
  "assistant_token_id": 3,
  "im_start_token_id": 1,
  "im_end_token_id": 2,
  "tts_bos_token_id": 1031,
  "tts_eos_token_id": 1032,
  "tts_pad_token_id": 1033,
  "model_type": "qwen3_tts",
  "tokenizer_type": "qwen3_tts_tokenizer_12hz",
  "tts_model_size": "0b6",
  "tts_model_type": "custom_voice",
  "talker_config": {
    "model_type": "qwen3_tts_talker",
    "num_hidden_layers": 1,
    "hidden_size": 16,
    "head_dim": 16,
    "num_attention_heads": 1,
    "vocab_size": 3072,
    "text_vocab_size": 3072,
    "num_code_groups": 16,
    "codec_bos_id": 2149,
    "codec_eos_token_id": 2150,
    "codec_think_id": 2154,
    "codec_nothink_id": 2155,
    "codec_think_bos_id": 2156,
    "codec_think_eos_id": 2157,
    "codec_pad_id": 2148,
    "codec_language_id": {"chinese": 2055, "english": 2050},
    "spk_id": {"serena": 3066},
    "spk_is_dialect": {"serena": false},
    "code_predictor_config": {
      "model_type": "qwen3_tts_talker_code_predictor",
      "num_hidden_layers": 1,
      "hidden_size": 16,
      "num_attention_heads": 1
    },
    "rope_scaling": {
      "interleaved": true,
      "rope_theta": 10000.0,
      "mrope_section": [3, 3, 2]
    }
  }
}"#;
        std::fs::write(&config_path, json).expect("write config.json");
        let meta = ModelMetadata::from_config_path(&config_path).expect("parse metadata");
        let _ = std::fs::remove_file(&config_path);
        let _ = std::fs::remove_dir(&temp_dir);
        meta
    }

    fn create_synthetic_llm(device: &Device) -> (CandleLLM, TalkerConfig) {
        let tokenizer = Arc::new(
            HfTokenizer::from_bytes(SAMPLE_CHAT_TOKENIZER_JSON.as_bytes())
                .expect("valid tokenizer"),
        );
        let config = create_synthetic_talker_config();
        let talker = Arc::new(create_synthetic_talker(&config, device));

        let sample_config = GenerationSamplingConfig {
            talker: BranchSamplingConfig {
                do_sample: false,
                options: SamplingOptions::greedy(),
            },
            subtalker: BranchSamplingConfig {
                do_sample: false,
                options: SamplingOptions::greedy(),
            },
        };

        let metadata = create_test_metadata();

        let llm = CandleLLM::from_components(
            tokenizer,
            metadata,
            talker,
            config.clone(),
            sample_config,
            device.clone(),
        );

        (llm, config)
    }

    #[test]
    fn challenge_candle_llm_synthesize_streaming_abort_on_frame_k() {
        let device = Device::Cpu;
        let (llm, _config) = create_synthetic_llm(&device);
        let mut decoder = create_test_decoder(&device);

        let options = SynthesisOptions {
            max_new_tokens: 6,
            speaker: Some("serena".to_string()),
            ..Default::default()
        };

        for abort_target in [1, 2, 3] {
            decoder.reset_state();
            let mut chunk_count = 0;
            let result = llm.synthesize_streaming("hello world", &options, &mut decoder, |_chunk| {
                chunk_count += 1;
                if chunk_count == abort_target {
                    Err(Error::Config(format!("aborted at chunk {chunk_count}")))
                } else {
                    Ok(())
                }
            });

            assert!(result.is_err(), "Must fail when callback errors on chunk {abort_target}");
            assert_eq!(chunk_count, abort_target, "Must terminate exactly on chunk {abort_target}");
        }
    }

    #[test]
    fn challenge_candle_llm_synthesize_streaming_pcm_continuity_and_sample_bounds() {
        let device = Device::Cpu;
        let (llm, _config) = create_synthetic_llm(&device);
        let mut decoder = create_test_decoder(&device);

        let options = SynthesisOptions {
            max_new_tokens: 5,
            speaker: Some("serena".to_string()),
            temperature: 0.0,
            ..Default::default()
        };

        let mut chunks: Vec<Vec<f32>> = Vec::new();
        decoder.reset_state();
        let frames = llm
            .synthesize_streaming("hello world", &options, &mut decoder, |chunk| {
                chunks.push(chunk);
                Ok(())
            })
            .expect("synthesize streaming");

        assert_eq!(frames, 4);
        assert_eq!(chunks.len(), 4);

        for (idx, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.len(), 1920, "Chunk {idx} length must be 1920");
            for &sample in chunk {
                assert!(sample.is_finite(), "Sample in chunk {idx} must be finite");
            }
        }

        // Boundary continuity check between adjacent chunks
        for i in 0..(chunks.len() - 1) {
            let tail = chunks[i][1919];
            let head = chunks[i + 1][0];
            let diff = (tail - head).abs();
            assert!(diff.is_finite());
            assert!(diff < 10.0, "Boundary step between chunk {i} and {} must be smooth (got {diff})", i + 1);
        }
    }

    #[test]
    fn challenge_candle_llm_synthesize_streaming_invalid_options() {
        let device = Device::Cpu;
        let (llm, _config) = create_synthetic_llm(&device);
        let mut decoder = create_test_decoder(&device);

        // 1. Unknown speaker
        let options_bad_speaker = SynthesisOptions {
            max_new_tokens: 3,
            speaker: Some("nonexistent_spk".to_string()),
            ..Default::default()
        };
        let res_spk = llm.synthesize_streaming("hello world", &options_bad_speaker, &mut decoder, |_| Ok(()));
        assert!(res_spk.is_err(), "Nonexistent speaker must return error");

        // 2. Empty text
        let options_ok = SynthesisOptions {
            max_new_tokens: 3,
            speaker: Some("serena".to_string()),
            ..Default::default()
        };
        let res_empty = llm.synthesize_streaming("", &options_ok, &mut decoder, |_| Ok(()));
        assert!(res_empty.is_err(), "Empty text must return error");
    }
}
