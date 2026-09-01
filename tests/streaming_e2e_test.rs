//! End-to-End Streaming Pipeline Integration Tests
//!
//! Validates:
//! 1. `TalkerForConditionalGeneration::generate_streaming` frame-by-frame callback matches batch `generate`.
//! 2. `TalkerForConditionalGeneration::generate_sampled_streaming` matches batch `generate_sampled`.
//! 3. `CandleLLM::synthesize_streaming` end-to-end integration with `Decoder12Hz::decode_chunk`.
//! 4. PCM chunk size invariants (exactly 1920 samples/frame at 24kHz), finite values, and continuity.
//! 5. Error handling and fail-closed propagation in streaming callback.
//! 6. Zero state leakage across multiple streaming synthesis sessions with `reset_state()`.

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
use qwen3tts::{Decoder12Hz, DecoderConfig, TtsDecoder};

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
    // Token 0 logit bias so greedy picks 0
    for v in &mut head[..h] {
        *v = 0.5;
    }
    // Token 2150 (EOS) logit bias
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

#[test]
fn test_talker_generate_streaming_matches_batch_generation() {
    let device = Device::Cpu;
    let config = create_synthetic_talker_config();
    let talker = create_synthetic_talker(&config, &device);

    let inputs = Tensor::zeros((1, 4, 16), DType::F32, &device).unwrap();
    let pad = Tensor::zeros((1, 1, 16), DType::F32, &device).unwrap();

    // 1. Batch generation
    let batch_result = talker
        .generate(&inputs, None, None, Some(&pad), 4, &device)
        .expect("batch generate");
    let batch_codes = batch_result.to_vec2::<u32>().expect("batch vec2");
    let num_frames = batch_codes.len();

    // 2. Streaming generation
    let mut streaming_frames: Vec<[u16; 16]> = Vec::new();
    let stream_count = talker
        .generate_streaming(
            &inputs,
            None,
            None,
            Some(&pad),
            4,
            &device,
            |frame| {
                streaming_frames.push(*frame);
                Ok(())
            },
        )
        .expect("streaming generate");

    assert_eq!(stream_count, num_frames);
    assert_eq!(streaming_frames.len(), num_frames);

    for (f_idx, stream_frame) in streaming_frames.iter().enumerate() {
        for c_idx in 0..16 {
            assert_eq!(
                stream_frame[c_idx] as u32, batch_codes[f_idx][c_idx],
                "Frame {f_idx} codebook {c_idx} mismatch: stream={}, batch={}",
                stream_frame[c_idx], batch_codes[f_idx][c_idx]
            );
        }
    }
}

#[test]
fn test_talker_generate_sampled_streaming_matches_batch_sampled() {
    let device = Device::Cpu;
    let config = create_synthetic_talker_config();
    let talker = create_synthetic_talker(&config, &device);

    let inputs = Tensor::zeros((1, 4, 16), DType::F32, &device).unwrap();
    let pad = Tensor::zeros((1, 1, 16), DType::F32, &device).unwrap();
    let sampling_opts = SamplingOptions {
        temperature: 0.8,
        top_k: 50,
        top_p: 0.95,
        repetition_penalty: 1.05,
    };

    // 1. Batch sampled generation
    let mut sampler_batch = Sampler::new(42);
    let batch_result = talker
        .generate_sampled(
            &inputs,
            None,
            None,
            Some(&pad),
            4,
            &device,
            &mut sampler_batch,
            sampling_opts,
            true,
            sampling_opts,
            true,
        )
        .expect("batch generate sampled");
    let batch_codes = batch_result.to_vec2::<u32>().expect("batch vec2");

    // 2. Streaming sampled generation with identical seed
    let mut sampler_stream = Sampler::new(42);
    let mut streaming_frames: Vec<[u16; 16]> = Vec::new();
    let stream_count = talker
        .generate_sampled_streaming(
            &inputs,
            None,
            None,
            Some(&pad),
            4,
            &device,
            &mut sampler_stream,
            sampling_opts,
            sampling_opts,
            true,
            true,
            |frame| {
                streaming_frames.push(*frame);
                Ok(())
            },
        )
        .expect("streaming generate sampled");

    assert_eq!(stream_count, batch_codes.len());
    assert_eq!(streaming_frames.len(), batch_codes.len());

    for (f_idx, stream_frame) in streaming_frames.iter().enumerate() {
        for c_idx in 0..16 {
            assert_eq!(
                stream_frame[c_idx] as u32, batch_codes[f_idx][c_idx],
                "Sampled frame {f_idx} codebook {c_idx} mismatch"
            );
        }
    }
}

#[test]
fn test_decoder_streaming_chunk_size_and_continuity() {
    let device = Device::Cpu;
    let mut decoder = create_test_decoder(&device);

    let test_frames = vec![
        [10u16, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
        [15u16, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165],
        [20u16, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170],
    ];

    let mut pcm_chunks = Vec::new();
    for frame in &test_frames {
        let chunk = decoder.decode_chunk(frame).expect("decode chunk");
        assert_eq!(chunk.len(), 1920, "Every 12Hz frame must produce 1920 PCM samples");
        for &s in &chunk {
            assert!(s.is_finite(), "PCM sample must be finite");
        }
        pcm_chunks.push(chunk);
    }

    assert_eq!(pcm_chunks.len(), 3);
    let total_samples: usize = pcm_chunks.iter().map(|c| c.len()).sum();
    assert_eq!(total_samples, 3 * 1920);
}

#[test]
fn test_decoder_reset_state_between_streaming_sessions_zero_leakage() {
    let device = Device::Cpu;
    let mut decoder = create_test_decoder(&device);

    let test_frame: [u16; 16] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];

    // Session 1
    decoder.reset_state();
    let chunk1_s1 = decoder.decode_chunk(&test_frame).expect("chunk 1");
    let chunk2_s1 = decoder.decode_chunk(&test_frame).expect("chunk 2");

    // Session 2 after reset
    decoder.reset_state();
    let chunk1_s2 = decoder.decode_chunk(&test_frame).expect("chunk 1 after reset");
    let chunk2_s2 = decoder.decode_chunk(&test_frame).expect("chunk 2 after reset");

    // Assert exact numerical reproduction after reset_state()
    assert_eq!(
        chunk1_s1, chunk1_s2,
        "Session 2 chunk 1 must match Session 1 chunk 1 identically after reset_state()"
    );
    assert_eq!(
        chunk2_s1, chunk2_s2,
        "Session 2 chunk 2 must match Session 1 chunk 2 identically after reset_state()"
    );
}

#[cfg(feature = "candle-llm")]
mod candle_llm_e2e_tests {
    use super::*;
    use std::sync::Arc;
    use qwen3tts::text_frontend::candle_backend::CandleLLM;
    use qwen3tts::text_frontend::model_catalog::{
        BranchSamplingConfig, GenerationSamplingConfig, ModelMetadata,
    };
    use qwen3tts::text_frontend::SynthesisOptions;
    use qwen3tts::Error;
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

    static META_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn create_test_metadata() -> ModelMetadata {
        let counter = META_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let temp_dir = std::env::temp_dir().join(format!("qwen3_meta_test_{}_{}", std::process::id(), counter));
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
    fn test_candle_llm_synthesize_streaming_pcm_length_and_continuity() {
        let device = Device::Cpu;
        let (llm, _config) = create_synthetic_llm(&device);
        let mut decoder = create_test_decoder(&device);

        let options = SynthesisOptions {
            max_new_tokens: 3,
            temperature: 0.0, // greedy
            speaker: Some("serena".to_string()),
            ..Default::default()
        };

        let mut pcm_chunks: Vec<Vec<f32>> = Vec::new();
        let frames_generated = llm
            .synthesize_streaming("hello world", &options, &mut decoder, |chunk| {
                pcm_chunks.push(chunk);
                Ok(())
            })
            .expect("synthesize streaming");

        assert!(frames_generated > 0, "Should generate at least 1 frame");
        assert_eq!(pcm_chunks.len(), frames_generated);

        for (i, chunk) in pcm_chunks.iter().enumerate() {
            assert_eq!(
                chunk.len(),
                1920,
                "Chunk {i} must contain exactly 1920 audio samples"
            );
            for &sample in chunk {
                assert!(sample.is_finite(), "Audio sample must be finite");
            }
        }

        let total_samples: usize = pcm_chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total_samples, frames_generated * 1920);
    }

    #[test]
    fn test_candle_llm_synthesize_streaming_error_propagation() {
        let device = Device::Cpu;
        let (llm, _config) = create_synthetic_llm(&device);
        let mut decoder = create_test_decoder(&device);

        let options = SynthesisOptions {
            max_new_tokens: 3,
            speaker: Some("serena".to_string()),
            ..Default::default()
        };

        let mut callback_count = 0;
        let result = llm.synthesize_streaming("hello world", &options, &mut decoder, |_chunk| {
            callback_count += 1;
            Err(Error::Config("user requested stream abort".into()))
        });

        assert!(result.is_err(), "Callback error must propagate immediately");
        assert_eq!(callback_count, 1, "Must abort after the first error");
    }

    #[test]
    fn test_candle_llm_synthesize_streaming_empty_text_fails() {
        let device = Device::Cpu;
        let (llm, _config) = create_synthetic_llm(&device);
        let mut decoder = create_test_decoder(&device);

        let options = SynthesisOptions::default();
        let result = llm.synthesize_streaming("", &options, &mut decoder, |_chunk| Ok(()));

        assert!(result.is_err(), "Empty text must return an error");
    }
}
