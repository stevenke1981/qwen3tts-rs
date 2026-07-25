#![cfg(feature = "stage-dump")]
use candle_core::{DType, Device, Tensor};
use qwen3tts::alignment_stage_dump::{StageDumpMetadata, StageDumpWriter};
use qwen3tts::talker::decoder_layer::StandardDecoderLayer;
use qwen3tts::talker::decoder_layer::TalkerDecoderLayer;
use qwen3tts::talker::primitives::{MultimodalRotaryEmbedding, RMSNorm, SwiGLUMLP};
use qwen3tts::talker::sampling::{Sampler, SamplingOptions};
use qwen3tts::talker::talker_attention::{StandardAttention, TalkerAttention};
use qwen3tts::talker::{CodePredictor, TalkerConfig, TalkerForConditionalGeneration, TalkerModel};

#[test]
fn synthetic_production_path_emits_ordered_stage_categories() {
    let d = Device::Cpu;
    let mut cfg = TalkerConfig::default();
    cfg.hidden_size = 6;
    cfg.text_hidden_size = 6;
    cfg.head_dim = 6;
    cfg.num_attention_heads = 1;
    cfg.num_key_value_heads = 1;
    cfg.num_hidden_layers = 1;
    cfg.vocab_size = 1028;
    cfg.num_code_groups = 16;
    cfg.mrope_section = vec![1, 1, 1];
    cfg.codec_eos_token_id = 1024;
    cfg.code_predictor.hidden_size = 6;
    cfg.code_predictor.head_dim = 6;
    cfg.code_predictor.num_attention_heads = 1;
    cfg.code_predictor.num_key_value_heads = 1;
    cfg.code_predictor.num_hidden_layers = 1;
    cfg.code_predictor.vocab_size = 8;
    cfg.code_predictor.num_code_groups = 16;
    let z = Tensor::zeros((1,), DType::F32, &d).unwrap();
    let norm = RMSNorm::new(Tensor::ones(6, DType::F32, &d).unwrap(), 1e-6);
    let cp_attn = StandardAttention::new(
        Tensor::zeros((6, 6), DType::F32, &d).unwrap(),
        Tensor::zeros((6, 6), DType::F32, &d).unwrap(),
        Tensor::zeros((6, 6), DType::F32, &d).unwrap(),
        Tensor::zeros((6, 6), DType::F32, &d).unwrap(),
        Tensor::ones(6, DType::F32, &d).unwrap(),
        Tensor::ones(6, DType::F32, &d).unwrap(),
        1,
        1,
        6,
        1e-6,
    );
    let cp_layer = StandardDecoderLayer::new(
        RMSNorm::new(Tensor::ones(6, DType::F32, &d).unwrap(), 1e-6),
        cp_attn,
        RMSNorm::new(Tensor::ones(6, DType::F32, &d).unwrap(), 1e-6),
        SwiGLUMLP::new(
            Tensor::zeros((6, 6), DType::F32, &d).unwrap(),
            Tensor::zeros((6, 6), DType::F32, &d).unwrap(),
            Tensor::zeros((6, 6), DType::F32, &d).unwrap(),
        ),
    );
    let cp = CodePredictor {
        codec_embeddings: (0..15)
            .map(|_| Tensor::zeros((8, 6), DType::F32, &d).unwrap())
            .collect(),
        lm_heads: (0..15)
            .map(|_| Tensor::zeros((8, 6), DType::F32, &d).unwrap())
            .collect(),
        layers: vec![cp_layer],
        norm: RMSNorm::new(Tensor::ones(6, DType::F32, &d).unwrap(), 1e-6),
        small_to_mtp_proj: None,
        config: cfg.code_predictor.clone(),
    };
    let attn = TalkerAttention::new(
        Tensor::zeros((6, 6), DType::F32, &d).unwrap(),
        Tensor::zeros((6, 6), DType::F32, &d).unwrap(),
        Tensor::zeros((6, 6), DType::F32, &d).unwrap(),
        Tensor::zeros((6, 6), DType::F32, &d).unwrap(),
        Tensor::ones(6, DType::F32, &d).unwrap(),
        Tensor::ones(6, DType::F32, &d).unwrap(),
        1,
        1,
        6,
        1e-6,
    );
    let layer = TalkerDecoderLayer::new(
        RMSNorm::new(Tensor::ones(6, DType::F32, &d).unwrap(), 1e-6),
        attn,
        RMSNorm::new(Tensor::ones(6, DType::F32, &d).unwrap(), 1e-6),
        SwiGLUMLP::new(
            Tensor::zeros((6, 6), DType::F32, &d).unwrap(),
            Tensor::zeros((6, 6), DType::F32, &d).unwrap(),
            Tensor::zeros((6, 6), DType::F32, &d).unwrap(),
        ),
    );
    let talker = TalkerForConditionalGeneration {
        model: TalkerModel::new(vec![layer], norm),
        text_embedding: z.clone(),
        text_proj_fc1_w: z.clone(),
        text_proj_fc1_b: z.clone(),
        text_proj_fc2_w: z.clone(),
        text_proj_fc2_b: z.clone(),
        codec_embedding: Tensor::zeros((1028, 6), DType::F32, &d).unwrap(),
        codec_head: Tensor::zeros((1028, 6), DType::F32, &d).unwrap(),
        code_predictor: cp,
        rope: MultimodalRotaryEmbedding::new(&cfg, &d).unwrap(),
        config: cfg,
    };
    let mut dir = std::env::temp_dir();
    dir.push(format!("qwen3tts-p03-synth-{}", std::process::id()));
    let mut w = StageDumpWriter::new(
        &dir,
        StageDumpMetadata {
            source: "synthetic".into(),
            revision: None,
            model: "synthetic".into(),
            case_id: "p03".into(),
            seed: Some(1),
        },
    )
    .unwrap();
    let mut s = Sampler::new(1);
    let o = SamplingOptions {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };
    let x = Tensor::zeros((1, 1, 6), DType::F32, &d).unwrap();
    talker
        .generate_sampled_with_observer(
            &x, None, None, None, 2, &d, &mut s, o, o, false, false, &mut w,
        )
        .unwrap();
    let names = w
        .manifest()
        .stages
        .iter()
        .map(|s| s.name.as_str())
        .collect::<Vec<_>>();
    assert!(
        names.iter().position(|n| *n == "talker-input-embed")
            < names.iter().position(|n| *n == "talker-logits-prefill")
    );
    assert!(names.iter().any(|n| *n == "next-emb-step0"));
    assert!(
        names
            .iter()
            .any(|n| n.starts_with("code-predictor-prefill-frame0"))
    );
    assert!(names.windows(2).all(|w| w[0] != w[1]));
}
