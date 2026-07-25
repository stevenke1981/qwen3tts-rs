use candle_core::{DType, Device, Error as CandleError, Result as CandleResult, Tensor};
use qwen3tts::talker::primitives::{MultimodalRotaryEmbedding, RMSNorm, embedding_lookup};
use qwen3tts::talker::{
    CodePredictor, CodePredictorConfig, InputBuilder, TalkerConfig, TalkerForConditionalGeneration,
    TalkerModel, VoiceClonePrompt,
};
use qwen3tts::text_frontend::model_catalog::{
    GenerationMode, ModelMetadata, TtsModelType, validate_generation_request,
};
use qwen3tts::text_frontend::prompt_templates::{
    build_assistant_prompt, build_instruction_prompt, build_reference_prompt,
    reference_text_tokens_from_prompt_ids,
};

const TEXT_VOCAB_SIZE: usize = 128;
const CODEC_VOCAB_SIZE: usize = 128;

#[test]
fn wrapper_strings_match_official_format() {
    let main_text = "你好世界";
    let ref_text = "這是參考文字\n含換行";
    let instruction = "請用溫和語氣";

    let main_prompt = build_assistant_prompt(main_text);
    let reference_prompt = build_reference_prompt(ref_text);
    let instruction_prompt = build_instruction_prompt(instruction);

    assert_eq!(
        main_prompt,
        "<|im_start|>assistant\n你好世界<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(
        reference_prompt,
        "<|im_start|>assistant\n這是參考文字\n含換行<|im_end|>\n"
    );
    assert_eq!(
        instruction_prompt,
        "<|im_start|>user\n請用溫和語氣<|im_end|>\n"
    );
}

#[test]
fn reference_text_slice_is_official_range_3_to_len_minus_2() {
    let ids = vec![100, 200, 300, 10, 11, 12, 13, 14, 15, 16, 17];
    let sliced = reference_text_tokens_from_prompt_ids(&ids).unwrap();
    assert_eq!(sliced, &[10, 11, 12, 13, 14, 15]);
}

#[test]
fn reference_text_slice_is_empty_or_too_short() {
    assert!(reference_text_tokens_from_prompt_ids(&[]).is_err());
    assert!(reference_text_tokens_from_prompt_ids(&[1, 2, 3, 4, 5]).is_err());
}

#[test]
fn validate_generation_request_mode_matrix_for_five_variants() {
    let base_06 = metadata_catalog("0b6", TtsModelType::Base, vec![], vec![]);
    let base_17 = metadata_catalog("1b7", TtsModelType::Base, vec![], vec![]);
    let custom_06 = metadata_catalog(
        "0b6",
        TtsModelType::CustomVoice,
        vec![("Dylan".into(), 3010)],
        vec![],
    );
    let custom_17 = metadata_catalog(
        "1b7",
        TtsModelType::CustomVoice,
        vec![("Dylan".into(), 3010)],
        vec![],
    );
    let voice_design_17 = metadata_catalog("1b7", TtsModelType::VoiceDesign, vec![], vec![]);

    assert!(
        validate_generation_request(
            &base_06,
            GenerationMode::VoiceClone,
            None,
            None,
            Some("ref.wav")
        )
        .is_ok()
    );
    assert!(
        validate_generation_request(&base_06, GenerationMode::VoiceClone, None, None, None)
            .is_err()
    );
    assert!(
        validate_generation_request(
            &base_06,
            GenerationMode::VoiceClone,
            Some("Dylan"),
            None,
            Some("ref.wav")
        )
        .is_err()
    );
    assert!(
        validate_generation_request(
            &base_06,
            GenerationMode::VoiceClone,
            None,
            Some("instr"),
            Some("ref.wav")
        )
        .is_err()
    );

    assert!(
        validate_generation_request(
            &base_17,
            GenerationMode::VoiceClone,
            None,
            None,
            Some("ref.wav")
        )
        .is_ok()
    );
    assert!(
        validate_generation_request(&base_17, GenerationMode::VoiceClone, None, None, None)
            .is_err()
    );

    assert!(
        validate_generation_request(
            &custom_06,
            GenerationMode::CustomVoice,
            Some("Dylan"),
            None,
            None
        )
        .is_ok()
    );
    assert!(
        validate_generation_request(&custom_06, GenerationMode::CustomVoice, None, None, None)
            .is_err()
    );
    assert!(
        validate_generation_request(
            &custom_06,
            GenerationMode::CustomVoice,
            Some("Dylan"),
            Some("instr"),
            None
        )
        .is_err()
    );

    assert!(
        validate_generation_request(
            &custom_17,
            GenerationMode::CustomVoice,
            Some("Dylan"),
            Some("instr"),
            None
        )
        .is_ok()
    );

    assert!(
        validate_generation_request(
            &voice_design_17,
            GenerationMode::VoiceDesign,
            None,
            Some("instr"),
            None
        )
        .is_ok()
    );
    assert!(
        validate_generation_request(
            &voice_design_17,
            GenerationMode::VoiceDesign,
            None,
            None,
            None
        )
        .is_err()
    );
    assert!(
        validate_generation_request(
            &voice_design_17,
            GenerationMode::VoiceDesign,
            Some("Dylan"),
            Some("instr"),
            None
        )
        .is_err()
    );
    assert!(
        validate_generation_request(
            &voice_design_17,
            GenerationMode::VoiceDesign,
            None,
            Some("instr"),
            Some("ref.wav")
        )
        .is_err()
    );
}

#[test]
fn input_builder_standard_and_custom_voice_base_layouts() {
    let (talker, config) = fake_talker();
    let builder = InputBuilder::new(&talker, &Device::Cpu);
    let prompt = build_synthetic_assistant_prompt_ids(&[10, 11, 12, 13, 14, 15]);

    let (expected_auto, expected_auto_trailing) =
        expected_standard_inputs(&talker, &prompt, None, "auto", None, &config);
    let (actual_auto, actual_auto_trailing, _) = {
        let (inputs, _mask, trailing, pad) = builder
            .build(&prompt, None, "auto", None)
            .expect("build standard auto");
        (
            to_scalar_vec(&inputs),
            to_scalar_vec(&trailing),
            to_scalar_vec(&pad),
        )
    };

    assert_eq!(actual_auto, expected_auto);
    assert_eq!(actual_auto_trailing, expected_auto_trailing);

    let instruction_ids = tokenizer_like_ids("請用溫和語氣");
    let (expected_voicedesign, expected_voicedesign_trailing) = expected_standard_inputs(
        &talker,
        &prompt,
        Some(&instruction_ids),
        "auto",
        None,
        &config,
    );
    let (actual_voicedesign, actual_voicedesign_trailing, _) = {
        let (inputs, _mask, trailing, pad) = builder
            .build(&prompt, Some(instruction_ids.as_slice()), "auto", None)
            .expect("build voicedesign");
        (
            to_scalar_vec(&inputs),
            to_scalar_vec(&trailing),
            to_scalar_vec(&pad),
        )
    };

    assert_eq!(actual_voicedesign, expected_voicedesign);
    assert_eq!(actual_voicedesign_trailing, expected_voicedesign_trailing);

    let (expected_custom, expected_custom_trailing) =
        expected_standard_inputs(&talker, &prompt, None, "chinese", Some("Dylan"), &config);
    let (actual_custom, actual_custom_trailing, _) = {
        let (inputs, _mask, trailing, pad) = builder
            .build(&prompt, None, "chinese", Some("Dylan"))
            .expect("build custom voice");
        (
            to_scalar_vec(&inputs),
            to_scalar_vec(&trailing),
            to_scalar_vec(&pad),
        )
    };

    assert_eq!(actual_custom, expected_custom);
    assert_eq!(actual_custom_trailing, expected_custom_trailing);
}

#[test]
fn input_builder_xvector_path_uses_speaker_embedding_without_invented_instruction() {
    let (talker, _config) = fake_talker();
    let builder = InputBuilder::new(&talker, &Device::Cpu);
    let prompt = build_synthetic_assistant_prompt_ids(&[10, 11, 12, 13, 14]);
    let xvector = Tensor::from_slice(&[777.0f32], (1, 1, 1), &Device::Cpu).unwrap();
    let voice_clone_prompt = VoiceClonePrompt {
        reference_text_token_ids: &[],
        reference_codes: &[],
        speaker_embedding: Some(&xvector),
    };

    let (actual_input, actual_trailing, actual_pad) = {
        let (inputs, _mask, trailing, pad) = builder
            .build_voice_clone(&prompt, None, "auto", None, &voice_clone_prompt)
            .expect("build voice clone xvector");
        (
            to_scalar_vec(&inputs),
            to_scalar_vec(&trailing),
            to_scalar_vec(&pad),
        )
    };

    let (expected_input, expected_trailing, expected_pad) = expected_standard_voice_clone_inputs(
        &talker,
        &prompt,
        None,
        "auto",
        None,
        Some(&voice_clone_prompt),
    );

    assert_eq!(actual_input, expected_input);
    assert_eq!(actual_trailing, expected_trailing);
    assert_eq!(actual_pad, expected_pad);
    assert_eq!(actual_pad.len(), 1);
    assert!(actual_pad[0].is_finite());
}

#[test]
fn input_builder_icl_geometry_handles_short_and_long_body() {
    let (talker, _config) = fake_talker();
    let builder = InputBuilder::new(&talker, &Device::Cpu);

    let short_body = vec![10, 11];
    let long_body = vec![10, 11, 12, 13, 14, 15, 16];

    let short_prompt = build_synthetic_assistant_prompt_ids(&short_body);
    let long_prompt = build_synthetic_assistant_prompt_ids(&long_body);

    let short_reference = vec![5, 6];
    let short_codes = vec![[1u16; 16], [2u16; 16], [3u16; 16], [4u16; 16], [5u16; 16]];
    let short_vox = VoiceClonePrompt {
        reference_text_token_ids: &short_reference,
        reference_codes: &short_codes,
        speaker_embedding: Some(&Tensor::from_slice(&[3.0f32], (1, 1, 1), &Device::Cpu).unwrap()),
    };

    let long_reference = vec![8, 9, 10];
    let long_codes = vec![[11u16; 16], [12u16; 16]];
    let long_vox = VoiceClonePrompt {
        reference_text_token_ids: &long_reference,
        reference_codes: &long_codes,
        speaker_embedding: None,
    };

    let short_actual = {
        let (inputs, _mask, trailing, _pad) = builder
            .build_voice_clone(&short_prompt, None, "english", None, &short_vox)
            .expect("short icl");
        (to_scalar_vec(&inputs), to_scalar_vec(&trailing))
    };
    let short_expected =
        expected_icl_voice_clone_inputs(&talker, &short_prompt, None, "english", None, &short_vox);
    assert_eq!(short_actual, short_expected);

    let long_actual = {
        let (inputs, _mask, trailing, _pad) = builder
            .build_voice_clone(&long_prompt, None, "english", None, &long_vox)
            .expect("long icl");
        (to_scalar_vec(&inputs), to_scalar_vec(&trailing))
    };
    let long_expected =
        expected_icl_voice_clone_inputs(&talker, &long_prompt, None, "english", None, &long_vox);
    assert_eq!(long_actual, long_expected);
}

#[test]
fn input_builder_rejects_malformed_prompt_geometry() {
    let (talker, _config) = fake_talker();
    let builder = InputBuilder::new(&talker, &Device::Cpu);

    let too_short = vec![2, 7, 3, 10, 11, 12, 13, 14];
    assert!(builder.build(&too_short, None, "auto", None).is_err());

    let malformed_ref_text_only = VoiceClonePrompt {
        reference_text_token_ids: &[],
        reference_codes: &[[1u16; 16]],
        speaker_embedding: None,
    };
    let prompt_ids = build_synthetic_assistant_prompt_ids(&[10, 11, 12, 13]);
    assert!(
        builder
            .build_voice_clone(&prompt_ids, None, "auto", None, &malformed_ref_text_only)
            .is_err()
    );
}

fn build_synthetic_assistant_prompt_ids(body: &[u32]) -> Vec<u32> {
    let mut ids = vec![2, 7, 3];
    ids.extend_from_slice(body);
    ids.extend_from_slice(&[100, 101, 102, 103, 104]);
    ids
}

fn metadata_catalog(
    tts_model_size: &str,
    tts_model_type: TtsModelType,
    spk_id: Vec<(String, u32)>,
    spk_is_dialect: Vec<(String, Option<String>)>,
) -> ModelMetadata {
    ModelMetadata {
        model_type: "qwen3_tts".into(),
        tokenizer_type: "qwen3_tts_tokenizer_12hz".into(),
        assistant_token_id: 7,
        im_start_token_id: 2,
        im_end_token_id: 3,
        tts_model_size: tts_model_size.into(),
        tts_model_type,
        talker_config_model_type: "qwen3_tts_talker".into(),
        talker_code_predictor_config_model_type: "qwen3_tts_talker_code_predictor".into(),
        tts_bos_token_id: 11,
        tts_eos_token_id: 12,
        tts_pad_token_id: 13,
        codec_bos_id: 21,
        codec_eos_token_id: 22,
        codec_think_id: 23,
        codec_nothink_id: 24,
        codec_think_bos_id: 25,
        codec_think_eos_id: 26,
        codec_pad_id: 20,
        talker_vocab_size: CODEC_VOCAB_SIZE,
        talker_text_vocab_size: TEXT_VOCAB_SIZE,
        talker_num_code_groups: 16,
        talker_num_hidden_layers: 1,
        talker_num_attention_heads: 1,
        talker_hidden_size: 1,
        talker_head_dim: 6,
        talker_code_predictor_num_hidden_layers: 1,
        talker_code_predictor_num_attention_heads: 1,
        talker_code_predictor_hidden_size: 1,
        rope_scaling_interleaved: true,
        rope_scaling_mrope_section: vec![1, 2, 2],
        rope_scaling_rope_theta: 1_000_000.0,
        codec_language_id: vec![
            ("chinese".into(), 2055u32),
            ("english".into(), 2054u32),
            ("beijing_dialect".into(), 9001u32),
        ],
        spk_id,
        spk_is_dialect,
    }
}

fn fake_talker() -> (TalkerForConditionalGeneration, TalkerConfig) {
    let device = Device::Cpu;

    let text_embedding = embedding_from_sequence(0.0, TEXT_VOCAB_SIZE, &device);
    let codec_embedding = embedding_from_sequence(10.0, CODEC_VOCAB_SIZE, &device);
    let tts_bias = Tensor::zeros((1,), DType::F32, &device).unwrap();
    let text_weight = Tensor::ones((1, 1), DType::F32, &device).unwrap();
    let codec_head = Tensor::ones((1, 1), DType::F32, &device).unwrap();

    let mut code_embeddings = Vec::with_capacity(15);
    for i in 0..15usize {
        code_embeddings.push(embedding_from_sequence(i as f32, CODEC_VOCAB_SIZE, &device));
    }
    let mut lm_heads = Vec::with_capacity(15);
    for _ in 0..15 {
        lm_heads.push(Tensor::ones((1, CODEC_VOCAB_SIZE), DType::F32, &device).unwrap());
    }
    let cp_config = CodePredictorConfig {
        hidden_size: 1,
        intermediate_size: 2,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        num_hidden_layers: 1,
        vocab_size: CODEC_VOCAB_SIZE,
        num_code_groups: 16,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        hidden_act: "silu".into(),
        attention_bias: false,
        attention_dropout: 0.0,
        layer_types: vec!["full_attention".into()],
    };

    let code_predictor = CodePredictor {
        codec_embeddings: code_embeddings,
        lm_heads,
        layers: vec![],
        norm: RMSNorm::new(Tensor::ones((1,), DType::F32, &device).unwrap(), 1e-6),
        small_to_mtp_proj: None,
        config: cp_config.clone(),
    };

    let config = TalkerConfig {
        hidden_size: 1,
        intermediate_size: 2,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 6,
        num_hidden_layers: 1,
        text_hidden_size: 1,
        text_vocab_size: TEXT_VOCAB_SIZE,
        vocab_size: CODEC_VOCAB_SIZE,
        num_code_groups: 16,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        mrope_section: vec![1, 1, 1],
        rope_interleaved: true,
        assistant_token_id: 7,
        im_start_token_id: 2,
        im_end_token_id: 3,
        hidden_act: "silu".into(),
        attention_bias: false,
        attention_dropout: 0.0,
        sliding_window: None,
        codec_bos_id: 21,
        codec_eos_token_id: 22,
        codec_think_id: 23,
        codec_nothink_id: 24,
        codec_think_bos_id: 25,
        codec_think_eos_id: 26,
        codec_pad_id: 20,
        tts_bos_token_id: 11,
        tts_eos_token_id: 12,
        tts_pad_token_id: 13,
        codec_language_id: vec![
            ("chinese".into(), 10u32),
            ("english".into(), 11u32),
            ("beijing_dialect".into(), 12u32),
            ("sichuan_dialect".into(), 13u32),
        ],
        spk_id: vec![("Dylan".into(), 14u32)],
        spk_is_dialect: vec![("Dylan".into(), Some("beijing_dialect".into()))],
        code_predictor: cp_config,
    };

    let model = TalkerModel::new(
        vec![],
        RMSNorm::new(Tensor::ones((1,), DType::F32, &device).unwrap(), 1e-6),
    );

    let talker = TalkerForConditionalGeneration {
        model,
        text_embedding,
        text_proj_fc1_w: text_weight.clone(),
        text_proj_fc1_b: tts_bias.clone(),
        text_proj_fc2_w: text_weight,
        text_proj_fc2_b: tts_bias.clone(),
        codec_embedding,
        codec_head,
        code_predictor,
        rope: MultimodalRotaryEmbedding::new(&config, &device).unwrap(),
        config: config.clone(),
    };
    (talker, config)
}

fn expected_standard_inputs(
    talker: &TalkerForConditionalGeneration,
    text_token_ids: &[u32],
    instruct_ids: Option<&[u32]>,
    language: &str,
    speaker: Option<&str>,
    config: &TalkerConfig,
) -> (Vec<f32>, Vec<f32>) {
    let device = &Device::Cpu;
    let text_seq_len = text_token_ids.len();
    assert!(text_seq_len >= 8);

    let role_emb = embed_text_ids(talker, &text_token_ids[0..3], device).unwrap();
    let (codec_input, codec_last) = codec_prefix(talker, language, speaker, None, config);
    let first_text = {
        let text_emb = embed_text_ids(talker, &text_token_ids[3..4], device).unwrap();
        text_emb
            .into_iter()
            .map(|value| value + codec_last)
            .collect::<Vec<_>>()
    };
    let text_content_ids = &text_token_ids[3..text_seq_len - 5];
    let trailing_len = text_content_ids.len().checked_sub(1).unwrap_or(0);
    let trailing_ids = &text_content_ids[1..1 + trailing_len];
    let mut trailing_emb = embed_text_ids(talker, trailing_ids, device).unwrap_or_default();
    let tts_eos_emb = embed_text_ids(talker, &[config.tts_eos_token_id], device).unwrap();

    let mut input_embeds = if let Some(ids) = instruct_ids {
        let mut out = embed_text_ids(talker, ids, device).unwrap();
        out.extend_from_slice(&role_emb);
        out.extend_from_slice(&codec_input);
        out
    } else {
        let mut out = role_emb;
        out.extend_from_slice(&codec_input);
        out
    };
    input_embeds.extend_from_slice(&first_text);

    trailing_emb.extend_from_slice(&tts_eos_emb);
    (input_embeds, trailing_emb)
}

fn expected_standard_voice_clone_inputs(
    talker: &TalkerForConditionalGeneration,
    text_token_ids: &[u32],
    instruct_ids: Option<&[u32]>,
    language: &str,
    speaker: Option<&str>,
    prompt: Option<&VoiceClonePrompt<'_>>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let device = &Device::Cpu;
    let text_seq_len = text_token_ids.len();
    assert!(text_seq_len >= 8);

    let role_emb = embed_text_ids(talker, &text_token_ids[0..3], device).unwrap();
    let prompt = prompt.expect("prompt required");
    let (codec_input, codec_last) = codec_prefix(
        talker,
        language,
        speaker,
        speaker_embedding(prompt),
        &talker.config,
    );
    let tts_pad = embed_text_ids(talker, &[talker.config.tts_pad_token_id], device).unwrap()[0];

    let mut base_input = if let Some(ids) = instruct_ids {
        let mut out = embed_text_ids(talker, ids, device).unwrap();
        out.extend_from_slice(&role_emb);
        out.extend_from_slice(&codec_input);
        out
    } else {
        let mut out = role_emb.clone();
        out.extend_from_slice(&codec_input);
        out
    };

    let uses_icl =
        !prompt.reference_text_token_ids.is_empty() || !prompt.reference_codes.is_empty();
    assert!(!uses_icl);

    let text_content_len = text_seq_len - 8;
    let text_content = &text_token_ids[3..3 + text_content_len];
    let first_text = embed_text_ids(talker, &[text_content[0]], device).unwrap();
    let first_text = vec![first_text[0] + codec_last];
    base_input.push(first_text[0]);

    let trailing_len = text_content_len - 1;
    let trailing_ids = &text_content[1..1 + trailing_len];
    let mut trailing = embed_text_ids(talker, trailing_ids, device).unwrap_or_default();
    trailing.extend_from_slice(
        &embed_text_ids(talker, &[talker.config.tts_eos_token_id], device).unwrap(),
    );

    let tts_pad_emb = vec![tts_pad];
    (base_input, trailing, tts_pad_emb)
}

fn expected_icl_voice_clone_inputs(
    talker: &TalkerForConditionalGeneration,
    text_token_ids: &[u32],
    instruct_ids: Option<&[u32]>,
    language: &str,
    speaker: Option<&str>,
    prompt: &VoiceClonePrompt<'_>,
) -> (Vec<f32>, Vec<f32>) {
    let device = &Device::Cpu;
    let config = &talker.config;
    let text_seq_len = text_token_ids.len();
    assert!(text_seq_len >= 8);
    let role_emb = embed_text_ids(talker, &text_token_ids[0..3], device).unwrap();
    let (codec_input, _) =
        codec_prefix(talker, language, speaker, speaker_embedding(prompt), config);
    let tts_pad = embed_text_ids(talker, &[config.tts_pad_token_id], device).unwrap()[0];

    let mut input = if let Some(ids) = instruct_ids {
        let mut out = embed_text_ids(talker, ids, device).unwrap();
        out.extend_from_slice(&role_emb);
        out.extend_from_slice(&codec_input);
        out
    } else {
        let mut out = role_emb;
        out.extend_from_slice(&codec_input);
        out
    };

    let body = &text_token_ids[3..text_token_ids.len() - 5];
    let (icl_input, trailing) = expected_icl_prompt(
        talker,
        prompt.reference_text_token_ids,
        body,
        prompt.reference_codes,
        tts_pad,
        config.tts_eos_token_id,
    )
    .unwrap();
    input.extend_from_slice(&icl_input);
    (input, trailing)
}

fn codec_prefix(
    talker: &TalkerForConditionalGeneration,
    language: &str,
    speaker: Option<&str>,
    explicit_speaker_embedding: Option<f32>,
    config: &TalkerConfig,
) -> (Vec<f32>, f32) {
    let codec_lang = resolve_codec_language_id_like(config, language, speaker);
    let codec_prefill = if let Some(lid) = codec_lang {
        vec![
            config.codec_think_id,
            config.codec_think_bos_id,
            lid,
            config.codec_think_eos_id,
        ]
    } else {
        vec![
            config.codec_nothink_id,
            config.codec_think_bos_id,
            config.codec_think_eos_id,
        ]
    };
    let mut codec_emb = embed_codec_ids(talker, &codec_prefill, &Device::Cpu).unwrap();
    let pad_bos = embed_codec_ids(
        talker,
        &[config.codec_pad_id, config.codec_bos_id],
        &Device::Cpu,
    )
    .unwrap();
    codec_emb.extend_from_slice(&pad_bos);
    let split = codec_prefill.len();

    if let Some(emb) = explicit_speaker_embedding {
        let spk = vec![emb];
        let mut out = codec_emb[..split.min(codec_emb.len())].to_vec();
        out.extend_from_slice(&spk);
        out.extend_from_slice(&codec_emb[split.min(codec_emb.len())..]);
        codec_emb = out;
    } else if let Some(id) = resolve_speaker_id_like(config, speaker) {
        let spk = embed_codec_ids(talker, &[id], &Device::Cpu).unwrap();
        let mut out = codec_emb[..split.min(codec_emb.len())].to_vec();
        out.extend_from_slice(&spk);
        out.extend_from_slice(&codec_emb[split.min(codec_emb.len())..]);
        codec_emb = out;
    }

    let codec_len = codec_emb.len();
    let mut pads = std::iter::repeat(
        embed_text_ids(talker, &[config.tts_pad_token_id], &Device::Cpu).unwrap()[0],
    )
    .take(codec_len - 2)
    .collect::<Vec<_>>();
    let tts_bos = embed_text_ids(talker, &[config.tts_bos_token_id], &Device::Cpu).unwrap()[0];
    pads.push(tts_bos);

    let codec_last = codec_emb.last().copied().unwrap_or(0.0);
    let codec_input = codec_emb
        .iter()
        .take(codec_len - 1)
        .zip(pads.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    (codec_input, codec_last)
}

fn speaker_embedding(prompt: &VoiceClonePrompt<'_>) -> Option<f32> {
    prompt
        .speaker_embedding
        .map(|tensor| to_scalar_vec(tensor).first().copied().unwrap_or(0.0))
}

fn expected_icl_prompt(
    talker: &TalkerForConditionalGeneration,
    reference_text_token_ids: &[u32],
    text_content_ids: &[u32],
    reference_codes: &[[u16; 16]],
    tts_pad: f32,
    eos_token_id: u32,
) -> CandleResult<(Vec<f32>, Vec<f32>)> {
    if reference_codes.is_empty() {
        return Err(CandleError::Msg(
            "voice clone reference codec tokens cannot be empty".into(),
        ));
    }

    let mut joined = Vec::with_capacity(reference_text_token_ids.len() + text_content_ids.len());
    joined.extend_from_slice(reference_text_token_ids);
    joined.extend_from_slice(text_content_ids);
    let mut text_embed = embed_text_ids(talker, &joined, &Device::Cpu)?;
    text_embed.extend_from_slice(&embed_text_ids(talker, &[eos_token_id], &Device::Cpu)?);

    let codec_bos = embed_codec_ids(talker, &[talker.config.codec_bos_id], &Device::Cpu)?;
    let frames = codec_reference_embeddings(talker, reference_codes)?;
    let codec_embed: Vec<f32> = std::iter::once(codec_bos[0])
        .chain(frames.iter().copied())
        .collect();

    if text_embed.len() > codec_embed.len() {
        let icl_input = add_vectors(&text_embed[0..codec_embed.len()], &codec_embed);
        let trailing = text_embed[codec_embed.len()..].to_vec();
        Ok((icl_input, trailing))
    } else {
        let pad_len = codec_embed.len().saturating_sub(text_embed.len());
        let mut padded = text_embed;
        padded.extend(std::iter::repeat(tts_pad).take(pad_len));
        Ok((
            padded
                .iter()
                .zip(&codec_embed)
                .map(|(a, b)| a + b)
                .collect(),
            vec![tts_pad],
        ))
    }
}

fn codec_reference_embeddings(
    talker: &TalkerForConditionalGeneration,
    reference_codes: &[[u16; 16]],
) -> CandleResult<Vec<f32>> {
    if reference_codes.is_empty() {
        return Err(CandleError::Msg(
            "voice clone reference codec tokens cannot be empty".into(),
        ));
    }
    let frames = reference_codes.len();
    let first_ids: Vec<u32> = reference_codes
        .iter()
        .map(|frame| frame[0] as u32)
        .collect();
    let mut sum = embed_codec_ids(talker, &first_ids, &Device::Cpu)?;
    for codebook in 1..talker.config.num_code_groups {
        let ids: Vec<u32> = reference_codes
            .iter()
            .map(|frame| frame[codebook] as u32)
            .collect();
        let codebook_tensor = Tensor::from_slice(&ids, (1, frames), &Device::Cpu)?;
        let next = embedding_lookup(
            &talker.code_predictor.codec_embeddings[codebook - 1],
            &codebook_tensor,
        )?;
        sum = add_vectors(&sum, &to_scalar_vec(&next));
    }
    Ok(sum)
}

fn embed_text_ids(
    talker: &TalkerForConditionalGeneration,
    ids: &[u32],
    device: &Device,
) -> CandleResult<Vec<f32>> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let ids_t = Tensor::from_slice(ids, (1, ids.len()), device)?;
    let embedded = talker.embed_text(&ids_t)?;
    embedded.flatten_all()?.to_vec1::<f32>().map(|v| v.to_vec())
}

fn embed_codec_ids(
    talker: &TalkerForConditionalGeneration,
    ids: &[u32],
    device: &Device,
) -> CandleResult<Vec<f32>> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let ids_t = Tensor::from_slice(ids, (1, ids.len()), device)?;
    let embedded = talker.embed_codec(&ids_t)?;
    embedded.flatten_all()?.to_vec1::<f32>().map(|v| v.to_vec())
}

fn to_scalar_vec(tensor: &Tensor) -> Vec<f32> {
    tensor
        .flatten_all()
        .and_then(|v| v.to_vec1::<f32>())
        .expect("tensor to scalar vec")
}

fn add_vectors(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(l, r)| l + r).collect()
}

fn embedding_from_sequence(offset: f32, vocab: usize, device: &Device) -> Tensor {
    let values = (0..vocab).map(|id| id as f32 + offset).collect::<Vec<_>>();
    Tensor::from_slice(&values, (vocab, 1), device).unwrap()
}

fn resolve_codec_language_id_like(
    config: &TalkerConfig,
    language: &str,
    speaker: Option<&str>,
) -> Option<u32> {
    if language.eq_ignore_ascii_case("auto") {
        speaker
            .and_then(|speaker| {
                config
                    .spk_is_dialect
                    .iter()
                    .find(|(name, _)| normalize_key(name) == normalize_key(speaker))
                    .and_then(|(_, dialect)| dialect.as_deref())
                    .and_then(|dialect| config.language_id_for(dialect))
            })
            .or_else(|| {
                config
                    .codec_language_id
                    .iter()
                    .find(|(name, _)| normalize_key(name) == normalize_key(language))
                    .map(|(_, id)| *id)
            })
    } else {
        let request_id = config.language_id_for(language);
        if is_chinese_language_like(config, language) {
            speaker
                .and_then(|speaker| {
                    config
                        .spk_is_dialect
                        .iter()
                        .find(|(name, _)| normalize_key(name) == normalize_key(speaker))
                        .and_then(|(_, dialect)| dialect.as_deref())
                        .and_then(|dialect| config.language_id_for(dialect))
                })
                .or(request_id)
        } else {
            request_id.or_else(|| config.language_id_for(speaker.unwrap_or(language)))
        }
    }
}

fn is_chinese_language_like(config: &TalkerConfig, language: &str) -> bool {
    let requested = config.language_id_for(language);
    requested == config.language_id_for("chinese") || requested == config.language_id_for("zh")
}

fn resolve_speaker_id_like(config: &TalkerConfig, speaker: Option<&str>) -> Option<u32> {
    speaker.and_then(|name| config.speaker_id_for(name))
}

fn normalize_key(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn tokenizer_like_ids(text: &str) -> Vec<u32> {
    let mut ids = Vec::with_capacity(text.len());
    let mut next_id = 10u32;
    let vocab = TEXT_VOCAB_SIZE as u32;
    for b in text.bytes() {
        ids.push((next_id.wrapping_add(b as u32 % 17)) % vocab);
        next_id = (next_id + 7) % vocab;
    }
    ids
}
