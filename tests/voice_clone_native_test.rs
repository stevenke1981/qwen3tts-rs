use candle_core::{Device, Tensor};
use qwen3tts::text_frontend::SynthesisOptions;
use qwen3tts::text_frontend::voice_clone::{
    NativeReferenceCodes, NativeVoiceCloneCondition, NativeVoiceClonePlan, VoiceCloneMode,
    reference_prefix_samples,
};

#[test]
fn native_voice_clone_plan_requires_reference_text_for_icl_mode() {
    let options = SynthesisOptions {
        reference_audio: Some("reference.wav".to_string()),
        reference_text: Some("這是一段參考語音".to_string()),
        ..SynthesisOptions::default()
    };

    let plan = NativeVoiceClonePlan::from_options(&options)
        .unwrap()
        .unwrap();

    assert_eq!(plan.mode, VoiceCloneMode::InContextLearning);
    assert_eq!(plan.reference_audio.as_os_str(), "reference.wav");
    assert_eq!(plan.reference_text.as_deref(), Some("這是一段參考語音"));
    assert!(plan.requires_reference_codec_tokens());
    assert!(plan.requires_speaker_embedding());
}

#[test]
fn native_voice_clone_plan_without_reference_text_uses_speaker_embedding_only_mode() {
    let options = SynthesisOptions {
        reference_audio: Some("reference.wav".to_string()),
        reference_text: None,
        ..SynthesisOptions::default()
    };

    let plan = NativeVoiceClonePlan::from_options(&options)
        .unwrap()
        .unwrap();

    assert_eq!(plan.mode, VoiceCloneMode::SpeakerEmbeddingOnly);
    assert_eq!(plan.reference_audio.as_os_str(), "reference.wav");
    assert_eq!(plan.reference_text, None);
    assert!(!plan.requires_reference_codec_tokens());
    assert!(plan.requires_speaker_embedding());
}

#[test]
fn native_voice_clone_plan_without_reference_audio_is_none() {
    let options = SynthesisOptions {
        reference_audio: None,
        reference_text: Some("這是一段參考文字".to_string()),
        ..SynthesisOptions::default()
    };

    let plan = NativeVoiceClonePlan::from_options(&options);
    assert!(plan.is_ok());
    assert!(plan.unwrap().is_none());
}

#[test]
fn native_voice_clone_plan_rejects_empty_reference_audio() {
    let options = SynthesisOptions {
        reference_audio: Some("   ".to_string()),
        reference_text: Some("這是一段參考文字".to_string()),
        ..SynthesisOptions::default()
    };

    assert!(NativeVoiceClonePlan::from_options(&options).is_err());
}

#[test]
fn trims_reference_audio_prefix_using_frame_ratio() {
    let cut = reference_prefix_samples(12_000, 12, 40);

    assert_eq!(cut, 3_600);
}

#[test]
fn reference_codes_reject_tokens_outside_codec_vocab() {
    let mut frame = [0u16; 16];
    frame[7] = 2048;

    let err = NativeReferenceCodes::from_frames(vec![frame])
        .expect_err("reference codec tokens must stay inside 0..2047");

    assert!(err.to_string().contains("Invalid token 2048 at layer 7"));
}

#[test]
fn native_condition_exports_talker_prompt_slices() {
    let speaker = Tensor::zeros((1, 1, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
    let codes = NativeReferenceCodes::from_frames(vec![[1u16; 16], [2u16; 16]]).unwrap();
    let condition = NativeVoiceCloneCondition::new(vec![10, 11], codes, speaker);

    let prompt = condition.as_talker_prompt();

    assert_eq!(prompt.reference_text_token_ids, &[10, 11]);
    assert_eq!(prompt.reference_codes.len(), 2);
    assert!(prompt.speaker_embedding.is_some());
}
