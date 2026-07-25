use qwen3tts::text_frontend::model_catalog::{
    GenerationMode, ModelMetadata, TtsModelType, infer_mode,
};
use std::path::PathBuf;

#[test]
#[ignore]
fn parse_real_model_06b_base_metadata() {
    let model_dir = std::env::var("QWEN3_TTS_REAL_MODEL_DIR")
        .expect("missing env var QWEN3_TTS_REAL_MODEL_DIR; set it to a local 0.6B Base model directory containing config.json");
    let model_dir = PathBuf::from(model_dir);
    assert!(model_dir.join("config.json").exists());

    let metadata = ModelMetadata::from_model_dir(&model_dir).expect("valid model metadata");
    assert_eq!(metadata.model_type, "qwen3_tts");
    assert_eq!(metadata.tokenizer_type, "qwen3_tts_tokenizer_12hz");
    assert_eq!(metadata.tts_model_size, "0b6");
    assert_eq!(metadata.tts_model_type, TtsModelType::Base);
    assert!(metadata.supports_voice_clone());
    assert!(!metadata.supports_speaker_presets());
    assert!(!metadata.supports_voice_design());
    assert_eq!(infer_mode(&metadata), GenerationMode::VoiceClone);
    assert_eq!(
        metadata.runtime_generation_mode(),
        GenerationMode::VoiceClone
    );
    assert_eq!(metadata.talker_num_hidden_layers, 28);
    assert_eq!(metadata.talker_num_attention_heads, 16);
    assert_eq!(metadata.talker_hidden_size, 1024);
    assert_eq!(metadata.talker_head_dim, 128);
}
