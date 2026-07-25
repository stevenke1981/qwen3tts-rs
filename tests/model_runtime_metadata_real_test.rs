use qwen3tts::text_frontend::model_catalog::{
    GenerationMode, ModelMetadata, TtsModelType, infer_mode,
};
use std::path::PathBuf;

fn codec_language_id(metadata: &ModelMetadata, language: &str) -> Option<u32> {
    metadata
        .codec_language_id
        .iter()
        .find(|(name, _)| name == &language)
        .map(|(_, id)| *id)
}

#[test]
#[ignore]
fn runtime_metadata_contract_is_loaded_from_real_model_dir() {
    let model_dir = std::env::var("QWEN3_TTS_REAL_MODEL_DIR")
        .expect("missing env var QWEN3_TTS_REAL_MODEL_DIR; set it to a local Qwen3-TTS directory containing config.json");
    let model_dir = PathBuf::from(model_dir);
    assert!(model_dir.join("config.json").exists());

    let metadata = ModelMetadata::from_model_dir(&model_dir).expect("valid model metadata");

    assert_eq!(metadata.model_type, "qwen3_tts");
    assert_eq!(metadata.tokenizer_type, "qwen3_tts_tokenizer_12hz");
    assert_eq!(metadata.tts_model_size, "0b6");
    assert_eq!(metadata.tts_model_type, TtsModelType::Base);
    assert_eq!(infer_mode(&metadata), metadata.runtime_generation_mode());
    assert_eq!(infer_mode(&metadata), GenerationMode::VoiceClone);

    assert_eq!(metadata.assistant_token_id, 77091);
    assert_eq!(metadata.im_start_token_id, 151644);
    assert_eq!(metadata.im_end_token_id, 151645);
    assert_eq!(metadata.tts_bos_token_id, 151672);
    assert_eq!(metadata.tts_eos_token_id, 151673);
    assert_eq!(metadata.tts_pad_token_id, 151671);
    assert_eq!(metadata.codec_bos_id, 2149);
    assert_eq!(metadata.codec_eos_token_id, 2150);
    assert_eq!(metadata.codec_pad_id, 2148);
    assert_eq!(metadata.codec_think_id, 2154);
    assert_eq!(metadata.codec_nothink_id, 2155);
    assert_eq!(metadata.codec_think_bos_id, 2156);
    assert_eq!(metadata.codec_think_eos_id, 2157);
    assert_eq!(metadata.talker_text_vocab_size, 151936);
    assert_eq!(metadata.talker_vocab_size, 3072);

    assert_eq!(metadata.codec_language_id.len(), 10);
    let expected_language_ids = [
        ("chinese", 2055),
        ("english", 2050),
        ("german", 2053),
        ("italian", 2070),
        ("portuguese", 2071),
        ("spanish", 2054),
        ("japanese", 2058),
        ("korean", 2064),
        ("french", 2061),
        ("russian", 2069),
    ];
    for (language, token_id) in expected_language_ids {
        assert_eq!(codec_language_id(&metadata, language), Some(token_id));
    }

    assert_eq!(metadata.spk_id.len(), 0);
    assert_eq!(metadata.spk_is_dialect.len(), 0);
}
