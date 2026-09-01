use std::path::Path;
use qwen3tts::downloader::{
    detect_download_tool, hf_hub_cache_dir, is_model_ready, is_tokenizer_ready,
    locate_model_snapshot, HF_MIRROR_ENDPOINT, HF_OFFICIAL_ENDPOINT,
};
use qwen3tts::gui::{BackendKind, SynthesisParams, TtsGuiApp};

#[test]
fn test_downloader_tool_detection() {
    let tool = detect_download_tool();
    assert!(tool.is_ok(), "系統應能偵測到 hf 或 python 下載工具");
    let tool = tool.unwrap();
    assert!(tool.path().exists(), "偵測到的下載工具可執行路徑應存在");
    println!("成功偵測下載工具：{} ({:?})", tool.name(), tool.path());
}

#[test]
fn test_downloader_hf_hub_cache() {
    let hub = hf_hub_cache_dir();
    assert!(hub.is_some(), "應能定位 HuggingFace hub 本地快取路徑");
    let hub = hub.unwrap();
    assert!(hub.is_absolute(), "快取路徑必須是絕對路徑");
}

#[test]
fn test_is_model_ready_contract() {
    // 假模型應該回傳 false
    assert!(!is_model_ready("Qwen/Fake-Model-That-Does-Not-Exist", None));

    // 自訂路徑不存在 model.safetensors 應回傳 false
    assert!(!is_model_ready(
        "Qwen/Fake-Model",
        Some(Path::new("C:/NonExistentPath12345"))
    ));
}

#[test]
fn test_is_tokenizer_ready_callable() {
    let _ready = is_tokenizer_ready();
    // 函式應正常呼叫不 panic
}

#[test]
fn test_locate_model_snapshot_nonexistent() {
    let snapshot = locate_model_snapshot("Qwen/NonExistent-Model-XYZ", None);
    assert!(snapshot.is_none());
}

#[test]
fn test_locate_model_in_dir() {
    let temp_dir = std::env::temp_dir().join("qwen3tts_test_models");
    let model_subdir = temp_dir.join("Qwen3-TTS-12Hz-0.6B-Base");
    std::fs::create_dir_all(&model_subdir).unwrap();
    let fake_weight = model_subdir.join("model.safetensors");
    std::fs::write(&fake_weight, b"fake weight").unwrap();

    let found = qwen3tts::downloader::locate_model_in_dir("Qwen/Qwen3-TTS-12Hz-0.6B-Base", &temp_dir);
    assert!(found.is_some(), "應能在 models 資料夾下找到模型子目錄");
    assert_eq!(found.unwrap(), model_subdir);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_endpoints_constants() {
    assert_eq!(HF_OFFICIAL_ENDPOINT, "https://huggingface.co");
    assert_eq!(HF_MIRROR_ENDPOINT, "https://hf-mirror.com");
}

#[test]
fn test_gui_app_default_models_dir() {
    let dir = qwen3tts::gui::default_models_dir();
    assert!(!dir.as_os_str().is_empty(), "預設 models 目錄不應為空");
    println!("預設 models 目錄: {}", dir.display());

    let app = TtsGuiApp::default();
    let _ = app;
}

#[test]
fn test_synthesis_params_auto_download_fields() {
    let params = SynthesisParams {
        text: "測試自動下載設定".into(),
        model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-Base".into(),
        model_dir: None,
        models_base_dir: std::path::PathBuf::from("models"),
        backend: BackendKind::Candle,
        language: "auto".into(),
        speaker: None,
        instruct: None,
        speed: 1.0,
        output_path: std::path::PathBuf::from("test.wav"),
        reference_audio: None,
        reference_text: None,
        seed: None,
        max_new_tokens: 4096,
        auto_download: true,
        hf_mirror: Some(HF_MIRROR_ENDPOINT.to_string()),
    };

    assert_eq!(params.models_base_dir, std::path::PathBuf::from("models"));
    assert_eq!(params.backend, BackendKind::Candle);
    assert!(params.auto_download);
    assert_eq!(params.hf_mirror.as_deref(), Some("https://hf-mirror.com"));
}
