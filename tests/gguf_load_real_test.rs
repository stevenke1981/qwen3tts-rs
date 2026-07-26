//! 端到端驗證：從真實 GGUF 檔案載入 TalkerWeightLoader
//!
//! 使用 Serveurperso/Qwen3-TTS-GGUF 的 qwen-talker-0.6b-base-Q4_K_M.gguf
//! 驗證 from_gguf → gguf_key_to_safetensors → build_talker 完整管線。
//!
//! 需要先下載檔案至 gguf-test/ 目錄。
//! 若檔案不存在，測試會自動 skip。

use candle_core::Device;
use qwen3tts::talker::weight_loader::TalkerWeightLoader;

/// 必要 tensor 清單（按類別分組）
const ESSENTIAL_TENSORS: &[&str] = &[
    // ── 頂層 ──
    "talker.model.text_embedding.weight",
    "talker.model.codec_embedding.weight",
    "talker.codec_head.weight",
    "talker.model.norm.weight",
    "talker.text_projection.linear_fc1.weight",
    "talker.text_projection.linear_fc1.bias",
    "talker.text_projection.linear_fc2.weight",
    "talker.text_projection.linear_fc2.bias",
    // ── Talker 主模型第 0 層（11 種 tensor）──
    "talker.model.layers.0.self_attn.q_proj.weight",
    "talker.model.layers.0.self_attn.k_proj.weight",
    "talker.model.layers.0.self_attn.v_proj.weight",
    "talker.model.layers.0.self_attn.o_proj.weight",
    "talker.model.layers.0.self_attn.q_norm.weight",
    "talker.model.layers.0.self_attn.k_norm.weight",
    "talker.model.layers.0.input_layernorm.weight",
    "talker.model.layers.0.post_attention_layernorm.weight",
    "talker.model.layers.0.mlp.gate_proj.weight",
    "talker.model.layers.0.mlp.up_proj.weight",
    "talker.model.layers.0.mlp.down_proj.weight",
    // ── Talker 最後一層 ──
    "talker.model.layers.27.self_attn.q_proj.weight",
    "talker.model.layers.27.self_attn.o_proj.weight",
    // ── Code Predictor 第 0 層 ──
    "talker.code_predictor.model.layers.0.self_attn.q_proj.weight",
    "talker.code_predictor.model.layers.0.self_attn.o_proj.weight",
    // ── Code Predictor 最後層 ──
    "talker.code_predictor.model.layers.4.mlp.down_proj.weight",
    "talker.code_predictor.model.norm.weight",
    // ── Code Predictor 子碼本嵌入 ──
    "talker.code_predictor.model.codec_embedding.0.weight",
    "talker.code_predictor.model.codec_embedding.14.weight",
    "talker.code_predictor.lm_head.0.weight",
    "talker.code_predictor.lm_head.14.weight",
];

fn gguf_path() -> Option<std::path::PathBuf> {
    let candidates = [
        "gguf-test/qwen-talker-0.6b-base-Q4_K_M.gguf",
        "../gguf-test/qwen-talker-0.6b-base-Q4_K_M.gguf",
        "E:/qwen3tts-rs/gguf-test/qwen-talker-0.6b-base-Q4_K_M.gguf",
    ];
    candidates
        .iter()
        .map(std::path::PathBuf::from)
        .find(|p| p.exists())
}

#[test]
fn gguf_weight_loader_loads_all_tensors() {
    let Some(path) = gguf_path() else {
        eprintln!("⚠️  GGUF 測試檔案不存在，跳過測試。");
        eprintln!("   下載: https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF");
        return;
    };

    let device = Device::Cpu;
    let loader = TalkerWeightLoader::from_gguf(&path, &device).expect("from_gguf 載入失敗");

    // ── 驗證全部必要 tensor 存在 ──
    let mut missing: Vec<&str> = Vec::new();
    for key in ESSENTIAL_TENSORS {
        if loader.get(key).is_err() {
            missing.push(key);
        }
    }

    assert!(
        missing.is_empty(),
        "缺少 {} 個 tensor: {:?}",
        missing.len(),
        missing
    );

    // ── Shape 驗證 ──
    let codec_emb = loader.get("talker.model.codec_embedding.weight").unwrap();
    assert_eq!(codec_emb.dims(), &[3072, 1024]);

    let text_emb = loader.get("talker.model.text_embedding.weight").unwrap();
    assert_eq!(text_emb.dims(), &[151936, 2048]);

    let q_proj_0 = loader
        .get("talker.model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(q_proj_0.dims(), &[2048, 1024]);

    let cp_codec_0 = loader
        .get("talker.code_predictor.model.codec_embedding.0.weight")
        .unwrap();
    assert_eq!(cp_codec_0.dims(), &[2048, 1024]);

    let lm_head_0 = loader
        .get("talker.code_predictor.lm_head.0.weight")
        .unwrap();
    assert_eq!(lm_head_0.dims(), &[2048, 1024]);

    // 0.6B 沒有 mtp_proj
    assert!(
        loader
            .get("talker.code_predictor.small_to_mtp_projection.weight")
            .is_err(),
        "0.6B 不應有 mtp_projection"
    );
}

#[test]
fn gguf_infer_config_matches_0_6b() {
    let Some(path) = gguf_path() else {
        eprintln!("⚠️  GGUF 測試檔案不存在，跳過測試。");
        return;
    };

    let device = Device::Cpu;
    let loader = TalkerWeightLoader::from_gguf(&path, &device).expect("from_gguf 載入失敗");

    let config = loader.infer_config().expect("infer_config 失敗");

    // 0.6B 預期值
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.intermediate_size, 3072);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.num_hidden_layers, 28);
    assert_eq!(config.text_hidden_size, 2048);
    assert_eq!(config.text_vocab_size, 151936);
    assert_eq!(config.vocab_size, 3072);
    assert_eq!(config.num_code_groups, 16);
    assert_eq!(config.code_predictor.hidden_size, 1024);
    assert_eq!(config.code_predictor.intermediate_size, 3072);
    assert_eq!(config.code_predictor.num_hidden_layers, 5);
    assert_eq!(config.code_predictor.vocab_size, 2048);
    assert_eq!(config.code_predictor.num_code_groups, 16);
}

#[test]
fn gguf_build_talker_creates_full_model() {
    let Some(path) = gguf_path() else {
        eprintln!("⚠️  GGUF 測試檔案不存在，跳過測試。");
        return;
    };

    let device = Device::Cpu;
    let loader = TalkerWeightLoader::from_gguf(&path, &device).expect("from_gguf 載入失敗");

    let config = loader.infer_config().expect("infer_config 失敗");
    let talker = loader.build_talker(&config).expect("build_talker 失敗");

    // 驗證結構
    assert_eq!(talker.model.layers.len(), 28);
    assert_eq!(talker.code_predictor.layers.len(), 5);
    assert_eq!(talker.code_predictor.codec_embeddings.len(), 15);
    assert_eq!(talker.code_predictor.lm_heads.len(), 15);
    assert!(talker.code_predictor.small_to_mtp_proj.is_none());
}
