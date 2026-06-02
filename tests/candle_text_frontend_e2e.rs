//! # 文字前端端到端測試
//!
//! 驗證 `CandleLLM` 從純文字到多碼本 Token 的整個推理流程，
//! 與 `tests/fixtures/talker_prompt_generation.json` 的 PyTorch 參考輸出位元對齊。
//!
//! 僅在 `candle-llm` feature 啟用時編譯（需 `tokenizers` crate 與 0.6B 模型權重）。
//!
//! 執行：
//! ```bash
//! cargo test --features candle-llm --test candle_text_frontend_e2e -- --ignored --nocapture
//! ```

#![cfg(feature = "candle-llm")]

use std::path::PathBuf;

use candle_core::Device;
use qwen3tts::text_frontend::{CandleLLM, SynthesisOptions, TextFrontend};

// ---------------------------------------------------------------------------
// 模型路徑
// ---------------------------------------------------------------------------

fn default_model_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("QWEN3_TTS_MODEL_SAFETENSORS") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    let home = std::env::var("USERPROFILE").ok().map(PathBuf::from)?;
    let snapshots = home
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join("models--Qwen--Qwen3-TTS-12Hz-0.6B-Base")
        .join("snapshots");
    let entries = std::fs::read_dir(snapshots).ok()?;
    for entry in entries.flatten() {
        let candidate = entry.path().join("model.safetensors");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn default_tokenizer_path() -> Option<PathBuf> {
    // 優先使用環境變數
    if let Ok(path) = std::env::var("QWEN3_TTS_TOKENIZER") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    // 嘗試 repo 內的 models/tokenizer.json
    let dev = PathBuf::from("models/tokenizer.json");
    if dev.exists() {
        return Some(dev);
    }
    // 嘗試在模型目錄
    if let Some(model) = default_model_path() {
        let in_model = model.parent()?.join("tokenizer.json");
        if in_model.exists() {
            return Some(in_model);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Fixture 載入
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
struct PromptGenerationFixture {
    text: String,
    #[allow(dead_code)]
    language: String,
    num_frames: usize,
    generated: Vec<Vec<u16>>,
    #[allow(dead_code)]
    #[serde(default)]
    c0_tokens: Vec<u32>,
}

fn load_fixture(path: &std::path::Path) -> PromptGenerationFixture {
    let data = std::fs::read_to_string(path).expect("read fixture");
    serde_json::from_str(&data).expect("parse fixture")
}

// ---------------------------------------------------------------------------
// 測試
// ---------------------------------------------------------------------------

/// 端到端：文字 → CandleLLM.synthesize → TokenStream
/// 與 PyTorch 參考實作生成的 3 幀做精確比對
#[test]
#[ignore = "loads the full 0.6B talker model + tokenizer; run with --ignored"]
fn candle_text_frontend_e2e_matches_pytorch() {
    let fixture_path = std::path::Path::new("tests/fixtures/talker_prompt_generation.json");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found at {fixture_path:?}");
        return;
    }

    let model_path = match default_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: Qwen3-TTS model.safetensors not found");
            return;
        }
    };
    let tokenizer_path = match default_tokenizer_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: tokenizer.json not found (run tools/build_tokenizer.py)");
            return;
        }
    };

    let fixture = load_fixture(fixture_path);
    let device = Device::Cpu;
    let backend = CandleLLM::from_files(&model_path, &tokenizer_path, &device)
        .expect("failed to load CandleLLM");

    let opts = SynthesisOptions {
        language: "chinese".to_string(),
        max_new_tokens: (fixture.num_frames * 8) as u32, // 允許提前停止
        temperature: 0.0,                                // greedy
        ..Default::default()
    };

    let stream = backend
        .synthesize(&fixture.text, &opts)
        .expect("synthesize failed");

    let actual: Vec<Vec<u16>> = stream.frames.iter().map(|f| f.to_vec()).collect();
    let expected = &fixture.generated;

    println!(
        "Generated {} frames (PyTorch generated {})",
        actual.len(),
        expected.len()
    );

    // 比較 fixture 中所有幀（PyTorch 截斷到 N 幀，native 會繼續生成）
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        println!("Frame {i}:");
        println!("  actual:   {:?}", a);
        println!("  expected: {:?}", e);
        assert_eq!(a, e, "Frame {i} mismatch");
    }

    assert!(
        actual.len() >= expected.len(),
        "Native backend generated {} frames, fewer than PyTorch's {}",
        actual.len(),
        expected.len()
    );
}

/// 輕量煙霧測試：只要能跑通基本流程，不比較具體 token
#[test]
#[ignore = "loads the full 0.6B talker model; run with --ignored"]
fn candle_text_frontend_smoke_test() {
    let model_path = match default_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: Qwen3-TTS model.safetensors not found");
            return;
        }
    };
    let tokenizer_path = match default_tokenizer_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: tokenizer.json not found");
            return;
        }
    };

    let device = Device::Cpu;
    let backend = CandleLLM::from_files(&model_path, &tokenizer_path, &device)
        .expect("failed to load CandleLLM");

    let opts = SynthesisOptions {
        language: "chinese".to_string(),
        max_new_tokens: 32, // 跑 ~4 幀或更少（會因 EOS 提前停止）
        temperature: 0.0,
        ..Default::default()
    };

    let stream = backend
        .synthesize("你好世界", &opts)
        .expect("synthesize failed");

    assert!(
        !stream.frames.is_empty(),
        "Should generate at least one frame"
    );
    assert_eq!(
        stream.frames[0].len(),
        16,
        "Each frame should have 16 codebook tokens"
    );
    println!(
        "Smoke test: generated {} frames in ~0.16s of audio ({} Hz)",
        stream.num_frames(),
        stream.sample_rate
    );
}

/// 測試 tokenizer 對齊：使用 Qwen2Tokenizer 行為，
/// 確認 prompt 構造與 PyTorch 完全一致
#[test]
#[ignore = "loads tokenizer + talker; run with --ignored"]
fn candle_text_frontend_prompt_ids_match_pytorch() {
    let model_path = match default_model_path() {
        Some(p) => p,
        None => return,
    };
    let tokenizer_path = match default_tokenizer_path() {
        Some(p) => p,
        None => return,
    };

    let device = Device::Cpu;
    let backend = CandleLLM::from_files(&model_path, &tokenizer_path, &device).unwrap();

    // 「你好」: 預期 [151644, 77091, 198, 108386, 151645, 198, 151644, 77091, 198]
    let prompt = "<|im_start|>assistant\n你好<|im_end|>\n<|im_start|>assistant\n";
    let encoding = backend.tokenizer().encode(prompt, false).unwrap();
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    assert_eq!(
        ids,
        vec![151644, 77091, 198, 108386, 151645, 198, 151644, 77091, 198],
        "Token IDs for '你好' chat template must match PyTorch fixture"
    );
    assert_eq!(ids.len(), 9, "Total prompt length must be 9 tokens");
}
