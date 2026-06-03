//! # Qwen3-TTS 文字轉語音範例
//!
//! 展示完整的文字 → 語音管線：
//!
//! 1. TextFrontend (PythonBridge 或 CandleLLM): 文字 → 多碼本 Token
//! 2. Decoder12Hz: Token → PCM 音訊
//! 3. hound: PCM → WAV 檔案
//!
//! ## 使用方式
//!
//! ```bash
//! # 需要先下載 Tokenizer 權重
//! # huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir weights/tokenizer
//!
//! # 基本用法（使用 Python 橋接，0.6B Base 模型）
//! cargo run --example synthesize -- --text "你好，今天天氣真好。"
//!
//! # 純 Rust / Candle 後端（無 Python 依賴，需 --features candle-llm）
//! cargo run --example synthesize --features candle-llm -- \
//!     --text "你好" --backend candle --model-dir <path-to-Qwen3-TTS-12Hz-0.6B-Base>
//!
//! # 指定說話者與語言
//! cargo run --example synthesize -- --text "Hello world" --language en --speaker default
//!
//! # 使用 1.7B CustomVoice 模型（需 GPU）
//! cargo run --example synthesize -- --text "你好" --model "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
//!
//! # 指定輸出檔案
//! cargo run --example synthesize -- --text "測試" --output test.wav
//!
//! # 保存語義 Token，之後可跳過 Python 前端、用 Rust 原生 decode 重跑
//! cargo run --example synthesize -- --text "你好" --save-tokens tokens.bin
//! cargo run --example synthesize -- --tokens tokens.bin --output native.wav
//! ```

use std::path::Path;
#[cfg(feature = "candle-llm")]
use std::path::PathBuf;

use qwen3tts::text_frontend::{PythonBridge, SynthesisOptions, TextFrontend, TokenStream};
use qwen3tts::{Decoder12Hz, DecoderConfig};

// ---------------------------------------------------------------------------
// 文字前端後端選擇
// ---------------------------------------------------------------------------

/// 文字前端後端類型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendKind {
    /// Python 子行程橋接（依賴 qwen-tts Python 套件）
    Python,
    /// 純 Rust / Candle 原生 LLM（需 candle-llm feature + tokenizer.json）
    #[cfg(feature = "candle-llm")]
    Candle,
}

#[cfg(feature = "candle-llm")]
fn default_backend() -> BackendKind {
    BackendKind::Candle
}

#[cfg(not(feature = "candle-llm"))]
fn default_backend() -> BackendKind {
    BackendKind::Python
}

/// 自動尋找 HuggingFace 快取中的 Qwen3-TTS 模型 snapshot 目錄
#[cfg(feature = "candle-llm")]
fn locate_model_snapshot(model_id: &str) -> Option<PathBuf> {
    if let Ok(env_dir) = std::env::var("QWEN3_TTS_MODEL_DIR") {
        let p = PathBuf::from(env_dir);
        if p.join("model.safetensors").exists() {
            return Some(p);
        }
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()
        .map(PathBuf::from)?;
    let hub = home.join(".cache").join("huggingface").join("hub");
    let repo_dir = hub.join(format!("models--{}", model_id.replace('/', "--")));
    let snapshots = repo_dir.join("snapshots");
    let Ok(entries) = std::fs::read_dir(&snapshots) else {
        return None;
    };
    for entry in entries.flatten() {
        let candidate = entry.path().join("model.safetensors");
        if candidate.exists() {
            return Some(entry.path());
        }
    }
    None
}

#[cfg(feature = "candle-llm")]
fn display_model_id(model_id: &str, model_dir: &Path) -> String {
    for component in model_dir.components().rev() {
        let name = component.as_os_str().to_string_lossy();
        if let Some(cache_name) = name.strip_prefix("models--") {
            return cache_name.replace("--", "/");
        }
    }
    model_id.to_string()
}

fn main() {
    // ----- 解析命令列參數 -----
    let args: Vec<String> = std::env::args().collect();
    let mut text = String::new();
    let mut model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base".to_string();
    let mut output_path = "output.wav".to_string();
    let mut language = "auto".to_string();
    let mut speaker: Option<String> = None;
    let mut tokens_path: Option<String> = None;
    let mut save_tokens_path: Option<String> = None;
    let mut text_only = false; // 只跑到 LLM 階段，產生 Token 後直接結束
    let mut max_new_tokens: u32 = 4096; // 最大生成 Token 數
    #[cfg(feature = "candle-llm")]
    let mut model_dir: Option<String> = None; // 給 CandleLLM 用
    let mut backend = default_backend();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--text" => {
                require_arg(&args, i, "--text");
                text = args[i + 1].clone();
                i += 2;
            }
            "--model" => {
                require_arg(&args, i, "--model");
                model_id = args[i + 1].clone();
                i += 2;
            }
            "--output" | "-o" => {
                require_arg(&args, i, "--output");
                output_path = args[i + 1].clone();
                i += 2;
            }
            "--language" | "-l" => {
                require_arg(&args, i, "--language");
                language = args[i + 1].clone();
                i += 2;
            }
            "--speaker" | "-s" => {
                require_arg(&args, i, "--speaker");
                speaker = Some(args[i + 1].clone());
                i += 2;
            }
            "--tokens" => {
                require_arg(&args, i, "--tokens");
                tokens_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--save-tokens" => {
                require_arg(&args, i, "--save-tokens");
                save_tokens_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--text-only" => {
                text_only = true;
                i += 1;
            }
            "--max-new-tokens" => {
                require_arg(&args, i, "--max-new-tokens");
                max_new_tokens = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("--max-new-tokens 必須是正整數");
                    std::process::exit(1);
                });
                i += 2;
            }
            "--model-dir" => {
                require_arg(&args, i, "--model-dir");
                #[cfg(feature = "candle-llm")]
                {
                    model_dir = Some(args[i + 1].clone());
                    i += 2;
                }
                #[cfg(not(feature = "candle-llm"))]
                {
                    eprintln!(
                        "錯誤: --model-dir 需要 --features candle-llm。\n\
                         重新編譯: cargo run --example synthesize --features candle-llm -- ..."
                    );
                    std::process::exit(1);
                }
            }
            "--backend" | "-b" => {
                require_arg(&args, i, "--backend");
                backend = match args[i + 1].as_str() {
                    "python" | "py" => BackendKind::Python,
                    #[cfg(feature = "candle-llm")]
                    "candle" | "rust" | "native" => BackendKind::Candle,
                    #[cfg(not(feature = "candle-llm"))]
                    "candle" | "rust" | "native" => {
                        eprintln!(
                            "錯誤: --backend candle 需要 --features candle-llm。\n\
                             重新編譯: cargo run --example synthesize --features candle-llm -- ..."
                        );
                        std::process::exit(1);
                    }
                    other => {
                        eprintln!("未知後端: {other}（僅支援 python / candle）");
                        std::process::exit(1);
                    }
                };
                i += 2;
            }
            "--help" | "-h" => {
                print_usage();
                return;
            }
            _ => {
                eprintln!("未知參數: {}", args[i]);
                print_usage();
                std::process::exit(1);
            }
        }
    }

    // --tokens 與 --text 二選一
    if text.is_empty() && tokens_path.is_none() {
        eprintln!("請使用 --text 指定要合成的文字（或用 --tokens 指定 Token 檔）");
        print_usage();
        std::process::exit(1);
    }

    println!("╔══════════════════════════════════════╗");
    println!("║    Qwen3-TTS Rust 文字轉語音        ║");
    println!("╚══════════════════════════════════════╝");
    println!("文字    : {text}");
    println!("模型    : {model_id}");
    println!("後端    : {:?}", backend);
    println!("語言    : {language}");
    println!("輸出    : {output_path}");
    if let Some(tp) = &save_tokens_path {
        println!("保存Token: {tp}");
    }
    println!();

    // ----- 步驟 1: 載入 Tokenizer 解碼器權重 -----
    let weight_dir = Path::new("weights/tokenizer");
    if !weight_dir.join("codebook.safetensors").exists() {
        eprintln!("錯誤: {weight_dir:?} 不存在。");
        eprintln!("請先下載 Tokenizer 權重:");
        eprintln!(
            "  huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir weights/tokenizer"
        );
        std::process::exit(1);
    }

    let device = candle_core::Device::Cpu;
    let config = DecoderConfig::realtime();

    println!("[1/3] 載入 Tokenizer 解碼器…");
    let mut decoder = Decoder12Hz::from_safetensors(config, weight_dir, &device)
        .expect("載入 Tokenizer 權重失敗");

    // ----- 步驟 2: 獲取 Token（透過 LLM 或從檔案載入）-----
    let stream: TokenStream = if let Some(tp) = tokens_path {
        println!("[2/3] 從二進位檔載入 Token ({tp})…");
        let data = std::fs::read(&tp).expect("讀取 Token 檔失敗");
        qwen3tts::text_frontend::TokenParser::parse_binary(&data).expect("解析 Token 檔失敗")
    } else {
        match backend {
            BackendKind::Python => {
                println!("[2/3] 載入 Python 橋接 LLM ({model_id}) 並生成 Token…");
                println!("      （首次載入需下載權重，約 1-5 分鐘）");
                let bridge = PythonBridge::new(&model_id)
                    .expect("建立 PythonBridge 失敗")
                    .with_python("python");
                let options = SynthesisOptions {
                    language,
                    speaker,
                    temperature: 0.9,
                    top_k: 50,
                    top_p: 1.0,
                    max_new_tokens,
                };
                bridge
                    .synthesize(&text, &options)
                    .expect("Python LLM Token 生成失敗")
            }
            #[cfg(feature = "candle-llm")]
            BackendKind::Candle => {
                use qwen3tts::text_frontend::CandleLLM;

                // 決定模型目錄：--model-dir > QWEN3_TTS_MODEL_DIR > HF 快取掃描
                let dir = if let Some(d) = model_dir.as_ref() {
                    PathBuf::from(d)
                } else {
                    locate_model_snapshot(&model_id).unwrap_or_else(|| {
                        eprintln!(
                            "錯誤: 找不到 {model_id} 的本地 snapshot。請提供 --model-dir。\n\
                             範例: --model-dir ~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/<sha>"
                        );
                        std::process::exit(1);
                    })
                };

                let sf_path = dir.join("model.safetensors");
                if !sf_path.exists() {
                    eprintln!("錯誤: {sf_path:?} 不存在");
                    std::process::exit(1);
                }
                let candidate1 = dir.join("tokenizer.json");
                let candidate2 = PathBuf::from("models/tokenizer.json");
                let tok_path = if candidate1.exists() {
                    candidate1
                } else if candidate2.exists() {
                    candidate2
                } else {
                    eprintln!(
                        "錯誤: 找不到 tokenizer.json。\n\
                         請先產生：\n  \
                         python tools/build_tokenizer.py --model-dir {dir:?}\n\
                         或複製到 models/tokenizer.json"
                    );
                    std::process::exit(1);
                };

                let display_model = display_model_id(&model_id, &dir);
                println!("[2/3] 載入 Candle LLM ({display_model}) 並生成 Token…");
                println!("      model dir : {dir:?}");
                println!("      （首次載入時間依模型大小與裝置而定，包含權重 BF16→F32）");

                let backend = CandleLLM::from_files(&sf_path, &tok_path, &device)
                    .expect("載入 CandleLLM 失敗");
                let options = SynthesisOptions {
                    language,
                    speaker,
                    temperature: 0.9,
                    top_k: 50,
                    top_p: 1.0,
                    max_new_tokens,
                };
                backend
                    .synthesize(&text, &options)
                    .expect("Candle LLM Token 生成失敗")
            }
        }
    };

    let num_frames = stream.num_frames();
    if num_frames == 0 {
        eprintln!("錯誤: 未產生任何 Token");
        std::process::exit(1);
    }
    if let Some(tp) = &save_tokens_path {
        stream.write_binary(tp).expect("保存 Token 檔失敗");
        println!("      → 已保存 Token: {tp}");
    }
    println!(
        "      → {num_frames} 幀 ({:.1} 秒語音)",
        stream.duration_sec()
    );

    // --text-only: 產生 Token 後直接結束，跳過解碼器（debug / 煙霧測試用）
    if text_only {
        println!();
        println!("✅ 完成 (--text-only): 已產生 {num_frames} 幀 Token，未執行解碼");
        if save_tokens_path.is_none() {
            println!("   提示：用 --save-tokens <檔案> 保存 Token");
        }
        return;
    }

    // ----- 步驟 3: 解碼 Token → PCM 音訊 -----
    println!("[3/3] 解碼 Token → 音訊…");
    let sample_rate = 24000u32;
    let all_samples = decoder.decode_frames(&stream.frames).expect("解碼失敗");

    let duration_sec = all_samples.len() as f64 / sample_rate as f64;
    println!(
        "      產出 {:.2} 秒音頻 ({} 個樣本)",
        duration_sec,
        all_samples.len()
    );

    // ----- 步驟 4: 寫入 WAV 檔案 -----
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&output_path, spec).expect("無法建立 WAV 檔案");

    for &sample in &all_samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let int_sample = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(int_sample).expect("寫入樣本失敗");
    }

    writer.finalize().expect("關閉 WAV 檔案失敗");
    println!();
    println!(
        "✅ 完成! 已儲存: {output_path} ({:.1} MB)",
        all_samples.len() as f64 * 2.0 / 1_048_576.0
    );
}

fn require_arg(args: &[String], i: usize, flag: &str) {
    if i + 1 >= args.len() {
        eprintln!("{flag} 需要參數");
        std::process::exit(1);
    }
}

fn print_usage() {
    let backend_default = if cfg!(feature = "candle-llm") {
        "candle"
    } else {
        "python"
    };
    eprintln!(
        "用法: cargo run --example synthesize -- --text \"合成文字\" [選項]

選項:
  --text <文字>      要合成的文字（與 --tokens 二選一）
  --tokens <檔案>    從二進位 Token 檔載入（跳過 LLM 階段）
  --save-tokens <檔案>
                    保存 Token 二進位檔
  --model <ID>       HuggingFace 模型 ID（預設: Qwen/Qwen3-TTS-12Hz-0.6B-Base）
  --model-dir <路徑> 本地模型目錄（給 Candle 後端用，自動從 HF 快取找）
  --backend / -b     文字前端後端：python | candle（預設: {backend_default}）
  --language / -l    語言（預設: auto）
  --speaker / -s     說話者名稱（可選）
  --output / -o      輸出 WAV 路徑（預設: output.wav）
  --max-new-tokens N 最大生成 Token 數（預設: 4096）
  --text-only        只跑到 LLM 階段產生 Token，不解碼成音訊
  --help / -h        顯示此說明

範例:
  # Python 橋接（無需 candle-llm feature）
  cargo run --example synthesize -- --text \"你好\"

  # 純 Rust / Candle 後端（需 --features candle-llm，無 Python 依賴）
  cargo run --example synthesize --features candle-llm -- \\
      --text \"你好\" --backend candle

  # 指定模型目錄（給 Candle 後端）
  cargo run --example synthesize --features candle-llm -- \\
      --text \"你好\" --backend candle \\
      --model-dir ~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/<sha>

模型清單:
  0.6B（無 GPU，約 1.2GB RAM）:
    Qwen/Qwen3-TTS-12Hz-0.6B-Base
    Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

  1.7B（建議 GPU，約 4GB VRAM）:
    Qwen/Qwen3-TTS-12Hz-1.7B-Base
    Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
"
    );
}
