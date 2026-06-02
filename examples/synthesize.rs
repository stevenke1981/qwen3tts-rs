//! # Qwen3-TTS 文字轉語音範例
//!
//! 展示完整的文字 → 語音管線：
//!
//! 1. TextFrontend (PythonBridge): 文字 → 多碼本 Token
//! 2. Decoder12Hz: Token → PCM 音訊
//! 3. hound: PCM → WAV 檔案
//!
//! ## 使用方式
//!
//! ```bash
//! # 需要先下載 Tokenizer 權重
//! # huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir weights/tokenizer
//!
//! # 基本用法（使用 0.6B Base 模型）
//! cargo run --example synthesize -- --text "你好，今天天氣真好。"
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

use qwen3tts::text_frontend::{PythonBridge, SynthesisOptions, TextFrontend};
use qwen3tts::{Decoder12Hz, DecoderConfig};

fn main() {
    // ----- 解析命令列參數 -----
    let args: Vec<String> = std::env::args().collect();
    let mut text = String::new();
    let mut model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base".to_string();
    let mut output_path = "output.wav".to_string();
    let mut language = "auto".to_string();
    let mut speaker: Option<String> = None;
    let mut tokens_path: Option<String> = None; // 可選：直接從二進位檔載入 Token
    let mut save_tokens_path: Option<String> = None; // 可選：保存 Token 供 Rust 原生 decode 重跑

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--text" => {
                if i + 1 < args.len() {
                    text = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("--text 需要參數");
                    return;
                }
            }
            "--model" => {
                if i + 1 < args.len() {
                    model_id = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("--model 需要參數");
                    return;
                }
            }
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    output_path = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("--output 需要參數");
                    return;
                }
            }
            "--language" | "-l" => {
                if i + 1 < args.len() {
                    language = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("--language 需要參數");
                    return;
                }
            }
            "--speaker" | "-s" => {
                if i + 1 < args.len() {
                    speaker = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("--speaker 需要參數");
                    return;
                }
            }
            "--tokens" => {
                if i + 1 < args.len() {
                    tokens_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("--tokens 需要參數");
                    return;
                }
            }
            "--save-tokens" => {
                if i + 1 < args.len() {
                    save_tokens_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("--save-tokens 需要參數");
                    return;
                }
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

    // 如果提供了 --tokens，則不須 --text
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
    let stream = if let Some(tp) = tokens_path {
        println!("[2/3] 從二進位檔載入 Token ({tp})…");
        let data = std::fs::read(&tp).expect("讀取 Token 檔失敗");
        qwen3tts::text_frontend::TokenParser::parse_binary(&data).expect("解析 Token 檔失敗")
    } else {
        println!("[2/3] 載入 LLM ({model_id}) 並生成 Token…");
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
            max_new_tokens: 4096,
        };

        bridge
            .synthesize(&text, &options)
            .expect("LLM Token 生成失敗")
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

fn print_usage() {
    eprintln!(
        "用法: cargo run --example synthesize -- --text \"合成文字\" [選項]

選項:
  --text <文字>      要合成的文字（與 --tokens 二選一）
  --tokens <檔案>    從二進位 Token 檔載入（跳過 LLM 階段，可與 --text 互斥）
  --save-tokens <檔案>
                    保存 Token 二進位檔，之後可用 --tokens 走 Rust 原生 decode
  --model <ID>       HuggingFace 模型 ID（預設: Qwen/Qwen3-TTS-12Hz-0.6B-Base）
  --language / -l    語言（預設: auto）
  --speaker / -s     說話者名稱（可選）
  --output / -o      輸出 WAV 路徑（預設: output.wav）
  --help / -h        顯示此說明

使用 0.6B 模型（無 GPU，約 1.2GB RAM）:
  Qwen/Qwen3-TTS-12Hz-0.6B-Base
  Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

使用 1.7B 模型（建議 GPU，約 4GB VRAM）:
  Qwen/Qwen3-TTS-12Hz-1.7B-Base
  Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
  Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
"
    );
}
