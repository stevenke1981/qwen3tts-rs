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

#[cfg(feature = "candle-llm")]
use std::path::{Path, PathBuf};

use qwen3tts::paths::ensure_tokenizer_weight_dir;
use qwen3tts::text_frontend::model_catalog::{
    GenerationMode, SUPPORTED_LANGUAGES, model_capability, model_table, validate_generation_request,
};
use qwen3tts::text_frontend::speaker_presets;
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
    locate_hf_model_snapshot(model_id)
}

#[cfg(feature = "candle-llm")]
fn locate_hf_model_snapshot(model_id: &str) -> Option<PathBuf> {
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
fn find_tokenizer_json_for_model(model_id: &str, model_dir: &Path) -> Option<PathBuf> {
    let local = model_dir.join("tokenizer.json");
    if local.exists() {
        return Some(local);
    }

    let snapshots = model_dir.join("snapshots");
    if snapshots.is_dir() {
        for entry in std::fs::read_dir(snapshots).ok()?.flatten() {
            let path = entry.path().join("tokenizer.json");
            if path.exists() {
                return Some(path);
            }
        }
    }

    for fallback in ["models/tokenizer_1.7b.json", "models/tokenizer.json"] {
        let path = PathBuf::from(fallback);
        if path.exists() {
            return Some(path);
        }
    }

    let base_ids: &[&str] = if model_id.contains("0.6B") {
        &[
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ]
    } else {
        &[
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ]
    };

    for base_id in base_ids {
        if let Some(base_dir) = locate_hf_model_snapshot(base_id) {
            let path = base_dir.join("tokenizer.json");
            if path.exists() {
                return Some(path);
            }
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
    let mut instruct: Option<String> = None;
    let mut instruct_file: Option<String> = None;
    let mut mode = GenerationMode::Auto;
    let mut reference_audio: Option<String> = None;
    let mut reference_text: Option<String> = None;
    let mut seed: Option<u64> = None;
    let mut tokens_path: Option<String> = None;
    let mut save_tokens_path: Option<String> = None;
    let mut text_only = false; // 只跑到 LLM 階段，產生 Token 後直接結束
    let mut max_new_tokens: u32 = 4096; // 最大生成 Token 數
    #[cfg(feature = "candle-llm")]
    let mut model_dir: Option<String> = None; // 給 CandleLLM 用
    let mut backend = default_backend();
    let mut speed = 1.0;

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
            "--instruct" => {
                require_arg(&args, i, "--instruct");
                instruct = Some(args[i + 1].clone());
                i += 2;
            }
            "--instruct-file" => {
                require_arg(&args, i, "--instruct-file");
                instruct_file = Some(args[i + 1].clone());
                i += 2;
            }
            "--mode" => {
                require_arg(&args, i, "--mode");
                mode = GenerationMode::parse(&args[i + 1]).unwrap_or_else(|err| {
                    eprintln!("錯誤: {err}");
                    std::process::exit(1);
                });
                i += 2;
            }
            "--reference-audio" => {
                require_arg(&args, i, "--reference-audio");
                reference_audio = Some(args[i + 1].clone());
                i += 2;
            }
            "--reference-text" => {
                require_arg(&args, i, "--reference-text");
                reference_text = Some(args[i + 1].clone());
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
            "--seed" => {
                require_arg(&args, i, "--seed");
                seed = Some(args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("--seed 必須是 0..18446744073709551615 的整數");
                    std::process::exit(1);
                }));
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
            "--speed" => {
                require_arg(&args, i, "--speed");
                speed = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("--speed 必須是浮點數");
                    std::process::exit(1);
                });
                if speed <= 0.0 {
                    eprintln!("--speed 必須大於 0.0");
                    std::process::exit(1);
                }
                i += 2;
            }
            "--version" | "-V" => {
                println!("qwen3tts-rs {}", env!("CARGO_PKG_VERSION"));
                return;
            }
            "--list-speakers" => {
                print_speakers();
                return;
            }
            "--list-models" => {
                print_models();
                return;
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

    let instruct = resolve_instruct(instruct, instruct_file);
    let validation_model = model_dir.as_deref().unwrap_or(&model_id);
    if let Err(err) = validate_generation_request(
        validation_model,
        mode,
        speaker.as_deref(),
        instruct.as_deref(),
        reference_audio.as_deref(),
    ) {
        eprintln!("錯誤: {err}");
        std::process::exit(1);
    }

    // --tokens 與 --text 二選一
    if text.is_empty() && tokens_path.is_none() {
        eprintln!("請使用 --text 指定要合成的文字（或用 --tokens 指定 Token 檔）");
        print_usage();
        std::process::exit(1);
    }

    #[cfg(feature = "candle-llm")]
    let banner_model = if backend == BackendKind::Candle {
        model_dir
            .as_deref()
            .map(|dir| display_model_id(&model_id, Path::new(dir)))
            .unwrap_or_else(|| model_id.clone())
    } else {
        model_id.clone()
    };
    #[cfg(not(feature = "candle-llm"))]
    let banner_model = model_id.clone();

    println!("╔══════════════════════════════════════╗");
    println!("║    Qwen3-TTS Rust 文字轉語音        ║");
    println!("╚══════════════════════════════════════╝");
    println!("文字    : {text}");
    println!("模型    : {banner_model}");
    if let Some(capability) =
        model_capability(&banner_model).or_else(|| model_capability(validation_model))
    {
        println!(
            "能力    : {} / {} / {} languages / streaming={}",
            capability.parameters,
            capability.main_function,
            capability.languages,
            if capability.streaming { "yes" } else { "no" }
        );
    }
    println!("後端    : {:?}", backend);
    println!("模式    : {}", mode.as_str());
    println!("語言    : {language}");
    println!("輸出    : {output_path}");
    if let Some(spk) = &speaker {
        println!("speaker : {spk}");
        if let Some(preset) = speaker_presets::lookup(spk) {
            println!(
                "preset  : {} / {} / {}",
                preset.name, preset.description, preset.native_language
            );
        }
    }
    if speed != 1.0 {
        println!("語速    : {speed}x");
    }
    if let Some(ins) = &instruct {
        println!("指令    : {ins}");
    }
    if let Some(path) = &reference_audio {
        println!("參考音訊: {path}");
    }
    if let Some(text) = &reference_text {
        println!("參考逐字: {text}");
    }
    if mode == GenerationMode::VoiceClone && reference_text.is_none() {
        eprintln!(
            "提示: Voice Clone 最佳效果建議提供 --reference-text；未提供時只能使用 speaker-embedding-only 模式，音色/內容穩定性可能較差。"
        );
    }
    if let Some(seed) = seed {
        println!("seed    : {seed}");
    }
    if let Some(tp) = &save_tokens_path {
        println!("保存Token: {tp}");
    }
    println!();

    if tokens_path.is_none() && mode == GenerationMode::VoiceClone && backend == BackendKind::Python
    {
        println!("[1/1] 使用 Python 官方 Voice Clone 路徑產生 WAV…");
        println!("      （首次載入需下載權重，約 1-5 分鐘）");
        let bridge = PythonBridge::new(&model_id)
            .expect("建立 PythonBridge 失敗")
            .with_python("python");
        let options = SynthesisOptions {
            language,
            speaker: None,
            instruct: None,
            reference_audio,
            reference_text,
            seed,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            max_new_tokens,
        };
        bridge
            .synthesize_voice_clone_wav(&text, &options, &output_path)
            .expect("Python Voice Clone 生成失敗");
        println!();
        println!("✅ 完成! 已儲存: {output_path}");
        return;
    }

    let device = runtime_device();

    // ----- 步驟 2: 獲取 Token（透過 LLM 或從檔案載入）-----
    let stream: TokenStream = if let Some(tp) = tokens_path {
        println!("[1/3] 從二進位檔載入 Token ({tp})…");
        let data = std::fs::read(&tp).unwrap_or_else(|err| {
            eprintln!("錯誤: 讀取 Token 檔失敗 {tp}: {err}");
            std::process::exit(1);
        });
        qwen3tts::text_frontend::TokenParser::parse_binary(&data).unwrap_or_else(|err| {
            eprintln!("錯誤: 解析 Token 檔失敗 {tp}: {err}");
            std::process::exit(1);
        })
    } else {
        match backend {
            BackendKind::Python => {
                println!("[1/3] 載入 Python 橋接 LLM ({model_id}) 並生成 Token…");
                println!("      （首次載入需下載權重，約 1-5 分鐘）");
                let bridge = PythonBridge::new(&model_id)
                    .expect("建立 PythonBridge 失敗")
                    .with_python("python");
                let options = SynthesisOptions {
                    language,
                    speaker,
                    instruct,
                    reference_audio,
                    reference_text,
                    seed,
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
                let tok_path = if let Some(path) = find_tokenizer_json_for_model(&model_id, &dir) {
                    path
                } else {
                    eprintln!(
                        "錯誤: 找不到 tokenizer.json。\n\
                         請先用 convert_tokenizer.exe / tools/build_tokenizer.py 產生，\n\
                         或把 Base 模型的 tokenizer.json 複製到 models/tokenizer.json。"
                    );
                    std::process::exit(1);
                };

                let display_model = display_model_id(&model_id, &dir);
                println!("[1/3] 載入 Candle LLM ({display_model}) 並生成 Token…");
                println!("      model dir : {dir:?}");
                println!("      （首次載入時間依模型大小與裝置而定，包含權重 BF16→F32）");

                let backend = CandleLLM::from_files(&sf_path, &tok_path, &device)
                    .expect("載入 CandleLLM 失敗");
                let options = SynthesisOptions {
                    language,
                    speaker,
                    instruct,
                    reference_audio,
                    reference_text,
                    seed,
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
    warn_if_truncated(&text, max_new_tokens, num_frames);

    // --text-only: 產生 Token 後直接結束，跳過解碼器（debug / 煙霧測試用）
    if text_only {
        println!();
        println!("✅ 完成 (--text-only): 已產生 {num_frames} 幀 Token，未執行解碼");
        if save_tokens_path.is_none() {
            println!("   提示：用 --save-tokens <檔案> 保存 Token");
        }
        return;
    }

    // ----- 步驟 2: 依實際幀數載入 Tokenizer 解碼器 -----
    let weight_dir = match ensure_tokenizer_weight_dir() {
        Ok(path) => path,
        Err(err) => {
            eprintln!("錯誤: {err}");
            eprintln!("可手動轉換 Tokenizer 權重:");
            eprintln!("  convert_tokenizer.exe --output weights/tokenizer");
            eprintln!("或使用 Python fallback:");
            eprintln!("  python tools/convert_weights.py tokenizer --output weights/tokenizer");
            std::process::exit(1);
        }
    };
    let target_frames = if speed != 1.0 {
        (num_frames as f64 / speed).round() as usize
    } else {
        num_frames
    };
    let mut config = DecoderConfig::realtime_with_capacity(num_frames.max(target_frames));
    config.speed = speed;

    println!("[2/3] 載入 Tokenizer 解碼器…");
    println!("      tokenizer weights: {}", weight_dir.display());
    println!(
        "      decoder capacity: {} frames",
        config.ring_buffer_capacity
    );
    let mut decoder = Decoder12Hz::from_safetensors(config, &weight_dir, &device)
        .expect("載入 Tokenizer 權重失敗");

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

fn resolve_instruct(instruct: Option<String>, instruct_file: Option<String>) -> Option<String> {
    match (instruct, instruct_file) {
        (Some(_), Some(_)) => {
            eprintln!("--instruct 與 --instruct-file 只能擇一使用");
            std::process::exit(1);
        }
        (Some(value), None) => Some(value),
        (None, Some(path)) => {
            let text = std::fs::read_to_string(&path).unwrap_or_else(|err| {
                eprintln!("讀取 --instruct-file 失敗: {path}: {err}");
                std::process::exit(1);
            });
            let text = text.trim().to_string();
            if text.is_empty() {
                eprintln!("--instruct-file 內容不可為空: {path}");
                std::process::exit(1);
            }
            Some(text)
        }
        (None, None) => None,
    }
}

fn runtime_device() -> candle_core::Device {
    #[cfg(feature = "cuda")]
    {
        match candle_core::Device::new_cuda(0) {
            Ok(device) => {
                println!("裝置    : CUDA:0");
                device
            }
            Err(err) => {
                eprintln!("警告: CUDA 初始化失敗 ({err}); 改用 CPU");
                println!("裝置    : CPU");
                candle_core::Device::Cpu
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("裝置    : CPU");
        candle_core::Device::Cpu
    }
}

fn warn_if_truncated(text: &str, max_new_tokens: u32, num_frames: usize) {
    if num_frames >= max_new_tokens as usize {
        let zh_chars = text
            .chars()
            .filter(|&ch| ('\u{4e00}'..='\u{9fff}').contains(&ch))
            .count();
        let recommended = zh_chars.saturating_mul(4).max(16);
        eprintln!(
            "⚠️ 警告: 已達到最大生成 Token 數 --max-new-tokens={}，語音後半段可能被截斷！",
            max_new_tokens
        );
        if zh_chars > 0 {
            eprintln!(
                "   提示：對於中文，建議 --max-new-tokens 至少設定為字數的 4 倍（目前字數 {}，建議值為 {}）。",
                zh_chars, recommended
            );
        } else {
            eprintln!("   提示：請增加 --max-new-tokens 參數值。");
        }
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
                    內建: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
                    CustomVoice 有 speaker id 時走真實 speaker；其他模型轉成 instruct preset
  --list-speakers    顯示內建 speaker preset 清單
  --list-models      顯示 Qwen3-TTS 模型能力表
  --mode             生成模式：auto | custom-voice | voice-design | voice-clone
  --reference-audio  Voice Clone 參考音訊（建議 3 秒以上；Candle/Rust 原生與 Python 參考路徑皆可用）
  --reference-text   Voice Clone 參考音訊逐字稿；最佳效果強烈建議提供
  --instruct         VoiceDesign/CustomVoice 音色或語氣指令
  --instruct-file    從 UTF-8 文字檔讀取 VoiceDesign/CustomVoice 指令
  --seed N           固定取樣 seed，讓相同文字/條件更容易重現
  --output / -o      輸出 WAV 路徑（預設: output.wav）
  --max-new-tokens N 最大生成 Token 數（預設: 4096）
  --speed N          調整語音語速，例如 1.2 變快，0.8 變慢（預設: 1.0）
  --text-only        只跑到 LLM 階段產生 Token，不解碼成音訊
  --version / -V     顯示版本號
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

  # VoiceDesign 音色描述（建議使用 1.7B-VoiceDesign）
  cargo run --example synthesize --features \"candle-llm cuda\" -- \\
      --text \"你好，今天想和你聊聊天\" --backend candle \\
      --model-dir <Qwen3-TTS-12Hz-1.7B-VoiceDesign-snapshot> \\
      --language chinese \\
      --instruct \"年輕女性，台灣口語，溫柔親切，語速自然\"

  # 內建 speaker preset；VoiceDesign/Base 會自動轉成 instruct fallback
  cargo run --example synthesize --features \"candle-llm cuda\" -- \\
      --text \"你好，今天想和你聊聊天\" --backend candle \\
      --model-dir <Qwen3-TTS-12Hz-1.7B-VoiceDesign-snapshot> \\
      --language chinese \\
      --speaker Vivian

  # 從檔案讀音色設定，並固定 seed
  cargo run --example synthesize --features \"candle-llm cuda\" -- \\
      --text \"你好\" --backend candle \\
      --model-dir <Qwen3-TTS-12Hz-1.7B-VoiceDesign-snapshot> \\
      --language chinese \\
      --instruct-file instruct.txt \\
      --seed 20260603

  # Voice Clone（Candle/Rust 原生路徑，無 Python runtime）
  cargo run --example synthesize --features candle-llm -- \\
      --backend candle \\
      --mode voice-clone \\
      --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
      --text \"測試文字\" \\
      --language Chinese \\
      --reference-audio reference.wav \\
      --reference-text \"參考音訊的逐字稿\" \\
      --output clone.wav

  # Voice Clone（Python 官方 qwen_tts 參考路徑，用於對齊比較）
  cargo run --example synthesize -- \\
      --backend python \\
      --mode voice-clone \\
      --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
      --text \"測試文字\" \\
      --language Chinese \\
      --reference-audio reference.wav \\
      --reference-text \"參考音訊的逐字稿\" \\
      --output clone.wav

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

fn print_models() {
    println!("Qwen3-TTS model capability table:");
    println!(
        "{:<42} {:<5} {:<46} {:<7} {:<9} {:<9} {}",
        "Model", "Params", "Main function", "Langs", "Streaming", "Instruct", "Recommended"
    );
    for model in model_table() {
        println!(
            "{:<42} {:<5} {:<46} {:<7} {:<9} {:<9} {}",
            model
                .model_id
                .strip_prefix("Qwen/")
                .unwrap_or(model.model_id),
            model.parameters,
            model.main_function,
            model.languages,
            if model.streaming { "yes" } else { "no" },
            model.instruction_control.label(),
            model.recommended_scenario
        );
    }
    println!();
    println!(
        "Supported languages (10): {}",
        SUPPORTED_LANGUAGES.join(", ")
    );
    println!(
        "Voice clone requires a Base model plus --reference-audio, and best quality should include --reference-text. Candle/Rust native voice-clone conditioning is available for Base models with converted tokenizer encoder and speaker encoder weights."
    );
}

fn print_speakers() {
    println!("Built-in Qwen CustomVoice speaker presets:");
    for name in speaker_presets::speaker_names() {
        let preset = speaker_presets::lookup(name).expect("known speaker preset");
        println!(
            "  {:<10} {:<18} {}",
            preset.name, preset.native_language, preset.description
        );
    }
    println!();
    println!("CustomVoice models use these as real speaker ids.");
    println!("Base/VoiceDesign models use the same names as instruct presets.");
}
