//! Load-once Candle batch synthesis runner.
//!
//! This example is meant for repeated local smoke tests. It loads the 12Hz
//! tokenizer decoder and Candle talker once, then synthesizes multiple lines.

use std::error::Error;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use qwen3tts::paths::ensure_tokenizer_weight_dir;
use qwen3tts::text_frontend::{CandleLLM, SynthesisOptions, TextFrontend};
use qwen3tts::{Decoder12Hz, DecoderConfig};

#[derive(Debug, Clone)]
struct Args {
    model_dir: PathBuf,
    output_dir: PathBuf,
    prefix: String,
    texts: Vec<String>,
    texts_file: Option<PathBuf>,
    language: String,
    speaker: Option<String>,
    instruct: Option<String>,
    max_new_tokens: u32,
    temperature: f64,
    top_k: u32,
    top_p: f64,
    save_tokens_dir: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            output_dir: PathBuf::from("batch-output"),
            prefix: "clip".to_string(),
            texts: Vec::new(),
            texts_file: None,
            language: "auto".to_string(),
            speaker: None,
            instruct: None,
            max_new_tokens: 128,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            save_tokens_dir: None,
        }
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args(std::env::args().skip(1).collect())?;
    let texts = collect_texts(&args)?;
    if texts.is_empty() {
        return Err("no text lines provided".into());
    }

    fs::create_dir_all(&args.output_dir)?;
    if let Some(dir) = &args.save_tokens_dir {
        fs::create_dir_all(dir)?;
    }

    let model_safetensors = args.model_dir.join("model.safetensors");
    if !model_safetensors.exists() {
        return Err(format!("missing {}", model_safetensors.display()).into());
    }
    let tokenizer_json = find_tokenizer_json(&args.model_dir)?;

    let weight_dir = ensure_tokenizer_weight_dir()?;

    let device = runtime_device();
    let decoder_config = DecoderConfig::realtime_with_capacity(args.max_new_tokens as usize);
    println!("loading tokenizer decoder once...");
    println!("  tokenizer weights: {}", weight_dir.display());
    println!(
        "  decoder capacity: {} frames",
        decoder_config.ring_buffer_capacity
    );
    let mut decoder = Decoder12Hz::from_safetensors(decoder_config, &weight_dir, &device)?;

    println!("loading Candle model once...");
    println!("  model dir: {}", args.model_dir.display());
    println!("  tokenizer: {}", tokenizer_json.display());
    let load_start = Instant::now();
    let frontend = CandleLLM::from_files(&model_safetensors, &tokenizer_json, &device)?;
    println!("  load time: {:.2}s", load_start.elapsed().as_secs_f64());

    let options = SynthesisOptions {
        language: args.language.clone(),
        speaker: args.speaker.clone(),
        instruct: args.instruct.clone(),
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        max_new_tokens: args.max_new_tokens,
    };

    for (idx, text) in texts.iter().enumerate() {
        let item_start = Instant::now();
        let one_based = idx + 1;
        let wav_path = make_output_path(&args.output_dir, &args.prefix, one_based);
        println!("[{one_based}/{}] {}", texts.len(), text);

        let synth_start = Instant::now();
        let stream = frontend.synthesize(text, &options)?;
        let synth_elapsed = synth_start.elapsed();
        if stream.frames.is_empty() {
            println!("  skipped: generated zero frames");
            continue;
        }

        if let Some(tokens_dir) = &args.save_tokens_dir {
            let token_path = make_token_path(tokens_dir, &args.prefix, one_based);
            stream.write_binary(&token_path)?;
            println!("  tokens: {}", token_path.display());
        }

        let decode_start = Instant::now();
        let samples = decoder.decode_frames(&stream.frames)?;
        let decode_elapsed = decode_start.elapsed();
        write_wav(&wav_path, &samples, 24_000)?;

        println!(
            "  frames={} synth={:.2}s decode={:.2}s total={:.2}s -> {}",
            stream.frames.len(),
            synth_elapsed.as_secs_f64(),
            decode_elapsed.as_secs_f64(),
            item_start.elapsed().as_secs_f64(),
            wav_path.display()
        );
    }

    Ok(())
}

fn parse_args(raw: Vec<String>) -> Result<Args, Box<dyn Error>> {
    let mut args = Args::default();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--model-dir" => {
                args.model_dir = PathBuf::from(require_arg(&raw, i, "--model-dir")?);
                i += 2;
            }
            "--output-dir" => {
                args.output_dir = PathBuf::from(require_arg(&raw, i, "--output-dir")?);
                i += 2;
            }
            "--prefix" => {
                args.prefix = require_arg(&raw, i, "--prefix")?;
                i += 2;
            }
            "--text" => {
                args.texts.push(require_arg(&raw, i, "--text")?);
                i += 2;
            }
            "--texts" => {
                args.texts_file = Some(PathBuf::from(require_arg(&raw, i, "--texts")?));
                i += 2;
            }
            "--language" => {
                args.language = require_arg(&raw, i, "--language")?;
                i += 2;
            }
            "--speaker" => {
                args.speaker = Some(require_arg(&raw, i, "--speaker")?);
                i += 2;
            }
            "--instruct" => {
                args.instruct = Some(require_arg(&raw, i, "--instruct")?);
                i += 2;
            }
            "--max-new-tokens" => {
                args.max_new_tokens = require_arg(&raw, i, "--max-new-tokens")?.parse()?;
                i += 2;
            }
            "--temperature" => {
                args.temperature = require_arg(&raw, i, "--temperature")?.parse()?;
                i += 2;
            }
            "--top-k" => {
                args.top_k = require_arg(&raw, i, "--top-k")?.parse()?;
                i += 2;
            }
            "--top-p" => {
                args.top_p = require_arg(&raw, i, "--top-p")?.parse()?;
                i += 2;
            }
            "--save-tokens-dir" => {
                args.save_tokens_dir =
                    Some(PathBuf::from(require_arg(&raw, i, "--save-tokens-dir")?));
                i += 2;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            flag => return Err(format!("unknown argument: {flag}").into()),
        }
    }

    if args.model_dir.as_os_str().is_empty() {
        return Err("--model-dir is required".into());
    }
    Ok(args)
}

fn require_arg(raw: &[String], i: usize, flag: &str) -> Result<String, Box<dyn Error>> {
    raw.get(i + 1)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn collect_texts(args: &Args) -> Result<Vec<String>, Box<dyn Error>> {
    let mut texts = args.texts.clone();
    if let Some(path) = &args.texts_file {
        texts.extend(parse_text_lines(&fs::read_to_string(path)?));
    }
    if texts.is_empty() {
        let mut stdin = String::new();
        io::stdin().read_to_string(&mut stdin)?;
        texts.extend(parse_text_lines(&stdin));
    }
    Ok(texts)
}

fn parse_text_lines(input: &str) -> Vec<String> {
    input
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect()
}

fn find_tokenizer_json(model_dir: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let local = model_dir.join("tokenizer.json");
    if local.exists() {
        return Ok(local);
    }
    let snapshots = model_dir.join("snapshots");
    if snapshots.is_dir() {
        for entry in fs::read_dir(snapshots)?.flatten() {
            let path = entry.path().join("tokenizer.json");
            if path.exists() {
                return Ok(path);
            }
        }
    }
    for fallback in ["models/tokenizer_1.7b.json", "models/tokenizer.json"] {
        let path = PathBuf::from(fallback);
        if path.exists() {
            return Ok(path);
        }
    }
    for base_id in [
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ] {
        if let Some(base_dir) = locate_hf_model_snapshot(base_id) {
            let path = base_dir.join("tokenizer.json");
            if path.exists() {
                return Ok(path);
            }
        }
    }
    Err("missing tokenizer.json; pass a model dir with tokenizer.json, create models/tokenizer.json, or keep a Base model snapshot in the HuggingFace cache".into())
}

fn locate_hf_model_snapshot(model_id: &str) -> Option<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()
        .map(PathBuf::from)?;
    let snapshots = home
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join(format!("models--{}", model_id.replace('/', "--")))
        .join("snapshots");
    let entries = fs::read_dir(snapshots).ok()?;
    for entry in entries.flatten() {
        let candidate = entry.path().join("tokenizer.json");
        if candidate.exists() {
            return Some(entry.path());
        }
    }
    None
}

fn make_output_path(output_dir: &Path, prefix: &str, index: usize) -> PathBuf {
    output_dir.join(format!("{prefix}_{index:04}.wav"))
}

fn make_token_path(output_dir: &Path, prefix: &str, index: usize) -> PathBuf {
    output_dir.join(format!("{prefix}_{index:04}.tokens"))
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), Box<dyn Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn print_usage() {
    eprintln!(
        "Usage: synthesize_batch --model-dir <snapshot> [--text <text> ...] [--texts lines.txt]\n\
         Options:\n\
           --output-dir <dir>       Output WAV directory (default: batch-output)\n\
           --prefix <name>          Output file prefix (default: clip)\n\
           --language <name>        Language (default: auto)\n\
           --speaker <name>         Speaker condition\n\
           --instruct <text>        VoiceDesign/CustomVoice style instruction\n\
           --max-new-tokens <n>     Max generated frames (default: 128)\n\
           --temperature <f>        Sampling temperature (default: 0.9)\n\
           --top-k <n>              Top-k sampling (default: 50)\n\
           --top-p <f>              Top-p sampling (default: 1.0)\n\
           --save-tokens-dir <dir>  Also write token files"
    );
}

fn runtime_device() -> candle_core::Device {
    #[cfg(feature = "cuda")]
    {
        match candle_core::Device::new_cuda(0) {
            Ok(device) => {
                println!("device: CUDA:0");
                device
            }
            Err(err) => {
                eprintln!("warning: CUDA init failed ({err}); falling back to CPU");
                println!("device: CPU");
                candle_core::Device::Cpu
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("device: CPU");
        candle_core::Device::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::{make_output_path, parse_text_lines};
    use std::path::Path;

    #[test]
    fn parse_text_lines_skips_blank_and_comment_lines() {
        let lines = parse_text_lines("你好\n\n# comment\n  今天天氣真好  \n");
        assert_eq!(lines, vec!["你好", "今天天氣真好"]);
    }

    #[test]
    fn make_output_path_uses_index_only() {
        let path = make_output_path(Path::new("out"), "clip", 7);
        assert_eq!(path, Path::new("out").join("clip_0007.wav"));
    }
}
