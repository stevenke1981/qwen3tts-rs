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
use qwen3tts::text_frontend::model_catalog::{
    GenerationMode, ModelMetadata, SUPPORTED_LANGUAGES, model_capability, model_table,
    resolve_generation_mode, validate_generation_request,
};
use qwen3tts::text_frontend::speaker_presets;
use qwen3tts::text_frontend::{CandleLLM, SynthesisOptions, TextFrontend};
use qwen3tts::{Decoder12Hz, DecoderConfig};

#[derive(Debug, Clone)]
struct Args {
    model_id: String,
    model_dir: Option<PathBuf>,
    output_dir: PathBuf,
    prefix: String,
    texts: Vec<String>,
    texts_file: Option<PathBuf>,
    language: String,
    speaker: Option<String>,
    mode: GenerationMode,
    reference_audio: Option<PathBuf>,
    instruct: Option<String>,
    instruct_files: Vec<PathBuf>,
    instruct_values: Vec<String>,
    seeds: Vec<u64>,
    max_new_tokens: u32,
    temperature: f64,
    top_k: u32,
    top_p: f64,
    save_tokens_dir: Option<PathBuf>,
    save_tokens: bool,
    speed: f64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-Base".to_string(),
            model_dir: None,
            output_dir: PathBuf::from("batch-output"),
            prefix: "clip".to_string(),
            texts: Vec::new(),
            texts_file: None,
            language: "auto".to_string(),
            speaker: None,
            mode: GenerationMode::Auto,
            reference_audio: None,
            instruct: None,
            instruct_files: Vec::new(),
            instruct_values: Vec::new(),
            seeds: Vec::new(),
            max_new_tokens: 4096,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            save_tokens_dir: None,
            save_tokens: true,
            speed: 1.0,
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
    validate_per_line_options(&args, texts.len())?;

    let model_dir = resolve_model_dir(&args)?;
    let metadata = ModelMetadata::from_model_dir(&model_dir)?;
    let effective_mode = resolve_generation_mode(&metadata, args.mode);
    validate_generation_request(
        &metadata,
        effective_mode,
        args.speaker.as_deref(),
        select_instruct(&args, 0).map(String::as_str),
        args.reference_audio
            .as_ref()
            .map(|path| path.to_string_lossy())
            .as_deref(),
    )?;

    if effective_mode == GenerationMode::VoiceClone {
        return Err(
            "batch voice-clone is not implemented yet, and Candle/Rust native voice-clone is not implemented yet. For best reference cloning quality in the temporary Python path, use synthesize.exe --backend python --mode voice-clone with both --reference-audio and --reference-text"
                .into(),
        );
    }

    fs::create_dir_all(&args.output_dir)?;
    if args.save_tokens {
        if let Some(dir) = &args.save_tokens_dir {
            fs::create_dir_all(dir)?;
        }
    }

    let model_safetensors = model_dir.join("model.safetensors");
    if !model_safetensors.exists() {
        return Err(format!("missing {}", model_safetensors.display()).into());
    }
    let tokenizer_json = find_tokenizer_json(&model_dir)?;

    let weight_dir = ensure_tokenizer_weight_dir()?;

    let device = runtime_device();
    let mut decoder_config = DecoderConfig::realtime_with_capacity(args.max_new_tokens as usize);
    decoder_config.speed = args.speed;
    println!("loading tokenizer decoder once...");
    println!("  tokenizer weights: {}", weight_dir.display());
    println!(
        "  decoder capacity: {} frames",
        decoder_config.ring_buffer_capacity
    );
    let mut decoder = Decoder12Hz::from_safetensors(decoder_config, &weight_dir, &device)?;

    println!("loading Candle model once...");
    println!("  model: {}", args.model_id);
    if let Some(capability) = model_capability(&args.model_id) {
        println!(
            "  capability: {} / {} / {} languages / streaming={}",
            capability.parameters,
            capability.main_function,
            capability.languages,
            if capability.streaming { "yes" } else { "no" }
        );
    }
    println!("  mode: {}", effective_mode.as_str());
    println!("  model dir: {}", model_dir.display());
    println!("  tokenizer: {}", tokenizer_json.display());
    let load_start = Instant::now();
    let frontend = CandleLLM::from_files(&model_safetensors, &tokenizer_json, &device)?;
    println!("  load time: {:.2}s", load_start.elapsed().as_secs_f64());

    for (idx, text) in texts.iter().enumerate() {
        let item_start = Instant::now();
        let one_based = idx + 1;
        let wav_path = make_output_path(&args.output_dir, &args.prefix, one_based);
        println!("[{one_based}/{}] {}", texts.len(), text);
        warn_if_max_new_tokens_low(text, args.max_new_tokens);

        let options = SynthesisOptions {
            language: args.language.clone(),
            speaker: args.speaker.clone(),
            instruct: select_instruct(&args, idx).cloned(),
            reference_audio: args
                .reference_audio
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned()),
            reference_text: None,
            seed: select_seed(&args, idx),
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            max_new_tokens: args.max_new_tokens,
        };
        if let Some(instruct) = &options.instruct {
            println!("  instruct: {}", abbreviate(instruct, 80));
        }
        if let Some(speaker) = &options.speaker {
            println!("  speaker: {speaker}");
        }
        if let Some(seed) = options.seed {
            println!("  seed: {seed}");
        }

        let synth_start = Instant::now();
        let stream = frontend.synthesize(text, &options)?;
        let synth_elapsed = synth_start.elapsed();
        if stream.frames.is_empty() {
            println!("  skipped: generated zero frames");
            continue;
        }

        warn_if_truncated(text, args.max_new_tokens, stream.frames.len());

        if args.save_tokens {
            if let Some(tokens_dir) = &args.save_tokens_dir {
                let token_path = make_token_path(tokens_dir, &args.prefix, one_based);
                stream.write_binary(&token_path)?;
                println!("  tokens: {}", token_path.display());
            }
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
                args.model_dir = Some(PathBuf::from(require_arg(&raw, i, "--model-dir")?));
                i += 2;
            }
            "--model" => {
                args.model_id = require_arg(&raw, i, "--model")?;
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
            "--mode" => {
                args.mode = GenerationMode::parse(&require_arg(&raw, i, "--mode")?)?;
                i += 2;
            }
            "--reference-audio" => {
                args.reference_audio =
                    Some(PathBuf::from(require_arg(&raw, i, "--reference-audio")?));
                i += 2;
            }
            "--instruct" => {
                args.instruct = Some(require_arg(&raw, i, "--instruct")?);
                i += 2;
            }
            "--instruct-file" => {
                args.instruct_files
                    .push(PathBuf::from(require_arg(&raw, i, "--instruct-file")?));
                i += 2;
            }
            "--seed" => {
                args.seeds.push(require_arg(&raw, i, "--seed")?.parse()?);
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
            "--no-save-tokens" => {
                args.save_tokens = false;
                i += 1;
            }
            "--speed" => {
                args.speed = require_arg(&raw, i, "--speed")?.parse()?;
                if args.speed <= 0.0 {
                    return Err("--speed must be greater than 0.0".into());
                }
                i += 2;
            }
            "--version" | "-V" => {
                println!("qwen3tts-rs {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--list-speakers" => {
                print_speakers();
                std::process::exit(0);
            }
            "--list-models" => {
                print_models();
                std::process::exit(0);
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            flag => return Err(format!("unknown argument: {flag}").into()),
        }
    }

    if args.instruct.is_some() && !args.instruct_files.is_empty() {
        return Err("--instruct and --instruct-file are mutually exclusive".into());
    }
    if !args.instruct_files.is_empty() {
        for path in &args.instruct_files {
            let value = fs::read_to_string(path)?.trim().to_string();
            if value.is_empty() {
                return Err(format!("--instruct-file is empty: {}", path.display()).into());
            }
            args.instruct_values.push(value);
        }
    }
    if !args.save_tokens {
        args.save_tokens_dir = None;
    }
    Ok(args)
}

fn resolve_model_dir(args: &Args) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(model_dir) = &args.model_dir {
        return Ok(model_dir.clone());
    }
    locate_hf_model_snapshot(&args.model_id, "model.safetensors").ok_or_else(|| {
        format!(
            "missing --model-dir and no local HuggingFace snapshot found for {}",
            args.model_id
        )
        .into()
    })
}

fn validate_per_line_options(args: &Args, text_count: usize) -> Result<(), Box<dyn Error>> {
    if args.instruct_values.len() > 1 && args.instruct_values.len() != text_count {
        return Err(format!(
            "--instruct-file was provided {} times but there are {text_count} text lines; pass one file for all lines or one per line",
            args.instruct_values.len()
        )
        .into());
    }
    if args.seeds.len() > 1 && args.seeds.len() != text_count {
        return Err(format!(
            "--seed was provided {} times but there are {text_count} text lines; pass one seed for all lines or one per line",
            args.seeds.len()
        )
        .into());
    }
    Ok(())
}

fn select_instruct<'a>(args: &'a Args, idx: usize) -> Option<&'a String> {
    if let Some(global) = &args.instruct {
        return Some(global);
    }
    if args.instruct_values.is_empty() {
        return None;
    }
    args.instruct_values
        .get(if args.instruct_values.len() == 1 {
            0
        } else {
            idx
        })
}

fn select_seed(args: &Args, idx: usize) -> Option<u64> {
    if args.seeds.is_empty() {
        return None;
    }
    args.seeds
        .get(if args.seeds.len() == 1 { 0 } else { idx })
        .copied()
}

fn abbreviate(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let abbreviated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{abbreviated}...")
    } else {
        abbreviated
    }
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
        if let Some(base_dir) = locate_hf_model_snapshot(base_id, "tokenizer.json") {
            let path = base_dir.join("tokenizer.json");
            if path.exists() {
                return Ok(path);
            }
        }
    }
    Err("missing tokenizer.json; pass a model dir with tokenizer.json, create models/tokenizer.json, or keep a Base model snapshot in the HuggingFace cache".into())
}

fn locate_hf_model_snapshot(model_id: &str, required_file: &str) -> Option<PathBuf> {
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
        let candidate = entry.path().join(required_file);
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

fn recommended_max_new_tokens(text: &str) -> usize {
    text.chars()
        .filter(|&ch| ('\u{4e00}'..='\u{9fff}').contains(&ch))
        .count()
        .saturating_mul(4)
        .max(16)
}

fn warn_if_max_new_tokens_low(text: &str, max_new_tokens: u32) {
    let recommended = recommended_max_new_tokens(text);
    let zh_chars = text
        .chars()
        .filter(|&ch| ('\u{4e00}'..='\u{9fff}').contains(&ch))
        .count();
    if zh_chars > 0 && (max_new_tokens as usize) < recommended {
        eprintln!(
            "  warning: --max-new-tokens={max_new_tokens} may be low for this Chinese text; suggest at least {recommended}"
        );
    }
}

fn warn_if_truncated(text: &str, max_new_tokens: u32, num_frames: usize) {
    if num_frames >= max_new_tokens as usize {
        let recommended = recommended_max_new_tokens(text);
        eprintln!(
            "  warning: generated frames reached --max-new-tokens={}; text may be truncated!",
            max_new_tokens
        );
        eprintln!(
            "  suggest increasing --max-new-tokens to at least {}",
            recommended
        );
    }
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
        "Usage: synthesize_batch [--model-dir <snapshot>] [--text <text> ...] [--texts lines.txt]\n\
         Options:\n\
           --model <id>             HuggingFace model id (default: Qwen/Qwen3-TTS-12Hz-0.6B-Base)\n\
           --model-dir <snapshot>   Local model snapshot directory; auto-searches HF cache if omitted\n\
           --output-dir <dir>       Output WAV directory (default: batch-output)\n\
           --prefix <name>          Output file prefix (default: clip)\n\
           --language <name>        Language (default: auto)\n\
           --speaker <name>         Speaker condition or built-in preset: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee\n\
           --list-speakers          Show built-in speaker presets\n\
           --list-models            Show Qwen3-TTS model capability table\n\
           --mode <name>            auto | custom-voice | voice-design | voice-clone\n\
           --reference-audio <wav>  Voice Clone reference audio (3s+; batch/Candle voice-clone is not implemented yet)\n\
           --instruct <text>        VoiceDesign/CustomVoice style instruction\n\
           --instruct-file <path>   Read instruction from UTF-8 text file; repeat once per line to switch voices\n\
           --seed <n>               Fixed sampling seed; repeat once per line to vary seeds\n\
           --max-new-tokens <n>     Max generated frames (default: 4096)\n\
           --temperature <f>        Sampling temperature (default: 0.9)\n\
           --top-k <n>              Top-k sampling (default: 50)\n\
           --top-p <f>              Top-p sampling (default: 1.0)\n\
           --speed <f>              Speech rate speed factor (default: 1.0)\n\
           --save-tokens-dir <dir>  Also write token files\n\
           --no-save-tokens         Disable token file output even if a wrapper passes --save-tokens-dir\n\
           --version / -V           Show version info"
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
        "Voice clone requires a Base model plus --reference-audio, and best quality should include --reference-text. Batch voice-clone and Candle/Rust native conditioning are not implemented yet."
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
    use super::{
        make_output_path, parse_args, parse_text_lines, recommended_max_new_tokens,
        select_instruct, select_seed, validate_per_line_options,
    };
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

    #[test]
    fn parse_seed_and_no_save_tokens() {
        let args = parse_args(vec![
            "--model-dir".into(),
            "model".into(),
            "--seed".into(),
            "42".into(),
            "--save-tokens-dir".into(),
            "tokens".into(),
            "--no-save-tokens".into(),
        ])
        .unwrap();
        assert_eq!(args.seeds, vec![42]);
        assert_eq!(select_seed(&args, 5), Some(42));
        assert!(!args.save_tokens);
        assert!(args.save_tokens_dir.is_none());
    }

    #[test]
    fn repeated_seed_can_match_each_line() {
        let args = parse_args(vec![
            "--model-dir".into(),
            "model".into(),
            "--seed".into(),
            "11".into(),
            "--seed".into(),
            "22".into(),
        ])
        .unwrap();
        validate_per_line_options(&args, 2).unwrap();
        assert_eq!(select_seed(&args, 0), Some(11));
        assert_eq!(select_seed(&args, 1), Some(22));
        assert!(validate_per_line_options(&args, 3).is_err());
    }

    #[test]
    fn repeated_instruct_file_can_match_each_line() {
        let base = std::env::temp_dir().join(format!(
            "qwen3tts-batch-instruct-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let a = base.join("voice_a.txt");
        let b = base.join("voice_b.txt");
        std::fs::write(&a, "voice A").unwrap();
        std::fs::write(&b, "voice B").unwrap();

        let args = parse_args(vec![
            "--model-dir".into(),
            "model".into(),
            "--instruct-file".into(),
            a.display().to_string(),
            "--instruct-file".into(),
            b.display().to_string(),
        ])
        .unwrap();
        validate_per_line_options(&args, 2).unwrap();
        assert_eq!(
            select_instruct(&args, 0).map(String::as_str),
            Some("voice A")
        );
        assert_eq!(
            select_instruct(&args, 1).map(String::as_str),
            Some("voice B")
        );
        assert!(validate_per_line_options(&args, 3).is_err());

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn chinese_token_recommendation_scales_by_four() {
        assert_eq!(recommended_max_new_tokens("今天天氣真好"), 24);
        assert_eq!(recommended_max_new_tokens("hello"), 16);
    }
}
