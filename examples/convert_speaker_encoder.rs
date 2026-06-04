use std::path::PathBuf;

use qwen3tts::speaker_converter::{SpeakerConverterOptions, convert_speaker_encoder_weights};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> qwen3tts::Result<()> {
    let mut input_dir: Option<PathBuf> = None;
    let mut output_dir = PathBuf::from("weights/speaker");

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                input_dir = Some(PathBuf::from(require_arg(&args, i, "--input")?));
                i += 2;
            }
            "--output" | "-o" => {
                output_dir = PathBuf::from(require_arg(&args, i, "--output")?);
                i += 2;
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            "--version" | "-V" => {
                println!("qwen3tts-rs {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            other => {
                return Err(qwen3tts::Error::Config(format!(
                    "Unknown argument: {other}"
                )));
            }
        }
    }

    let converted = convert_speaker_encoder_weights(&SpeakerConverterOptions {
        input_dir,
        output_dir,
    })?;
    println!(
        "Converted speaker encoder weights -> {}",
        converted.output_dir.display()
    );
    for file in converted.files {
        println!("  {}", file.display());
    }
    Ok(())
}

fn require_arg(args: &[String], i: usize, flag: &str) -> qwen3tts::Result<String> {
    args.get(i + 1)
        .cloned()
        .ok_or_else(|| qwen3tts::Error::Config(format!("{flag} requires a value")))
}

fn print_usage() {
    println!(
        "Usage: convert_speaker_encoder.exe [--input <base-model-snapshot>] [--output weights/speaker] [--version | -V]\n\
         If --input is omitted, the converter searches the HuggingFace cache for\n\
         Qwen/Qwen3-TTS-12Hz-0.6B-Base or Qwen/Qwen3-TTS-12Hz-1.7B-Base."
    );
}
