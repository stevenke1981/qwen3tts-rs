use std::path::PathBuf;

use qwen3tts::tokenizer_converter::{ConverterOptions, convert_tokenizer_weights};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> qwen3tts::Result<()> {
    let mut input_dir: Option<PathBuf> = None;
    let mut output_dir = PathBuf::from("weights/tokenizer");

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

    let converted = convert_tokenizer_weights(&ConverterOptions {
        input_dir,
        output_dir,
    })?;
    println!(
        "Converted tokenizer decoder weights -> {}",
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
        "Usage: convert_tokenizer.exe [--input <hf-tokenizer-snapshot>] [--output weights/tokenizer] [--version | -V]\n\
         If --input is omitted, the converter searches the HuggingFace cache for\n\
         Qwen/Qwen3-TTS-Tokenizer-12Hz."
    );
}
