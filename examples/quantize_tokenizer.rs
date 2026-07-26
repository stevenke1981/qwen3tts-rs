#![allow(dead_code)]

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use qwen3tts::quantization::{
    QuantizationFormat, QuantizeFileOptions, QuantizedTensorReport, quantize_safetensors_file,
};
use serde::Serialize;

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Args {
    input_dir: PathBuf,
    output_dir: PathBuf,
    format: QuantizationFormat,
    group_size: usize,
    min_elements: usize,
    min_cosine: f64,
    fail_low_cosine: bool,
    calibration_texts: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            input_dir: PathBuf::from("weights/tokenizer"),
            output_dir: PathBuf::from("weights/tokenizer-q8"),
            format: QuantizationFormat::Q8_0,
            group_size: 64,
            min_elements: 1024,
            min_cosine: 0.995,
            fail_low_cosine: false,
            calibration_texts: None,
        }
    }
}

#[derive(Debug, Serialize)]
struct QuantizationRunReport {
    version: String,
    input_dir: String,
    output_dir: String,
    format: String,
    group_size: usize,
    min_elements: usize,
    min_cosine: f64,
    preserve_low_cosine: bool,
    calibration_texts: Option<String>,
    files: Vec<FileReport>,
    total_original_bytes: usize,
    total_stored_bytes: usize,
    quantized_tensors: usize,
    preserved_tensors: usize,
}

#[derive(Debug, Serialize)]
struct FileReport {
    input: String,
    output: String,
    tensors: Vec<QuantizedTensorReport>,
}

fn main() {
    eprintln!(
        "error: codec/vocoder 整數量化已停用（違反 AGENTS.md §3.2，聲碼器永不整數量化）。\n       \
         強制使用 F32 權重；此程式保留僅供未來 debug 用途，不再執行量化動作。"
    );
    std::process::exit(1);
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args(std::env::args().skip(1).collect())?;
    fs::create_dir_all(&args.output_dir)?;

    let inputs = collect_safetensors(&args.input_dir)?;
    if inputs.is_empty() {
        return Err(format!(
            "no .safetensors files found in {}",
            args.input_dir.display()
        )
        .into());
    }

    let mut files = Vec::new();
    let mut total_original_bytes = 0usize;
    let mut total_stored_bytes = 0usize;
    let mut quantized_tensors = 0usize;
    let mut preserved_tensors = 0usize;
    let mut low_cosine = Vec::new();

    for input in inputs {
        let file_name = input
            .file_name()
            .ok_or_else(|| format!("invalid input path: {}", input.display()))?;
        let output = args.output_dir.join(file_name);
        let tensor_report = quantize_safetensors_file(
            &input,
            &output,
            &QuantizeFileOptions {
                format: args.format,
                group_size: args.group_size,
                min_elements: args.min_elements,
                min_cosine: Some(args.min_cosine),
                preserve_low_cosine: !args.fail_low_cosine,
            },
        )?;
        for item in &tensor_report {
            total_original_bytes += item.original_bytes;
            total_stored_bytes += item.stored_bytes;
            if item.quantized {
                quantized_tensors += 1;
                if item.cosine.unwrap_or(0.0) < args.min_cosine {
                    low_cosine.push(format!(
                        "{}:{} cosine={:.6}",
                        input.display(),
                        item.name,
                        item.cosine.unwrap_or(0.0)
                    ));
                }
            } else {
                preserved_tensors += 1;
            }
        }
        files.push(FileReport {
            input: input.display().to_string(),
            output: output.display().to_string(),
            tensors: tensor_report,
        });
    }

    if let Some(path) = &args.calibration_texts {
        let lines = fs::read_to_string(path)?
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .count();
        println!(
            "calibration texts: {} ({} usable lines)",
            path.display(),
            lines
        );
    }

    let report = QuantizationRunReport {
        version: env!("CARGO_PKG_VERSION").to_string(),
        input_dir: args.input_dir.display().to_string(),
        output_dir: args.output_dir.display().to_string(),
        format: args.format.as_str().to_string(),
        group_size: args.group_size,
        min_elements: args.min_elements,
        min_cosine: args.min_cosine,
        preserve_low_cosine: !args.fail_low_cosine,
        calibration_texts: args
            .calibration_texts
            .as_ref()
            .map(|path| path.display().to_string()),
        files,
        total_original_bytes,
        total_stored_bytes,
        quantized_tensors,
        preserved_tensors,
    };
    let report_path = args.output_dir.join("quantization_report.json");
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;

    if !low_cosine.is_empty() {
        eprintln!("low cosine tensors below --min-cosine {}:", args.min_cosine);
        for item in low_cosine {
            eprintln!("  {item}");
        }
        return Err("quantization quality gate failed".into());
    }

    println!(
        "quantized {} tensors, preserved {} tensors",
        quantized_tensors, preserved_tensors
    );
    println!(
        "bytes: original={} stored={} ratio={:.3}",
        total_original_bytes,
        total_stored_bytes,
        total_stored_bytes as f64 / total_original_bytes.max(1) as f64
    );
    println!("output: {}", args.output_dir.display());
    println!("report: {}", report_path.display());
    Ok(())
}

fn parse_args(raw: Vec<String>) -> Result<Args, Box<dyn Error>> {
    let mut args = Args::default();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--input" => {
                args.input_dir = PathBuf::from(require_arg(&raw, i, "--input")?);
                i += 2;
            }
            "--output" | "-o" => {
                args.output_dir = PathBuf::from(require_arg(&raw, i, "--output")?);
                i += 2;
            }
            "--format" => {
                args.format = QuantizationFormat::from_str(&require_arg(&raw, i, "--format")?)?;
                if args.output_dir == PathBuf::from("weights/tokenizer-q8") {
                    args.output_dir = match args.format {
                        QuantizationFormat::Q8_0 => PathBuf::from("weights/tokenizer-q8"),
                        QuantizationFormat::Q4_0 => PathBuf::from("weights/tokenizer-q4"),
                    };
                }
                i += 2;
            }
            "--group-size" => {
                args.group_size = require_arg(&raw, i, "--group-size")?.parse()?;
                if args.group_size == 0 {
                    return Err("--group-size must be greater than 0".into());
                }
                i += 2;
            }
            "--min-elements" => {
                args.min_elements = require_arg(&raw, i, "--min-elements")?.parse()?;
                i += 2;
            }
            "--min-cosine" => {
                args.min_cosine = require_arg(&raw, i, "--min-cosine")?.parse()?;
                i += 2;
            }
            "--fail-low-cosine" => {
                args.fail_low_cosine = true;
                i += 1;
            }
            "--calibration-texts" => {
                args.calibration_texts =
                    Some(PathBuf::from(require_arg(&raw, i, "--calibration-texts")?));
                i += 2;
            }
            "--version" | "-V" => {
                println!("qwen3tts-rs {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(args)
}

fn collect_safetensors(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().map_or(false, |ext| ext == "safetensors") {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn require_arg(raw: &[String], i: usize, flag: &str) -> Result<String, Box<dyn Error>> {
    raw.get(i + 1)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_usage() {
    println!(
        "Usage: quantize_tokenizer.exe [--input weights/tokenizer] [--output weights/tokenizer-q8] [options]\n\
         Options:\n\
           --format q8_0|q4_0         Quantization format (default: q8_0)\n\
           --group-size <n>           Per-group quantization size (default: 64)\n\
           --min-elements <n>         Preserve tensors smaller than this (default: 1024)\n\
           --min-cosine <f>           Preserve/fail tensors below this cosine (default: 0.995)\n\
           --fail-low-cosine          Do not auto-preserve low-cosine tensors as F32 anchors\n\
           --calibration-texts <file> Record a text calibration manifest in the report\n\
           --version / -V             Show version info\n\
           --help / -h                Show this help"
    );
}
