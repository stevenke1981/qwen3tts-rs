//! Runtime path helpers for CLI examples and release builds.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{Error, Result};
use serde::Deserialize;

const TOKENIZER_WEIGHT_FILES: &[&str] = &[
    "codebook.safetensors",
    "lightweight.safetensors",
    "pre_transformer.safetensors",
    "upsample.safetensors",
    "decoder_blocks.safetensors",
];

/// Returns tokenizer decoder weight directory candidates in search order.
///
/// The release executables are often launched from a directory different from
/// the directory containing the `.exe`. Search the process working directory
/// first for source-tree workflows, then the executable directory for unpacked
/// release zips.
pub fn resolve_tokenizer_weight_dir_from(cwd: &Path, exe_path: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(path) = std::env::var("QWEN3TTS_TOKENIZER_WEIGHT_DIR") {
        push_unique(&mut candidates, PathBuf::from(path));
    }

    push_tokenizer_weight_candidates(&mut candidates, cwd);

    if let Some(exe_path) = exe_path {
        if let Some(exe_dir) = exe_path.parent() {
            push_tokenizer_weight_candidates(&mut candidates, exe_dir);
        }
    }

    if let Some(cache_dir) = default_tokenizer_q8_cache_dir() {
        push_unique(&mut candidates, cache_dir);
    }
    if let Some(cache_dir) = default_tokenizer_cache_dir() {
        push_unique(&mut candidates, cache_dir);
    }

    candidates
}

/// Finds an existing tokenizer decoder weight directory without conversion.
pub fn find_existing_tokenizer_weight_dir() -> Result<PathBuf> {
    let cwd = std::env::current_dir()?;
    let exe = std::env::current_exe().ok();
    let candidates = resolve_tokenizer_weight_dir_from(&cwd, exe.as_deref());

    for candidate in &candidates {
        if is_complete_tokenizer_weight_dir(candidate) {
            log_tokenizer_weight_dir(candidate);
            return Ok(candidate.clone());
        }
    }

    Err(missing_weights_error(&candidates))
}

/// Finds tokenizer decoder weights, automatically converting from HuggingFace
/// format with the bundled Python converter when needed.
pub fn ensure_tokenizer_weight_dir() -> Result<PathBuf> {
    if let Ok(path) = find_existing_tokenizer_weight_dir() {
        return Ok(path);
    }

    let cwd = std::env::current_dir()?;
    let exe = std::env::current_exe().ok();
    let candidates = resolve_tokenizer_weight_dir_from(&cwd, exe.as_deref());
    let converter = find_converter_program_from(&cwd, exe.as_deref()).ok_or_else(|| {
        Error::Config(format!(
            "{}\n\nNo bundled converter was found. Expected convert_tokenizer.exe or tools/convert_weights.py next to the app or in the current repository.",
            missing_weights_error(&candidates)
        ))
    })?;
    let output = default_tokenizer_cache_dir()
        .unwrap_or_else(|| converter.base_dir.join("weights").join("tokenizer"));

    run_tokenizer_converter(&converter, &output)?;
    if is_complete_tokenizer_weight_dir(&output) {
        if let Some(q8_output) = try_build_q8_tokenizer_cache(&cwd, exe.as_deref(), &output) {
            log_tokenizer_weight_dir(&q8_output);
            return Ok(q8_output);
        }
        log_tokenizer_weight_dir(&output);
        return Ok(output);
    }

    Err(Error::Config(format!(
        "Tokenizer conversion finished but required Rust weights are still incomplete in {}",
        output.display()
    )))
}

/// Returns converter script candidates in search order.
pub fn resolve_converter_script_from(cwd: &Path, exe_path: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    push_unique(
        &mut candidates,
        cwd.join("tools").join("convert_weights.py"),
    );

    if let Some(exe_path) = exe_path {
        if let Some(exe_dir) = exe_path.parent() {
            push_unique(
                &mut candidates,
                exe_dir.join("tools").join("convert_weights.py"),
            );
        }
    }

    candidates
}

/// Returns Rust converter executable candidates in search order.
pub fn resolve_converter_exe_from(cwd: &Path, exe_path: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    push_unique(&mut candidates, cwd.join("convert_tokenizer.exe"));

    if let Some(exe_path) = exe_path {
        if let Some(exe_dir) = exe_path.parent() {
            push_unique(&mut candidates, exe_dir.join("convert_tokenizer.exe"));
        }
    }

    candidates
}

/// Returns Rust tokenizer quantizer executable candidates in search order.
pub fn resolve_quantizer_exe_from(cwd: &Path, exe_path: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    push_unique(&mut candidates, cwd.join("quantize_tokenizer.exe"));

    if let Some(exe_path) = exe_path {
        if let Some(exe_dir) = exe_path.parent() {
            push_unique(&mut candidates, exe_dir.join("quantize_tokenizer.exe"));
        }
    }

    candidates
}

#[derive(Debug, Clone)]
struct ConverterProgram {
    path: PathBuf,
    base_dir: PathBuf,
    kind: ConverterKind,
}

#[derive(Debug, Clone, Copy)]
enum ConverterKind {
    RustExe,
    PythonScript,
}

fn find_converter_program_from(cwd: &Path, exe_path: Option<&Path>) -> Option<ConverterProgram> {
    for path in resolve_converter_exe_from(cwd, exe_path) {
        if path.exists() {
            let base_dir = path.parent().unwrap_or(cwd).to_path_buf();
            return Some(ConverterProgram {
                path,
                base_dir,
                kind: ConverterKind::RustExe,
            });
        }
    }

    for path in resolve_converter_script_from(cwd, exe_path) {
        if path.exists() {
            let base_dir = path
                .parent()
                .and_then(Path::parent)
                .unwrap_or(cwd)
                .to_path_buf();
            return Some(ConverterProgram {
                path,
                base_dir,
                kind: ConverterKind::PythonScript,
            });
        }
    }

    None
}

fn find_quantizer_program_from(cwd: &Path, exe_path: Option<&Path>) -> Option<PathBuf> {
    resolve_quantizer_exe_from(cwd, exe_path)
        .into_iter()
        .find(|path| path.exists())
}

fn run_tokenizer_converter(converter: &ConverterProgram, output: &Path) -> Result<()> {
    eprintln!(
        "Tokenizer decoder weights not found; attempting automatic conversion with {}",
        converter.path.display()
    );
    eprintln!("Tokenizer converter output: {}", output.display());

    match converter.kind {
        ConverterKind::RustExe => {
            let status = Command::new(&converter.path)
                .arg("--output")
                .arg(output)
                .status()
                .map_err(|err| {
                    Error::Config(format!(
                        "Could not launch Rust tokenizer converter {}: {err}",
                        converter.path.display()
                    ))
                })?;
            if status.success() {
                return Ok(());
            }
            Err(Error::Config(format!(
                "Rust tokenizer converter exited with {status}"
            )))
        }
        ConverterKind::PythonScript => {
            for python in ["python", "py"] {
                let mut cmd = Command::new(python);
                if python == "py" {
                    cmd.arg("-3");
                }
                let status = cmd
                    .arg(&converter.path)
                    .arg("tokenizer")
                    .arg("--output")
                    .arg(output)
                    .status();
                match status {
                    Ok(status) if status.success() => return Ok(()),
                    Ok(status) => {
                        eprintln!("Converter via {python} exited with {status}");
                    }
                    Err(err) => {
                        eprintln!("Could not launch {python}: {err}");
                    }
                }
            }

            Err(Error::Config(
                "Automatic tokenizer conversion failed. Run convert_tokenizer.exe manually, or install Python with torch, safetensors, huggingface_hub, and numpy for the Python fallback.".into(),
            ))
        }
    }
}

fn try_build_q8_tokenizer_cache(
    cwd: &Path,
    exe_path: Option<&Path>,
    f32_output: &Path,
) -> Option<PathBuf> {
    let q8_output = q8_dir_for_f32_dir(f32_output);
    if is_complete_tokenizer_weight_dir(&q8_output) {
        return Some(q8_output);
    }

    let Some(quantizer) = find_quantizer_program_from(cwd, exe_path) else {
        eprintln!(
            "Q8 tokenizer auto-cache skipped: quantize_tokenizer.exe was not found next to the app."
        );
        return None;
    };

    eprintln!(
        "Building Q8 tokenizer decoder cache with {}",
        quantizer.display()
    );
    eprintln!("Q8 tokenizer cache output: {}", q8_output.display());

    let status = Command::new(&quantizer)
        .arg("--input")
        .arg(f32_output)
        .arg("--output")
        .arg(&q8_output)
        .arg("--format")
        .arg("q8_0")
        .arg("--group-size")
        .arg("64")
        .arg("--min-cosine")
        .arg("0.995")
        .status();

    match status {
        Ok(status) if status.success() && is_complete_tokenizer_weight_dir(&q8_output) => {
            Some(q8_output)
        }
        Ok(status) => {
            eprintln!(
                "Q8 tokenizer auto-cache skipped: quantize_tokenizer.exe exited with {status}; using F32 tokenizer weights."
            );
            None
        }
        Err(err) => {
            eprintln!(
                "Q8 tokenizer auto-cache skipped: could not launch {}: {err}; using F32 tokenizer weights.",
                quantizer.display()
            );
            None
        }
    }
}

fn is_complete_tokenizer_weight_dir(path: &Path) -> bool {
    TOKENIZER_WEIGHT_FILES
        .iter()
        .all(|file| path.join(file).exists())
}

fn q8_dir_for_f32_dir(path: &Path) -> PathBuf {
    match path.file_name().and_then(|name| name.to_str()) {
        Some("tokenizer-12hz") => path.with_file_name("tokenizer-12hz-q8"),
        Some("tokenizer") => path.with_file_name("tokenizer-q8"),
        Some(name) => path.with_file_name(format!("{name}-q8")),
        None => path.join("tokenizer-q8"),
    }
}

#[derive(Debug, Deserialize)]
struct QuantizationSummary {
    total_original_bytes: usize,
    total_stored_bytes: usize,
    quantized_tensors: usize,
    preserved_tensors: usize,
}

fn q8_weight_summary(path: &Path) -> Option<String> {
    if !is_q8_tokenizer_dir(path) {
        return None;
    }

    let report_path = path.join("quantization_report.json");
    let report = std::fs::read_to_string(report_path)
        .ok()
        .and_then(|text| serde_json::from_str::<QuantizationSummary>(&text).ok());

    let Some(report) = report else {
        return Some(format!(
            "已使用 Q8 量化 tokenizer decoder 權重: {}",
            path.display()
        ));
    };

    let stored_mb = (report.total_stored_bytes as f64 / 1_000_000.0).round() as usize;
    let saved_pct = if report.total_original_bytes == 0 {
        0
    } else {
        (100.0 * (1.0 - report.total_stored_bytes as f64 / report.total_original_bytes as f64))
            .round() as isize
    };
    let tensor_total = report.quantized_tensors + report.preserved_tensors;

    Some(format!(
        "已使用 Q8 量化 tokenizer decoder 權重: {} (約 {} MB，-{}%，{}/{} tensors quantized)",
        path.display(),
        stored_mb,
        saved_pct.max(0),
        report.quantized_tensors,
        tensor_total
    ))
}

fn is_q8_tokenizer_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.ends_with("tokenizer-q8") || name.ends_with("tokenizer-12hz-q8"))
        .unwrap_or(false)
}

fn log_tokenizer_weight_dir(path: &Path) {
    if let Some(summary) = q8_weight_summary(path) {
        eprintln!("{summary}");
    }
}

fn missing_weights_error(candidates: &[PathBuf]) -> Error {
    let checked = candidates
        .iter()
        .map(|p| {
            let missing = missing_tokenizer_files(p);
            if missing.is_empty() {
                format!("  - {}", p.display())
            } else {
                format!("  - {} (missing: {})", p.display(), missing.join(", "))
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    Error::Config(format!(
        "Tokenizer decoder weights not found. Expected converted Rust safetensors in one of:\n{checked}"
    ))
}

fn missing_tokenizer_files(path: &Path) -> Vec<&'static str> {
    TOKENIZER_WEIGHT_FILES
        .iter()
        .copied()
        .filter(|file| !path.join(file).exists())
        .collect()
}

fn push_unique(candidates: &mut Vec<PathBuf>, path: PathBuf) {
    if !candidates.iter().any(|existing| existing == &path) {
        candidates.push(path);
    }
}

fn push_tokenizer_weight_candidates(candidates: &mut Vec<PathBuf>, root: &Path) {
    let weights = root.join("weights");
    push_unique(candidates, weights.join("tokenizer-q8"));
    push_unique(candidates, weights.join("tokenizer"));
}

fn default_tokenizer_q8_cache_dir() -> Option<PathBuf> {
    default_tokenizer_cache_root().map(|root| root.join("tokenizer-12hz-q8"))
}

fn default_tokenizer_cache_dir() -> Option<PathBuf> {
    default_tokenizer_cache_root().map(|root| root.join("tokenizer-12hz"))
}

fn default_tokenizer_cache_root() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("QWEN3TTS_TOKENIZER_CACHE_DIR") {
        return Some(PathBuf::from(path));
    }
    if let Ok(path) = std::env::var("LOCALAPPDATA") {
        return Some(PathBuf::from(path).join("qwen3tts-rs"));
    }
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()
        .map(PathBuf::from)
        .map(|home| home.join(".cache").join("qwen3tts-rs"))
}

#[cfg(test)]
mod tests {
    use super::{
        is_complete_tokenizer_weight_dir, q8_dir_for_f32_dir, q8_weight_summary,
        resolve_converter_exe_from, resolve_converter_script_from, resolve_quantizer_exe_from,
        resolve_tokenizer_weight_dir_from,
    };
    use std::path::Path;

    #[test]
    fn resolves_q8_tokenizer_weights_before_f32_weights() {
        let candidates = resolve_tokenizer_weight_dir_from(
            Path::new("C:/run"),
            Some(Path::new("C:/app/qwen3tts.exe")),
        );

        assert_eq!(candidates[0], Path::new("C:/run/weights/tokenizer-q8"));
        assert_eq!(candidates[1], Path::new("C:/run/weights/tokenizer"));
        assert_eq!(candidates[2], Path::new("C:/app/weights/tokenizer-q8"));
        assert_eq!(candidates[3], Path::new("C:/app/weights/tokenizer"));
    }

    #[test]
    fn deduplicates_when_exe_is_in_current_directory() {
        let candidates = resolve_tokenizer_weight_dir_from(
            Path::new("C:/app"),
            Some(Path::new("C:/app/synthesize.exe")),
        );

        assert_eq!(candidates[0], Path::new("C:/app/weights/tokenizer-q8"));
        assert_eq!(candidates[1], Path::new("C:/app/weights/tokenizer"));
        assert_eq!(
            candidates
                .iter()
                .filter(|p| *p == Path::new("C:/app/weights/tokenizer"))
                .count(),
            1
        );
    }

    #[test]
    fn resolves_converter_script_next_to_cwd_and_exe() {
        let candidates = resolve_converter_script_from(
            Path::new("C:/run"),
            Some(Path::new("C:/app/synthesize.exe")),
        );

        assert_eq!(candidates[0], Path::new("C:/run/tools/convert_weights.py"));
        assert_eq!(candidates[1], Path::new("C:/app/tools/convert_weights.py"));
    }

    #[test]
    fn resolves_rust_converter_exe_next_to_cwd_and_exe() {
        let candidates = resolve_converter_exe_from(
            Path::new("C:/run"),
            Some(Path::new("C:/app/synthesize.exe")),
        );

        assert_eq!(candidates[0], Path::new("C:/run/convert_tokenizer.exe"));
        assert_eq!(candidates[1], Path::new("C:/app/convert_tokenizer.exe"));
    }

    #[test]
    fn resolves_quantizer_exe_next_to_cwd_and_exe() {
        let candidates = resolve_quantizer_exe_from(
            Path::new("C:/run"),
            Some(Path::new("C:/app/synthesize.exe")),
        );

        assert_eq!(candidates[0], Path::new("C:/run/quantize_tokenizer.exe"));
        assert_eq!(candidates[1], Path::new("C:/app/quantize_tokenizer.exe"));
    }

    #[test]
    fn derives_q8_output_dir_from_f32_cache_or_weight_dir() {
        assert_eq!(
            q8_dir_for_f32_dir(Path::new("C:/cache/tokenizer-12hz")),
            Path::new("C:/cache/tokenizer-12hz-q8")
        );
        assert_eq!(
            q8_dir_for_f32_dir(Path::new("C:/app/weights/tokenizer")),
            Path::new("C:/app/weights/tokenizer-q8")
        );
    }

    #[test]
    fn q8_weight_summary_uses_quantization_report_size_and_ratio() {
        let base = std::env::temp_dir()
            .join(format!("qwen3tts-q8-summary-test-{}", std::process::id()))
            .join("tokenizer-q8");
        let root = base.parent().unwrap().to_path_buf();
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(
            base.join("quantization_report.json"),
            r#"{
                "total_original_bytes": 467000000,
                "total_stored_bytes": 125000000,
                "quantized_tensors": 99,
                "preserved_tensors": 137
            }"#,
        )
        .unwrap();

        let summary = q8_weight_summary(&base).expect("summary");

        assert!(summary.contains("已使用 Q8"));
        assert!(summary.contains("125 MB"));
        assert!(summary.contains("-73%"));
        assert!(summary.contains("99/236"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn complete_tokenizer_weight_dir_requires_all_decoder_files() {
        let base = std::env::temp_dir().join(format!("qwen3tts-path-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("codebook.safetensors"), []).unwrap();

        assert!(!is_complete_tokenizer_weight_dir(&base));

        for file in [
            "lightweight.safetensors",
            "pre_transformer.safetensors",
            "upsample.safetensors",
            "decoder_blocks.safetensors",
        ] {
            std::fs::write(base.join(file), []).unwrap();
        }
        assert!(is_complete_tokenizer_weight_dir(&base));

        let _ = std::fs::remove_dir_all(&base);
    }
}
