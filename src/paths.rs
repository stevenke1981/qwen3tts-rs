//! Runtime path helpers for CLI examples and release builds.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{Error, Result};

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
    push_unique(&mut candidates, cwd.join("weights").join("tokenizer"));

    if let Some(exe_path) = exe_path {
        if let Some(exe_dir) = exe_path.parent() {
            push_unique(&mut candidates, exe_dir.join("weights").join("tokenizer"));
        }
    }

    if let Ok(path) = std::env::var("QWEN3TTS_TOKENIZER_WEIGHT_DIR") {
        push_unique(&mut candidates, PathBuf::from(path));
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

fn is_complete_tokenizer_weight_dir(path: &Path) -> bool {
    TOKENIZER_WEIGHT_FILES
        .iter()
        .all(|file| path.join(file).exists())
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

fn default_tokenizer_cache_dir() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("QWEN3TTS_TOKENIZER_CACHE_DIR") {
        return Some(PathBuf::from(path).join("tokenizer-12hz"));
    }
    if let Ok(path) = std::env::var("LOCALAPPDATA") {
        return Some(
            PathBuf::from(path)
                .join("qwen3tts-rs")
                .join("tokenizer-12hz"),
        );
    }
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()
        .map(PathBuf::from)
        .map(|home| {
            home.join(".cache")
                .join("qwen3tts-rs")
                .join("tokenizer-12hz")
        })
}

#[cfg(test)]
mod tests {
    use super::{
        is_complete_tokenizer_weight_dir, resolve_converter_exe_from,
        resolve_converter_script_from, resolve_tokenizer_weight_dir_from,
    };
    use std::path::Path;

    #[test]
    fn resolves_cwd_tokenizer_weights_before_exe_relative_weights() {
        let candidates = resolve_tokenizer_weight_dir_from(
            Path::new("C:/run"),
            Some(Path::new("C:/app/qwen3tts.exe")),
        );

        assert_eq!(candidates[0], Path::new("C:/run/weights/tokenizer"));
        assert_eq!(candidates[1], Path::new("C:/app/weights/tokenizer"));
    }

    #[test]
    fn deduplicates_when_exe_is_in_current_directory() {
        let candidates = resolve_tokenizer_weight_dir_from(
            Path::new("C:/app"),
            Some(Path::new("C:/app/synthesize.exe")),
        );

        assert_eq!(candidates[0], Path::new("C:/app/weights/tokenizer"));
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
