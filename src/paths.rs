//! Runtime path helpers for CLI examples and release builds.

use std::path::{Path, PathBuf};

use crate::{Error, Result};

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

    candidates
}

/// Finds an existing tokenizer decoder weight directory.
pub fn find_existing_tokenizer_weight_dir() -> Result<PathBuf> {
    let cwd = std::env::current_dir()?;
    let exe = std::env::current_exe().ok();
    let candidates = resolve_tokenizer_weight_dir_from(&cwd, exe.as_deref());

    for candidate in &candidates {
        if candidate.join("codebook.safetensors").exists() {
            return Ok(candidate.clone());
        }
    }

    let checked = candidates
        .iter()
        .map(|p| format!("  - {}", p.display()))
        .collect::<Vec<_>>()
        .join("\n");
    Err(Error::Config(format!(
        "Tokenizer decoder weights not found. Expected codebook.safetensors in one of:\n{checked}"
    )))
}

fn push_unique(candidates: &mut Vec<PathBuf>, path: PathBuf) {
    if !candidates.iter().any(|existing| existing == &path) {
        candidates.push(path);
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_tokenizer_weight_dir_from;
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

        assert_eq!(candidates, vec![Path::new("C:/app/weights/tokenizer")]);
    }
}
