//! # Python 橋接後端
//!
//! 透過子行程調用 `tools/generate_tokens.py`，
//! 利用現成的 qwen-tts Python 套件產生語義 Token。
//!
//! ## 設計要點
//! - Python 行程以 binary stdout 輸出 Token 幀
//! - Rust 端以 `read_exact` 高效讀取（無文字解析開銷）
//! - 支援 Model ID 配置（可選 0.6B / 1.7B）
//! - LLM 載入後持續在行程記憶體中（多次呼叫共享行程）

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::text_frontend::speaker_presets;
use crate::text_frontend::token_parser::TokenParser;
use crate::text_frontend::{SynthesisOptions, TextFrontend, TokenStream};
use crate::Result;

// ---------------------------------------------------------------------------
// PythonBridge
// ---------------------------------------------------------------------------

/// 透過子行程調用 qwen-tts Python 套件的文本前端
///
/// # 範例
/// ```no_run
/// use qwen3tts::text_frontend::{PythonBridge, SynthesisOptions, TextFrontend};
///
/// let bridge = PythonBridge::new("Qwen/Qwen3-TTS-12Hz-0.6B-Base")?;
/// let stream = bridge.synthesize("你好世界", &SynthesisOptions::default())?;
/// # Ok::<_, qwen3tts::Error>(())
/// ```
pub struct PythonBridge {
    /// HuggingFace model ID
    model_id: String,

    /// Python 腳本路徑
    script_path: PathBuf,

    /// Python 直譯器路徑
    python_path: String,

    /// Token 解析器
    parser: TokenParser,
}

impl PythonBridge {
    /// 建立新的 PythonBridge
    ///
    /// # 參數
    /// - `model_id`: HuggingFace model ID（如 `"Qwen/Qwen3-TTS-12Hz-0.6B-Base"`）
    pub fn new(model_id: impl Into<String>) -> Result<Self> {
        let model_id = model_id.into();

        // 自動偵測 Python 直譯器
        let python_path = detect_python();

        // 腳本路徑：相對於專案根目錄
        let script_path = find_script("tools/generate_tokens.py")
            .or_else(|| find_script("../tools/generate_tokens.py"))
            .unwrap_or_else(|| PathBuf::from("tools/generate_tokens.py"));

        Ok(Self {
            model_id,
            script_path,
            python_path,
            parser: TokenParser::new(24000),
        })
    }

    /// 指定 Python 直譯器路徑
    pub fn with_python(mut self, python: impl Into<String>) -> Self {
        self.python_path = python.into();
        self
    }

    /// 指定 Token 腳本路徑
    pub fn with_script(mut self, path: impl Into<PathBuf>) -> Self {
        self.script_path = path.into();
        self
    }

    fn effective_conditions(options: &SynthesisOptions) -> (Option<String>, Option<String>) {
        let requested_speaker = options.speaker.as_deref();
        let effective_speaker = requested_speaker
            .and_then(speaker_presets::canonical_name)
            .or(requested_speaker)
            .map(ToOwned::to_owned);
        let effective_instruct = speaker_presets::effective_instruct(
            options.instruct.clone(),
            effective_speaker.as_deref(),
        );
        (effective_speaker, effective_instruct)
    }

    fn build_token_args(&self, text: &str, options: &SynthesisOptions) -> Vec<String> {
        let (effective_speaker, effective_instruct) = Self::effective_conditions(options);
        let mut args = vec![
            self.script_path.to_string_lossy().into_owned(),
            "--text".into(),
            text.into(),
            "--model".into(),
            self.model_id.clone(),
            "--language".into(),
            options.language.clone(),
            "--speaker".into(),
            effective_speaker.unwrap_or_default(),
            "--instruct".into(),
            effective_instruct.unwrap_or_default(),
            "--temperature".into(),
            options.temperature.to_string(),
            "--top-k".into(),
            options.top_k.to_string(),
            "--top-p".into(),
            options.top_p.to_string(),
            "--max-new-tokens".into(),
            options.max_new_tokens.to_string(),
        ];

        if let Some(seed) = options.seed {
            args.push("--seed".into());
            args.push(seed.to_string());
        }

        args
    }

    fn build_voice_clone_args(
        &self,
        text: &str,
        options: &SynthesisOptions,
        output_path: &Path,
    ) -> Result<Vec<String>> {
        let reference_audio = options.reference_audio.as_deref().ok_or_else(|| {
            crate::Error::Config(
                "voice-clone Python bridge requires options.reference_audio".into(),
            )
        })?;

        let mut args = vec![
            self.script_path.to_string_lossy().into_owned(),
            "--voice-clone".into(),
            "--text".into(),
            text.into(),
            "--model".into(),
            self.model_id.clone(),
            "--language".into(),
            options.language.clone(),
            "--reference-audio".into(),
            reference_audio.into(),
            "--output-wav".into(),
            output_path.to_string_lossy().into_owned(),
            "--temperature".into(),
            options.temperature.to_string(),
            "--top-k".into(),
            options.top_k.to_string(),
            "--top-p".into(),
            options.top_p.to_string(),
            "--max-new-tokens".into(),
            options.max_new_tokens.to_string(),
        ];

        if let Some(reference_text) = options.reference_text.as_deref() {
            if !reference_text.trim().is_empty() {
                args.push("--reference-text".into());
                args.push(reference_text.into());
            }
        }
        if let Some(seed) = options.seed {
            args.push("--seed".into());
            args.push(seed.to_string());
        }

        Ok(args)
    }

    /// 執行 Python 子行程並回傳 Token 位元組
    fn run_python(&self, text: &str, options: &SynthesisOptions) -> Result<Vec<u8>> {
        let mut cmd = Command::new(&self.python_path);
        cmd.args(self.build_token_args(text, options));

        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            crate::Error::Config(format!(
                "Failed to launch Python bridge: {e}.\n\
                 Make sure Python with qwen-tts package is installed:\n\
                 pip install qwen-tts"
            ))
        })?;

        // 讀取 stdout（binary token data）
        let mut stdout = child.stdout.take().unwrap();
        let mut output = Vec::new();
        stdout.read_to_end(&mut output)?;

        // 讀取 stderr（log 訊息）
        let stderr_output = child.wait()?;
        if !stderr_output.success() {
            let mut stderr = String::new();
            if let Some(ref mut stderr_pipe) = child.stderr {
                stderr_pipe.read_to_string(&mut stderr)?;
            }
            log::warn!("Python bridge stderr: {stderr}");
        }

        Ok(output)
    }

    /// Run the official qwen-tts voice-clone path and write a WAV directly.
    ///
    /// This path is intentionally separate from token generation: official
    /// Voice Clone needs the speech tokenizer encoder and speaker encoder,
    /// which are still pending in the native Candle implementation.
    pub fn synthesize_voice_clone_wav(
        &self,
        text: &str,
        options: &SynthesisOptions,
        output_path: impl AsRef<Path>,
    ) -> Result<()> {
        let mut cmd = Command::new(&self.python_path);
        cmd.args(self.build_voice_clone_args(text, options, output_path.as_ref())?);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let output = cmd.output().map_err(|e| {
            crate::Error::Config(format!(
                "Failed to launch Python voice-clone bridge: {e}.\n\
                 Make sure Python with qwen-tts package is installed:\n\
                 pip install qwen-tts"
            ))
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(crate::Error::Config(format!(
                "Python voice-clone bridge failed: {stderr}"
            )));
        }
        if !output.stderr.is_empty() {
            log::info!(
                "Python voice-clone bridge stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }
}

impl TextFrontend for PythonBridge {
    fn synthesize(&self, text: &str, options: &SynthesisOptions) -> Result<TokenStream> {
        let data = self.run_python(text, options)?;

        if data.is_empty() {
            return Err(crate::Error::Config(
                "Python bridge returned empty output".into(),
            ));
        }

        self.parser.parse_bytes(&data, options)
    }
}

impl std::fmt::Debug for PythonBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PythonBridge")
            .field("model_id", &self.model_id)
            .field("script_path", &self.script_path)
            .field("python_path", &self.python_path)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// 輔助函式
// ---------------------------------------------------------------------------

/// 自動偵測系統中的 Python 直譯器
fn detect_python() -> String {
    // 優先順序: python3 > python
    for candidate in &["python3", "python"] {
        if Command::new(candidate)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
        {
            return candidate.to_string();
        }
    }
    "python".to_string()
}

/// 搜尋腳本檔案（從 CWD 和常見位置）
fn find_script(path: &str) -> Option<PathBuf> {
    let p = PathBuf::from(path);
    if p.exists() {
        return Some(p);
    }
    None
}

// ---------------------------------------------------------------------------
// 測試
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_detection() {
        let python = detect_python();
        assert!(!python.is_empty(), "Python should be available");
    }

    #[test]
    fn test_bridge_creation() {
        let bridge = PythonBridge::new("test-model").unwrap();
        assert_eq!(bridge.model_id, "test-model");
    }

    #[test]
    fn test_script_exists() {
        // 檢查 generate_tokens.py 是否存在
        let found = find_script("tools/generate_tokens.py")
            .or_else(|| find_script("../tools/generate_tokens.py"));
        assert!(
            found.is_some(),
            "generate_tokens.py should exist at tools/generate_tokens.py"
        );
    }

    #[test]
    fn voice_clone_args_include_reference_audio_and_output() {
        let bridge = PythonBridge::new("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
            .unwrap()
            .with_script("tools/generate_tokens.py");
        let options = SynthesisOptions {
            language: "Chinese".into(),
            reference_audio: Some("ref.wav".into()),
            reference_text: Some("參考文字".into()),
            seed: Some(1234),
            ..SynthesisOptions::default()
        };

        let args = bridge
            .build_voice_clone_args("要合成的文字", &options, Path::new("clone.wav"))
            .unwrap();

        assert!(args.contains(&"--voice-clone".into()));
        assert_arg_pair(&args, "--reference-audio", "ref.wav");
        assert_arg_pair(&args, "--reference-text", "參考文字");
        assert_arg_pair(&args, "--output-wav", "clone.wav");
        assert_arg_pair(&args, "--seed", "1234");
    }

    #[test]
    fn voice_clone_args_require_reference_audio() {
        let bridge = PythonBridge::new("Qwen/Qwen3-TTS-12Hz-0.6B-Base").unwrap();
        let err = bridge
            .build_voice_clone_args("text", &SynthesisOptions::default(), Path::new("out.wav"))
            .unwrap_err();
        assert!(err.to_string().contains("reference_audio"));
    }

    fn assert_arg_pair(args: &[String], flag: &str, value: &str) {
        let idx = args
            .iter()
            .position(|arg| arg == flag)
            .unwrap_or_else(|| panic!("missing arg {flag} in {args:?}"));
        assert_eq!(args.get(idx + 1).map(String::as_str), Some(value));
    }
}
