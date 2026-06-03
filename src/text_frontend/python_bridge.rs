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
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::Result;
use crate::text_frontend::token_parser::TokenParser;
use crate::text_frontend::{SynthesisOptions, TextFrontend, TokenStream};

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

    /// 執行 Python 子行程並回傳 Token 位元組
    fn run_python(&self, text: &str, options: &SynthesisOptions) -> Result<Vec<u8>> {
        let mut cmd = Command::new(&self.python_path);

        cmd.arg(&self.script_path)
            .arg("--text")
            .arg(text)
            .arg("--model")
            .arg(&self.model_id)
            .arg("--language")
            .arg(&options.language)
            .arg("--speaker")
            .arg(options.speaker.as_deref().unwrap_or(""))
            .arg("--temperature")
            .arg(options.temperature.to_string())
            .arg("--top-k")
            .arg(options.top_k.to_string())
            .arg("--top-p")
            .arg(options.top_p.to_string())
            .arg("--max-new-tokens")
            .arg(options.max_new_tokens.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

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
}
