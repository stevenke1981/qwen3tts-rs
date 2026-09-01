//! # Tokenizer 載入與解析模組
//!
//! 提供對 HuggingFace `tokenizers::Tokenizer` 的安全封裝，
//! 支援從 `tokenizer.json` 載入、文字與 Token ID 序列雙向轉換、特殊 Token 處理與詞表查詢。

use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

use crate::{Error, Result};

/// Tokenizer 配置
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Tokenizer 設定檔路徑
    pub tokenizer_path: String,
    /// Token 最大長度
    pub max_length: usize,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            tokenizer_path: "tokenizer.json".into(),
            max_length: 2048,
        }
    }
}

/// Tokenizer 狀態包裝器
#[derive(Clone)]
pub struct Tokenizer {
    inner: HfTokenizer,
    #[allow(dead_code)]
    config: Option<TokenizerConfig>,
}

impl Tokenizer {
    /// 從 HuggingFace `tokenizers::Tokenizer` 實例建立
    pub fn new(inner: HfTokenizer) -> Self {
        Self {
            inner,
            config: None,
        }
    }

    /// 從配置檔案路徑建立 Tokenizer
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let p = path.as_ref();
        let inner = HfTokenizer::from_file(p)
            .map_err(|e| Error::Config(format!("Failed to load tokenizer from {}: {e}", p.display())))?;
        Ok(Self {
            inner,
            config: Some(TokenizerConfig {
                tokenizer_path: p.to_string_lossy().to_string(),
                max_length: 2048,
            }),
        })
    }

    /// 從 JSON 字串建立 Tokenizer
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(json: &str) -> Result<Self> {
        Self::from_bytes(json.as_bytes())
    }

    /// 從位元組切片建立 Tokenizer
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let inner = HfTokenizer::from_bytes(bytes)
            .map_err(|e| Error::Config(format!("Failed to parse tokenizer from bytes: {e}")))?;
        Ok(Self {
            inner,
            config: None,
        })
    }

    /// 從 TokenizerConfig 建立 Tokenizer
    pub fn from_config(config: TokenizerConfig) -> Result<Self> {
        let mut tok = Self::from_file(&config.tokenizer_path)?;
        tok.config = Some(config);
        Ok(tok)
    }

    /// 將文字編碼為 Token ID 序列（預設包含特殊 Token）
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_with_special_tokens(text, true)
    }

    /// 將文字編碼為 Token ID 序列，可指定是否包含特殊 Token
    pub fn encode_with_special_tokens(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| Error::Config(format!("Failed to encode text: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// 將 Token ID 序列解碼為文字（預設跳過特殊 Token）
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.decode_with_special_tokens(tokens, true)
    }

    /// 將 Token ID 序列解碼為文字，可指定是否跳過特殊 Token
    pub fn decode_with_special_tokens(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(tokens, skip_special_tokens)
            .map_err(|e| Error::Config(format!("Failed to decode tokens: {e}")))
    }

    /// 取得詞表大小
    pub fn vocab_size(&self, with_added_tokens: bool) -> usize {
        self.inner.get_vocab_size(with_added_tokens)
    }

    /// 查詢 Token 對應的 ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// 查詢 ID 對應的 Token 字串
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// 取得內部 `tokenizers::Tokenizer` 的不可變參考
    pub fn inner(&self) -> &HfTokenizer {
        &self.inner
    }

    /// 消費自身並轉化為內部 `tokenizers::Tokenizer`
    pub fn into_inner(self) -> HfTokenizer {
        self.inner
    }
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size(true))
            .field("config", &self.config)
            .finish()
    }
}

impl std::str::FromStr for Tokenizer {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_bytes(s.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    { "id": 0, "content": "<|pad|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
    { "id": 1, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
    { "id": 2, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true }
  ],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "unk_token": "<|pad|>",
    "vocab": {
      "<|pad|>": 0,
      "<|im_start|>": 1,
      "<|im_end|>": 2,
      "hello": 3,
      "world": 4,
      "qwen": 5
    }
  }
}"#;

    fn create_test_tokenizer() -> Tokenizer {
        Tokenizer::from_str(SAMPLE_TOKENIZER_JSON).expect("valid sample tokenizer JSON")
    }

    #[test]
    fn test_tokenizer_encode_decode_roundtrip() {
        let tok = create_test_tokenizer();
        let tokens = tok.encode("hello world").expect("encode should succeed");
        assert_eq!(tokens, vec![3, 4]);

        let decoded = tok.decode(&tokens).expect("decode should succeed");
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_tokenizer_vocab_and_token_lookups() {
        let tok = create_test_tokenizer();
        assert_eq!(tok.vocab_size(true), 6);
        assert_eq!(tok.token_to_id("hello"), Some(3));
        assert_eq!(tok.token_to_id("qwen"), Some(5));
        assert_eq!(tok.token_to_id("unknown_token"), None);

        assert_eq!(tok.id_to_token(3), Some("hello".to_string()));
        assert_eq!(tok.id_to_token(5), Some("qwen".to_string()));
        assert_eq!(tok.id_to_token(999), None);
    }

    #[test]
    fn test_tokenizer_from_nonexistent_file_fails() {
        let result = Tokenizer::from_file("nonexistent_path_to_tokenizer.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenizer_config_default_and_debug() {
        let config = TokenizerConfig::default();
        assert_eq!(config.tokenizer_path, "tokenizer.json");
        assert_eq!(config.max_length, 2048);

        let tok = create_test_tokenizer();
        let debug_str = format!("{tok:?}");
        assert!(debug_str.contains("Tokenizer"));
        assert!(debug_str.contains("vocab_size: 6"));
    }
}

