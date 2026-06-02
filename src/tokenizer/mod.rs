//! # Tokenizer 載入與解析模組
//!
//! 從官方 `tokenizer.json` 載入預分詞器規則。
//!
//! ## 設計原則
//! - 必須從配置檔案完整載入，禁止硬編碼
//! - 支援 12Hz 與 25Hz 雙模式 Tokenizer

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

/// Tokenizer 狀態
#[derive(Debug, Clone)]
pub struct Tokenizer {
    #[allow(dead_code)]
    config: TokenizerConfig,
    // TODO(Phase 0): 載入真實 tokenizer.json
    // vocab: HashMap<String, u32>,
}

impl Tokenizer {
    /// 從設定檔路徑建立 Tokenizer
    pub fn new(config: TokenizerConfig) -> Result<Self> {
        // TODO(Phase 0): 實作 tokenizer.json 解析
        Ok(Self { config })
    }

    /// 將文字編碼為 Token ID 序列
    pub fn encode(&self, _text: &str) -> Result<Vec<u32>> {
        // TODO(Phase 0): 實作編碼邏輯
        Err(Error::Config("Tokenizer encode not yet implemented".into()))
    }

    /// 將 Token ID 序列解碼為文字
    pub fn decode(&self, _tokens: &[u32]) -> Result<String> {
        // TODO(Phase 0): 實作解碼邏輯
        Err(Error::Config("Tokenizer decode not yet implemented".into()))
    }
}
