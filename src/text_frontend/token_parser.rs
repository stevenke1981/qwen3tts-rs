//! # Token 解析器
//!
//! 將 LLM 原始輸出（位元組串流或 `Vec<Vec<u16>>`）轉換為 `TokenStream`。

use crate::Result;
use crate::text_frontend::{SynthesisOptions, TokenStream};

// ---------------------------------------------------------------------------
// 常數
// ---------------------------------------------------------------------------

/// 12Hz 模式碼本層數
const NUM_CODEBOOKS: usize = 16;

/// EOS token ID（Qwen3-TTS 特殊 token）
const CODEC_EOS_TOKEN_ID: u16 = 0x7FFF; // 32767, 實際值由 config 決定

// ---------------------------------------------------------------------------
// TokenParser
// ---------------------------------------------------------------------------

/// 將 LLM 回傳的原始碼本張量解析為 `TokenStream`
pub struct TokenParser {
    sample_rate: u32,
}

impl TokenParser {
    /// 建立新的 TokenParser
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// 從 2D 碼本陣列解析 TokenStream
    ///
    /// # 參數
    /// - `codes`: 形狀 `(num_frames, 16)` 的碼本 token 陣列
    ///
    /// 自動移除 EOS token 之後的幀。
    pub fn parse(&self, codes: &[Vec<u16>], _options: &SynthesisOptions) -> Result<TokenStream> {
        let mut stream = TokenStream::new(self.sample_rate);

        for frame_tokens in codes {
            // 檢查是否為 EOS 幀（任一 token 為 EOS）
            let is_eos = frame_tokens.iter().any(|&t| t == CODEC_EOS_TOKEN_ID);
            if is_eos {
                break;
            }

            if frame_tokens.len() != NUM_CODEBOOKS {
                return Err(crate::Error::Config(format!(
                    "Expected {NUM_CODEBOOKS} codebook tokens per frame, got {}",
                    frame_tokens.len()
                )));
            }

            let mut frame = [0u16; NUM_CODEBOOKS];
            frame.copy_from_slice(frame_tokens);
            stream.frames.push(frame);
        }

        Ok(stream)
    }

    /// 從原始位元組串流解析（靜態方法，不需實例）
    ///
    /// 位元組格式（little-endian）：
    /// ```text
    /// [num_frames: u32]
    /// [frame_0_token_0: u16] [frame_0_token_1: u16] ... [frame_0_token_15: u16]
    /// [frame_1_token_0: u16] ...
    /// ```
    ///
    /// 使用預設 24000 Hz 取樣率。
    pub fn parse_binary(data: &[u8]) -> Result<TokenStream> {
        TokenParser::new(24000).parse_bytes(data, &SynthesisOptions::default())
    }

    /// 從原始位元組串流解析
    ///
    /// 位元組格式（little-endian）：
    /// ```text
    /// [num_frames: u32]
    /// [frame_0_token_0: u16] [frame_0_token_1: u16] ... [frame_0_token_15: u16]
    /// [frame_1_token_0: u16] ...
    /// ```
    pub fn parse_bytes(&self, data: &[u8], _options: &SynthesisOptions) -> Result<TokenStream> {
        let header_size = 4; // u32
        if data.len() < header_size {
            return Err(crate::Error::Config("Empty token data".into()));
        }

        let num_frames = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let expected_size = header_size + num_frames * NUM_CODEBOOKS * 2;
        let frame_size = NUM_CODEBOOKS * 2; // 16 u16 = 32 bytes

        if num_frames == 0 && data.len() > header_size && data.len() % frame_size == 0 {
            return self.parse_frame_bytes(data, data.len() / frame_size, 0);
        }

        if data.len() < expected_size {
            if data.len() % frame_size == 0 {
                return self.parse_frame_bytes(data, data.len() / frame_size, 0);
            }
            return Err(crate::Error::Config(format!(
                "Token data too short: {} bytes, expected {}",
                data.len(),
                expected_size
            )));
        }

        self.parse_frame_bytes(data, num_frames, header_size)
    }

    fn parse_frame_bytes(
        &self,
        data: &[u8],
        num_frames: usize,
        base_offset: usize,
    ) -> Result<TokenStream> {
        let mut stream = TokenStream::new(self.sample_rate);
        let frame_size = NUM_CODEBOOKS * 2;

        for i in 0..num_frames {
            let offset = base_offset + i * frame_size;
            let raw = &data[offset..offset + frame_size];

            let mut frame = [0u16; NUM_CODEBOOKS];
            for j in 0..NUM_CODEBOOKS {
                frame[j] = u16::from_le_bytes(raw[j * 2..j * 2 + 2].try_into().unwrap());
            }

            // 檢查 EOS
            let is_eos = frame.iter().any(|&t| t == CODEC_EOS_TOKEN_ID);
            if is_eos {
                break;
            }

            stream.frames.push(frame);
        }

        Ok(stream)
    }
}

// ---------------------------------------------------------------------------
// 測試
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_parser_simple() {
        let parser = TokenParser::new(24000);
        let codes = vec![vec![42; 16], vec![100; 16], vec![200; 16]];
        let stream = parser.parse(&codes, &SynthesisOptions::default()).unwrap();
        assert_eq!(stream.num_frames(), 3);
        assert!((stream.duration_sec() - 0.24).abs() < 0.01);
    }

    #[test]
    fn test_token_parser_eos_stop() {
        let parser = TokenParser::new(24000);
        let mut codes: Vec<Vec<u16>> = (0..5).map(|i| vec![i as u16 * 10; 16]).collect();
        codes.push(vec![CODEC_EOS_TOKEN_ID; 16]); // EOS frame
        codes.push(vec![999; 16]); // should be ignored

        let stream = parser.parse(&codes, &SynthesisOptions::default()).unwrap();
        assert_eq!(stream.num_frames(), 5);
    }

    #[test]
    fn test_parse_bytes_roundtrip() {
        let parser = TokenParser::new(24000);
        let mut data = Vec::new();
        data.extend_from_slice(&3u32.to_le_bytes()); // 3 frames
        for i in 0..3 {
            for _ in 0..16 {
                data.extend_from_slice(&(i as u16 * 10).to_le_bytes());
            }
        }

        let stream = parser
            .parse_bytes(&data, &SynthesisOptions::default())
            .unwrap();
        assert_eq!(stream.num_frames(), 3);
        assert_eq!(stream.frames[0][0], 0);
        assert_eq!(stream.frames[1][0], 10);
        assert_eq!(stream.frames[2][0], 20);
    }

    #[test]
    fn test_parse_raw_u16_frames() {
        let parser = TokenParser::new(24000);
        let mut data = Vec::new();
        for i in 0..3 {
            for _ in 0..16 {
                data.extend_from_slice(&(i as u16 * 10).to_le_bytes());
            }
        }

        let stream = parser
            .parse_bytes(&data, &SynthesisOptions::default())
            .unwrap();
        assert_eq!(stream.num_frames(), 3);
        assert_eq!(stream.frames[0][0], 0);
        assert_eq!(stream.frames[1][0], 10);
        assert_eq!(stream.frames[2][0], 20);
    }

    #[test]
    fn test_wrong_codebook_count() {
        let parser = TokenParser::new(24000);
        let codes = vec![vec![1, 2, 3]]; // only 3 codebooks
        let result = parser.parse(&codes, &SynthesisOptions::default());
        assert!(result.is_err());
    }
}
