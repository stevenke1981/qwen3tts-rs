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

/// 預設 EOS token ID（Qwen3-TTS 官方標準）
pub const DEFAULT_CODEC_EOS_TOKEN_ID: u16 = 2150;

// ---------------------------------------------------------------------------
// TokenParser
// ---------------------------------------------------------------------------

/// 將 LLM 回傳的原始碼本張量解析為 `TokenStream`
#[derive(Debug, Clone)]
pub struct TokenParser {
    sample_rate: u32,
    codec_eos_token_id: u16,
}

impl TokenParser {
    /// 建立新的 TokenParser（使用預設 EOS token ID 2150）
    pub fn new(sample_rate: u32) -> Self {
        Self::with_eos(sample_rate, DEFAULT_CODEC_EOS_TOKEN_ID)
    }

    /// 建立指定 EOS token ID 的 TokenParser
    pub fn with_eos(sample_rate: u32, codec_eos_token_id: u16) -> Self {
        Self {
            sample_rate,
            codec_eos_token_id,
        }
    }

    /// 取得目前的 codec EOS token ID
    pub fn codec_eos_token_id(&self) -> u16 {
        self.codec_eos_token_id
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
            let is_eos = frame_tokens.iter().any(|&t| t == self.codec_eos_token_id);
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
    /// 使用預設 24000 Hz 取樣率與預設 EOS ID (2150)。
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
            let is_eos = frame.iter().any(|&t| t == self.codec_eos_token_id);
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
        assert_eq!(parser.codec_eos_token_id(), DEFAULT_CODEC_EOS_TOKEN_ID);
        assert_eq!(parser.codec_eos_token_id(), 2150);

        let mut codes: Vec<Vec<u16>> = (0..5).map(|i| vec![i as u16 * 10; 16]).collect();
        codes.push(vec![DEFAULT_CODEC_EOS_TOKEN_ID; 16]); // EOS frame (2150)
        codes.push(vec![999; 16]); // should be ignored

        let stream = parser.parse(&codes, &SynthesisOptions::default()).unwrap();
        assert_eq!(stream.num_frames(), 5);
    }

    #[test]
    fn test_token_parser_custom_eos() {
        let custom_eos = 9999u16;
        let parser = TokenParser::with_eos(24000, custom_eos);
        assert_eq!(parser.codec_eos_token_id(), custom_eos);

        let codes = vec![
            vec![10; 16],
            vec![DEFAULT_CODEC_EOS_TOKEN_ID; 16], // 2150 should NOT stop custom parser
            vec![20; 16],
            vec![custom_eos; 16],                 // custom EOS should stop here
            vec![30; 16],                         // should be ignored
        ];

        let stream = parser.parse(&codes, &SynthesisOptions::default()).unwrap();
        assert_eq!(stream.num_frames(), 3);
        assert_eq!(stream.frames[0][0], 10);
        assert_eq!(stream.frames[1][0], DEFAULT_CODEC_EOS_TOKEN_ID);
        assert_eq!(stream.frames[2][0], 20);
    }

    #[test]
    fn test_parse_bytes_eos_stop() {
        let parser = TokenParser::new(24000);
        let mut data = Vec::new();
        data.extend_from_slice(&4u32.to_le_bytes()); // 4 frames declared
        // Frame 0: normal
        for _ in 0..16 {
            data.extend_from_slice(&10u16.to_le_bytes());
        }
        // Frame 1: normal
        for _ in 0..16 {
            data.extend_from_slice(&20u16.to_le_bytes());
        }
        // Frame 2: EOS (2150)
        for _ in 0..16 {
            data.extend_from_slice(&DEFAULT_CODEC_EOS_TOKEN_ID.to_le_bytes());
        }
        // Frame 3: trailing
        for _ in 0..16 {
            data.extend_from_slice(&30u16.to_le_bytes());
        }

        let stream = parser
            .parse_bytes(&data, &SynthesisOptions::default())
            .unwrap();
        assert_eq!(stream.num_frames(), 2);
        assert_eq!(stream.frames[0][0], 10);
        assert_eq!(stream.frames[1][0], 20);
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
    fn test_token_stream_binary_roundtrip() {
        let parser = TokenParser::new(24000);
        let mut original = TokenStream::new(24000);
        original.frames.push([7; 16]);
        original.frames.push([42; 16]);

        let parsed = parser
            .parse_bytes(&original.to_binary(), &SynthesisOptions::default())
            .unwrap();

        assert_eq!(parsed.num_frames(), 2);
        assert_eq!(parsed.frames, original.frames);
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
