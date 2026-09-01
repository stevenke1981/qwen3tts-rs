//! Empirical Tokenizer Adversarial Stress Test Suite
//!
//! Rigorously validates HuggingFace Tokenizer wrapper robustness against:
//! 1. Empty strings and empty token lists (both with and without special tokens).
//! 2. Production Qwen tokenizer (`models/tokenizer.json`) and synthetic tokenizers on
//!    multilingual UTF-8 (Traditional Chinese, Simplified Chinese, Japanese, Korean,
//!    Arabic RTL, Hebrew RTL, European accents, mathematical symbols, Emojis, ZWJ compound emojis).
//! 3. Ultra-long texts (100,000+ characters) and dense whitespace stress.
//! 4. Invalid UTF-8 byte slices, truncated multibyte sequences, and malformed JSON payloads (fail-closed, zero panic).
//! 5. Nonexistent file paths, empty paths, directory paths, and invalid configs (fail-closed, zero panic).
//! 6. Boundary token IDs, out-of-vocab lookups, and inner wrapper ownership transitions.

use std::path::Path;
use qwen3tts::tokenizer::{Tokenizer, TokenizerConfig};

const SAMPLE_WORDLEVEL_TOKENIZER_JSON: &str = r#"{
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

fn create_synthetic_tokenizer() -> Tokenizer {
    Tokenizer::from_str(SAMPLE_WORDLEVEL_TOKENIZER_JSON).expect("valid sample WordLevel tokenizer JSON")
}

fn get_production_tokenizer_or_skip() -> Option<Tokenizer> {
    let path = Path::new("models/tokenizer.json");
    if path.exists() {
        Tokenizer::from_file(path).ok()
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// 1. Empty Strings & Empty Token Lists
// ---------------------------------------------------------------------------

#[test]
fn test_tokenizer_empty_strings_and_empty_tokens() {
    let tok = create_synthetic_tokenizer();

    // Empty text encoding
    let enc_empty = tok.encode("").expect("encode empty string must succeed");
    assert!(enc_empty.is_empty(), "Empty string should encode to empty token list");

    let enc_empty_no_special = tok
        .encode_with_special_tokens("", false)
        .expect("encode empty without special must succeed");
    assert!(enc_empty_no_special.is_empty());

    // Empty token decoding
    let dec_empty = tok.decode(&[]).expect("decode empty token list must succeed");
    assert_eq!(dec_empty, "");

    let dec_empty_no_special = tok
        .decode_with_special_tokens(&[], false)
        .expect("decode empty without special must succeed");
    assert_eq!(dec_empty_no_special, "");

    // Also test production tokenizer on empty strings if available
    if let Some(prod_tok) = get_production_tokenizer_or_skip() {
        let prod_enc_empty = prod_tok.encode("").expect("prod encode empty string");
        assert!(prod_enc_empty.is_empty());
        let prod_dec_empty = prod_tok.decode(&[]).expect("prod decode empty tokens");
        assert_eq!(prod_dec_empty, "");
    }
}

// ---------------------------------------------------------------------------
// 2. Multi-byte Unicode, CJK, RTL, Emojis, and Combining Characters
// ---------------------------------------------------------------------------

#[test]
fn test_tokenizer_unicode_multilingual_and_emojis() {
    let test_strings = [
        // Traditional Chinese
        "語音合成技術深度測試（繁體中文）",
        // Simplified Chinese
        "语音合成技术深度测试（简体中文）",
        // Japanese Kanji, Hiragana, Katakana
        "音声合成技術のテスト：ひらがな、カタカナ、漢字！",
        // Korean Hangul
        "음성 합성 기술 심층 테스트 (한국어)",
        // Emojis & complex glyphs
        "🎉 🚀 🦀 ⚡ ✨ 🎯 🔥 💡 🤖 🎧",
        // Zero-Width-Joiner compound emoji (family / technologist)
        "👨‍👩‍👧‍👦 👩‍💻 🏃‍♀️",
        // Arabic (RTL)
        "مرحبا بكم في عالم الذكاء الاصطناعي والتعلم الآلي",
        // Hebrew (RTL)
        "שלום עולם, בדיקת סינתזת דיבור",
        // European multilingual accents & diacritics
        "Café, mañana, résumé, naïve, façade, Über, København, Łódź, Çorlu",
        // Combining characters
        "e\u{0301} a\u{0308} o\u{0303} c\u{0327}",
        // Math / Greek / Cyrillic
        "∀x ∈ ℝ: ∑(x_i) = ∫ f(t)dt, α+β=γ, Привет мир!",
        // Mixed non-printable and special whitespace
        "line1\r\nline2\tline3 \u{200B}zero-width\u{FEFF}BOM",
    ];

    if let Some(prod_tok) = get_production_tokenizer_or_skip() {
        for (idx, &s) in test_strings.iter().enumerate() {
            let tokens = prod_tok
                .encode(s)
                .unwrap_or_else(|e| panic!("Failed to encode string {idx} ('{s}'): {e}"));
            assert!(!tokens.is_empty(), "Non-empty string '{s}' must produce tokens");

            let decoded = prod_tok
                .decode(&tokens)
                .unwrap_or_else(|e| panic!("Failed to decode tokens for string {idx}: {e}"));

            // Decoded text must match original string
            assert_eq!(
                decoded.trim(),
                s.trim(),
                "Decoded text must match original for string {idx} ('{s}')"
            );
        }
    } else {
        // Fallback with synthetic tokenizer
        let tok = create_synthetic_tokenizer();
        for &s in &["hello world", "hello", "world", "qwen"] {
            let tokens = tok.encode(s).expect("synthetic encode");
            assert!(!tokens.is_empty());
            let decoded = tok.decode(&tokens).expect("synthetic decode");
            assert_eq!(decoded, s);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Ultra-Long Texts and Dense Whitespace
// ---------------------------------------------------------------------------

#[test]
fn test_tokenizer_ultra_long_texts_and_whitespace_stress() {
    let unit = "hello world qwen 語音合成測試 ";
    let repeat_count = 100_000 / unit.len() + 1;
    let huge_text: String = unit.repeat(repeat_count);
    assert!(huge_text.len() >= 100_000);

    if let Some(prod_tok) = get_production_tokenizer_or_skip() {
        let start = std::time::Instant::now();
        let tokens = prod_tok.encode(&huge_text).expect("encode 100k text");
        let encode_duration = start.elapsed();
        assert!(!tokens.is_empty());
        assert!(
            encode_duration.as_secs() < 5,
            "100k chars encoding should be fast (took {:?})",
            encode_duration
        );

        let decoded = prod_tok.decode(&tokens).expect("decode 100k tokens");
        assert_eq!(decoded.trim(), huge_text.trim());

        // Dense whitespace string (50,000 chars)
        let whitespace_stress: String = " \t\r\n   ".repeat(10_000);
        let ws_tokens = prod_tok.encode(&whitespace_stress).expect("encode dense whitespace");
        let ws_decoded = prod_tok.decode(&ws_tokens).expect("decode dense whitespace");
        assert_eq!(ws_decoded.len(), whitespace_stress.len());
    } else {
        let syn_tok = create_synthetic_tokenizer();
        let syn_unit = "hello world ";
        let syn_huge = syn_unit.repeat(10_000);
        let tokens = syn_tok.encode(&syn_huge).expect("syn encode");
        assert!(!tokens.is_empty());
        let decoded = syn_tok.decode(&tokens).expect("syn decode");
        assert_eq!(decoded.len(), syn_huge.len());
    }
}

// ---------------------------------------------------------------------------
// 4. Invalid UTF-8 Bytes and Malformed JSON Payloads
// ---------------------------------------------------------------------------

#[test]
fn test_tokenizer_invalid_utf8_bytes_and_malformed_json_fail_closed() {
    // 1. Completely invalid UTF-8 byte slices
    let invalid_utf8_cases: &[&[u8]] = &[
        &[0xFF, 0xFE, 0xFD],
        &[0x80, 0x81, 0x82],
        &[0xC0, 0xAF],             // Overlong 2-byte sequence
        &[0xE0, 0x80, 0xAF],       // Overlong 3-byte sequence
        &[0xF0, 0x80, 0x80, 0xAF], // Overlong 4-byte sequence
        &[0xF4, 0x90, 0x80, 0x80], // Out of Unicode range (> U+10FFFF)
        &[0xED, 0xA0, 0x80],       // UTF-16 surrogate half
        &[0xE2, 0x82],             // Truncated 3-byte sequence
        &[0xF0, 0x9F, 0x98],       // Truncated 4-byte sequence
    ];

    for (i, &bytes) in invalid_utf8_cases.iter().enumerate() {
        let res = Tokenizer::from_bytes(bytes);
        assert!(
            res.is_err(),
            "Invalid UTF-8 case {i} ({:02X?}) must fail gracefully with Err, not panic",
            bytes
        );
    }

    // 2. Empty byte slice
    assert!(Tokenizer::from_bytes(&[]).is_err(), "Empty bytes must return Err");

    // 3. Malformed JSON payloads
    let malformed_json_cases = [
        "{",
        "{\"version\": 1.0",
        "{\"model\": null}",
        "random non-json content 1234567890",
        "<xml><tokenizer></tokenizer></xml>",
        "{\"version\": \"1.0\", \"model\": {\"type\": \"UnknownType\"}}",
    ];

    for (i, &json_str) in malformed_json_cases.iter().enumerate() {
        let res_str = Tokenizer::from_str(json_str);
        assert!(
            res_str.is_err(),
            "Malformed JSON case {i} ('{json_str}') must return Err via from_str"
        );

        let res_bytes = Tokenizer::from_bytes(json_str.as_bytes());
        assert!(
            res_bytes.is_err(),
            "Malformed JSON case {i} ('{json_str}') must return Err via from_bytes"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Nonexistent and Invalid File Paths
// ---------------------------------------------------------------------------

#[test]
fn test_tokenizer_nonexistent_and_invalid_file_paths_fail_closed() {
    let invalid_paths = [
        "nonexistent_file_path_1234567890.json",
        "./deeply/nested/nonexistent/directory/tokenizer.json",
        "",
        ".",
        "..",
        "src",
        "tests",
        "Cargo.toml", // Valid file, but invalid tokenizer JSON schema
    ];

    for (i, &path) in invalid_paths.iter().enumerate() {
        let res = Tokenizer::from_file(path);
        assert!(
            res.is_err(),
            "Path case {i} ('{path}') must fail gracefully with Err"
        );

        // Also test from_config with invalid path
        let config = TokenizerConfig {
            tokenizer_path: path.to_string(),
            max_length: 1024,
        };
        let cfg_res = Tokenizer::from_config(config);
        assert!(
            cfg_res.is_err(),
            "Config path case {i} ('{path}') must fail gracefully with Err"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. Boundary Token IDs, Out-of-Vocab Lookups, and Inner Conversions
// ---------------------------------------------------------------------------

#[test]
fn test_tokenizer_vocab_boundary_queries_and_inner_conversion() {
    let tok = create_synthetic_tokenizer();

    // 1. Boundary token lookups
    assert_eq!(tok.token_to_id("hello"), Some(3));
    assert_eq!(tok.token_to_id("world"), Some(4));
    assert_eq!(tok.token_to_id("completely_unseen_token_xyz_999"), None);
    assert_eq!(tok.token_to_id(""), None);

    assert_eq!(tok.id_to_token(3), Some("hello".to_string()));
    assert_eq!(tok.id_to_token(4), Some("world".to_string()));
    assert_eq!(tok.id_to_token(u32::MAX), None);
    assert_eq!(tok.id_to_token(u32::MAX - 1), None);
    assert_eq!(tok.id_to_token(100_000), None);

    // 2. Vocab size
    let vocab_size = tok.vocab_size(true);
    assert_eq!(vocab_size, 6);

    // 3. Inner and IntoInner conversions
    let inner_ref = tok.inner();
    assert_eq!(inner_ref.get_vocab_size(true), vocab_size);

    let inner_owned = tok.clone().into_inner();
    assert_eq!(inner_owned.get_vocab_size(true), vocab_size);

    // 4. Debug formatting
    let debug_str = format!("{tok:?}");
    assert!(debug_str.contains("Tokenizer"));
    assert!(debug_str.contains("vocab_size: 6"));
}
