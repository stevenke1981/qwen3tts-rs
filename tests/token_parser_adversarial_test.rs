use qwen3tts::text_frontend::{SynthesisOptions, TokenParser};

const DEFAULT_EOS: u16 = 2150;

#[test]
fn test_empty_frame_inputs() {
    let parser = TokenParser::new(24000);
    let options = SynthesisOptions::default();

    // 1. Empty slice of Vec<u16>
    let empty_codes: Vec<Vec<u16>> = vec![];
    let stream = parser
        .parse(&empty_codes, &options)
        .expect("empty input should succeed");
    assert_eq!(stream.num_frames(), 0);
    assert_eq!(stream.frames.len(), 0);
    assert_eq!(stream.duration_sec(), 0.0);

    // 2. Empty binary with 0 declared frames
    let mut zero_header = Vec::new();
    zero_header.extend_from_slice(&0u32.to_le_bytes());
    let stream_bin = parser
        .parse_bytes(&zero_header, &options)
        .expect("0 frames binary should succeed");
    assert_eq!(stream_bin.num_frames(), 0);
}

#[test]
fn test_single_frame_with_eos() {
    let parser = TokenParser::new(24000);
    let options = SynthesisOptions::default();

    // Case A: All 16 tokens in the single frame are EOS
    let codes_all_eos = vec![vec![DEFAULT_EOS; 16]];
    let stream_a = parser
        .parse(&codes_all_eos, &options)
        .expect("single frame all EOS should succeed");
    assert_eq!(
        stream_a.num_frames(),
        0,
        "single frame with all EOS must yield 0 frames"
    );

    // Case B: EOS token at index 0
    let mut frame_eos_0 = vec![100u16; 16];
    frame_eos_0[0] = DEFAULT_EOS;
    let stream_b = parser
        .parse(&[frame_eos_0], &options)
        .expect("single frame EOS at index 0 should succeed");
    assert_eq!(
        stream_b.num_frames(),
        0,
        "single frame with EOS at index 0 must yield 0 frames"
    );

    // Case C: EOS token at index 15 (last codebook)
    let mut frame_eos_15 = vec![100u16; 16];
    frame_eos_15[15] = DEFAULT_EOS;
    let stream_c = parser
        .parse(&[frame_eos_15], &options)
        .expect("single frame EOS at index 15 should succeed");
    assert_eq!(
        stream_c.num_frames(),
        0,
        "single frame with EOS at index 15 must yield 0 frames"
    );

    // Case D: EOS token at index 7 (middle codebook)
    let mut frame_eos_7 = vec![100u16; 16];
    frame_eos_7[7] = DEFAULT_EOS;
    let stream_d = parser
        .parse(&[frame_eos_7], &options)
        .expect("single frame EOS at index 7 should succeed");
    assert_eq!(
        stream_d.num_frames(),
        0,
        "single frame with EOS at index 7 must yield 0 frames"
    );
}

#[test]
fn test_multi_frame_with_eos_at_middle() {
    let parser = TokenParser::new(24000);
    let options = SynthesisOptions::default();

    let num_total = 10;
    let eos_index = 4;

    let mut codes: Vec<Vec<u16>> = (0..num_total)
        .map(|frame_idx| (0..16).map(|cb_idx| (frame_idx * 100 + cb_idx) as u16).collect())
        .collect();

    // Insert EOS in frame 4 at codebook index 8
    codes[eos_index][8] = DEFAULT_EOS;

    let stream = parser
        .parse(&codes, &options)
        .expect("multi-frame parse with EOS at middle should succeed");

    assert_eq!(
        stream.num_frames(),
        eos_index,
        "should truncate exactly at frame index {}",
        eos_index
    );

    // Verify frames 0..4 match original content exactly
    for (i, frame) in stream.frames.iter().enumerate() {
        for (cb_idx, &tok) in frame.iter().enumerate() {
            let expected_tok = (i * 100 + cb_idx) as u16;
            assert_eq!(
                tok, expected_tok,
                "frame {} cb {} mismatch: got {}, expected {}",
                i, cb_idx, tok, expected_tok
            );
        }
    }
}

#[test]
fn test_multi_frame_with_no_eos() {
    let parser = TokenParser::new(24000);
    let options = SynthesisOptions::default();

    let num_frames = 50;
    let codes: Vec<Vec<u16>> = (0..num_frames)
        .map(|frame_idx| {
            (0..16)
                .map(|cb_idx| ((frame_idx * 31 + cb_idx * 7) % 2048) as u16)
                .map(|tok| if tok == DEFAULT_EOS { 2149 } else { tok })
                .collect()
        })
        .collect();

    let stream = parser
        .parse(&codes, &options)
        .expect("multi-frame parse with no EOS should succeed");

    assert_eq!(
        stream.num_frames(),
        num_frames,
        "all frames must be retained when no EOS is present"
    );

    for (i, frame) in stream.frames.iter().enumerate() {
        for (cb_idx, &tok) in frame.iter().enumerate() {
            assert_eq!(tok, codes[i][cb_idx]);
        }
    }
}

#[test]
fn test_non_default_eos_tokens() {
    let test_eos_candidates = [0u16, 100u16, 2047u16, 32767u16, 65535u16];

    for &custom_eos in &test_eos_candidates {
        let parser = TokenParser::with_eos(24000, custom_eos);
        assert_eq!(parser.codec_eos_token_id(), custom_eos);

        let options = SynthesisOptions::default();

        // Frame 0: normal
        // Frame 1: contains DEFAULT_EOS (2150) -> should NOT truncate
        // Frame 2: normal
        // Frame 3: contains custom_eos -> SHOULD truncate
        // Frame 4: normal
        let mut codes = vec![
            vec![10u16; 16],
            vec![20u16; 16],
            vec![30u16; 16],
            vec![40u16; 16],
            vec![50u16; 16],
        ];

        // Put DEFAULT_EOS in frame 1
        if custom_eos != DEFAULT_EOS {
            codes[1][5] = DEFAULT_EOS;
        } else {
            // If custom_eos happens to be 2150, replace frame 1 token with safe value
            codes[1][5] = 999;
        }

        // Put custom_eos in frame 3
        codes[3][11] = custom_eos;

        let stream = parser
            .parse(&codes, &options)
            .unwrap_or_else(|e| panic!("parse failed for custom_eos {}: {}", custom_eos, e));

        assert_eq!(
            stream.num_frames(),
            3,
            "custom_eos {} must truncate at frame index 3, but got {} frames",
            custom_eos,
            stream.num_frames()
        );
        assert_eq!(stream.frames[0][0], 10);
        if custom_eos != DEFAULT_EOS {
            assert_eq!(stream.frames[1][5], DEFAULT_EOS);
        }
        assert_eq!(stream.frames[2][0], 30);
    }
}

#[test]
fn test_binary_parsing_adversarial_suite() {
    let options = SynthesisOptions::default();

    // 1. Binary parse with default EOS
    let parser = TokenParser::new(24000);

    // Empty buffer must fail
    assert!(parser.parse_bytes(&[], &options).is_err());
    assert!(parser.parse_bytes(&[0, 1, 2], &options).is_err());

    // 4 frames binary data
    let mut data = Vec::new();
    data.extend_from_slice(&4u32.to_le_bytes());
    // Frame 0: normal
    for _ in 0..16 {
        data.extend_from_slice(&500u16.to_le_bytes());
    }
    // Frame 1: EOS (2150) at position 3
    for i in 0..16 {
        let tok = if i == 3 { DEFAULT_EOS } else { 600u16 };
        data.extend_from_slice(&tok.to_le_bytes());
    }
    // Frame 2: trailing
    for _ in 0..16 {
        data.extend_from_slice(&700u16.to_le_bytes());
    }
    // Frame 3: trailing
    for _ in 0..16 {
        data.extend_from_slice(&800u16.to_le_bytes());
    }

    let stream = parser.parse_bytes(&data, &options).expect("parse_bytes should succeed");
    assert_eq!(stream.num_frames(), 1, "binary parser must truncate at frame 1 containing EOS");
    assert_eq!(stream.frames[0][0], 500);

    // 2. Binary parse with custom EOS
    let custom_eos = 1234u16;
    let custom_parser = TokenParser::with_eos(24000, custom_eos);

    let mut custom_data = Vec::new();
    custom_data.extend_from_slice(&3u32.to_le_bytes());
    // Frame 0: contains 2150 (should NOT stop)
    for i in 0..16 {
        let tok = if i == 0 { DEFAULT_EOS } else { 100u16 };
        custom_data.extend_from_slice(&tok.to_le_bytes());
    }
    // Frame 1: contains 1234 (SHOULD stop)
    for i in 0..16 {
        let tok = if i == 15 { custom_eos } else { 200u16 };
        custom_data.extend_from_slice(&tok.to_le_bytes());
    }
    // Frame 2: trailing
    for _ in 0..16 {
        custom_data.extend_from_slice(&300u16.to_le_bytes());
    }

    let custom_stream = custom_parser
        .parse_bytes(&custom_data, &options)
        .expect("custom binary parse should succeed");
    assert_eq!(
        custom_stream.num_frames(),
        1,
        "custom parser must not stop on 2150, but stop on 1234"
    );
    assert_eq!(custom_stream.frames[0][0], DEFAULT_EOS);

    // 3. Static parse_binary uses default EOS (2150)
    let static_stream = TokenParser::parse_binary(&data).expect("parse_binary should succeed");
    assert_eq!(static_stream.num_frames(), 1);
    assert_eq!(static_stream.frames[0][0], 500);
}

#[test]
fn test_error_handling_and_validation() {
    let parser = TokenParser::new(24000);
    let options = SynthesisOptions::default();

    // 1. Frame with 15 tokens (invalid codebook count)
    let invalid_short_frame = vec![vec![100u16; 15]];
    let err_short = parser.parse(&invalid_short_frame, &options);
    assert!(err_short.is_err(), "15 tokens should return error");

    // 2. Frame with 17 tokens (invalid codebook count)
    let invalid_long_frame = vec![vec![100u16; 17]];
    let err_long = parser.parse(&invalid_long_frame, &options);
    assert!(err_long.is_err(), "17 tokens should return error");

    // 3. Truncated binary data
    let mut incomplete_data = Vec::new();
    incomplete_data.extend_from_slice(&2u32.to_le_bytes()); // declares 2 frames (64 bytes payload)
    incomplete_data.extend_from_slice(&[0u8; 16]); // only 16 bytes provided
    let err_bin = parser.parse_bytes(&incomplete_data, &options);
    assert!(err_bin.is_err(), "incomplete binary data should return error");
}
