use candle_core::{Device, Tensor};
use qwen3tts::talker::primitives::apply_multimodal_rotary_pos_emb;
use qwen3tts::text_frontend::{SynthesisOptions, TokenParser};

const EXPECTED_DEFAULT_EOS: u16 = 2150;

// ===========================================================================
// CHALLENGE 1: M-RoPE Mathematical Correctness & Index 6 Exact Value Proof
// ===========================================================================

#[test]
#[allow(clippy::approx_constant)]
fn challenge_mrope_index_6_mathematical_precision() {
    let device = Device::Cpu;

    // Test vectors from tests/mrope_reference_test.rs
    let cos_data = [0.6f32, 0.8, 0.6, 0.8, 0.2, 0.98, 0.2, 0.98];
    let sin_data = [0.8f32, -0.6, 0.8, -0.6, 0.98, -0.2, 0.98, -0.2];
    let q_data = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let k_data = [8f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    // Mathematical verification of Index 6:
    // head_dim = 8, half = 4
    // k = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    // rot_k = [-4.0, -3.0, -2.0, -1.0, 8.0, 7.0, 6.0, 5.0]
    // For index 6:
    // k[6] = 2.0, cos[6] = 0.2 => 2.0 * 0.2 = 0.40
    // rot_k[6] = k[2] = 6.0, sin[6] = 0.98 => 6.0 * 0.98 = 5.88
    // Result = 0.40 + 5.88 = 6.28f32
    let term1 = k_data[6] * cos_data[6]; // 2.0 * 0.2 = 0.4
    let term2 = k_data[2] * sin_data[6]; // 6.0 * 0.98 = 5.88
    let exact_sum = term1 + term2; // 6.28

    assert_eq!(term1, 0.4f32);
    assert_eq!(term2, 5.88f32);
    assert_eq!(exact_sum, 6.28f32);
    assert_eq!(exact_sum.to_bits(), 6.28f32.to_bits());

    // Execute via Candle tensor primitives
    let cos = Tensor::from_slice(&cos_data, (1, 1, 1, 8), &device).unwrap();
    let sin = Tensor::from_slice(&sin_data, (1, 1, 1, 8), &device).unwrap();
    let q = Tensor::from_slice(&q_data, (1, 1, 1, 8), &device).unwrap();
    let k = Tensor::from_slice(&k_data, (1, 1, 1, 8), &device).unwrap();

    let (rot_q, rot_k) = apply_multimodal_rotary_pos_emb(&q, &k, &cos, &sin).unwrap();

    let rot_k_vec = rot_k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let rot_q_vec = rot_q.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Verify index 6 specifically
    assert_eq!(rot_k_vec[6], 6.28f32, "Rotary index 6 must equal exactly 6.28f32");
    assert_eq!(rot_k_vec[6].to_bits(), 6.28f32.to_bits());

    // Verify all q and k outputs
    let expected_q = [-3.4f32, 5.2, -3.8, 8.0, 1.98, 5.48, 4.34, 7.04];
    let expected_k = [1.6f32, 7.4, 2.0, 4.6, 8.64, 1.54, 6.28, -0.02];

    for i in 0..8 {
        assert!(
            (rot_q_vec[i] - expected_q[i]).abs() <= 1e-6,
            "q mismatch at index {i}: got {}, expected {}",
            rot_q_vec[i],
            expected_q[i]
        );
        assert!(
            (rot_k_vec[i] - expected_k[i]).abs() <= 1e-6,
            "k mismatch at index {i}: got {}, expected {}",
            rot_k_vec[i],
            expected_k[i]
        );
    }
}

// ===========================================================================
// CHALLENGE 2: TokenParser Binary Little-Endian Byte Stream Decoding & EOS
// ===========================================================================

#[test]
fn challenge_token_parser_little_endian_eos_decoding() {
    let options = SynthesisOptions::default();
    let parser = TokenParser::new(24000);

    // Verify default EOS id
    assert_eq!(parser.codec_eos_token_id(), EXPECTED_DEFAULT_EOS);
    assert_eq!(parser.codec_eos_token_id(), 2150);

    // 2150 in hex is 0x0866.
    // Little-endian byte order: low byte 0x66, high byte 0x08
    let eos_le_bytes = 2150u16.to_le_bytes();
    assert_eq!(eos_le_bytes, [0x66, 0x08]);

    // Construct a binary stream with 5 frames:
    // Frame 0: normal tokens (all 100)
    // Frame 1: normal tokens (all 200)
    // Frame 2: EOS token (2150) placed at codebook index 5 (middle)
    // Frame 3: trailing normal tokens (all 300)
    // Frame 4: trailing normal tokens (all 400)
    let mut byte_stream = Vec::new();
    byte_stream.extend_from_slice(&5u32.to_le_bytes()); // Declared 5 frames header

    // Frame 0
    for _ in 0..16 {
        byte_stream.extend_from_slice(&100u16.to_le_bytes());
    }
    // Frame 1
    for _ in 0..16 {
        byte_stream.extend_from_slice(&200u16.to_le_bytes());
    }
    // Frame 2: contains EOS at index 5
    for cb in 0..16 {
        let tok: u16 = if cb == 5 { 2150 } else { 250 };
        byte_stream.extend_from_slice(&tok.to_le_bytes());
    }
    // Frame 3
    for _ in 0..16 {
        byte_stream.extend_from_slice(&300u16.to_le_bytes());
    }
    // Frame 4
    for _ in 0..16 {
        byte_stream.extend_from_slice(&400u16.to_le_bytes());
    }

    let stream = parser
        .parse_bytes(&byte_stream, &options)
        .expect("binary decoding should succeed");

    assert_eq!(
        stream.num_frames(),
        2,
        "TokenParser must decode exactly 2 frames prior to the EOS frame"
    );
    assert_eq!(stream.frames[0], [100u16; 16]);
    assert_eq!(stream.frames[1], [200u16; 16]);
}

#[test]
fn challenge_token_parser_eos_at_all_16_codebook_positions() {
    let options = SynthesisOptions::default();
    let parser = TokenParser::new(24000);

    // Stress test: test putting EOS token at each of the 16 codebook positions in frame 1
    for eos_pos in 0..16 {
        let mut byte_stream = Vec::new();
        byte_stream.extend_from_slice(&3u32.to_le_bytes()); // 3 frames

        // Frame 0: valid
        for _ in 0..16 {
            byte_stream.extend_from_slice(&50u16.to_le_bytes());
        }

        // Frame 1: EOS placed at position `eos_pos`
        for cb in 0..16 {
            let tok: u16 = if cb == eos_pos { 2150 } else { 77u16 };
            byte_stream.extend_from_slice(&tok.to_le_bytes());
        }

        // Frame 2: trailing
        for _ in 0..16 {
            byte_stream.extend_from_slice(&99u16.to_le_bytes());
        }

        let stream = parser
            .parse_bytes(&byte_stream, &options)
            .unwrap_or_else(|e| panic!("Failed on eos_pos {}: {}", eos_pos, e));

        assert_eq!(
            stream.num_frames(),
            1,
            "EOS at codebook index {} must truncate and leave exactly 1 frame",
            eos_pos
        );
        assert_eq!(stream.frames[0], [50u16; 16]);
    }
}

#[test]
fn challenge_token_parser_malformed_byte_streams() {
    let options = SynthesisOptions::default();
    let parser = TokenParser::new(24000);

    // 1. Empty byte array
    assert!(parser.parse_bytes(&[], &options).is_err());

    // 2. Odd byte length (1, 3, 5, 7, 31, 33 bytes)
    for &len in &[1, 3, 5, 7, 15, 31, 33, 63, 65] {
        let odd_bytes = vec![0xABu8; len];
        let res = parser.parse_bytes(&odd_bytes, &options);
        assert!(res.is_err(), "Odd byte length {} must be rejected", len);
    }

    // 3. Huge declared frame count with truncated payload
    let mut huge_decl = Vec::new();
    huge_decl.extend_from_slice(&1_000_000u32.to_le_bytes());
    huge_decl.extend_from_slice(&[0u8; 32]); // only 1 frame provided
    assert!(parser.parse_bytes(&huge_decl, &options).is_err());

    // 4. Raw bytes without 4-byte header (32 bytes = 1 raw frame)
    let mut raw_frame = Vec::new();
    for i in 0..16 {
        raw_frame.extend_from_slice(&(i as u16 * 10).to_le_bytes());
    }
    let raw_stream = parser
        .parse_bytes(&raw_frame, &options)
        .expect("32-byte raw frame should succeed");
    assert_eq!(raw_stream.num_frames(), 1);
    assert_eq!(raw_stream.frames[0][0], 0);
    assert_eq!(raw_stream.frames[0][1], 10);
}

#[test]
fn challenge_token_parser_custom_eos_byte_decoding() {
    let options = SynthesisOptions::default();
    let custom_eos: u16 = 0xABCD; // 43981
    let parser = TokenParser::with_eos(24000, custom_eos);

    let mut byte_stream = Vec::new();
    byte_stream.extend_from_slice(&4u32.to_le_bytes());

    // Frame 0: contains 2150 (default EOS) -> should NOT truncate because custom EOS is 0xABCD
    for _ in 0..16 {
        byte_stream.extend_from_slice(&2150u16.to_le_bytes());
    }
    // Frame 1: normal
    for _ in 0..16 {
        byte_stream.extend_from_slice(&111u16.to_le_bytes());
    }
    // Frame 2: contains custom_eos (0xABCD) -> SHOULD truncate
    for cb in 0..16 {
        let tok = if cb == 12 { custom_eos } else { 222u16 };
        byte_stream.extend_from_slice(&tok.to_le_bytes());
    }
    // Frame 3: trailing
    for _ in 0..16 {
        byte_stream.extend_from_slice(&333u16.to_le_bytes());
    }

    let stream = parser
        .parse_bytes(&byte_stream, &options)
        .expect("custom EOS byte stream parse should succeed");

    assert_eq!(
        stream.num_frames(),
        2,
        "Custom parser must not stop on 2150, but stop on custom EOS 0xABCD at frame 2"
    );
    assert_eq!(stream.frames[0], [2150u16; 16]);
    assert_eq!(stream.frames[1], [111u16; 16]);
}
