//! Integration tests for memory-mapped weight loaders and Rayon BF16 transcoding.

use std::borrow::Cow;
use std::path::PathBuf;

use candle_core::Device;
use safetensors::tensor::{Dtype, View};

use qwen3tts::talker::weight_loader::{bf16_bytes_to_f32_vec, bf16_to_f32, f32_bytes_to_f32_vec, TalkerWeightLoader};
use qwen3tts::weights::WeightLoader;

#[derive(Debug, Clone)]
struct SimpleTensorView {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for SimpleTensorView {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn create_temp_safetensors_file(name_prefix: &str, tensors: Vec<(String, SimpleTensorView)>) -> PathBuf {
    let tmp_dir = std::env::temp_dir();
    let file_path = tmp_dir.join(format!(
        "qwen3tts_{}_{}_{}.safetensors",
        name_prefix,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    safetensors::serialize_to_file(tensors, None, &file_path).expect("failed to write temp safetensors file");
    file_path
}

#[test]
fn test_weight_loader_from_file_mmap_bf16_and_f32() {
    let device = Device::Cpu;

    // Create a 2x3 BF16 tensor and a 4-element F32 tensor
    // BF16 values: [[1.0, -1.0, 2.0], [0.5, -0.5, 0.0]]
    // In BF16 LE:
    // 1.0  = 0x3F80 -> [0x80, 0x3F]
    // -1.0 = 0xBF80 -> [0x80, 0xBF]
    // 2.0  = 0x4000 -> [0x00, 0x40]
    // 0.5  = 0x3F00 -> [0x00, 0x3F]
    // -0.5 = 0xBF00 -> [0x00, 0xBF]
    // 0.0  = 0x0000 -> [0x00, 0x00]
    let bf16_data = vec![
        0x80, 0x3F, 0x80, 0xBF, 0x00, 0x40,
        0x00, 0x3F, 0x00, 0xBF, 0x00, 0x00,
    ];
    let bf16_view = SimpleTensorView {
        dtype: Dtype::BF16,
        shape: vec![2, 3],
        data: bf16_data,
    };

    let f32_vals = vec![10.0f32, -20.5f32, 30.25f32, 0.125f32];
    let mut f32_data = Vec::new();
    for &v in &f32_vals {
        f32_data.extend_from_slice(&v.to_le_bytes());
    }
    let f32_view = SimpleTensorView {
        dtype: Dtype::F32,
        shape: vec![4],
        data: f32_data,
    };

    let path = create_temp_safetensors_file(
        "weight_loader_test",
        vec![
            ("pre_conv.weight".to_string(), bf16_view),
            ("pre_conv.bias".to_string(), f32_view),
        ],
    );

    let loader = WeightLoader::from_file(&path, &device).expect("WeightLoader::from_file failed");
    assert_eq!(loader.len(), 2);
    assert!(loader.has("pre_conv.weight"));
    assert!(loader.has("pre_conv.bias"));

    let w = loader.conv1d_weight("pre_conv").expect("get pre_conv.weight");
    assert_eq!(w.dims(), &[2, 3]);
    let w_vec = w.to_vec2::<f32>().expect("to_vec2");
    assert_eq!(w_vec, vec![vec![1.0, -1.0, 2.0], vec![0.5, -0.5, 0.0]]);

    let b = loader.conv1d_bias("pre_conv").expect("get pre_conv.bias").expect("bias present");
    assert_eq!(b.dims(), &[4]);
    assert_eq!(b.to_vec1::<f32>().expect("to_vec1"), f32_vals);

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_weight_loader_from_dir_mmap() {
    let device = Device::Cpu;
    let tmp_dir = std::env::temp_dir().join(format!(
        "qwen3tts_dir_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let f1_path = tmp_dir.join("part1.safetensors");
    let t1 = SimpleTensorView {
        dtype: Dtype::F32,
        shape: vec![2],
        data: vec![0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40], // [1.0, 2.0]
    };
    safetensors::serialize_to_file(vec![("tensor_a".to_string(), t1)], None, &f1_path).unwrap();

    let f2_path = tmp_dir.join("part2.safetensors");
    let t2 = SimpleTensorView {
        dtype: Dtype::F32,
        shape: vec![2],
        data: vec![0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x40], // [3.0, 4.0]
    };
    safetensors::serialize_to_file(vec![("tensor_b".to_string(), t2)], None, &f2_path).unwrap();

    let loader = WeightLoader::from_dir(&tmp_dir, &device).expect("WeightLoader::from_dir failed");
    assert_eq!(loader.len(), 2);
    assert!(loader.has("tensor_a"));
    assert!(loader.has("tensor_b"));

    assert_eq!(loader.get("tensor_a").unwrap().to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(loader.get("tensor_b").unwrap().to_vec1::<f32>().unwrap(), vec![3.0, 4.0]);

    let _ = std::fs::remove_dir_all(tmp_dir);
}

#[test]
fn test_talker_weight_loader_from_safetensors_mmap() {
    let device = Device::Cpu;

    // Create mock talker weights
    let emb_data = vec![
        0x80, 0x3F, 0x00, 0x40, // 1.0, 2.0 in BF16
        0x00, 0xBF, 0x00, 0x00, // -0.5, 0.0 in BF16
    ];
    let emb_view = SimpleTensorView {
        dtype: Dtype::BF16,
        shape: vec![2, 2],
        data: emb_data,
    };

    let norm_vals = vec![1.0f32, 1.0f32];
    let mut norm_data = Vec::new();
    for &v in &norm_vals {
        norm_data.extend_from_slice(&v.to_le_bytes());
    }
    let norm_view = SimpleTensorView {
        dtype: Dtype::F32,
        shape: vec![2],
        data: norm_data,
    };

    let path = create_temp_safetensors_file(
        "talker_mmap_test",
        vec![
            ("talker.model.text_embedding.weight".to_string(), emb_view),
            ("talker.model.norm.weight".to_string(), norm_view),
        ],
    );

    let loader = TalkerWeightLoader::from_safetensors(&path, &device).expect("TalkerWeightLoader::from_safetensors failed");
    let emb = loader.get("talker.model.text_embedding.weight").expect("get text_embedding");
    assert_eq!(emb.dims(), &[2, 2]);
    assert_eq!(emb.to_vec2::<f32>().unwrap(), vec![vec![1.0, 2.0], vec![-0.5, 0.0]]);

    let norm = loader.get("talker.model.norm.weight").expect("get norm");
    assert_eq!(norm.dims(), &[2]);
    assert_eq!(norm.to_vec1::<f32>().unwrap(), vec![1.0, 1.0]);

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_bf16_transcoding_ieee754_exactness_all_ranges() {
    // Test all 65536 possible 16-bit BF16 patterns for exact bit-level IEEE 754 fidelity
    let mut all_bf16_bytes = Vec::with_capacity(65536 * 2);
    let mut expected_floats = Vec::with_capacity(65536);

    for bits in 0..=0xFFFFu16 {
        all_bf16_bytes.push((bits & 0xFF) as u8);
        all_bf16_bytes.push(((bits >> 8) & 0xFF) as u8);

        let ref_f32 = f32::from_bits((bits as u32) << 16);
        expected_floats.push(ref_f32);
    }

    // 65536 >= 16384, so this tests the Rayon parallel chunked conversion path
    let converted = bf16_bytes_to_f32_vec(&all_bf16_bytes).expect("bf16_bytes_to_f32_vec failed");
    assert_eq!(converted.len(), 65536);

    for (i, (&actual, &expected)) in converted.iter().zip(expected_floats.iter()).enumerate() {
        let actual_bits = actual.to_bits();
        let expected_bits = expected.to_bits();
        assert_eq!(
            actual_bits, expected_bits,
            "Mismatch at index {i} (BF16 0x{:04X}): actual 0x{actual_bits:08X} ({actual}) vs expected 0x{expected_bits:08X} ({expected})",
            i as u16
        );
    }
}

#[test]
fn test_bf16_individual_bit_shift_helper() {
    let test_cases: &[(u16, f32)] = &[
        (0x0000, 0.0),
        (0x8000, -0.0),
        (0x3F80, 1.0),
        (0xBF80, -1.0),
        (0x4000, 2.0),
        (0xC000, -2.0),
        (0x3F00, 0.5),
        (0x7F80, f32::INFINITY),
        (0xFF80, f32::NEG_INFINITY),
    ];

    for &(bits, expected) in test_cases {
        let actual = bf16_to_f32(bits);
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "bits 0x{bits:04X}: actual {actual} vs expected {expected}"
        );
    }
}

#[test]
fn test_f32_fast_copy_exactness_large_rayon() {
    let count = 30000;
    let mut bytes = Vec::with_capacity(count * 4);
    let mut expected = Vec::with_capacity(count);

    for i in 0..count {
        let val = (i as f32) * std::f32::consts::PI - 12345.678;
        bytes.extend_from_slice(&val.to_le_bytes());
        expected.push(val);
    }

    let result = f32_bytes_to_f32_vec(&bytes).expect("f32_bytes_to_f32_vec failed");
    assert_eq!(result.len(), count);
    for (i, (&actual, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual.to_bits(),
            exp.to_bits(),
            "Mismatch at index {i}: actual {actual} vs expected {exp}"
        );
    }
}

#[test]
fn test_invalid_byte_lengths() {
    // Odd length for BF16
    let odd_bytes = vec![1, 2, 3];
    assert!(bf16_bytes_to_f32_vec(&odd_bytes).is_err());

    // Non-multiple of 4 for F32
    let bad_f32_bytes = vec![1, 2, 3, 4, 5];
    assert!(f32_bytes_to_f32_vec(&bad_f32_bytes).is_err());

    // Empty slice is valid and produces empty vector
    assert_eq!(bf16_bytes_to_f32_vec(&[]).unwrap(), Vec::<f32>::new());
    assert_eq!(f32_bytes_to_f32_vec(&[]).unwrap(), Vec::<f32>::new());
}
