//! Empirical Challenger M2 Stress Test Suite
//!
//! Stress-tests:
//! 1. Memory-mapped weight loading on corrupted, empty, truncated, invalid-offset safetensors files.
//! 2. Zero-sized tensors and large tensors (N > 1,000,000).
//! 3. BF16 and F32 parallel Rayon transcoding vs. sequential bit-exact reference across all boundary sizes and special float values.
//! 4. Concurrent multi-threaded reads from Mmap-backed loaders.

use std::borrow::Cow;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use candle_core::Device;
use safetensors::tensor::{Dtype, View};

use qwen3tts::talker::weight_loader::{
    bf16_bytes_to_f32_vec, bf16_to_f32, f32_bytes_to_f32_vec, TalkerWeightLoader,
};
use qwen3tts::weights::WeightLoader;

#[derive(Debug, Clone)]
struct MockTensorView {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for MockTensorView {
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

fn temp_file_path(name: &str) -> PathBuf {
    let tmp_dir = std::env::temp_dir();
    tmp_dir.join(format!(
        "qwen3tts_challenger_{}_{}_{}.safetensors",
        name,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

// ============================================================================
// 1. Corrupted & Empty Safetensors Tests
// ============================================================================

#[test]
fn challenge_empty_file_handling() {
    let device = Device::Cpu;
    let path = temp_file_path("empty_file");

    // Create a 0-byte file
    File::create(&path).expect("create empty file");

    // Both loaders must return Err without panicking
    let res_weight = WeightLoader::from_file(&path, &device);
    assert!(
        res_weight.is_err(),
        "WeightLoader::from_file should fail gracefully on empty file"
    );

    let res_talker = TalkerWeightLoader::from_safetensors(&path, &device);
    assert!(
        res_talker.is_err(),
        "TalkerWeightLoader::from_safetensors should fail gracefully on empty file"
    );

    let _ = std::fs::remove_file(path);
}

#[test]
fn challenge_empty_bytes_handling() {
    let device = Device::Cpu;
    let empty_bytes: &[u8] = &[];

    let res_weight = WeightLoader::from_bytes(empty_bytes, &device);
    assert!(
        res_weight.is_err(),
        "WeightLoader::from_bytes must return Err on 0 bytes"
    );

    let res_talker = TalkerWeightLoader::from_bytes(empty_bytes, &device);
    assert!(
        res_talker.is_err(),
        "TalkerWeightLoader::from_bytes must return Err on 0 bytes"
    );
}

#[test]
fn challenge_truncated_and_corrupted_safetensors_headers() {
    let device = Device::Cpu;

    let corrupted_scenarios: &[(&str, &[u8])] = &[
        // Incomplete 8-byte header size
        ("incomplete_u64_header", &[0x01, 0x02, 0x03]),
        // Header size says 100 bytes, but file only has 10 bytes
        ("header_size_overflows_file", &[100, 0, 0, 0, 0, 0, 0, 0, b'{', b'}']),
        // Header JSON is malformed
        (
            "invalid_json_header",
            &[
                10, 0, 0, 0, 0, 0, 0, 0, b'{', b'n', b'o', b't', b'_', b'j', b's', b'o', b'n',
                b'!',
            ],
        ),
        // Header JSON is valid JSON but not a valid safetensors metadata schema
        (
            "invalid_schema_header",
            &[
                14, 0, 0, 0, 0, 0, 0, 0, b'[', b'1', b',', b'2', b',', b'3', b',', b'4', b',',
                b'5', b',', b'6', b',', b'7', b']',
            ],
        ),
        // Header points to data offsets out of file bounds
        ("data_offsets_out_of_bounds", {
            b"\x44\x00\x00\x00\x00\x00\x00\x00{\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[1000,1004]}}"
        }),
        // Tensor data length does not match shape * dtype size
        ("mismatched_tensor_data_length", {
            b"\x43\x00\x00\x00\x00\x00\x00\x00{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,8]}}12345678"
        }),
    ];

    for (name, bytes) in corrupted_scenarios {
        let path = temp_file_path(name);
        let mut f = File::create(&path).expect("create corrupted file");
        f.write_all(bytes).expect("write corrupted bytes");
        drop(f);

        // Test from_file
        let w_res = WeightLoader::from_file(&path, &device);
        assert!(
            w_res.is_err(),
            "WeightLoader::from_file should fail for scenario {name}"
        );

        let t_res = TalkerWeightLoader::from_safetensors(&path, &device);
        assert!(
            t_res.is_err(),
            "TalkerWeightLoader::from_safetensors should fail for scenario {name}"
        );

        // Test from_bytes
        let wb_res = WeightLoader::from_bytes(bytes, &device);
        assert!(
            wb_res.is_err(),
            "WeightLoader::from_bytes should fail for scenario {name}"
        );

        let tb_res = TalkerWeightLoader::from_bytes(bytes, &device);
        assert!(
            tb_res.is_err(),
            "TalkerWeightLoader::from_bytes should fail for scenario {name}"
        );

        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn challenge_nonexistent_file_and_empty_dir() {
    let device = Device::Cpu;
    let non_existent = PathBuf::from("D:/qwen3tts-rs/does_not_exist_12345.safetensors");
    assert!(WeightLoader::from_file(&non_existent, &device).is_err());
    assert!(TalkerWeightLoader::from_safetensors(&non_existent, &device).is_err());

    let empty_dir = std::env::temp_dir().join(format!("qwen3tts_empty_dir_{}", std::process::id()));
    std::fs::create_dir_all(&empty_dir).expect("create empty dir");

    let dir_loader =
        WeightLoader::from_dir(&empty_dir, &device).expect("empty dir should succeed with 0 tensors");
    assert_eq!(dir_loader.len(), 0);
    assert!(dir_loader.is_empty());

    let _ = std::fs::remove_dir_all(empty_dir);
}

// ============================================================================
// 2. Zero-Sized and Very Large Tensors (N > 1,000,000)
// ============================================================================

#[test]
fn challenge_zero_sized_tensors() {
    let device = Device::Cpu;

    // Zero-sized tensors: shape [0], shape [0, 512], shape [16, 0]
    let t_zero_1 = MockTensorView {
        dtype: Dtype::F32,
        shape: vec![0],
        data: vec![],
    };
    let t_zero_2 = MockTensorView {
        dtype: Dtype::BF16,
        shape: vec![0, 512],
        data: vec![],
    };
    let t_zero_3 = MockTensorView {
        dtype: Dtype::BF16,
        shape: vec![16, 0],
        data: vec![],
    };

    let path = temp_file_path("zero_sized");
    safetensors::serialize_to_file(
        vec![
            ("zero_f32".to_string(), t_zero_1),
            ("zero_bf16_2d".to_string(), t_zero_2),
            ("zero_bf16_matrix".to_string(), t_zero_3),
        ],
        None,
        &path,
    )
    .expect("serialize zero-sized tensors");

    // Test WeightLoader
    let loader =
        WeightLoader::from_file(&path, &device).expect("WeightLoader should load zero-sized tensors");
    assert_eq!(loader.len(), 3);
    assert_eq!(loader.get("zero_f32").unwrap().dims(), &[0]);
    assert_eq!(loader.get("zero_bf16_2d").unwrap().dims(), &[0, 512]);
    assert_eq!(loader.get("zero_bf16_matrix").unwrap().dims(), &[16, 0]);

    // Test TalkerWeightLoader
    let talker_loader = TalkerWeightLoader::from_safetensors(&path, &device)
        .expect("TalkerWeightLoader should load zero-sized tensors");
    assert_eq!(talker_loader.get("zero_f32").unwrap().dims(), &[0]);
    assert_eq!(talker_loader.get("zero_bf16_2d").unwrap().dims(), &[0, 512]);

    let _ = std::fs::remove_file(path);
}

#[test]
fn challenge_mega_tensor_mmap_and_transcoding() {
    let device = Device::Cpu;

    // N = 1,048,576 elements (1024 x 1024), scaling far beyond 65,536
    let n_elements = 1048576;
    let mut bf16_data = Vec::with_capacity(n_elements * 2);
    let mut expected_f32 = Vec::with_capacity(n_elements);

    for i in 0..n_elements {
        let f_val = ((i % 1000) as f32) * 0.125f32 - 50.0f32;
        let bits_32 = f_val.to_bits();
        let bf16_bits = (bits_32 >> 16) as u16;
        let recon_f32 = f32::from_bits((bf16_bits as u32) << 16);

        bf16_data.extend_from_slice(&bf16_bits.to_le_bytes());
        expected_f32.push(recon_f32);
    }

    let large_view = MockTensorView {
        dtype: Dtype::BF16,
        shape: vec![1024, 1024],
        data: bf16_data,
    };

    let path = temp_file_path("mega_tensor");
    safetensors::serialize_to_file(
        vec![("mega_weight".to_string(), large_view)],
        None,
        &path,
    )
    .expect("serialize mega tensor");

    // Load via WeightLoader
    let loader =
        WeightLoader::from_file(&path, &device).expect("load mega tensor in WeightLoader");
    let tensor = loader.get("mega_weight").expect("get mega_weight");
    assert_eq!(tensor.dims(), &[1024, 1024]);
    let actual_flat = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(actual_flat.len(), n_elements);

    // Verify bit-exact equality across all 1,048,576 elements
    for (i, (&actual, &expected)) in actual_flat.iter().zip(expected_f32.iter()).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "Mega tensor element mismatch at index {i}: actual {actual} vs expected {expected}"
        );
    }

    // Load via TalkerWeightLoader
    let talker_loader = TalkerWeightLoader::from_safetensors(&path, &device)
        .expect("load mega tensor in TalkerWeightLoader");
    let talker_tensor = talker_loader.get("mega_weight").expect("get mega_weight");
    let talker_flat = talker_tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(talker_flat.len(), n_elements);
    for (i, (&actual, &expected)) in talker_flat.iter().zip(expected_f32.iter()).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "Talker mega tensor element mismatch at index {i}: actual {actual} vs expected {expected}"
        );
    }

    let _ = std::fs::remove_file(path);
}

// ============================================================================
// 3. Parallel Rayon Transcoding vs Sequential Reference Oracle
// ============================================================================

/// Ground-truth sequential reference implementation for BF16 to F32
fn sequential_bf16_to_f32(data: &[u8]) -> Result<Vec<f32>, String> {
    if !data.len().is_multiple_of(2) {
        return Err("invalid length".to_string());
    }
    let n = data.len() / 2;
    let mut out = Vec::with_capacity(n);
    for chunk in data.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
        out.push(f32::from_bits(bits << 16));
    }
    Ok(out)
}

/// Ground-truth sequential reference implementation for F32 byte copy
fn sequential_f32_from_le_bytes(data: &[u8]) -> Result<Vec<f32>, String> {
    if !data.len().is_multiple_of(4) {
        return Err("invalid length".to_string());
    }
    let n = data.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in data.chunks_exact(4) {
        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(f);
    }
    Ok(out)
}

#[test]
fn challenge_individual_helper_vs_sequential() {
    for bits in 0..=0xFFFFu16 {
        let bytes = bits.to_le_bytes();
        let from_helper = bf16_to_f32(bits);
        let from_seq = sequential_bf16_to_f32(&bytes).unwrap();
        assert_eq!(from_helper.to_bits(), from_seq[0].to_bits());
    }
}

#[test]
fn challenge_rayon_bf16_transcoding_all_boundary_sizes() {
    let boundary_sizes: &[usize] = &[
        0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 1023, 1024, 1025, 4095, 4096, 4097, 8191, 8192,
        8193, 16383, 16384, 16385, 32767, 32768, 32769, 65535, 65536, 65537, 100000, 200000,
    ];

    for &size in boundary_sizes {
        let mut raw_bytes = Vec::with_capacity(size * 2);
        for i in 0..size {
            let u = (i * 37 + 0x1234) as u16;
            raw_bytes.extend_from_slice(&u.to_le_bytes());
        }

        let ref_result = sequential_bf16_to_f32(&raw_bytes).unwrap();
        let rayon_result = bf16_bytes_to_f32_vec(&raw_bytes).unwrap();

        assert_eq!(
            rayon_result.len(),
            ref_result.len(),
            "Length mismatch for size {size}"
        );

        for (idx, (&actual, &expected)) in rayon_result.iter().zip(ref_result.iter()).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "Bit mismatch at size {size}, index {idx}: 0x{:08X} vs 0x{:08X}",
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }
}

#[test]
fn challenge_rayon_f32_copy_all_boundary_sizes() {
    let boundary_sizes: &[usize] = &[
        0, 1, 2, 3, 15, 16, 17, 4095, 4096, 4097, 16383, 16384, 16385, 32768, 65536, 100000,
    ];

    for &size in boundary_sizes {
        let mut raw_bytes = Vec::with_capacity(size * 4);
        for i in 0..size {
            let val = (i as f32) * 1.5 - 777.0;
            raw_bytes.extend_from_slice(&val.to_le_bytes());
        }

        let ref_result = sequential_f32_from_le_bytes(&raw_bytes).unwrap();
        let rayon_result = f32_bytes_to_f32_vec(&raw_bytes).unwrap();

        assert_eq!(rayon_result.len(), ref_result.len());
        for (idx, (&actual, &expected)) in rayon_result.iter().zip(ref_result.iter()).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "F32 bit mismatch at size {size}, index {idx}"
            );
        }
    }
}

#[test]
fn challenge_bf16_special_float_values_parallel() {
    let special_u16s: &[u16] = &[
        0x0000, // +0.0
        0x8000, // -0.0
        0x0001, // min positive subnormal
        0x007F, // max positive subnormal
        0x8001, // min negative subnormal
        0x807F, // max negative subnormal
        0x0080, // min positive normal (2^-126)
        0x8080, // min negative normal (-2^-126)
        0x7F7F, // max positive normal (~3.3895314e38)
        0xFF7F, // max negative normal (~-3.3895314e38)
        0x7F80, // +infinity
        0xFF80, // -infinity
        0x7FC0, // canonical quiet NaN
        0xFFC0, // negative quiet NaN
        0x7F81, // signaling NaN with payload 1
        0x7FFF, // quiet NaN with all 1s payload
        0xFF81, // negative signaling NaN
        0xFFFF, // negative NaN with all 1s payload
        0x3F80, // 1.0
        0xBF80, // -1.0
        0x4000, // 2.0
        0xC000, // -2.0
    ];

    let reps = 1000;
    let mut large_special_bytes = Vec::with_capacity(special_u16s.len() * reps * 2);
    for _ in 0..reps {
        for &val in special_u16s {
            large_special_bytes.extend_from_slice(&val.to_le_bytes());
        }
    }

    let total_elements = special_u16s.len() * reps;
    assert!(total_elements >= 16384, "Must exceed Rayon parallel threshold");

    let parallel_res =
        bf16_bytes_to_f32_vec(&large_special_bytes).expect("parallel conversion");
    let seq_res =
        sequential_bf16_to_f32(&large_special_bytes).expect("sequential conversion");

    assert_eq!(parallel_res.len(), total_elements);

    for (i, (&actual, &expected)) in parallel_res.iter().zip(seq_res.iter()).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "Special float bit mismatch at index {i}: actual 0x{:08X} vs expected 0x{:08X}",
            actual.to_bits(),
            expected.to_bits()
        );
    }
}

// ============================================================================
// 4. Concurrent Access Stress Test
// ============================================================================

#[test]
fn challenge_concurrent_loader_access() {
    let device = Device::Cpu;

    let view = MockTensorView {
        dtype: Dtype::F32,
        shape: vec![4],
        data: vec![0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64], // [1.0, 2.0, 3.0, 4.0]
    };

    let path = temp_file_path("concurrent");
    safetensors::serialize_to_file(vec![("shared_weight".to_string(), view)], None, &path)
        .expect("serialize file for concurrent test");

    let loader = Arc::new(WeightLoader::from_file(&path, &device).expect("load for concurrent test"));

    let mut handles = Vec::new();
    for thread_id in 0..8 {
        let l = Arc::clone(&loader);
        handles.push(std::thread::spawn(move || {
            for _ in 0..100 {
                let t = l.get("shared_weight").expect("get shared_weight in thread");
                let v = t.to_vec1::<f32>().expect("to_vec1");
                assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0], "Thread {thread_id} read corruption");
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join failed");
    }

    let _ = std::fs::remove_file(path);
}
