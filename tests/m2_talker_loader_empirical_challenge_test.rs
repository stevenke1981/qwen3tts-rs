//! Empirical Challenge Suite for Milestone M2:
//! - TalkerWeightLoader memory mapping with mixed BF16 and F32 datatypes.
//! - Odd-length / misaligned byte slice error handling.
//! - Extreme numerical edge cases, large Rayon parallel chunk verification, and concurrency stress testing.

use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use candle_core::Device;
use safetensors::tensor::{Dtype, View};

use qwen3tts::talker::weight_loader::{
    bf16_bytes_to_f32_vec, f32_bytes_to_f32_vec, TalkerWeightLoader,
};
use qwen3tts::weights::WeightLoader;
use qwen3tts::Error;

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

fn create_temp_safetensors_file(
    prefix: &str,
    tensors: Vec<(String, MockTensorView)>,
) -> PathBuf {
    let tmp_dir = std::env::temp_dir();
    let file_path = tmp_dir.join(format!(
        "qwen3tts_m2_challenge_{}_{}_{}.safetensors",
        prefix,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    safetensors::serialize_to_file(tensors, None, &file_path)
        .expect("Failed to serialize challenge safetensors file");
    file_path
}

// ===========================================================================
// CHALLENGE 1: Multi-Tensor Safetensors with Mixed BF16 and F32 Datatypes
// ===========================================================================

#[test]
fn challenge_talker_loader_mixed_bf16_f32_multi_tensor_mmap() {
    let device = Device::Cpu;

    // 1. Construct BF16 tensor (2D: 4x4 = 16 elements)
    let bf16_raw_floats = [
        1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25, 10.0, -10.0, 0.0, -0.0, 100.0,
        -100.0, 3.140625, -3.140625,
    ];
    let mut bf16_bytes = Vec::new();
    for &f in &bf16_raw_floats {
        let bits = (f.to_bits() >> 16) as u16;
        bf16_bytes.extend_from_slice(&bits.to_le_bytes());
    }
    let t_bf16 = MockTensorView {
        dtype: Dtype::BF16,
        shape: vec![4, 4],
        data: bf16_bytes,
    };

    // 2. Construct F32 tensor (1D: 8 elements)
    let f32_raw_floats = [
        0.125f32, 0.875, -42.5, 1337.0, 1e-4, -1e-4, 99999.0, -99999.0,
    ];
    let mut f32_bytes = Vec::new();
    for &f in &f32_raw_floats {
        f32_bytes.extend_from_slice(&f.to_le_bytes());
    }
    let t_f32 = MockTensorView {
        dtype: Dtype::F32,
        shape: vec![8],
        data: f32_bytes,
    };

    // 3. Construct 3D BF16 tensor (2x2x3 = 12 elements)
    let mut bf16_3d_bytes = Vec::new();
    let mut expected_3d = Vec::new();
    for i in 0..12 {
        let val = (i as f32) * 0.5;
        let bits = (val.to_bits() >> 16) as u16;
        bf16_3d_bytes.extend_from_slice(&bits.to_le_bytes());
        expected_3d.push(val);
    }
    let t_bf16_3d = MockTensorView {
        dtype: Dtype::BF16,
        shape: vec![2, 2, 3],
        data: bf16_3d_bytes,
    };

    // 4. Construct Large BF16 tensor (N = 20000 >= 16384 to trigger Rayon parallel path)
    let large_n = 20000;
    let mut large_bf16_bytes = Vec::with_capacity(large_n * 2);
    let mut large_expected_f32 = Vec::with_capacity(large_n);
    for i in 0..large_n {
        let bits = ((i as u16) ^ 0x4000) & 0xFFFE;
        large_bf16_bytes.extend_from_slice(&bits.to_le_bytes());
        large_expected_f32.push(f32::from_bits((bits as u32) << 16));
    }
    let t_large_bf16 = MockTensorView {
        dtype: Dtype::BF16,
        shape: vec![100, 200],
        data: large_bf16_bytes,
    };

    // 5. Construct Large F32 tensor (N = 25000 >= 16384 to trigger Rayon parallel path)
    let large_f32_n = 25000;
    let mut large_f32_bytes = Vec::with_capacity(large_f32_n * 4);
    let mut large_f32_expected = Vec::with_capacity(large_f32_n);
    for i in 0..large_f32_n {
        let val = (i as f32) * 1.25 - 5000.0;
        large_f32_bytes.extend_from_slice(&val.to_le_bytes());
        large_f32_expected.push(val);
    }
    let t_large_f32 = MockTensorView {
        dtype: Dtype::F32,
        shape: vec![50, 500],
        data: large_f32_bytes,
    };

    let path = create_temp_safetensors_file(
        "mixed_dtypes",
        vec![
            ("talker.model.text_embedding.weight".to_string(), t_bf16),
            ("talker.model.norm.weight".to_string(), t_f32),
            ("talker.text_projection.linear_fc1.weight".to_string(), t_bf16_3d),
            ("talker.model.layers.0.mlp.gate_proj.weight".to_string(), t_large_bf16),
            ("talker.model.layers.0.mlp.up_proj.weight".to_string(), t_large_f32),
        ],
    );

    // Test A: TalkerWeightLoader::from_safetensors (Zero-Copy Mmap)
    let loader = TalkerWeightLoader::from_safetensors(&path, &device)
        .expect("TalkerWeightLoader::from_safetensors failed on mixed BF16/F32");

    // Verify tensor 1 (BF16 2D)
    let t1 = loader.get("talker.model.text_embedding.weight").unwrap();
    assert_eq!(t1.dims(), &[4, 4]);
    let t1_flat = t1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for (i, (&actual, &expected)) in t1_flat.iter().zip(bf16_raw_floats.iter()).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "BF16 mismatch at index {i}: got {actual}, expected {expected}"
        );
    }

    // Verify tensor 2 (F32 1D)
    let t2 = loader.get("talker.model.norm.weight").unwrap();
    assert_eq!(t2.dims(), &[8]);
    assert_eq!(
        t2.to_vec1::<f32>().unwrap(),
        f32_raw_floats.to_vec()
    );

    // Verify tensor 3 (BF16 3D)
    let t3 = loader.get("talker.text_projection.linear_fc1.weight").unwrap();
    assert_eq!(t3.dims(), &[2, 2, 3]);
    let t3_flat = t3.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(t3_flat, expected_3d);

    // Verify tensor 4 (Large BF16 with Rayon parallelization)
    let t4 = loader.get("talker.model.layers.0.mlp.gate_proj.weight").unwrap();
    assert_eq!(t4.dims(), &[100, 200]);
    let t4_flat = t4.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(t4_flat.len(), large_n);
    for (i, (&actual, &expected)) in t4_flat.iter().zip(large_expected_f32.iter()).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "Large BF16 mismatch at index {i}"
        );
    }

    // Verify tensor 5 (Large F32 with Rayon parallelization)
    let t5 = loader.get("talker.model.layers.0.mlp.up_proj.weight").unwrap();
    assert_eq!(t5.dims(), &[50, 500]);
    let t5_flat = t5.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(t5_flat.len(), large_f32_n);
    assert_eq!(t5_flat, large_f32_expected);

    // Test B: TalkerWeightLoader::from_bytes (Zero-copy in-memory slice)
    let file_bytes = std::fs::read(&path).unwrap();
    let bytes_loader = TalkerWeightLoader::from_bytes(&file_bytes, &device)
        .expect("TalkerWeightLoader::from_bytes failed on mixed BF16/F32");
    assert_eq!(bytes_loader.get("talker.model.norm.weight").unwrap().dims(), &[8]);

    // Test C: WeightLoader::from_file on the same mixed file
    let wl = WeightLoader::from_file(&path, &device)
        .expect("WeightLoader::from_file failed on mixed BF16/F32");
    assert_eq!(wl.len(), 5);
    assert_eq!(wl.get("talker.model.norm.weight").unwrap().dims(), &[8]);

    let _ = std::fs::remove_file(path);
}

// ===========================================================================
// CHALLENGE 2: Odd-Length and Misaligned Byte Slice Error Rejection
// ===========================================================================

#[test]
fn challenge_odd_and_misaligned_byte_slices_rejected_with_error() {
    // 1. Test bf16_bytes_to_f32_vec with odd lengths: must return Err(Error::Weight(...))
    let odd_lengths = [1, 3, 5, 7, 9, 11, 15, 31, 63, 127, 255, 1023, 16383, 16385, 32767, 65535];
    for &len in &odd_lengths {
        let slice = vec![0x55u8; len];
        let res = bf16_bytes_to_f32_vec(&slice);
        assert!(
            res.is_err(),
            "bf16_bytes_to_f32_vec MUST return Err for odd byte length {len}"
        );
        match res.unwrap_err() {
            Error::Weight(msg) => {
                assert!(
                    msg.contains("Invalid BF16 tensor byte length"),
                    "Expected 'Invalid BF16 tensor byte length' in error message, got: {msg}"
                );
            }
            other => panic!("Expected Error::Weight, got {other:?}"),
        }
    }

    // 2. Test f32_bytes_to_f32_vec with non-multiple-of-4 lengths: must return Err(Error::Weight(...))
    let invalid_f32_lengths = [
        1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 1021, 1022, 1023, 16381, 16382,
        16383, 16385, 16386, 16387, 32765, 32766, 32767,
    ];
    for &len in &invalid_f32_lengths {
        let slice = vec![0xAAu8; len];
        let res = f32_bytes_to_f32_vec(&slice);
        assert!(
            res.is_err(),
            "f32_bytes_to_f32_vec MUST return Err for non-multiple-of-4 byte length {len}"
        );
        match res.unwrap_err() {
            Error::Weight(msg) => {
                assert!(
                    msg.contains("Invalid F32 tensor byte length"),
                    "Expected 'Invalid F32 tensor byte length' in error message, got: {msg}"
                );
            }
            other => panic!("Expected Error::Weight, got {other:?}"),
        }
    }

    // 3. Test TalkerWeightLoader::from_bytes with corrupted / odd-length raw header
    let device = Device::Cpu;
    let corrupted_raw_slices: &[&[u8]] = &[
        &[0u8],
        &[0u8, 1, 2],
        &[0u8, 1, 2, 3, 4, 5, 6, 7, 8],
        &[0xFF; 15],
        &[0x7B, 0x22, 0x7D],
    ];
    for slice in corrupted_raw_slices {
        let res = TalkerWeightLoader::from_bytes(slice, &device);
        assert!(
            res.is_err(),
            "TalkerWeightLoader::from_bytes must return error on corrupted bytes"
        );
    }
}

// ===========================================================================
// CHALLENGE 3: Unsupported Dtypes in Safetensors
// ===========================================================================

#[test]
fn challenge_unsupported_safetensors_dtypes_rejected() {
    let device = Device::Cpu;

    // Create a safetensors file containing an I32 tensor
    let t_i32 = MockTensorView {
        dtype: Dtype::I32,
        shape: vec![4],
        data: vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
    };

    let path = create_temp_safetensors_file(
        "unsupported_i32",
        vec![("talker.unsupported.weight".to_string(), t_i32)],
    );

    let res = TalkerWeightLoader::from_safetensors(&path, &device);
    assert!(
        res.is_err(),
        "TalkerWeightLoader must reject unsupported I32 dtype with error"
    );
    if let Err(Error::Weight(msg)) = res {
        assert!(
            msg.contains("Unsupported dtype"),
            "Error message must specify unsupported dtype, got: {msg}"
        );
    } else {
        panic!("Expected Error::Weight, got unexpected result");
    }

    let _ = std::fs::remove_file(path);
}

// ===========================================================================
// CHALLENGE 4: Extreme IEEE 754 Floating-Point Edge Cases
// ===========================================================================

#[test]
fn challenge_bf16_extreme_floating_point_values() {
    let edge_bit_patterns: &[u16] = &[
        0x0000, // +0.0
        0x8000, // -0.0
        0x7F80, // +inf
        0xFF80, // -inf
        0x0001, // min subnormal
        0x007F, // max subnormal
        0x0080, // min normal
        0x7F7F, // max normal
        0x8001, // min negative subnormal
        0x8080, // min negative normal
        0xFF7F, // max negative normal
        0x7FC0, // canonical quiet NaN
        0x7F81, // signaling NaN
        0xFFC0, // negative NaN
    ];

    let mut raw_bytes = Vec::new();
    for &bits in edge_bit_patterns {
        raw_bytes.extend_from_slice(&bits.to_le_bytes());
    }

    let decoded = bf16_bytes_to_f32_vec(&raw_bytes).expect("decoding edge cases failed");
    assert_eq!(decoded.len(), edge_bit_patterns.len());

    for (i, (&bits, &actual_f32)) in edge_bit_patterns.iter().zip(decoded.iter()).enumerate() {
        let expected_bits = (bits as u32) << 16;
        let actual_bits = actual_f32.to_bits();
        assert_eq!(
            actual_bits, expected_bits,
            "Edge case pattern mismatch at index {i}"
        );
    }
}

// ===========================================================================
// CHALLENGE 5: Concurrent Multi-Threaded Loader Access
// ===========================================================================

#[test]
fn challenge_talker_weight_loader_thread_safety_and_concurrency() {
    let device = Device::Cpu;
    let count = 1000;
    let mut data = Vec::with_capacity(count * 4);
    for i in 0..count {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let t_norm = MockTensorView {
        dtype: Dtype::F32,
        shape: vec![count],
        data,
    };

    let path = create_temp_safetensors_file(
        "concurrency_test",
        vec![("talker.model.norm.weight".to_string(), t_norm)],
    );

    let loader = Arc::new(
        TalkerWeightLoader::from_safetensors(&path, &device)
            .expect("TalkerWeightLoader failed to load"),
    );

    // Spawn 16 threads accessing get() concurrently
    let mut handles = Vec::new();
    for thread_id in 0..16 {
        let loader_clone = Arc::clone(&loader);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let tensor = loader_clone
                    .get("talker.model.norm.weight")
                    .expect("get in thread failed");
                assert_eq!(tensor.dims(), &[1000]);
            }
            thread_id
        }));
    }

    for h in handles {
        h.join().expect("thread join failed");
    }

    let _ = std::fs::remove_file(path);
}
