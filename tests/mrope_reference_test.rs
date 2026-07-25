use candle_core::{DType, Device, Tensor};

use qwen3tts::talker::primitives::{
    MultimodalRotaryEmbedding, RMSNorm, apply_multimodal_rotary_pos_emb,
};
use qwen3tts::talker::{CodePredictor, TalkerConfig, TalkerForConditionalGeneration, TalkerModel};

fn inv_freq_values(half_dim: usize, theta: f64) -> Vec<f32> {
    (0..half_dim)
        .map(|i| 1.0 / theta.powf(i as f64 / half_dim as f64) as f32)
        .collect()
}

fn expected_interleaved_axis(dim: usize, mrope_section: &[usize], head_dim: usize) -> usize {
    let half = head_dim / 2;
    let mut axis = 0;
    let axis_dim = dim % half;
    for axis_idx in 1..mrope_section.len() {
        let start = axis_idx;
        let end = mrope_section[axis_idx] * 3;
        if axis_dim >= start && axis_dim < end && (axis_dim - start) % 3 == 0 {
            axis = axis_idx;
            break;
        }
    }
    axis
}

fn expected_non_interleaved_axis(dim: usize, mrope_section: &[usize], head_dim: usize) -> usize {
    let mut offset = 0usize;
    let modality_num = mrope_section.len();
    for chunk_idx in 0..(modality_num * 2) {
        let axis = chunk_idx % modality_num;
        let section = mrope_section[axis];
        if dim < offset + section {
            return axis;
        }
        offset += section;
    }
    panic!("failed to map dim {dim} for head_dim {head_dim}");
}

#[test]
fn compute_position_ids_prefill_and_delta_match_expected_formula() {
    let device = Device::Cpu;
    let config = TalkerConfig::default();
    let rope = MultimodalRotaryEmbedding::new(&config, &device).unwrap();
    let talker = minimal_talker(&rope, &config, &device);

    let all_valid = Tensor::from_slice(&[1u32, 1, 1, 1, 1], (1, 5), &device).unwrap();
    let left_padded = Tensor::from_slice(&[0u32, 0, 1, 1, 1], (1, 5), &device).unwrap();
    let right_padded = Tensor::from_slice(&[1u32, 1, 1, 0, 0], (1, 5), &device).unwrap();
    let two_batch = Tensor::from_slice(
        &[
            1u32, 1, 1, 1, 1, // all valid
            0, 0, 1, 1, 1, // left-padded
            1, 1, 1, 0, 0, // right-padded
            0, 1, 1, 0, 1, // mixed
        ],
        (4, 5),
        &device,
    )
    .unwrap();

    let (all_positions, all_delta) = talker
        .compute_position_ids(&all_valid.to_dtype(DType::I64).unwrap())
        .unwrap();
    let (left_positions, left_delta) = talker
        .compute_position_ids(&left_padded.to_dtype(DType::I64).unwrap())
        .unwrap();
    let (right_positions, right_delta) = talker
        .compute_position_ids(&right_padded.to_dtype(DType::I64).unwrap())
        .unwrap();
    let (batch_positions, batch_delta) = talker
        .compute_position_ids(&two_batch.to_dtype(DType::I64).unwrap())
        .unwrap();

    assert_eq!(
        all_positions.to_vec3::<u32>().unwrap()[0][0],
        vec![0, 1, 2, 3, 4]
    );
    assert_eq!(
        left_positions.to_vec3::<u32>().unwrap()[0][0],
        vec![1, 1, 0, 1, 2]
    );
    assert_eq!(
        right_positions.to_vec3::<u32>().unwrap()[0][0],
        vec![0, 1, 2, 1, 1]
    );
    assert_eq!(all_delta.to_vec2::<u32>().unwrap(), vec![vec![0u32]]);
    assert_eq!(left_delta.to_vec2::<u32>().unwrap(), vec![vec![0u32]]);
    assert_eq!(right_delta.to_vec2::<u32>().unwrap(), vec![vec![0u32]]);
    assert_eq!(
        batch_positions.to_vec3::<u32>().unwrap(),
        vec![
            vec![
                vec![0, 1, 2, 3, 4],
                vec![1, 1, 0, 1, 2],
                vec![0, 1, 2, 1, 1],
                vec![1, 0, 1, 1, 2]
            ],
            vec![
                vec![0, 1, 2, 3, 4],
                vec![1, 1, 0, 1, 2],
                vec![0, 1, 2, 1, 1],
                vec![1, 0, 1, 1, 2]
            ],
            vec![
                vec![0, 1, 2, 3, 4],
                vec![1, 1, 0, 1, 2],
                vec![0, 1, 2, 1, 1],
                vec![1, 0, 1, 1, 2]
            ],
        ]
    );
    assert_eq!(
        batch_delta.to_vec2::<u32>().unwrap(),
        vec![vec![0u32], vec![0u32], vec![0u32], vec![0u32]]
    );

    assert!(all_positions.dims3().is_ok());
    assert_eq!(all_positions.dim(0).unwrap(), 3);
}

#[test]
fn cached_positions_follow_cache_position_plus_delta_formula() {
    let device = Device::Cpu;

    let rope_delta = Tensor::from_slice(&[3u32, 0u32], (2, 1), &device).unwrap();
    let positions =
        MultimodalRotaryEmbedding::cached_positions_from_delta(7, &rope_delta, 4, &device).unwrap();
    let positions = positions.to_vec2::<u32>().unwrap();

    assert_eq!(
        positions,
        vec![vec![10u32, 11, 12, 13], vec![7u32, 8, 9, 10]]
    );

    let wrong_delta = Tensor::from_slice(&[0u32, 2u32], (2, 2), &device).unwrap();
    assert!(
        MultimodalRotaryEmbedding::cached_positions_from_delta(1, &wrong_delta, 2, &device)
            .is_err()
    );
}

#[test]
fn interleaved_axis_boundary_selection_matches_formula() {
    let device = Device::Cpu;
    let mut config = TalkerConfig::default();
    config.head_dim = 128;
    config.mrope_section = vec![24, 20, 20];
    config.rope_interleaved = true;
    let rope = MultimodalRotaryEmbedding::new(&config, &device).unwrap();

    let position_ids = Tensor::from_slice(&[0u32, 1, 2], (3, 1, 1), &device).unwrap();
    let x = Tensor::zeros((1, 1, 1, 128), DType::F32, &device).unwrap();
    let (cos, sin) = rope.forward(&x, &position_ids).unwrap();
    let cos = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let sin = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let check_dims = [0usize, 1, 2, 57, 58, 59, 60, 63, 64, 65, 66, 127];
    let half = config.head_dim / 2;
    let freqs = inv_freq_values(half, config.rope_theta);

    for &dim in check_dims.iter() {
        let axis = expected_interleaved_axis(dim, &config.mrope_section, config.head_dim);
        let pos = axis as f32;
        let freq = freqs[dim % half];
        let expected_cos = (pos * freq).cos();
        let expected_sin = (pos * freq).sin();

        assert!((cos[dim] - expected_cos).abs() <= 1e-6);
        assert!((sin[dim] - expected_sin).abs() <= 1e-6);
    }
}

#[test]
fn non_interleaved_axis_repetition_matches_formula() {
    let device = Device::Cpu;
    let mut config = TalkerConfig::default();
    config.head_dim = 128;
    config.mrope_section = vec![24, 20, 20];
    config.rope_interleaved = false;
    let rope = MultimodalRotaryEmbedding::new(&config, &device).unwrap();

    let position_ids = Tensor::from_slice(&[0u32, 1, 2], (3, 1, 1), &device).unwrap();
    let x = Tensor::zeros((1, 1, 1, 128), DType::F32, &device).unwrap();
    let (cos, sin) = rope.forward(&x, &position_ids).unwrap();
    let cos = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let sin = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let check_dims = [0usize, 1, 2, 23, 24, 43, 44, 63, 64, 84, 108, 127];
    let half = config.head_dim / 2;
    let freqs = inv_freq_values(half, config.rope_theta);

    for &dim in check_dims.iter() {
        let axis = expected_non_interleaved_axis(dim, &config.mrope_section, config.head_dim);
        let pos = axis as f32;
        let freq = freqs[dim % half];
        let expected_cos = (pos * freq).cos();
        let expected_sin = (pos * freq).sin();

        assert!((cos[dim] - expected_cos).abs() <= 1e-6);
        assert!((sin[dim] - expected_sin).abs() <= 1e-6);
    }
}

#[test]
fn forward_single_position_matches_general_forward_with_equal_axes() {
    let device = Device::Cpu;
    let mut config = TalkerConfig::default();
    config.head_dim = 12;
    config.mrope_section = vec![2, 2, 2];
    let rope = MultimodalRotaryEmbedding::new(&config, &device).unwrap();

    let position_ids = Tensor::from_slice(
        &[4u32, 5, 7u32, 8, 4u32, 5, 7u32, 8, 4u32, 5, 7u32, 8],
        (3, 2, 2),
        &device,
    )
    .unwrap();

    let x = Tensor::zeros((2, 1, 2, 12), DType::F32, &device).unwrap();
    let (cos_full, sin_full) = rope.forward(&x, &position_ids).unwrap();

    let per_batch_positions = Tensor::from_slice(&[4u32, 7u32], (2, 1), &device).unwrap();
    let cached =
        MultimodalRotaryEmbedding::cached_positions_from_delta(0, &per_batch_positions, 2, &device)
            .unwrap();
    let (cos_fast, sin_fast) = rope.forward_single_position(&cached).unwrap();

    assert_tensors_close(&cos_full, &cos_fast, 1e-6);
    assert_tensors_close(&sin_full, &sin_fast, 1e-6);
}

#[test]
fn apply_multimodal_rotary_pos_emb_matches_manual_reference_values() {
    let device = Device::Cpu;
    let _config = TalkerConfig::default();
    let cos = Tensor::from_slice(
        &[
            0.6f32, 0.8f32, 0.6f32, 0.8f32, 0.2f32, 0.98f32, 0.2f32, 0.98f32,
        ],
        (1, 1, 1, 8),
        &device,
    )
    .unwrap();
    let sin = Tensor::from_slice(
        &[
            0.8f32, -0.6f32, 0.8f32, -0.6f32, 0.98f32, -0.2f32, 0.98f32, -0.2f32,
        ],
        (1, 1, 1, 8),
        &device,
    )
    .unwrap();

    let q = Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6., 7., 8.], (1, 1, 1, 8), &device).unwrap();
    let k = Tensor::from_slice(&[8f32, 7., 6., 5., 4., 3., 2., 1.], (1, 1, 1, 8), &device).unwrap();

    let (rot_q, rot_k) = apply_multimodal_rotary_pos_emb(&q, &k, &cos, &sin).unwrap();
    let ref_q = Tensor::from_slice(
        &[-3.4f32, 5.2, -3.8, 8.0, 1.98, 5.48, 4.34, 7.04],
        (1, 1, 1, 8),
        &device,
    )
    .unwrap();
    let ref_k = Tensor::from_slice(
        &[1.6f32, 7.4, 2.0, 4.6, 8.64, 1.54, 6.28, -0.02],
        (1, 1, 1, 8),
        &device,
    )
    .unwrap();

    assert_tensors_close(&rot_q, &ref_q, 1e-6);
    assert_tensors_close(&rot_k, &ref_k, 1e-6);
}

#[test]
fn invalid_mrope_inputs_fail_fast() {
    let device = Device::Cpu;

    let mut cfg = TalkerConfig::default();
    cfg.rope_theta = 0.0;
    assert!(MultimodalRotaryEmbedding::new(&cfg, &device).is_err());

    cfg = TalkerConfig::default();
    cfg.rope_theta = f64::NAN;
    assert!(MultimodalRotaryEmbedding::new(&cfg, &device).is_err());

    cfg = TalkerConfig::default();
    cfg.mrope_section = vec![24, 20];
    assert!(MultimodalRotaryEmbedding::new(&cfg, &device).is_err());

    cfg = TalkerConfig::default();
    cfg.mrope_section = vec![3, 3, 2];
    cfg.head_dim = 12;
    assert!(MultimodalRotaryEmbedding::new(&cfg, &device).is_err());

    cfg = TalkerConfig::default();
    let rope = MultimodalRotaryEmbedding::new(&cfg, &device).unwrap();
    let wrong_shape = Tensor::from_slice(&[1u32, 1, 2], (3, 1), &device).unwrap();
    assert!(
        rope.forward(
            &Tensor::zeros((1, 1, 1, 128), DType::F32, &device).unwrap(),
            &wrong_shape
        )
        .is_err()
    );
}

fn minimal_talker(
    rope: &MultimodalRotaryEmbedding,
    config: &TalkerConfig,
    device: &Device,
) -> TalkerForConditionalGeneration {
    let norm = RMSNorm::new(Tensor::ones((1,), DType::F32, device).unwrap(), 1e-6);
    let model = TalkerModel::new(Vec::new(), norm.clone());
    let dummy = Tensor::zeros((1,), DType::F32, device).unwrap();

    TalkerForConditionalGeneration {
        model,
        text_embedding: dummy.clone(),
        text_proj_fc1_w: dummy.clone(),
        text_proj_fc1_b: dummy.clone(),
        text_proj_fc2_w: dummy.clone(),
        text_proj_fc2_b: dummy.clone(),
        codec_embedding: dummy.clone(),
        codec_head: dummy.clone(),
        code_predictor: CodePredictor {
            codec_embeddings: Vec::new(),
            lm_heads: Vec::new(),
            layers: Vec::new(),
            norm,
            small_to_mtp_proj: None,
            config: config.code_predictor.clone(),
        },
        rope: rope.clone(),
        config: config.clone(),
    }
}

fn assert_tensors_close(a: &Tensor, b: &Tensor, tolerance: f32) {
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a.len(), b.len());
    for (idx, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() <= tolerance, "idx={idx} x={x} y={y}");
    }
}
