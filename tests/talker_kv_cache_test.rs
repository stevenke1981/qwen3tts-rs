use candle_core::{DType, Device, Tensor};
use qwen3tts::alignment_stage_dump::StageDumpObserver;
use qwen3tts::talker::decoder_layer::TalkerDecoderLayer;
use qwen3tts::talker::model::TalkerModel;
use qwen3tts::talker::primitives::{
    MultimodalRotaryEmbedding, RMSNorm, SwiGLUMLP, create_causal_mask,
};
use qwen3tts::talker::talker::compute_position_ids_from_attention_mask;
use qwen3tts::talker::talker_attention::TalkerAttention;

fn layer(d: &Device, offset: f32) -> TalkerDecoderLayer {
    let matrix = |scale: f32| {
        let values: Vec<f32> = (0..16).map(|i| offset + scale * (i as f32 + 1.0)).collect();
        Tensor::from_slice(&values, (4, 4), d).unwrap()
    };
    let norm = Tensor::from_slice(&[1.0f32, 1.1, 0.9, 1.2], 4, d).unwrap();
    TalkerDecoderLayer::new(
        RMSNorm::new(norm.clone(), 1e-6),
        TalkerAttention::new(
            matrix(0.01),
            matrix(0.02),
            matrix(0.03),
            matrix(0.04),
            norm.clone(),
            norm,
            1,
            1,
            4,
            1e-6,
        ),
        RMSNorm::new(
            Tensor::from_slice(&[1.2f32, 0.8, 1.1, 0.95], 4, d).unwrap(),
            1e-6,
        ),
        SwiGLUMLP::new(matrix(0.005), matrix(0.006), matrix(0.007)),
    )
}

struct NoopObserver;
impl StageDumpObserver for NoopObserver {}

fn model(d: &Device) -> TalkerModel {
    TalkerModel::new(
        vec![layer(d, 0.1), layer(d, -0.2)],
        RMSNorm::new(
            Tensor::from_slice(&[1.0f32, 0.9, 1.1, 1.05], 4, d).unwrap(),
            1e-6,
        ),
    )
}

fn rope(seq: usize, d: &Device) -> (Tensor, Tensor) {
    let cos = Tensor::ones((1, 1, seq, 4), DType::F32, d).unwrap();
    let sin = Tensor::zeros((1, 1, seq, 4), DType::F32, d).unwrap();
    (cos, sin)
}

fn rope_from_position_ids(position_ids: &Tensor, d: &Device) -> (Tensor, Tensor) {
    let positions = position_ids.to_vec3::<u32>().unwrap();
    let sequence = &positions[0][0];
    let mut cos = Vec::with_capacity(sequence.len() * 4);
    let mut sin = Vec::with_capacity(sequence.len() * 4);
    for &position in sequence {
        let angle = position as f32;
        let half_angle = angle * 0.5;
        cos.extend([angle.cos(), half_angle.cos(), angle.cos(), half_angle.cos()]);
        sin.extend([angle.sin(), half_angle.sin(), angle.sin(), half_angle.sin()]);
    }
    (
        Tensor::from_slice(&cos, (1, 1, sequence.len(), 4), d).unwrap(),
        Tensor::from_slice(&sin, (1, 1, sequence.len(), 4), d).unwrap(),
    )
}

fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
    a.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .zip(b.flatten_all().unwrap().to_vec1::<f32>().unwrap())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (mut dot, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (x, y) in av.iter().zip(bv.iter()) {
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    dot / (aa.sqrt() * bb.sqrt()).max(1e-12)
}

fn cache_snapshot(caches: &[Option<(Tensor, Tensor)>]) -> Vec<Option<(Vec<f32>, Vec<f32>)>> {
    caches
        .iter()
        .map(|entry| {
            entry.as_ref().map(|(k, v)| {
                (
                    k.to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                    v.to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                )
            })
        })
        .collect()
}

fn assert_snapshot(caches: &[Option<(Tensor, Tensor)>], snap: &[Option<(Vec<f32>, Vec<f32>)>]) {
    for (entry, old) in caches.iter().zip(snap) {
        match (entry, old) {
            (None, None) => {}
            (Some((k, v)), Some((ok, ov))) => {
                assert_eq!(
                    k.to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                    *ok
                );
                assert_eq!(
                    v.to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                    *ov
                );
            }
            _ => panic!("cache presence changed on rejected call"),
        }
    }
}

#[test]
fn two_layer_cache_matches_full_recompute_and_rejects_malformed_transactionally() {
    let d = Device::Cpu;
    let m = model(&d);
    let x = Tensor::from_slice(
        &[
            0.2f32, -0.4, 0.7, 1.1, 1.3, 0.5, -0.8, 0.9, -0.6, 0.3, 1.2, -1.0,
        ],
        (1, 3, 4),
        &d,
    )
    .unwrap();
    let (cos3, sin3) = rope(3, &d);
    let (cos2, sin2) = rope(2, &d);
    let (cos1, sin1) = rope(1, &d);

    let mut full_cache = vec![None, None];
    let full = m.forward(&x, &cos3, &sin3, None, &mut full_cache).unwrap();
    let mut full_prefill_cache = vec![None, None];
    let _ = m
        .forward(
            &x.narrow(1, 0, 2).unwrap(),
            &cos2,
            &sin2,
            None,
            &mut full_prefill_cache,
        )
        .unwrap();
    let mut prefill_cache = vec![None, None];
    let _ = m
        .forward(
            &x.narrow(1, 0, 2).unwrap(),
            &cos2,
            &sin2,
            None,
            &mut prefill_cache,
        )
        .unwrap();
    let prefill_snapshot = prefill_cache.clone();
    let cached = m
        .forward(
            &x.narrow(1, 2, 1).unwrap(),
            &cos1,
            &sin1,
            None,
            &mut prefill_cache,
        )
        .unwrap();
    let full_last = full.narrow(1, 2, 1).unwrap();
    assert!(cosine(&full_last, &cached) > 0.999);
    assert!(
        max_abs(&full_last, &cached) < 1e-3,
        "hidden max_abs={}",
        max_abs(&full_last, &cached)
    );
    for (idx, (fc, cc)) in full_cache.iter().zip(prefill_cache.iter()).enumerate() {
        let (fpk, fpv) = full_prefill_cache[idx].as_ref().unwrap();
        let (fk, fv) = fc.as_ref().unwrap();
        let (ck, cv) = cc.as_ref().unwrap();
        let seq = prefill_snapshot[0].as_ref().unwrap().0.dim(2).unwrap();
        let (pk, pv) = prefill_snapshot[idx].as_ref().unwrap();
        let fk_prefix = fpk.narrow(2, 0, seq).unwrap();
        let fv_prefix = fpv.narrow(2, 0, seq).unwrap();
        assert_eq!(
            fk_prefix.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            pk.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            fv_prefix.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            pv.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            ck.narrow(2, 0, seq)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            pk.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            cv.narrow(2, 0, seq)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            pv.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        let fk_last = fk.narrow(2, seq, 1).unwrap();
        let fv_last = fv.narrow(2, seq, 1).unwrap();
        let ck_last = ck.narrow(2, seq, 1).unwrap();
        let cv_last = cv.narrow(2, seq, 1).unwrap();
        assert!(cosine(&fk_last, &ck_last) > 0.999);
        assert!(cosine(&fv_last, &cv_last) > 0.999);
        assert!(max_abs(&fk_last, &ck_last) < 1e-5);
        assert!(max_abs(&fv_last, &cv_last) < 1e-5);
    }

    let mut observer_cache = prefill_snapshot.clone();
    let mut observer = NoopObserver;
    let observed = m
        .forward_with_observer(
            &x.narrow(1, 2, 1).unwrap(),
            &cos1,
            &sin1,
            None,
            &mut observer_cache,
            "step",
            &mut observer,
        )
        .unwrap();
    assert!(cosine(&observed, &cached) > 0.999);
    assert!(max_abs(&observed, &cached) < 1e-5);

    let valid = prefill_cache.clone();
    let bad_cases = [
        vec![None],
        vec![
            Some((
                Tensor::zeros((1, 1, 2, 3), DType::F32, &d).unwrap(),
                Tensor::zeros((1, 1, 2, 3), DType::F32, &d).unwrap(),
            )),
            Some(valid[1].as_ref().unwrap().clone()),
        ],
        vec![valid[0].clone(), None],
        vec![
            Some((
                Tensor::zeros((1, 1, 1, 4), DType::F32, &d).unwrap(),
                Tensor::zeros((1, 1, 1, 4), DType::F32, &d).unwrap(),
            )),
            Some((
                Tensor::zeros((1, 1, 2, 4), DType::F32, &d).unwrap(),
                Tensor::zeros((1, 1, 2, 4), DType::F32, &d).unwrap(),
            )),
        ],
        vec![
            Some((
                Tensor::zeros((1, 1, 2, 4), DType::F16, &d).unwrap(),
                Tensor::zeros((1, 1, 2, 4), DType::F32, &d).unwrap(),
            )),
            valid[1].clone(),
        ],
    ];
    for mut bad in bad_cases {
        let snap = cache_snapshot(&bad);
        assert!(
            m.forward(&x.narrow(1, 2, 1).unwrap(), &cos1, &sin1, None, &mut bad)
                .is_err()
        );
        assert_snapshot(&bad, &snap);
        let mut obs = NoopObserver;
        assert!(
            m.forward_with_observer(
                &x.narrow(1, 2, 1).unwrap(),
                &cos1,
                &sin1,
                None,
                &mut bad,
                "step",
                &mut obs
            )
            .is_err()
        );
        assert_snapshot(&bad, &snap);
    }
}

#[test]
fn cached_positions_from_delta_rejects_off_by_one_mutation() {
    let d = Device::Cpu;
    let m = model(&d);
    let x = Tensor::from_slice(
        &[
            0.2f32, -0.4, 0.7, 1.1, 1.3, 0.5, -0.8, 0.9, -0.6, 0.3, 1.2, -1.0,
        ],
        (1, 3, 4),
        &d,
    )
    .unwrap();
    let mask = Tensor::from_slice(&[0u32, 1], (1, 2), &d).unwrap();
    let (prefill_positions, delta) = compute_position_ids_from_attention_mask(&mask).unwrap();
    assert_eq!(
        prefill_positions.to_vec3::<u32>().unwrap()[0][0],
        vec![1, 0]
    );
    assert_eq!(delta.to_vec2::<u32>().unwrap(), vec![vec![1]]);
    let cached_position =
        MultimodalRotaryEmbedding::cached_positions_from_delta(2, &delta, 1, &d).unwrap();
    assert_eq!(cached_position.to_vec2::<u32>().unwrap(), vec![vec![3]]);

    let cached_position_3d = cached_position
        .unsqueeze(0)
        .unwrap()
        .expand((3, 1, 1))
        .unwrap();
    let full_positions = Tensor::cat(&[&prefill_positions, &cached_position_3d], 2).unwrap();
    let (full_cos, full_sin) = rope_from_position_ids(&full_positions, &d);
    let (prefill_cos, prefill_sin) = rope_from_position_ids(&prefill_positions, &d);
    let (cached_cos, cached_sin) = rope_from_position_ids(&cached_position_3d, &d);
    let mut full_cache = vec![None, None];
    let full = m
        .forward(
            &x,
            &full_cos,
            &full_sin,
            Some(&create_causal_mask(3, &d).unwrap()),
            &mut full_cache,
        )
        .unwrap();
    let mut cached_cache = vec![None, None];
    m.forward(
        &x.narrow(1, 0, 2).unwrap(),
        &prefill_cos,
        &prefill_sin,
        Some(&create_causal_mask(2, &d).unwrap()),
        &mut cached_cache,
    )
    .unwrap();
    let prefill_snapshot = cached_cache.clone();
    let cached = m
        .forward(
            &x.narrow(1, 2, 1).unwrap(),
            &cached_cos,
            &cached_sin,
            None,
            &mut cached_cache,
        )
        .unwrap();
    let full_last = full.narrow(1, 2, 1).unwrap();
    assert!(cosine(&full_last, &cached) >= 0.999);
    assert!(max_abs(&full_last, &cached) <= 1e-3);
    for (layer, (full_entry, cached_entry)) in full_cache.iter().zip(&cached_cache).enumerate() {
        let (full_k, full_v) = full_entry.as_ref().unwrap();
        let (cached_k, cached_v) = cached_entry.as_ref().unwrap();
        let (prefill_k, prefill_v) = prefill_snapshot[layer].as_ref().unwrap();
        assert_eq!(
            cached_k
                .narrow(2, 0, 2)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            prefill_k.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            cached_v
                .narrow(2, 0, 2)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            prefill_v.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert!(
            max_abs(
                &full_k.narrow(2, 2, 1).unwrap(),
                &cached_k.narrow(2, 2, 1).unwrap()
            ) <= 1e-5
        );
        assert!(
            max_abs(
                &full_v.narrow(2, 2, 1).unwrap(),
                &cached_v.narrow(2, 2, 1).unwrap()
            ) <= 1e-5
        );
    }

    let mutated_delta = Tensor::from_slice(&[0u32], (1, 1), &d).unwrap();
    let mutated_position =
        MultimodalRotaryEmbedding::cached_positions_from_delta(2, &mutated_delta, 1, &d).unwrap();
    assert_eq!(mutated_position.to_vec2::<u32>().unwrap(), vec![vec![2]]);
    let mutated_position_3d = mutated_position
        .unsqueeze(0)
        .unwrap()
        .expand((3, 1, 1))
        .unwrap();
    let (mutated_cos, mutated_sin) = rope_from_position_ids(&mutated_position_3d, &d);
    let mut mutated_cache = prefill_snapshot;
    let mutated = m
        .forward(
            &x.narrow(1, 2, 1).unwrap(),
            &mutated_cos,
            &mutated_sin,
            None,
            &mut mutated_cache,
        )
        .unwrap();
    assert!(
        max_abs(&cached, &mutated) > 1e-7,
        "off-by-one cached M-RoPE position must change the model output"
    );
}
