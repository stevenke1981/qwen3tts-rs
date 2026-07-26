use candle_core::{DType, Device, Tensor};
use qwen3tts::StageDumpObserver;
use qwen3tts::talker::decoder_layer::StandardDecoderLayer;
use qwen3tts::talker::primitives::{RMSNorm, SwiGLUMLP, embedding_lookup, linear};
use qwen3tts::talker::talker_attention::StandardAttention;
use qwen3tts::talker::{CodePredictor, CodePredictorConfig};

type CacheEntry = Option<(Tensor, Tensor)>;
type CacheSnapshot = Vec<CacheEntry>;

struct FailOnFinalHidden;

impl StageDumpObserver for FailOnFinalHidden {
    fn wants_capture(&self) -> bool {
        true
    }

    fn on_stage(&mut self, name: &str, _tensor: &Tensor, _layout: &str) -> candle_core::Result<()> {
        if name == "code-predictor-hidden-prefill-frame0-final" {
            return Err(candle_core::Error::Msg(
                "injected final observer failure".into(),
            ));
        }
        Ok(())
    }
}

fn predictor() -> CodePredictor {
    let device = Device::Cpu;
    let c = CodePredictorConfig {
        hidden_size: 8,
        intermediate_size: 16,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        num_hidden_layers: 2,
        vocab_size: 32,
        num_code_groups: 16,
        max_position_embeddings: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        hidden_act: "silu".into(),
        attention_bias: false,
        attention_dropout: 0.0,
        layer_types: vec!["full_attention".into(); 2],
    };
    let w = |m: usize, n: usize, seed: f32| {
        let data = (0..m * n)
            .map(|i| {
                let row = (i / n) as f32 + 1.0;
                let column = (i % n) as f32 + 1.0;
                seed * 0.01
                    + row * 0.001
                    + column * 0.0007
                    + (row * column * 0.37 + seed).sin() * 0.05
            })
            .collect::<Vec<_>>();
        Tensor::from_slice(&data, (m, n), &device).unwrap()
    };
    let o = || Tensor::ones(8, DType::F32, &device).unwrap();
    let layer = |seed: f32| {
        StandardDecoderLayer::new(
            RMSNorm::new(o(), c.rms_norm_eps),
            StandardAttention::new(
                w(8, 8, seed),
                w(4, 8, seed + 0.01),
                w(4, 8, seed + 0.02),
                w(8, 8, seed),
                Tensor::ones(4, DType::F32, &device).unwrap(),
                Tensor::ones(4, DType::F32, &device).unwrap(),
                2,
                1,
                4,
                c.rms_norm_eps,
            ),
            RMSNorm::new(o(), c.rms_norm_eps),
            SwiGLUMLP::new(w(16, 8, seed), w(16, 8, seed + 0.01), w(8, 16, seed)),
        )
    };
    CodePredictor {
        codec_embeddings: (0..15).map(|i| w(32, 8, 0.1 + i as f32 * 0.37)).collect(),
        lm_heads: (0..15).map(|i| w(32, 8, 0.2 + i as f32 * 0.41)).collect(),
        layers: vec![layer(0.01), layer(0.02)],
        norm: RMSNorm::new(o(), c.rms_norm_eps),
        small_to_mtp_proj: None,
        config: c,
    }
}

fn one_hot_row(vocab: usize, hidden: usize, row: usize) -> Tensor {
    let mut data = vec![0.0f32; vocab * hidden];
    for value in &mut data[row * hidden..(row + 1) * hidden] {
        *value = 1.0;
    }
    Tensor::from_slice(&data, (vocab, hidden), &Device::Cpu).unwrap()
}

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> f32 {
    (lhs - rhs)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

fn assert_bit_identical(lhs: &Tensor, rhs: &Tensor, context: &str) {
    assert_eq!(lhs.dims(), rhs.dims(), "{context} shape mismatch");
    let lhs = lhs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let rhs = rhs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        lhs.iter()
            .zip(&rhs)
            .all(|(left, right)| left.to_bits() == right.to_bits()),
        "{context} is not bit-identical"
    );
}

fn private_trace(
    predictor: &CodePredictor,
    position_override: Option<(usize, u32)>,
) -> (Vec<Tensor>, Vec<CacheSnapshot>) {
    let device = Device::Cpu;
    let talker_hidden = Tensor::from_slice(
        &[0.11f32, -0.23, 0.37, -0.41, 0.53, -0.67, 0.79, -0.83],
        (1, 1, 8),
        &device,
    )
    .unwrap();
    let codebook_0 = Tensor::from_slice(
        &[-0.19f32, 0.29, -0.31, 0.43, -0.47, 0.59, -0.61, 0.73],
        (1, 1, 8),
        &device,
    )
    .unwrap();
    let prefill = Tensor::cat(&[talker_hidden, codebook_0], 1).unwrap();
    let mut inputs = vec![prefill.clone()];
    let mut cache = vec![None, None];
    let hidden = predictor
        .forward_prefix_for_test(&prefill, &[0, 1], &mut cache)
        .unwrap();
    let last_hidden = hidden.narrow(1, 1, 1).unwrap();
    let mut logits = vec![
        linear(&last_hidden, &predictor.lm_heads[0])
            .unwrap()
            .squeeze(1)
            .unwrap(),
    ];
    let mut snapshots = vec![cache.clone()];
    let mut next = logits[0]
        .argmax(1)
        .unwrap()
        .reshape(())
        .unwrap()
        .to_scalar::<u32>()
        .unwrap();

    for group in 1..15 {
        let ids = Tensor::from_slice(&[next], (1, 1), &device).unwrap();
        let input = embedding_lookup(&predictor.codec_embeddings[group - 1], &ids).unwrap();
        inputs.push(input.clone());
        let position = position_override
            .filter(|(target, _)| *target == group)
            .map_or((group + 1) as u32, |(_, position)| position);
        let hidden = predictor
            .forward_prefix_for_test(&input, &[position], &mut cache)
            .unwrap();
        let step_logits = linear(&hidden, &predictor.lm_heads[group])
            .unwrap()
            .squeeze(1)
            .unwrap();
        next = step_logits
            .argmax(1)
            .unwrap()
            .reshape(())
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        logits.push(step_logits);
        snapshots.push(cache.clone());
    }

    (logits, snapshots)
}

#[test]
fn frame_prefill_and_fourteen_steps_grow_each_cache_to_sixteen() {
    let p = predictor();
    let d = Device::Cpu;
    let a = Tensor::ones((1, 1, 8), DType::F32, &d).unwrap();
    let b = Tensor::zeros((1, 1, 8), DType::F32, &d).unwrap();
    let mut cache = vec![None, None];
    let (codes, updated) = p.generate(&a, &b, &mut cache, &d).unwrap();
    assert_eq!(codes.dims(), &[1, 15]);
    assert_eq!(updated.len(), 2);
    for entry in updated {
        let (k, v) = entry.unwrap();
        assert_eq!(k.dims(), &[1, 1, 16, 4]);
        assert_eq!(v.dims(), &[1, 1, 16, 4]);
    }
}

#[test]
fn malformed_cache_shape_dtype_and_sequence_fail_without_mutation() {
    let p = predictor();
    let d = Device::Cpu;
    let x = Tensor::zeros((1, 1, 8), DType::F32, &d).unwrap();
    let bad_k = Tensor::zeros((1, 1, 1, 3), DType::F32, &d).unwrap();
    let bad_v = Tensor::zeros((1, 1, 1, 4), DType::F32, &d).unwrap();
    let mut shape = vec![Some((bad_k, bad_v)), None];
    assert!(p.generate(&x, &x, &mut shape, &d).is_err());
    assert!(shape[0].is_some() && shape[1].is_none());

    let mut too_long = vec![
        Some((
            Tensor::zeros((1, 1, 32, 4), DType::F32, &d).unwrap(),
            Tensor::zeros((1, 1, 32, 4), DType::F32, &d).unwrap(),
        )),
        Some((
            Tensor::zeros((1, 1, 32, 4), DType::F32, &d).unwrap(),
            Tensor::zeros((1, 1, 32, 4), DType::F32, &d).unwrap(),
        )),
    ];
    assert!(p.generate(&x, &x, &mut too_long, &d).is_err());
}

#[test]
fn distinct_frames_do_not_share_cache_state() {
    let p = predictor();
    let d = Device::Cpu;
    let a = Tensor::ones((1, 1, 8), DType::F32, &d).unwrap();
    let b = Tensor::zeros((1, 1, 8), DType::F32, &d).unwrap();
    let mut fresh_a = vec![None, None];
    let (codes_a, cache_a) = p.generate(&a, &b, &mut fresh_a, &d).unwrap();
    let mut fresh_b = vec![None, None];
    let (codes_b, _) = p.generate(&b, &a, &mut fresh_b, &d).unwrap();
    let mut reused = cache_a;
    assert!(p.generate(&b, &a, &mut reused, &d).is_err());
    let mut logits_a_cache = vec![None, None];
    let logits_a = p
        .first_step_logits(&a, &b, &mut logits_a_cache, &d)
        .unwrap();
    let mut logits_b_cache = vec![None, None];
    let logits_b = p
        .first_step_logits(&b, &a, &mut logits_b_cache, &d)
        .unwrap();
    assert_ne!(
        logits_a.to_vec2::<f32>().unwrap(),
        logits_b.to_vec2::<f32>().unwrap(),
        "distinct frame inputs collapsed to identical logits"
    );
    assert_eq!(codes_a.dims(), codes_b.dims());
}

#[test]
fn cached_steps_match_every_growing_prefix_and_mutations_break_parity() {
    let p = predictor();
    let device = Device::Cpu;
    let talker_hidden = Tensor::from_slice(
        &[0.11f32, -0.23, 0.37, -0.41, 0.53, -0.67, 0.79, -0.83],
        (1, 1, 8),
        &device,
    )
    .unwrap();
    let codebook_0 = Tensor::from_slice(
        &[-0.19f32, 0.29, -0.31, 0.43, -0.47, 0.59, -0.61, 0.73],
        (1, 1, 8),
        &device,
    )
    .unwrap();
    let mut inputs = vec![Tensor::cat(&[talker_hidden, codebook_0], 1).unwrap()];
    let mut cached = vec![None, None];
    let mut cached_logits = Vec::new();
    let mut cached_hidden = Vec::new();
    let mut cached_snapshots = Vec::new();
    let mut next = 0u32;

    for group in 0..15 {
        let previous_cache = cached.clone();
        let input = if group == 0 {
            inputs[0].clone()
        } else {
            let ids = Tensor::from_slice(&[next], (1, 1), &device).unwrap();
            let input = embedding_lookup(&p.codec_embeddings[group - 1], &ids).unwrap();
            inputs.push(input.clone());
            input
        };
        let positions = if group == 0 {
            vec![0, 1]
        } else {
            vec![(group + 1) as u32]
        };
        let hidden = p
            .forward_prefix_for_test(&input, &positions, &mut cached)
            .unwrap();
        if group > 0 {
            for (layer, (before, after)) in previous_cache.iter().zip(&cached).enumerate() {
                let (before_k, before_v) = before.as_ref().unwrap();
                let (after_k, after_v) = after.as_ref().unwrap();
                let prefix_len = before_k.dim(2).unwrap();
                assert_bit_identical(
                    before_k,
                    &after_k.narrow(2, 0, prefix_len).unwrap(),
                    &format!("K prefix group {group} layer {layer}"),
                );
                assert_bit_identical(
                    before_v,
                    &after_v.narrow(2, 0, prefix_len).unwrap(),
                    &format!("V prefix group {group} layer {layer}"),
                );
                assert_eq!(after_k.dim(2).unwrap(), prefix_len + 1);
                assert_eq!(after_v.dim(2).unwrap(), prefix_len + 1);
            }
        }
        let last = hidden.narrow(1, hidden.dim(1).unwrap() - 1, 1).unwrap();
        let logits = linear(&last, &p.lm_heads[group])
            .unwrap()
            .squeeze(1)
            .unwrap();
        next = logits
            .argmax(1)
            .unwrap()
            .reshape(())
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        cached_hidden.push(last);
        cached_logits.push(logits);
        cached_snapshots.push(cached.clone());

        let prefix = Tensor::cat(&inputs.iter().collect::<Vec<_>>(), 1).unwrap();
        let full_positions = (0..prefix.dim(1).unwrap() as u32).collect::<Vec<_>>();
        let mut fresh = vec![None, None];
        let full_hidden = p
            .forward_prefix_for_test(&prefix, &full_positions, &mut fresh)
            .unwrap();
        let full_cache = fresh;
        let full_last = full_hidden
            .narrow(1, full_hidden.dim(1).unwrap() - 1, 1)
            .unwrap();
        let full_logits = linear(&full_last, &p.lm_heads[group])
            .unwrap()
            .squeeze(1)
            .unwrap();
        assert!(
            max_abs_diff(&cached_hidden[group], &full_last) < 1e-5,
            "hidden mismatch at group {group}"
        );
        assert!(
            max_abs_diff(&cached_logits[group], &full_logits) < 1e-5,
            "logit mismatch at group {group}"
        );
        for (layer, (cached_layer, full_layer)) in cached_snapshots[group]
            .iter()
            .zip(full_cache.iter())
            .enumerate()
        {
            let (cached_k, cached_v) = cached_layer.as_ref().unwrap();
            let (full_k, full_v) = full_layer.as_ref().unwrap();
            assert!(
                max_abs_diff(cached_k, full_k) < 1e-5 && max_abs_diff(cached_v, full_v) < 1e-5,
                "KV mismatch at group {group}, layer {layer}"
            );
        }
    }

    let (baseline, _) = private_trace(&p, None);
    let mut embedding_mutant = predictor();
    embedding_mutant.codec_embeddings.swap(0, 1);
    let (embedding_logits, _) = private_trace(&embedding_mutant, None);
    assert!(
        baseline
            .iter()
            .zip(&embedding_logits)
            .any(|(a, b)| max_abs_diff(a, b) > 1e-6),
        "private embedding swap was not detected"
    );

    let mut head_mutant = predictor();
    head_mutant.lm_heads.swap(0, 1);
    let (head_logits, _) = private_trace(&head_mutant, None);
    assert!(
        baseline
            .iter()
            .zip(&head_logits)
            .any(|(a, b)| max_abs_diff(a, b) > 1e-6),
        "private head swap was not detected"
    );

    let (position_logits, position_caches) = private_trace(&p, Some((1, 3)));
    assert!(
        baseline
            .iter()
            .zip(&position_logits)
            .any(|(a, b)| max_abs_diff(a, b) > 1e-6)
            || cached_snapshots.iter().zip(position_caches.iter()).any(
                |(baseline_layers, mutant_layers)| baseline_layers.iter().zip(mutant_layers).any(
                    |(a, b)| {
                        let (ak, av) = a.as_ref().unwrap();
                        let (bk, bv) = b.as_ref().unwrap();
                        max_abs_diff(ak, bk) > 1e-6 || max_abs_diff(av, bv) > 1e-6
                    }
                )
            ),
        "position mutation was not detected"
    );
}

#[test]
fn prefill_rejects_cache_reuse_and_fresh_frame_is_deterministic() {
    let p = predictor();
    let d = Device::Cpu;
    let a = Tensor::ones((1, 1, 8), DType::F32, &d).unwrap();
    let b = Tensor::zeros((1, 1, 8), DType::F32, &d).unwrap();
    let mut first = vec![None, None];
    let (codes_a, first_cache) = p.generate(&a, &b, &mut first, &d).unwrap();
    let mut reused = first_cache;
    assert!(p.generate(&a, &b, &mut reused, &d).is_err());
    let mut fresh = vec![None, None];
    let (codes_b, _) = p.generate(&a, &b, &mut fresh, &d).unwrap();
    assert_eq!(
        codes_a.to_vec2::<u32>().unwrap(),
        codes_b.to_vec2::<u32>().unwrap()
    );
}

#[test]
fn malformed_cache_count_fails_closed() {
    let p = predictor();
    let d = Device::Cpu;
    let x = Tensor::zeros((1, 1, 8), DType::F32, &d).unwrap();
    let mut cache = vec![None];
    assert!(p.generate(&x, &x, &mut cache, &d).is_err());
}

#[test]
fn final_observer_failure_does_not_commit_staged_cache() {
    let predictor = predictor();
    let device = Device::Cpu;
    let hidden = Tensor::ones((1, 1, 8), DType::F32, &device).unwrap();
    let codebook_0 = Tensor::zeros((1, 1, 8), DType::F32, &device).unwrap();
    let mut cache = vec![None, None];
    let mut observer = FailOnFinalHidden;
    assert!(
        predictor
            .generate_with_observer(&hidden, &codebook_0, &mut cache, &device, 0, &mut observer,)
            .is_err()
    );
    assert!(cache.iter().all(Option::is_none));
}

#[test]
fn private_head_then_embedding_mapping_is_frame_local() {
    let mut p = predictor();
    for group in 0..15 {
        p.lm_heads[group] = one_hot_row(32, 8, group + 1);
        let mut emb = vec![0.0f32; 32 * 8];
        for value in &mut emb[(group + 1) * 8..(group + 2) * 8] {
            *value = 1.0;
        }
        p.codec_embeddings[group] = Tensor::from_slice(&emb, (32, 8), &Device::Cpu).unwrap();
    }
    let d = Device::Cpu;
    let hidden = Tensor::ones((1, 1, 8), DType::F32, &d).unwrap();
    let c0 = Tensor::ones((1, 1, 8), DType::F32, &d).unwrap();
    let mut cache = vec![None, None];
    let (codes, _) = p.generate(&hidden, &c0, &mut cache, &d).unwrap();
    let ids = codes.to_vec2::<u32>().unwrap()[0].clone();
    for (group, id) in ids.iter().enumerate() {
        assert_eq!(
            *id,
            (group + 1) as u32,
            "private mapping mismatch at group {group}"
        );
    }
}
