use candle_core::{DType, Device, Tensor};
use qwen3tts::talker::primitives::create_causal_mask;
use qwen3tts::talker::{InputBuilder, TalkerConfig, TalkerWeightLoader};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct F {
    prompt_ids: Vec<Vec<u32>>,
    language: String,
    snapshot_revision: String,
    frames: Vec<Vec<u32>>,
}
#[derive(Deserialize)]
struct P03 {
    snapshot_revision: String,
    official_qwen_revision: String,
    qwentts_cpp_revision: String,
    min_cosine: f64,
    max_abs_limit: f64,
    #[serde(default)]
    external_frame_index: usize,
}
fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut x, mut y) = (0., 0., 0.);
    for (u, v) in a.iter().zip(b) {
        d += (*u as f64) * (*v as f64);
        x += (*u as f64).powi(2);
        y += (*v as f64).powi(2);
    }
    d / (x.sqrt() * y.sqrt() + 1e-12)
}
fn raw_f32(path: &str) -> Vec<f32> {
    let bytes = fs::read(path).expect("FIXTURE_MISSING: qwentts raw tensor");
    assert!(bytes.len() >= 8);
    let rank = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let header = 4 + rank * 4;
    assert_eq!(header, 8);
    bytes[header..]
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "requires pinned real model"]
fn pinned_talker_kv_cache_full_vs_cached() {
    let dir = std::path::PathBuf::from(
        std::env::var("QWEN3_TTS_REAL_MODEL_DIR")
            .expect("FIXTURE_MISSING: QWEN3_TTS_REAL_MODEL_DIR"),
    );
    let dir = dir
        .canonicalize()
        .expect("FIXTURE_MISSING: canonical model dir");
    assert!(dir.ends_with("5d83992436eae1d760afd27aff78a71d676296fc"));
    for name in ["config.json", "generation_config.json", "model.safetensors"] {
        assert!(
            dir.join(name).is_file(),
            "FIXTURE_MISSING: {}",
            dir.join(name).display()
        );
    }
    let f: F = serde_json::from_str(
        &fs::read_to_string("fixtures/alignment/p02_deterministic_token_sequences_real.json")
            .expect("FIXTURE_MISSING: p02 fixture"),
    )
    .unwrap();
    let p03: P03 = serde_json::from_str(
        &fs::read_to_string("fixtures/alignment/p03_talker_kv_cache_real.json")
            .expect("FIXTURE_MISSING: p03 fixture"),
    )
    .unwrap();
    assert_eq!(p03.snapshot_revision, f.snapshot_revision);
    assert_eq!(
        p03.official_qwen_revision,
        "022e286b98fbec7e1e916cb940cdf532cd9f488e"
    );
    assert_eq!(
        p03.qwentts_cpp_revision,
        "82cd05b9f3a175612dc89fd6943e610fab096ef5"
    );
    assert_eq!(
        f.snapshot_revision,
        "5d83992436eae1d760afd27aff78a71d676296fc"
    );
    let d = Device::Cpu;
    let talker = TalkerWeightLoader::from_safetensors(dir.join("model.safetensors"), &d)
        .unwrap()
        .build_talker(&TalkerConfig::default())
        .unwrap();
    let (input, mask, trailing, _) = InputBuilder::new(&talker, &d)
        .build(&f.prompt_ids[0], None, &f.language, None)
        .unwrap();
    let seq = input.dim(1).unwrap();
    let (pos, delta) = talker.compute_position_ids(&mask).unwrap();
    let (cosr, sinr) = talker.rope.forward(&input, &pos).unwrap();
    let causal = create_causal_mask(seq, &d).unwrap();
    let mut full_cache = vec![None; talker.config.num_hidden_layers];
    let _ = talker
        .model
        .forward(&input, &cosr, &sinr, Some(&causal), &mut full_cache)
        .unwrap();
    let frame = &f.frames[p03.external_frame_index];
    let c0 = talker
        .embed_codec(
            &Tensor::new(&[frame[0]], &d)
                .unwrap()
                .reshape((1, 1))
                .unwrap(),
        )
        .unwrap();
    let mut next = c0;
    for (i, token) in frame.iter().skip(1).enumerate() {
        let ids = Tensor::new(&[*token], &d).unwrap().reshape((1, 1)).unwrap();
        next = (next
            + qwen3tts::talker::primitives::embedding_lookup(
                &talker.code_predictor.codec_embeddings[i],
                &ids,
            )
            .unwrap())
        .unwrap();
    }
    next = (next + trailing.narrow(1, 0, 1).unwrap()).unwrap();
    let full_input = Tensor::cat(&[&input, &next], 1).unwrap();
    let full_mask = Tensor::ones((1, seq + 1), DType::I64, &d).unwrap();
    let (full_pos, _) = talker.compute_position_ids(&full_mask).unwrap();
    let (full_cos, full_sin) = talker.rope.forward(&full_input, &full_pos).unwrap();
    let full_causal = create_causal_mask(seq + 1, &d).unwrap();
    let mut recompute_cache = vec![None; talker.config.num_hidden_layers];
    let recomputed = talker
        .model
        .forward(
            &full_input,
            &full_cos,
            &full_sin,
            Some(&full_causal),
            &mut recompute_cache,
        )
        .unwrap();
    let positions =
        qwen3tts::talker::primitives::MultimodalRotaryEmbedding::cached_positions_from_delta(
            seq as u32, &delta, 1, &d,
        )
        .unwrap();
    let (c2, s2) = talker.rope.forward_single_position(&positions).unwrap();
    let mut cache = vec![None; talker.config.num_hidden_layers];
    let _ = talker
        .model
        .forward(&input, &cosr, &sinr, Some(&causal), &mut cache)
        .unwrap();
    let cached = talker
        .model
        .forward(&next, &c2, &s2, None, &mut cache)
        .unwrap();
    assert_eq!(cache.len(), 28);
    let mut layer_metrics = Vec::new();
    for (idx, c) in cache.iter().enumerate() {
        let (k, v) = c.as_ref().unwrap();
        let kd = k.dims4().unwrap();
        assert_eq!(kd.0, 1);
        assert_eq!(kd.1, talker.model.layers[idx].self_attn.num_kv_heads);
        assert_eq!(kd.3, talker.model.layers[idx].self_attn.head_dim);
        assert_eq!(k.dtype(), DType::F32);
        assert_eq!(v.dtype(), DType::F32);
        assert_eq!(kd.2, seq + 1);
        assert_eq!(v.dims4().unwrap(), kd);
        let (fk, fv) = recompute_cache[idx].as_ref().unwrap();
        let prefix = k
            .narrow(2, 0, seq)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let full_prefix = fk
            .narrow(2, 0, seq)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(prefix, full_prefix, "layer {idx} KV prefix changed");
        let value_prefix = v
            .narrow(2, 0, seq)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let full_value_prefix = fv
            .narrow(2, 0, seq)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            value_prefix, full_value_prefix,
            "layer {idx} V prefix changed"
        );
        let appended = k
            .narrow(2, seq, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = fk
            .narrow(2, seq, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let kc = cos(&appended, &expected);
        let km = appended
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            kc >= p03.min_cosine && km <= p03.max_abs_limit as f32,
            "layer {idx} K metrics cosine={kc} max_abs={km}"
        );
        let va = v
            .narrow(2, seq, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let ve = fv
            .narrow(2, seq, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let vc = cos(&va, &ve);
        let vm = va
            .iter()
            .zip(&ve)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            vc >= p03.min_cosine && vm <= p03.max_abs_limit as f32,
            "layer {idx} V metrics cosine={vc} max_abs={vm}"
        );
        layer_metrics.push(format!(
            "{{\"layer\":{idx},\"k_cosine\":{kc:.12},\"k_max_abs\":{km:.9},\"v_cosine\":{vc:.12},\"v_max_abs\":{vm:.9}}}"
        ));
    }
    let a = recomputed
        .narrow(1, seq, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let b = cached.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let oracle_dir = "artifacts/alignment/P03/P03-T02/qwentts-direct-oracle/raw_tensor_dump";
    let oracle_next = raw_f32(&format!("{oracle_dir}/next-emb-step0.bin"));
    let oracle_hidden = raw_f32(&format!("{oracle_dir}/talker-hidden-step1.bin"));
    let rust_next = next.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let next_cos = cos(&rust_next, &oracle_next);
    let hidden_cos = cos(&b, &oracle_hidden);
    assert!(
        next_cos >= p03.min_cosine,
        "qwentts next embedding cosine={next_cos}"
    );
    assert!(
        hidden_cos >= p03.min_cosine,
        "qwentts step1 hidden cosine={hidden_cos}"
    );
    let next_max = rust_next
        .iter()
        .zip(&oracle_next)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let hidden_max = b
        .iter()
        .zip(&oracle_hidden)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        next_max <= p03.max_abs_limit as f32,
        "qwentts next embedding max_abs={next_max}"
    );
    assert!(
        hidden_max <= p03.max_abs_limit as f32,
        "qwentts step1 hidden max_abs={hidden_max}"
    );
    let hc = cos(&a, &b);
    let hm = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        hc >= p03.min_cosine && hm <= p03.max_abs_limit as f32,
        "hidden cosine={hc} max_abs={hm}"
    );
    let la = talker
        .codec_head_logits(&recomputed.narrow(1, seq, 1).unwrap())
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let lb = talker
        .codec_head_logits(&cached)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let lc = cos(&la, &lb);
    let lm = la
        .iter()
        .zip(&lb)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        lc >= p03.min_cosine && lm <= p03.max_abs_limit as f32,
        "logits cosine={lc} max_abs={lm}"
    );
    println!(
        "p03_kv_metrics={{\"snapshot\":\"{}\",\"layers\":28,\"hidden_cosine\":{hc},\"sequence\":{seq},\"per_layer\":[{}]}}",
        f.snapshot_revision,
        layer_metrics.join(",")
    );
    if let Ok(path) = std::env::var("P03_METRICS_OUT") {
        let metrics = format!(
            "{{\"schema_version\":1,\"provenance\":{{\"snapshot_revision\":\"{}\",\"qwentts_cpp_revision\":\"{}\"}},\"sequence\":{},\"next_embedding\":{{\"cosine\":{next_cos},\"max_abs\":{next_max}}},\"qwentts_hidden_step1\":{{\"cosine\":{hidden_cos},\"max_abs\":{hidden_max}}},\"final_hidden\":{{\"cosine\":{hc},\"max_abs\":{hm}}},\"logits\":{{\"cosine\":{lc},\"max_abs\":{lm}}},\"layers\":[{}]}}",
            p03.snapshot_revision,
            p03.qwentts_cpp_revision,
            seq,
            layer_metrics.join(",")
        );
        fs::write(path, metrics).expect("failed to write P03 metrics artifact");
    }
}
