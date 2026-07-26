//! Pinned official-Python versus Candle Code Predictor frame gate.

use candle_core::{Device, Tensor};
use qwen3tts::talker::primitives::{embedding_lookup, linear};
use qwen3tts::talker::{TalkerConfig, TalkerWeightLoader};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const EXPECTED_MODEL_DIR: &str = r"C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc";

fn model_dir() -> PathBuf {
    PathBuf::from(
        std::env::var_os("QWEN3_TTS_REAL_MODEL_DIR")
            .expect("FIXTURE_MISSING: QWEN3_TTS_REAL_MODEL_DIR"),
    )
}

fn read_f32(path: impl AsRef<Path>) -> Vec<f32> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("FIXTURE_MISSING: {}: {error}", path.display()));
    assert_eq!(
        bytes.len() % 4,
        0,
        "invalid f32 fixture: {}",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn sha256(path: impl AsRef<Path>) -> String {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("FIXTURE_MISSING: {}: {error}", path.display()));
    format!("{:X}", Sha256::digest(bytes))
}

fn verified_asset(entry: &Value, name: &str) -> PathBuf {
    let path = PathBuf::from(
        entry["path"]
            .as_str()
            .unwrap_or_else(|| panic!("FIXTURE_INVALID: {name} path")),
    );
    if !path.exists() {
        panic!("FIXTURE_MISSING: {}", path.display());
    }
    let expected_hash = entry["sha256"]
        .as_str()
        .unwrap_or_else(|| panic!("FIXTURE_INVALID: {name} sha256"));
    assert_eq!(
        sha256(&path),
        expected_hash,
        "FIXTURE_INVALID: {name} sha256"
    );
    path
}

fn verified_tensor(entry: &Value, name: &str) -> PathBuf {
    let path = verified_asset(entry, name);
    let shape = entry["shape"]
        .as_array()
        .unwrap_or_else(|| panic!("FIXTURE_INVALID: {name} shape"));
    let elements = shape
        .iter()
        .map(|dimension| dimension.as_u64().unwrap() as usize)
        .product::<usize>();
    let item_size = match entry["dtype"].as_str() {
        Some("float32") => 4,
        Some("int64") => 8,
        _ => panic!("FIXTURE_INVALID: {name} dtype"),
    };
    let actual_size = std::fs::metadata(&path)
        .unwrap_or_else(|error| panic!("FIXTURE_MISSING: {}: {error}", path.display()))
        .len() as usize;
    assert_eq!(
        actual_size,
        elements * item_size,
        "FIXTURE_INVALID: {name} byte size"
    );
    path
}

fn metrics(actual: &Tensor, expected: &Tensor) -> (f64, f64) {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&lhs, &rhs) in actual.iter().zip(&expected) {
        let lhs = lhs as f64;
        let rhs = rhs as f64;
        dot += lhs * rhs;
        actual_norm += lhs * lhs;
        expected_norm += rhs * rhs;
        max_abs = max_abs.max((lhs - rhs).abs());
    }
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt()).max(f64::MIN_POSITIVE);
    (cosine, max_abs)
}

fn assert_aligned(
    name: &str,
    actual: &Tensor,
    expected: &Tensor,
    min_cosine: f64,
    max_abs_limit: f64,
) -> Value {
    let (cosine, max_abs) = metrics(actual, expected);
    assert!(
        cosine >= min_cosine,
        "{name} cosine {cosine} is below {min_cosine}"
    );
    assert!(
        max_abs <= max_abs_limit,
        "{name} max_abs {max_abs} exceeds {max_abs_limit}"
    );
    json!({"name": name, "cosine": cosine, "max_abs": max_abs})
}

#[test]
#[ignore = "requires pinned 0.6B snapshot and independent official oracle"]
fn pinned_code_predictor_frame_matches_official_oracle() {
    let model_dir = model_dir();
    if !model_dir.exists() {
        panic!("FIXTURE_MISSING: {}", model_dir.display());
    }
    assert_eq!(
        model_dir
            .canonicalize()
            .unwrap_or_else(|error| panic!("FIXTURE_MISSING: {}: {error}", model_dir.display())),
        PathBuf::from(EXPECTED_MODEL_DIR)
            .canonicalize()
            .expect("FIXTURE_MISSING: pinned expected model directory")
    );
    for name in ["config.json", "generation_config.json", "model.safetensors"] {
        assert!(
            model_dir.join(name).exists(),
            "FIXTURE_MISSING: {}",
            model_dir.join(name).display()
        );
    }

    let fixture_path = Path::new("fixtures/alignment/p03_code_predictor_frame_real.json");
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_path)
            .unwrap_or_else(|error| panic!("FIXTURE_MISSING: {}: {error}", fixture_path.display())),
    )
    .unwrap();
    assert_eq!(fixture["oracle_status"], "ready");
    let min_cosine = fixture["min_cosine"].as_f64().unwrap();
    let max_abs_limit = fixture["max_abs_limit"].as_f64().unwrap();
    assert_eq!(min_cosine, 0.999);
    assert_eq!(max_abs_limit, 0.001);
    let manifest_path = Path::new(fixture["oracle_manifest"].as_str().unwrap());
    if !manifest_path.exists() {
        panic!("FIXTURE_MISSING: {}", manifest_path.display());
    }
    assert_eq!(
        sha256(manifest_path),
        fixture["oracle_manifest_sha256"].as_str().unwrap()
    );
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(manifest_path).unwrap_or_else(|error| {
            panic!("FIXTURE_MISSING: {}: {error}", manifest_path.display())
        }),
    )
    .unwrap();
    assert_eq!(
        manifest["cache_lengths"],
        json!((2..=16).collect::<Vec<_>>())
    );
    assert_eq!(manifest["manual_generate_codes_equal"], true);
    assert_eq!(manifest["manual_generate_logits_bit_equal"], true);

    for name in ["model.safetensors", "config.json", "generation_config.json"] {
        assert_eq!(
            sha256(model_dir.join(name)),
            manifest["model_hashes"][name].as_str().unwrap(),
            "FIXTURE_INVALID: {name} sha256"
        );
    }
    let tensors = &manifest["tensors"];
    for name in [
        "projected_prefill",
        "projected_private_inputs",
        "normalized_hidden",
        "logits",
        "final_cache_k",
        "final_cache_v",
        "positions",
    ] {
        verified_tensor(&tensors[name], name);
    }
    let prefill_values = read_f32(verified_tensor(
        &tensors["projected_prefill"],
        "projected_prefill",
    ));
    let private_values = read_f32(verified_tensor(
        &tensors["projected_private_inputs"],
        "projected_private_inputs",
    ));
    let hidden_values = read_f32(verified_tensor(
        &tensors["normalized_hidden"],
        "normalized_hidden",
    ));
    let logits_values = read_f32(verified_tensor(&tensors["logits"], "logits"));
    let cache_k_values = read_f32(verified_tensor(&tensors["final_cache_k"], "final_cache_k"));
    let cache_v_values = read_f32(verified_tensor(&tensors["final_cache_v"], "final_cache_v"));
    assert_eq!(prefill_values.len(), 2 * 1024);
    assert_eq!(private_values.len(), 14 * 1024);
    assert_eq!(hidden_values.len(), 15 * 1024);
    assert_eq!(logits_values.len(), 15 * 2048);
    assert_eq!(cache_k_values.len(), 5 * 8 * 16 * 128);
    assert_eq!(cache_v_values.len(), 5 * 8 * 16 * 128);

    let cache_manifest_path = verified_asset(&manifest["cache_manifest"], "cache_manifest");
    let cache_manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(&cache_manifest_path).unwrap()).unwrap();
    assert_eq!(
        cache_manifest["lengths"],
        json!((2..=16).collect::<Vec<_>>())
    );
    assert_eq!(cache_manifest["prefix_bit_identical"], true);
    let codes_path = verified_asset(&manifest["codes"], "codes");
    let codes: Value =
        serde_json::from_str(&std::fs::read_to_string(&codes_path).unwrap()).unwrap();
    let expected_codes = codes["generated"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as u32)
        .collect::<Vec<_>>();
    assert_eq!(expected_codes.len(), 15);

    let device = Device::Cpu;
    let talker = TalkerWeightLoader::from_safetensors(model_dir.join("model.safetensors"), &device)
        .unwrap()
        .build_talker(&TalkerConfig::default())
        .unwrap();
    let predictor = &talker.code_predictor;
    assert_eq!(predictor.layers.len(), 5);
    assert_eq!(predictor.codec_embeddings.len(), 15);
    assert_eq!(predictor.lm_heads.len(), 15);

    let prefill = Tensor::from_vec(prefill_values.clone(), (1, 2, 1024), &device).unwrap();
    let talker_hidden = prefill.narrow(1, 0, 1).unwrap();
    let expected_c0_embedding = prefill.narrow(1, 1, 1).unwrap();
    let c0_ids = Tensor::from_slice(&[1995u32], (1, 1), &device).unwrap();
    let actual_c0_embedding = embedding_lookup(&talker.codec_embedding, &c0_ids).unwrap();
    let mut tensor_metrics = vec![assert_aligned(
        "c0-embedding",
        &actual_c0_embedding,
        &expected_c0_embedding,
        min_cosine,
        max_abs_limit,
    )];
    let oracle_hidden = Tensor::from_vec(hidden_values, (15, 1024), &device).unwrap();
    let oracle_logits = Tensor::from_vec(logits_values, (15, 2048), &device).unwrap();
    let oracle_cache_k = Tensor::from_vec(cache_k_values, (5, 1, 8, 16, 128), &device).unwrap();
    let oracle_cache_v = Tensor::from_vec(cache_v_values, (5, 1, 8, 16, 128), &device).unwrap();

    let mut cache = vec![None; 5];
    let mut actual_codes = Vec::with_capacity(15);
    let mut next_code = 0u32;
    for group in 0..15 {
        let input = if group == 0 {
            prefill.clone()
        } else {
            let ids = Tensor::from_slice(&[next_code], (1, 1), &device).unwrap();
            let input = embedding_lookup(&predictor.codec_embeddings[group - 1], &ids).unwrap();
            let expected = Tensor::from_slice(
                &private_values[(group - 1) * 1024..group * 1024],
                (1, 1, 1024),
                &device,
            )
            .unwrap();
            tensor_metrics.push(assert_aligned(
                &format!("private-input-{group}"),
                &input,
                &expected,
                min_cosine,
                max_abs_limit,
            ));
            input
        };
        let positions = if group == 0 {
            vec![0, 1]
        } else {
            vec![(group + 1) as u32]
        };
        let hidden = predictor
            .forward_prefix_for_test(&input, &positions, &mut cache)
            .unwrap();
        let last_hidden = hidden
            .narrow(1, hidden.dim(1).unwrap() - 1, 1)
            .unwrap()
            .squeeze(1)
            .unwrap();
        let expected_hidden = oracle_hidden.narrow(0, group, 1).unwrap();
        tensor_metrics.push(assert_aligned(
            &format!("hidden-{group}"),
            &last_hidden,
            &expected_hidden,
            min_cosine,
            max_abs_limit,
        ));
        let logits = linear(&last_hidden, &predictor.lm_heads[group]).unwrap();
        let expected_logits = oracle_logits.narrow(0, group, 1).unwrap();
        tensor_metrics.push(assert_aligned(
            &format!("logits-{group}"),
            &logits,
            &expected_logits,
            min_cosine,
            max_abs_limit,
        ));
        next_code = logits
            .argmax(1)
            .unwrap()
            .reshape(())
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        assert_eq!(next_code, expected_codes[group], "code mismatch at {group}");
        actual_codes.push(next_code);

        let expected_len = group + 2;
        for (layer, entry) in cache.iter().enumerate() {
            let (actual_k, actual_v) = entry.as_ref().unwrap();
            assert_eq!(actual_k.dims(), &[1, 8, expected_len, 128]);
            assert_eq!(actual_v.dims(), &[1, 8, expected_len, 128]);
            let expected_k = oracle_cache_k
                .narrow(0, layer, 1)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .narrow(2, 0, expected_len)
                .unwrap();
            let expected_v = oracle_cache_v
                .narrow(0, layer, 1)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .narrow(2, 0, expected_len)
                .unwrap();
            tensor_metrics.push(assert_aligned(
                &format!("cache-k-{group}-{layer}"),
                actual_k,
                &expected_k,
                min_cosine,
                max_abs_limit,
            ));
            tensor_metrics.push(assert_aligned(
                &format!("cache-v-{group}-{layer}"),
                actual_v,
                &expected_v,
                min_cosine,
                max_abs_limit,
            ));
        }
    }
    assert_eq!(actual_codes, expected_codes);

    let mut production_cache = vec![None; 5];
    let (production_codes, _) = predictor
        .generate(
            &talker_hidden,
            &actual_c0_embedding,
            &mut production_cache,
            &device,
        )
        .unwrap();
    assert_eq!(
        production_codes.to_vec2::<u32>().unwrap()[0],
        expected_codes
    );
    for entry in &production_cache {
        assert_eq!(entry.as_ref().unwrap().0.dim(2).unwrap(), 16);
    }

    let minimum_cosine = tensor_metrics
        .iter()
        .map(|metric| metric["cosine"].as_f64().unwrap())
        .fold(1.0f64, f64::min);
    let maximum_abs = tensor_metrics
        .iter()
        .map(|metric| metric["max_abs"].as_f64().unwrap())
        .fold(0.0f64, f64::max);
    let metrics_document = json!({
        "task": "P03-T03",
        "source": "official-pinned-python-vs-candle-cpu-f32",
        "groups": 15,
        "layers": 5,
        "cache_lengths": (2..=16).collect::<Vec<_>>(),
        "minimum_cosine": minimum_cosine,
        "maximum_abs": maximum_abs,
        "min_cosine_limit": min_cosine,
        "max_abs_limit": max_abs_limit,
        "codes": actual_codes,
        "tensors": tensor_metrics,
    });
    let metrics_path = Path::new("artifacts/alignment/P03/P03-T03/real-metrics.json");
    std::fs::write(
        metrics_path,
        serde_json::to_string_pretty(&metrics_document).unwrap() + "\n",
    )
    .unwrap();
    println!("P03_T03_REAL_PASS minimum_cosine={minimum_cosine:.12} maximum_abs={maximum_abs:.12}");
}
