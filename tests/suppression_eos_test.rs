use candle_core::{Device, Tensor};
use qwen3tts::talker::sampling::{Sampler, SamplingOptions, greedy_select};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;

#[derive(Deserialize)]
struct Fixture {
    version: u32,
    fixture_id: String,
    source_revision: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    vocab_size: usize,
    suppress_from: usize,
    eos: usize,
    step: usize,
    allowed: Vec<usize>,
}

#[test]
fn fixture_provenance_and_manifest_are_exact() {
    let raw = fs::read_to_string("fixtures/alignment/p02_suppression_eos_vectors.json").unwrap();
    let fixture: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(
        fixture["official_qwen_revision"],
        "022e286b98fbec7e1e916cb940cdf532cd9f488e"
    );
    assert_eq!(
        fixture["qwentts_cpp_revision"],
        "82cd05b9f3a175612dc89fd6943e610fab096ef5"
    );
    let digest = format!("{:x}", Sha256::digest(raw.as_bytes()));
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string("config/fixtures.json").unwrap()).unwrap();
    let entry = manifest["fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["id"] == "p02-suppression-eos-vectors")
        .unwrap();
    assert_eq!(entry["sha256"], digest);
    assert_eq!(
        entry["revision"],
        "82cd05b9f3a175612dc89fd6943e610fab096ef5"
    );
}

#[test]
fn dynamic_reserved_suffix_vectors_match() {
    let fixture: Fixture = serde_json::from_str(
        &fs::read_to_string("fixtures/alignment/p02_suppression_eos_vectors.json").unwrap(),
    )
    .unwrap();
    assert_eq!(fixture.version, 1);
    assert_eq!(fixture.fixture_id, "p02-suppression-eos-vectors");
    assert_eq!(
        fixture.source_revision,
        "82cd05b9f3a175612dc89fd6943e610fab096ef5"
    );
    for case in fixture.cases {
        assert_eq!(case.suppress_from, case.vocab_size - 1024);
        assert_eq!(case.allowed.contains(&case.eos), case.step >= 2);
        let logits = Tensor::ones(case.vocab_size, candle_core::DType::F32, &Device::Cpu).unwrap();
        let token = greedy_select(
            &logits,
            Some(case.suppress_from),
            (case.step >= 2).then_some(case.eos),
        )
        .unwrap() as usize;
        assert!(case.allowed.contains(&token));
    }
}

#[test]
fn sampled_literal_vectors_select_only_dynamic_allowed_tokens() {
    let logits = Tensor::new(&[0.0_f32; 8], &Device::Cpu).unwrap();
    let options = SamplingOptions {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };
    let mut sampler = Sampler::new(77);
    let mut values = logits.to_vec1::<f32>().unwrap();
    values[3] = 10.0;
    values[7] = 20.0;
    let masked = Tensor::from_slice(&values, 8, &Device::Cpu).unwrap();
    assert_eq!(
        sampler
            .sample_with_mode(&masked, options, true, Some(4), None, &[])
            .unwrap(),
        3
    );
    let mut eos_values = vec![0.0_f32; 8];
    eos_values[7] = 20.0;
    let eos_logits = Tensor::from_slice(&eos_values, 8, &Device::Cpu).unwrap();
    assert_eq!(
        sampler
            .sample_with_mode(&eos_logits, options, true, Some(4), Some(7), &[])
            .unwrap(),
        7
    );
}

#[test]
fn suppression_no_candidate_fails_without_philox_draw() {
    let mut sampler = Sampler::new(1);
    let logits = Tensor::new(&[0.0_f32, 1.0], &Device::Cpu).unwrap();
    let options = SamplingOptions {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };
    assert!(
        sampler
            .sample_with_mode(&logits, options, true, Some(0), None, &[])
            .is_err()
    );
    assert_eq!(sampler.subsequence_counter(), 0);
    let nonfinite = Tensor::new(&[f32::NAN, f32::NEG_INFINITY], &Device::Cpu).unwrap();
    assert!(
        sampler
            .sample_with_mode(&nonfinite, options, true, Some(0), None, &[])
            .is_err()
    );
    assert_eq!(sampler.subsequence_counter(), 0);
}

#[test]
fn code_predictor_reserved_tokens_are_not_suppressed() {
    let logits = Tensor::new(&[0.0_f32, 10.0], &Device::Cpu).unwrap();
    assert_eq!(greedy_select(&logits, None, None).unwrap(), 1);
}
