use candle_core::{Device, Tensor};
use qwen3tts::talker::sampling::{Sampler, SamplingOptions};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

const REPO_ROOT_FIXTURE: &str = "fixtures/alignment/p02_repetition_penalty_vectors.json";
const FIXTURE_JSON: &str = REPO_ROOT_FIXTURE;
const ALIGNED_REVISION: &str = "82cd05b9f3a175612dc89fd6943e610fab096ef5";

#[derive(Debug, Deserialize)]
struct Fixture {
    version: u32,
    fixture_id: String,
    source_revision: String,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    logit_bits: Vec<String>,
    history: Vec<u16>,
    suppress_from: Option<usize>,
    allow_suppressed_token: Option<usize>,
    options: CaseOptions,
    candidates_before_repetition: Vec<BitLogit>,
    after_repetition_bits: Vec<String>,
    candidates_after_repetition: Vec<usize>,
    after_temperature_bits: Vec<String>,
    candidates_after_temperature: Vec<usize>,
    seed: String,
    samples: Vec<CaseSample>,
}

#[derive(Debug, Deserialize)]
struct BitLogit {
    idx: usize,
    logit_bits: String,
}

#[derive(Debug, Deserialize)]
struct CaseOptions {
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repetition_penalty: f64,
}

#[derive(Debug, Deserialize)]
struct CaseSample {
    uniform_bits: Option<String>,
    expected: u32,
    draw: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ConfigFixture {
    id: String,
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct FixtureManifest {
    fixtures: Vec<ConfigFixture>,
}

fn bits_to_f32(bits: &str) -> f32 {
    let bits = bits.trim_start_matches("0x");
    let raw = u32::from_str_radix(bits, 16).unwrap();
    f32::from_bits(raw)
}

fn bits_to_u32(bits: &str) -> u32 {
    u32::from_str_radix(bits.trim_start_matches("0x"), 16).expect("invalid hex bits")
}

fn load_fixture() -> Fixture {
    let text = fs::read_to_string(FIXTURE_JSON).expect("FIXTURE_MISSING: cannot read fixture file");
    serde_json::from_str(&text).expect("failed to parse repetition penalty fixture")
}

fn load_fixture_for_text(path: &str) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("FIXTURE_MISSING: {path} read failed: {err}"))
}

fn cmp_desc(a: &(usize, f32), b: &(usize, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.0.cmp(&b.0))
}

fn filter_candidates(
    logits: &[f32],
    suppress_from: Option<usize>,
    allow_suppressed_token: Option<usize>,
) -> Vec<(usize, f32)> {
    logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(idx, logit)| {
            logit.is_finite()
                && suppress_from
                    .map(|start| *idx < start || Some(*idx) == allow_suppressed_token)
                    .unwrap_or(true)
        })
        .collect()
}

fn apply_repetition_penalty(logits: &[f32], history: &[u16], penalty: f64) -> Vec<f32> {
    if (penalty - 1.0).abs() <= f64::EPSILON {
        return logits.to_vec();
    }

    let mut seen = vec![false; logits.len()];
    let mut out = logits.to_vec();
    for &raw in history {
        let idx = usize::from(raw);
        if idx >= out.len() || seen[idx] {
            continue;
        }
        seen[idx] = true;
        let score = out[idx];
        out[idx] = if score < 0.0 {
            score * penalty as f32
        } else {
            score / penalty as f32
        };
    }
    out
}

fn reference_probabilities(
    mut candidates: Vec<(usize, f32)>,
    top_k: usize,
    top_p: f64,
    temperature: f64,
) -> Vec<(usize, f64)> {
    if candidates.is_empty() {
        return Vec::new();
    }
    candidates.sort_by(cmp_desc);
    if top_k > 0 && candidates.len() > top_k {
        candidates.truncate(top_k);
    }
    candidates.sort_by(cmp_desc);

    let inv_temp = if temperature == 0.0 {
        0.0f32
    } else {
        1.0f32 / temperature as f32
    };
    for (_, logit) in candidates.iter_mut() {
        *logit *= inv_temp;
    }
    candidates.sort_by(cmp_desc);

    let max_logit = candidates[0].1 as f64;
    let mut probs: Vec<(usize, f64)> = candidates
        .iter()
        .map(|(idx, logit)| (*idx, ((*logit as f64) - max_logit).exp()))
        .collect();
    let total: f64 = probs.iter().map(|(_, p)| *p).sum();
    if total <= 0.0 || !total.is_finite() {
        return vec![(candidates[0].0, 1.0)];
    }
    for (_, p) in probs.iter_mut() {
        *p /= total;
    }

    if top_p >= 1.0 {
        probs.sort_by_key(|(idx, _)| *idx);
        return probs;
    }
    let mut cumulative = 0.0;
    let keep = probs
        .iter()
        .position(|(_, p)| {
            cumulative += *p;
            cumulative >= top_p
        })
        .map(|idx| idx + 1)
        .unwrap_or(probs.len());
    let mut kept = probs[..keep].to_vec();
    let renorm: f64 = kept.iter().map(|(_, p)| *p).sum();
    if renorm > 0.0 {
        for (_, p) in kept.iter_mut() {
            *p /= renorm;
        }
    }
    kept.sort_by_key(|(idx, _)| *idx);
    kept
}

fn sample_with_case(case: &Case, draw: f64) -> usize {
    let logits: Vec<f32> = case
        .logit_bits
        .iter()
        .map(|bits| bits_to_f32(bits))
        .collect();

    if logits.is_empty() {
        return 0;
    }
    let mut candidates =
        filter_candidates(&logits, case.suppress_from, case.allow_suppressed_token);
    if candidates.is_empty() {
        return 0;
    }
    candidates.sort_by(cmp_desc);

    if case.options.temperature <= 0.0 {
        return candidates[0].0;
    }

    let after_penalty =
        apply_repetition_penalty(&logits, &case.history, case.options.repetition_penalty);
    let scored = filter_candidates(
        &after_penalty,
        case.suppress_from,
        case.allow_suppressed_token,
    );
    if scored.is_empty() {
        return 0;
    }
    let probs = reference_probabilities(
        scored,
        case.options.top_k,
        case.options.top_p,
        case.options.temperature,
    );
    if probs.is_empty() {
        return 0;
    }
    let mut cumulative = 0.0;
    for (idx, p) in probs {
        cumulative += p;
        if cumulative >= draw {
            return idx;
        }
    }
    0
}

fn sha256_file(path: &Path) -> String {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .unwrap_or_else(|err| panic!("fixture file open failed {}: {err}", path.display()));
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf).expect("fixture read");
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    format!("{:x}", hasher.finalize())
}

#[test]
fn repetition_penalty_matches_official_oracle() {
    let fixture = load_fixture();
    assert_eq!(fixture.source_revision, ALIGNED_REVISION);
    assert_eq!(fixture.fixture_id, "p02-repetition-penalty-vectors");
    assert_eq!(fixture.version, 1);

    for case in &fixture.cases {
        let logits: Vec<f32> = case
            .logit_bits
            .iter()
            .map(|bits| bits_to_f32(bits))
            .collect();
        let options = SamplingOptions {
            temperature: case.options.temperature,
            top_k: case.options.top_k,
            top_p: case.options.top_p,
            repetition_penalty: case.options.repetition_penalty,
        };
        let seed = case.seed.parse::<u64>().unwrap();
        let logits_tensor = Tensor::from_slice(&logits, logits.len(), &Device::Cpu).unwrap();
        let mut sampler = Sampler::new(seed);

        let mut got = Vec::with_capacity(case.samples.len());
        for _sample in &case.samples {
            let out = sampler
                .sample(
                    &logits_tensor,
                    options,
                    case.suppress_from,
                    case.allow_suppressed_token,
                    &case.history,
                )
                .expect("sample should succeed");
            got.push(out as u32);
        }

        let expected: Vec<u32> = case.samples.iter().map(|s| s.expected).collect();
        assert_eq!(got, expected, "case={} unexpected sample", case.name);

        for s in case.samples.iter() {
            if let Some(uniform_bits) = s.uniform_bits.as_deref() {
                let draw = match s.draw {
                    Some(value) => value,
                    None => continue,
                };
                let expected_by_reference = sample_with_case(case, draw) as u32;
                assert_eq!(
                    bits_to_u32(uniform_bits),
                    (draw as f32).to_bits(),
                    "case={} draw mismatch with uniform_bits",
                    case.name
                );
                assert_eq!(
                    expected_by_reference, s.expected,
                    "case={} sample by draw not aligned",
                    case.name
                );
            }
        }
    }
}

#[test]
fn repetition_penalty_binary_transform_is_logged_with_literals() {
    let fixture = load_fixture();
    for case in &fixture.cases {
        let logits: Vec<f32> = case
            .logit_bits
            .iter()
            .map(|bits| bits_to_f32(bits))
            .collect();
        let mut base = filter_candidates(&logits, case.suppress_from, case.allow_suppressed_token);
        base.sort_by(cmp_desc);
        assert_eq!(
            case.candidates_before_repetition.len(),
            base.len(),
            "case={} candidate count changed",
            case.name
        );

        let expected_before: Vec<usize> = base.iter().map(|(idx, _)| *idx).collect();
        let actual_before: Vec<usize> = case
            .candidates_before_repetition
            .iter()
            .map(|item| item.idx)
            .collect();
        let expected_before_bits: Vec<String> = base
            .iter()
            .map(|(_, logit)| format!("0x{:08x}", (*logit).to_bits()))
            .collect();
        let actual_before_bits: Vec<&str> = case
            .candidates_before_repetition
            .iter()
            .map(|item| item.logit_bits.as_str())
            .collect();
        assert_eq!(
            actual_before, expected_before,
            "case={} base candidates must keep index order",
            case.name
        );
        assert_eq!(
            expected_before_bits, actual_before_bits,
            "case={} base candidate logits changed",
            case.name
        );

        let after_penalty = if case.options.temperature > 0.0 {
            apply_repetition_penalty(&logits, &case.history, case.options.repetition_penalty)
        } else {
            logits.clone()
        };
        let mut after = filter_candidates(
            &after_penalty,
            case.suppress_from,
            case.allow_suppressed_token,
        );
        after.sort_by(cmp_desc);
        let expected_after_bits: Vec<String> = after
            .iter()
            .map(|(_, logit)| format!("0x{:08x}", (*logit).to_bits()))
            .collect();
        assert_eq!(
            case.after_repetition_bits, expected_after_bits,
            "case={} after_penalty bits changed",
            case.name
        );
        assert_eq!(
            case.candidates_after_repetition,
            after.iter().map(|(idx, _)| *idx).collect::<Vec<_>>(),
            "case={} after_penalty candidate index order changed",
            case.name
        );

        let after_temp: Vec<String> = if case.options.temperature <= 0.0 {
            after
                .iter()
                .map(|(_, logit)| format!("0x{:08x}", (*logit).to_bits()))
                .collect()
        } else {
            let inv_temp = 1.0f32 / case.options.temperature as f32;
            after
                .iter()
                .map(|(_, logit)| format!("0x{:08x}", ((*logit * inv_temp).to_bits())))
                .collect()
        };
        assert_eq!(
            case.after_temperature_bits, after_temp,
            "case={} after_temperature bits changed",
            case.name
        );
        assert_eq!(
            case.candidates_after_temperature,
            after.iter().map(|(idx, _)| *idx).collect::<Vec<_>>(),
            "case={} candidate order after temperature changed",
            case.name
        );
    }
}

#[test]
fn invalid_repetition_penalty_is_rejected() {
    let logits = Tensor::from_slice(&[0.0_f32, 1.0_f32], 2, &Device::Cpu).unwrap();
    let options = SamplingOptions {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 0.0,
    };
    let err = Sampler::new(1).sample(&logits, options, None, None, &[]);
    assert!(err.is_err(), "repetition_penalty=0 should return Err");

    let mut options = SamplingOptions {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: -1.5,
    };
    assert!(
        Sampler::new(2)
            .sample(&logits, options, None, None, &[])
            .is_err(),
        "negative repetition_penalty should fail"
    );
    options.repetition_penalty = f64::NAN;
    assert!(
        Sampler::new(3)
            .sample(&logits, options, None, None, &[])
            .is_err()
    );
}

#[test]
fn talker_repetition_history_contract_is_locked_in_code_shape() {
    let cp_src = load_fixture_for_text("src/talker/code_predictor.rs");
    let cp_call = "sampler.sample(&logits, sampling, None, None, &[])";
    let cp_calls = cp_src.matches(cp_call).count();
    assert_eq!(
        cp_calls, 2,
        "code predictor must sample 15 steps with empty history"
    );
}

#[test]
fn repetition_penalty_fixture_is_hash_guarded() {
    let text = load_fixture_for_text("config/fixtures.json");
    let manifest: FixtureManifest =
        serde_json::from_str(&text).expect("parse config/fixtures.json");
    let target = manifest
        .fixtures
        .iter()
        .find(|f| f.id == "p02-repetition-penalty")
        .expect("missing p02-repetition-penalty entry");

    assert!(
        target
            .path
            .ends_with("fixtures/alignment/p02_repetition_penalty_vectors.json"),
        "fixture path in config does not point to p02 repetition penalty vectors"
    );
    let path = Path::new("fixtures/alignment/p02_repetition_penalty_vectors.json");

    let actual = sha256_file(path);
    assert_eq!(actual, target.sha256, "fixture hash mismatch");
}
