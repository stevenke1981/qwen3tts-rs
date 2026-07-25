use candle_core::{Device, Tensor};
use qwen3tts::talker::sampling::{Sampler, SamplingOptions};
use qwen3tts::text_frontend::model_catalog::{
    GenerationSamplingConfig, resolve_effective_sampling_plan,
};
use serde::Deserialize;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const FIXTURE_PATH: &str = "fixtures/alignment/p02_sampling_config_matrix.json";
const FIXTURE_REVISION: &str = "82cd05b9f3a175612dc89fd6943e610fab096ef5";

#[derive(Debug, Deserialize)]
struct SamplingConfigFixture {
    version: u32,
    fixture_id: String,
    source_revision: String,
    cases: Vec<SamplingCase>,
    models: Vec<ModelProvenance>,
}

#[derive(Debug, Deserialize)]
struct ModelProvenance {
    model_id: String,
    model_revision: String,
    generation_config: GenerationConfigProvenance,
}

#[derive(Debug, Deserialize)]
struct GenerationConfigProvenance {
    path: String,
    sha256: String,
    raw_bytes: String,
    resolved: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct SamplingCase {
    name: String,
    generation_config: serde_json::Value,
    synthesis_options: SynthesisOptions,
    expected: Option<ExpectedCase>,
    expect_error: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct SynthesisOptions {
    temperature: f64,
    top_k: usize,
    top_p: f64,
}

#[derive(Debug, Deserialize)]
struct ExpectedCase {
    route: String,
    talker: ExpectedBranch,
    subtalker: ExpectedBranch,
}

#[derive(Debug, Deserialize)]
struct ExpectedBranch {
    do_sample: bool,
    options: ExpectedSamplingOptions,
}

#[derive(Debug, Deserialize)]
struct ExpectedSamplingOptions {
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repetition_penalty: f64,
}

#[derive(Debug, Deserialize)]
struct ManifestFixture {
    id: String,
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    fixtures: Vec<ManifestFixture>,
}

fn load_matrix_fixture() -> SamplingConfigFixture {
    let text = fs::read_to_string(FIXTURE_PATH)
        .unwrap_or_else(|err| panic!("FIXTURE_MISSING: read matrix fixture failed: {err}"));
    serde_json::from_str(&text).expect("failed to parse p02 sampling config matrix fixture")
}

fn load_config_fixtures_manifest() -> Manifest {
    let text = fs::read_to_string("config/fixtures.json")
        .unwrap_or_else(|err| panic!("FIXTURE_MISSING: read config/fixtures.json failed: {err}"));
    serde_json::from_str(&text).expect("failed to parse config/fixtures.json")
}

fn write_generation_config(json: &str) -> PathBuf {
    let marker = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("qwen3tts-sampling-config-matrix-test-{marker}"));
    fs::create_dir_all(&dir)
        .unwrap_or_else(|err| panic!("failed to create temp config dir {}: {err}", dir.display()));
    let path = dir.join("generation_config.json");
    fs::write(&path, json).unwrap_or_else(|err| {
        panic!(
            "failed to write generation config {}: {err}",
            path.display()
        )
    });
    path
}

fn synthesize_route(cfg: &GenerationSamplingConfig, options: &SynthesisOptions) -> &'static str {
    let plan = resolve_effective_sampling_plan(
        *cfg,
        options.temperature,
        options.top_k as u32,
        options.top_p,
    )
    .expect("valid synthesis options");
    if plan.uses_explicit_sampler() {
        "generate_sampled"
    } else {
        "generate"
    }
}

#[test]
fn five_public_models_have_immutable_generation_config_provenance() {
    let fixture = load_matrix_fixture();
    assert_eq!(fixture.models.len(), 5);
    let expected = [
        (
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "5d83992436eae1d760afd27aff78a71d676296fc",
        ),
        (
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "85e237c12c027371202489a0ec509ded67b5e4b5",
        ),
        (
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "fd4b254389122332181a7c3db7f27e918eec64e3",
        ),
        (
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "0c0e3051f131929182e2c023b9537f8b1c68adfe",
        ),
        (
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "5ecdb67327fd37bb2e042aab12ff7391903235d3",
        ),
    ];
    for model in fixture.models {
        assert!(
            expected.iter().any(|(id, revision)| {
                *id == model.model_id && *revision == model.model_revision
            })
        );
        assert!(!model.model_id.is_empty());
        assert_eq!(model.model_revision.len(), 40);
        assert_eq!(model.generation_config.path, "generation_config.json");
        assert_eq!(
            model.generation_config.sha256,
            "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa"
        );
        use sha2::{Digest, Sha256};
        let digest = format!(
            "{:x}",
            Sha256::digest(model.generation_config.raw_bytes.as_bytes())
        );
        assert_eq!(digest, model.generation_config.sha256);
        assert!(model.generation_config.resolved.get("talker").is_some());
        assert!(model.generation_config.resolved.get("subtalker").is_some());
        assert_eq!(
            model.generation_config.resolved["talker"]["do_sample"],
            true
        );
        assert_eq!(
            model.generation_config.resolved["subtalker"]["do_sample"],
            true
        );
        assert_eq!(
            model.generation_config.resolved["talker"]["options"]["repetition_penalty"],
            1.05
        );
        assert_eq!(
            model.generation_config.resolved["subtalker"]["options"]["repetition_penalty"],
            1.0
        );
    }
}

#[test]
fn four_mode_cross_product_preserves_shared_philox_order() {
    let logits = Tensor::new(&[0.1_f32, 0.9_f32], &Device::Cpu).expect("logits");
    let sampled = SamplingOptions {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.5,
    };
    for talker_sample in [false, true] {
        for subtalker_sample in [false, true] {
            let mut sampler = Sampler::new(7);
            // Production Talker passes accumulated c0 history; CP deliberately
            // receives an empty history for every codebook 1..15 call.
            sampler
                .sample_with_mode(&logits, sampled, talker_sample, None, None, &[0_u16])
                .expect("c0 sample");
            for _ in 0..15 {
                sampler
                    .sample_with_mode(&logits, sampled, subtalker_sample, None, None, &[])
                    .expect("subtalker sample");
            }
            let expected = u64::from(talker_sample) + 15 * u64::from(subtalker_sample);
            assert_eq!(sampler.subsequence_counter(), expected);
        }
    }
}

#[test]
fn c0_history_penalty_is_observable_while_cp_history_stays_empty() {
    let logits = Tensor::new(&[0.0_f32, 2.0], &Device::Cpu).expect("logits");
    let options = SamplingOptions {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 2.0,
    };
    let mut sampler = Sampler::new(9);
    let token = sampler
        .sample_with_mode(&logits, options, false, None, None, &[1_u16])
        .expect("greedy c0");
    assert_eq!(token, 1);
    let cp_options = SamplingOptions {
        temperature: 1.0,
        ..options
    };
    sampler
        .sample_with_mode(&logits, cp_options, true, None, None, &[])
        .expect("sampled cp");
    let diagnostics = sampler.diagnostics();
    assert_eq!(diagnostics.history_nonempty_calls, 1);
    assert_eq!(diagnostics.call_count, 2);
}

#[test]
fn effective_plan_rejects_non_finite_and_invalid_public_overrides() {
    let cfg = GenerationSamplingConfig::default();
    for temperature in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(resolve_effective_sampling_plan(cfg, temperature, 50, 1.0).is_err());
    }
    for top_p in [0.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(resolve_effective_sampling_plan(cfg, 0.9, 50, top_p).is_err());
    }
    let greedy = resolve_effective_sampling_plan(cfg, 0.0, 0, 1.0).expect("greedy override");
    assert!(!greedy.talker.do_sample);
    assert!(greedy.subtalker.do_sample);
}

#[test]
fn sampling_config_matrix_matches_effective_contract() {
    let fixture = load_matrix_fixture();
    assert_eq!(fixture.version, 1);
    assert_eq!(fixture.fixture_id, "p02-sampling-config-matrix");
    assert_eq!(fixture.source_revision, FIXTURE_REVISION);

    for case in fixture.cases {
        let expect_error = case.expect_error.unwrap_or(false);
        let generation_config = serde_json::to_string_pretty(&case.generation_config)
            .expect("serialize generation config");
        let path = write_generation_config(&generation_config);
        let parsed = GenerationSamplingConfig::from_path(&path);

        if expect_error {
            assert!(
                parsed.is_err(),
                "{} should fail with expect_error=true",
                case.name
            );
            continue;
        }

        let cfg = parsed.expect("valid config should parse");
        let expected = case
            .expected
            .unwrap_or_else(|| panic!("missing expected in case {}", case.name));

        assert_eq!(
            cfg.talker.do_sample, expected.talker.do_sample,
            "{}",
            case.name
        );
        assert_eq!(
            cfg.subtalker.do_sample, expected.subtalker.do_sample,
            "{}",
            case.name
        );
        let plan = resolve_effective_sampling_plan(
            cfg,
            case.synthesis_options.temperature,
            case.synthesis_options.top_k as u32,
            case.synthesis_options.top_p,
        )
        .expect("valid effective plan");
        assert_eq!(
            synthesize_route(&cfg, &case.synthesis_options),
            expected.route,
            "{}",
            case.name
        );
        let actual_talker = plan.talker.options;
        assert_eq!(
            actual_talker,
            SamplingOptions {
                temperature: expected.talker.options.temperature,
                top_k: expected.talker.options.top_k,
                top_p: expected.talker.options.top_p,
                repetition_penalty: expected.talker.options.repetition_penalty,
            },
            "{}",
            case.name
        );

        let actual_subtalker = plan.subtalker.options;
        assert_eq!(
            actual_subtalker,
            SamplingOptions {
                temperature: expected.subtalker.options.temperature,
                top_k: expected.subtalker.options.top_k,
                top_p: expected.subtalker.options.top_p,
                repetition_penalty: expected.subtalker.options.repetition_penalty,
            },
            "{}",
            case.name
        );
    }
}

#[test]
fn malformed_generation_config_json_is_rejected() {
    let marker = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("qwen3tts-sampling-config-malformed-{marker}"));
    fs::create_dir_all(&dir)
        .unwrap_or_else(|err| panic!("failed to create temp config dir {}: {err}", dir.display()));
    let path = dir.join("generation_config.json");
    fs::write(&path, "{not-json}")
        .unwrap_or_else(|err| panic!("failed to write malformed config {}: {err}", path.display()));

    assert!(
        GenerationSamplingConfig::from_path(&path).is_err(),
        "malformed generation config should be rejected"
    );
}

#[test]
fn missing_generation_config_uses_model_default() {
    let marker = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("qwen3tts-sampling-config-missing-config-{marker}"));
    fs::create_dir_all(&dir)
        .unwrap_or_else(|err| panic!("failed to create temp model dir {}: {err}", dir.display()));

    let cfg = GenerationSamplingConfig::from_model_dir(&dir)
        .expect("missing generation_config.json should default");
    let defaults = GenerationSamplingConfig::default();
    assert_eq!(cfg, defaults);
}

#[test]
fn sampling_config_matrix_is_hashed_in_manifest() {
    let manifest = load_config_fixtures_manifest();
    let target = manifest
        .fixtures
        .iter()
        .find(|item| item.id == "p02-sampling-config-matrix")
        .expect("missing p02-sampling-config-matrix entry in config/fixtures.json");

    assert!(
        target
            .path
            .ends_with("fixtures/alignment/p02_sampling_config_matrix.json")
    );
    let actual_sha = compute_sha256(FIXTURE_PATH);
    assert_eq!(
        actual_sha, target.sha256,
        "p02 sampling matrix fixture hash mismatch"
    );
}

fn compute_sha256(path: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut file = fs::File::open(Path::new(path))
        .unwrap_or_else(|err| panic!("failed to open fixture file {path}: {err}"));
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buffer)
            .unwrap_or_else(|err| panic!("failed to read fixture file {path}: {err}"));
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    format!("{:x}", hasher.finalize())
}
