use qwen3tts::talker::sampling::SamplingOptions;
use qwen3tts::text_frontend::model_catalog::GenerationSamplingConfig;
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

fn effective_talker(
    cfg: &GenerationSamplingConfig,
    overrides: &SynthesisOptions,
) -> SamplingOptions {
    let mut talker = cfg.talker.options;
    talker.temperature = overrides.temperature;
    talker.top_k = overrides.top_k;
    talker.top_p = overrides.top_p;
    talker
}

fn effective_subtalker(cfg: &GenerationSamplingConfig) -> SamplingOptions {
    cfg.subtalker.options
}

fn synthesize_route(cfg: &GenerationSamplingConfig) -> &'static str {
    if cfg.talker.do_sample {
        "generate_sampled"
    } else {
        "generate"
    }
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
        assert_eq!(synthesize_route(&cfg), expected.route, "{}", case.name);

        let actual_talker = effective_talker(&cfg, &case.synthesis_options);
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

        let actual_subtalker = effective_subtalker(&cfg);
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
