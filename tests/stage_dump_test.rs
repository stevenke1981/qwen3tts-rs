use candle_core::{Device, Tensor};
#[cfg(feature = "stage-dump")]
use qwen3tts::StageDumpMetadata;
use qwen3tts::StageDumpObserver;
#[cfg(feature = "stage-dump")]
use qwen3tts::alignment_stage_dump::{StageDumpManifest, StageDumpWriter};
#[cfg(feature = "stage-dump")]
use sha2::{Digest, Sha256};
#[cfg(feature = "stage-dump")]
use std::fs;
#[cfg(feature = "stage-dump")]
use std::process::Command;
#[cfg(feature = "stage-dump")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "stage-dump")]
fn temp_stage_dir(test: &str) -> std::path::PathBuf {
    let mut base = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_else(|_| 0);
    base.push(format!("qwen3tts-stage-dump-{test}-{ts}"));
    base
}

fn sample_tensor(values: &[f32]) -> Tensor {
    Tensor::from_slice(values, values.len(), &Device::Cpu).unwrap()
}

#[cfg(feature = "stage-dump")]
use std::path::Path;

#[cfg(feature = "stage-dump")]
#[test]
fn writer_manifest_contains_schema_fields_and_hashes() {
    let dir = temp_stage_dir("manifest");
    let metadata = StageDumpMetadata {
        source: "qwen3tts-rs".to_string(),
        revision: Some("unit-test".to_string()),
        model: "unit-model".to_string(),
        case_id: "case-a".to_string(),
        seed: Some(777),
    };
    let mut writer = StageDumpWriter::new(&dir, metadata).unwrap();

    let logits = sample_tensor(&[0.1, 0.2, 0.3, 0.4]);
    let code_matrix = sample_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let pcm = sample_tensor(&[-0.1, 0.0, 0.2, -0.3]);

    writer.on_talker_codebook0_logits(0, &logits).unwrap();
    writer.on_talker_final_codes(&code_matrix).unwrap();
    writer.on_code_predictor_step_logits(0, 0, &logits).unwrap();
    writer.on_code_predictor_step_logits(0, 1, &logits).unwrap();
    writer.on_code_predictor_final_codes(0, &logits).unwrap();
    writer.on_code_predictor_step_logits(1, 0, &logits).unwrap();
    writer.on_code_predictor_final_codes(1, &logits).unwrap();
    writer.on_codec_input_codes(&code_matrix).unwrap();
    writer.on_codec_output_pcm(&pcm).unwrap();
    writer.commit().unwrap();

    let manifest = fs::read_to_string(dir.join("manifest.json")).unwrap();
    let parsed: StageDumpManifest = serde_json::from_str(&manifest).unwrap();

    assert_eq!(parsed.schema_version, 1);
    assert_eq!(parsed.source, "qwen3tts-rs");
    assert_eq!(parsed.model, "unit-model");
    assert_eq!(parsed.case_id, "case-a");
    assert_eq!(parsed.revision.as_deref(), Some("unit-test"));
    assert_eq!(parsed.seed, Some(777));
    assert!(
        parsed
            .stages
            .iter()
            .all(|s| s.byte_order == "little-endian")
    );
    assert_eq!(parsed.stages.len(), 9);

    assert_eq!(
        parsed
            .stages
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "talker_codebook0_logits_0000",
            "talker_final_code_matrix",
            "code_predictor_step_logits_0000_0000",
            "code_predictor_step_logits_0000_0001",
            "code_predictor_final_code_matrix_0000",
            "code_predictor_step_logits_0001_0000",
            "code_predictor_final_code_matrix_0001",
            "codec_input_code_matrix",
            "codec_final_pcm",
        ]
    );
    assert!(parsed.stages.iter().all(|s| s.dtype == "f32"));
    assert!(
        parsed
            .stages
            .iter()
            .all(|s| matches!(s.sha256.chars().next(), Some(c) if c.is_ascii_hexdigit()))
    );

    for stage in &parsed.stages {
        assert!(stage.file.ends_with(".f32.bin"));
        assert!(!stage.layout.is_empty());
        let path = dir.join(&stage.file);
        let bytes = fs::read(&path).unwrap();
        let hash = Sha256::digest(&bytes);
        assert_eq!(stage.sha256, format!("{:x}", hash));
        if stage.name == "codec_final_pcm" {
            let expected = [
                (-0.1f32).to_le_bytes(),
                (0.0f32).to_le_bytes(),
                (0.2f32).to_le_bytes(),
                (-0.3f32).to_le_bytes(),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
            assert_eq!(bytes, expected);
        }
    }
}

#[cfg(feature = "stage-dump")]
#[test]
fn writer_orders_code_predictor_frames_without_duplicates() {
    let dir = temp_stage_dir("frame-order");
    let metadata = StageDumpMetadata {
        source: "qwen3tts-rs".to_string(),
        revision: None,
        model: "unit-model".to_string(),
        case_id: "case-b".to_string(),
        seed: Some(888),
    };
    let mut writer = StageDumpWriter::new(&dir, metadata).unwrap();

    let logits = sample_tensor(&[0.1, 0.2, 0.3]);
    writer.on_code_predictor_step_logits(0, 0, &logits).unwrap();
    writer.on_code_predictor_step_logits(0, 1, &logits).unwrap();
    writer.on_code_predictor_final_codes(0, &logits).unwrap();
    writer.on_code_predictor_step_logits(1, 0, &logits).unwrap();
    writer.on_code_predictor_step_logits(1, 1, &logits).unwrap();
    writer.on_code_predictor_final_codes(1, &logits).unwrap();

    writer.commit().unwrap();

    let manifest = fs::read_to_string(dir.join("manifest.json")).unwrap();
    let parsed: StageDumpManifest = serde_json::from_str(&manifest).unwrap();
    let names: Vec<&str> = parsed
        .stages
        .iter()
        .map(|stage| stage.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec![
            "code_predictor_step_logits_0000_0000",
            "code_predictor_step_logits_0000_0001",
            "code_predictor_final_code_matrix_0000",
            "code_predictor_step_logits_0001_0000",
            "code_predictor_step_logits_0001_0001",
            "code_predictor_final_code_matrix_0001",
        ]
    );
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), names.len());
}

#[cfg(feature = "stage-dump")]
#[test]
fn writer_rejects_duplicates_and_double_commit() {
    let dir = temp_stage_dir("duplicates");
    let metadata = StageDumpMetadata {
        source: "qwen3tts-rs".to_string(),
        revision: None,
        model: "unit-model".to_string(),
        case_id: "case-a".to_string(),
        seed: None,
    };
    let mut writer = StageDumpWriter::new(&dir, metadata.clone()).unwrap();
    assert!(StageDumpWriter::new(&dir, metadata).is_err());

    let logits = sample_tensor(&[0.1, 0.2, 0.3, 0.4]);
    writer.on_talker_codebook0_logits(0, &logits).unwrap();
    assert!(writer.on_talker_codebook0_logits(0, &logits).is_err());

    assert!(writer.commit().is_ok());
    assert!(writer.commit().is_err());
}

#[cfg(feature = "stage-dump")]
#[test]
fn writer_rejects_invalid_structured_identifiers() {
    let dir = temp_stage_dir("invalid-identifiers");
    let mut writer = StageDumpWriter::new(
        &dir,
        StageDumpMetadata {
            source: "test".into(),
            revision: None,
            model: "test".into(),
            case_id: "bad".into(),
            seed: None,
        },
    )
    .unwrap();
    let tensor = sample_tensor(&[1.0]);
    assert!(
        writer
            .on_stage("talker-prefill-lx-input-norm", &tensor, "BTH")
            .is_err()
    );
    assert!(
        writer
            .on_stage("code-predictor-step-frame0-l0-input-norm", &tensor, "BTH")
            .is_err()
    );
    assert!(writer.on_stage("talker-foo", &tensor, "BTH").is_err());
    assert!(
        writer
            .on_stage("talker-prefill-l0-bogus", &tensor, "BTH")
            .is_err()
    );
    assert!(
        writer
            .on_stage("code-predictor-step1-frame0", &tensor, "BTH")
            .is_err()
    );
    assert!(writer.on_stage("next-emb-stepX", &tensor, "BTH").is_err());
    assert!(writer.on_stage("next-emb-bogus", &tensor, "BTH").is_err());
    assert!(writer.on_stage("talker-input-embed", &tensor, "").is_err());
    assert!(
        writer
            .on_stage("talker-input-embed", &tensor, "bogus")
            .is_err()
    );
    for invalid in [
        "talker-input-garbage",
        "talker-step1-l0-input-norm-extra",
        "talker-logits-step1-extra",
        "code-predictor-step1-frame0-l0-input-norm-extra",
    ] {
        assert!(
            writer.on_stage(invalid, &tensor, "BTH").is_err(),
            "{invalid} must be rejected"
        );
    }
}

#[cfg(not(feature = "stage-dump"))]
#[test]
fn default_build_noop_observer_never_touches_disk() {
    use qwen3tts::NoopStageDumpObserver;

    let mut observer = NoopStageDumpObserver::default();
    let t = sample_tensor(&[1.0, 2.0, 3.0, 4.0]);

    assert!(observer.on_talker_codebook0_logits(0, &t).is_ok());
    assert!(observer.on_code_predictor_step_logits(0, 0, &t).is_ok());
    assert!(observer.on_talker_final_codes(&t).is_ok());
    assert!(observer.commit().is_ok());
}

#[cfg(feature = "stage-dump")]
fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(feature = "stage-dump")]
fn require_python_command() -> String {
    if command_exists("python") {
        return "python".to_string();
    }
    if command_exists("python3") {
        return "python3".to_string();
    }
    panic!("python command not available; compare_stage_dumps requires python");
}

#[cfg(feature = "stage-dump")]
fn write_compat_manifests(dir_a: &Path, dir_b: &Path) -> (String, String) {
    let metadata = StageDumpMetadata {
        source: "qwen3tts-rs".to_string(),
        revision: None,
        model: "unit-model".to_string(),
        case_id: "case-compare".to_string(),
        seed: None,
    };
    let t = sample_tensor(&[0.5, 0.6, 0.7, 0.8]);
    let pcm = sample_tensor(&[0.1, -0.1, 0.2]);

    let mut left = StageDumpWriter::new(dir_a, metadata.clone()).unwrap();
    left.on_codec_input_codes(&t).unwrap();
    left.on_codec_output_pcm(&pcm).unwrap();
    left.commit().unwrap();

    let mut right = StageDumpWriter::new(dir_b, metadata).unwrap();
    right.on_codec_input_codes(&t).unwrap();
    right.on_codec_output_pcm(&pcm).unwrap();
    right.commit().unwrap();

    (
        dir_a.join("manifest.json").to_string_lossy().to_string(),
        dir_b.join("manifest.json").to_string_lossy().to_string(),
    )
}

#[cfg(feature = "stage-dump")]
#[test]
fn compare_tool_reports_identity_and_regression() {
    let left = temp_stage_dir("compare-left");
    let right = temp_stage_dir("compare-right");
    let (left_path, right_path) = write_compat_manifests(&left, &right);
    let python_cmd = require_python_command();

    let ok = Command::new(&python_cmd)
        .args([
            "tools/compare_stage_dumps.py",
            &left_path,
            &right_path,
            "--min-cosine",
            "0.999",
        ])
        .status()
        .unwrap();
    assert!(ok.success());

    let mut changed: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&right_path).unwrap()).unwrap();
    let stages = changed
        .get_mut("stages")
        .and_then(|value| value.as_array_mut())
        .expect("stages must be an array");
    stages[0]["sha256"] = serde_json::Value::String("00badcafe".to_string());
    fs::write(
        right.join("manifest.json"),
        serde_json::to_string_pretty(&changed).unwrap(),
    )
    .unwrap();

    let bad = Command::new(&python_cmd)
        .args([
            "tools/compare_stage_dumps.py",
            &left_path,
            &right_path,
            "--min-cosine",
            "0.999",
        ])
        .status()
        .unwrap();
    assert!(!bad.success());
}
