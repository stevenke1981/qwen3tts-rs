//! Mutation tests proving each comparator metric and missing-stage check can fail.
//!
//! These tests use synthetic stage dumps (no real model) to verify the
//! comparator's fail-closed behavior for every threshold and check added in
//! P03-T05.

#![cfg(feature = "stage-dump")]

use candle_core::{Device, Tensor};
use qwen3tts::alignment_stage_dump::{StageDumpMetadata, StageDumpWriter};
use qwen3tts::StageDumpObserver;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(tag: &str) -> PathBuf {
    let mut base = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    base.push(format!("qwen3tts-gate-{tag}-{ts}"));
    base
}

fn python_cmd() -> String {
    for name in ["python", "python3"] {
        if Command::new(name)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return name.to_string();
        }
    }
    panic!("python not available")
}

fn run_comparator(args: &[&str]) -> (bool, String) {
    let py = python_cmd();
    let mut cmd_args = vec!["tools/compare_stage_dumps.py"];
    cmd_args.extend_from_slice(args);
    let output = Command::new(&py)
        .args(&cmd_args)
        .output()
        .expect("failed to run comparator");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    (output.status.success(), stdout)
}

fn write_stage(dir: &Path, name: &str, values: &[f32], layout: &str) {
    let metadata = StageDumpMetadata {
        source: "test".into(),
        revision: None,
        model: "test".into(),
        case_id: "gate".into(),
        seed: None,
    };
    let mut w = StageDumpWriter::new(dir, metadata).unwrap();
    let t = Tensor::from_slice(values, values.len(), &Device::Cpu).unwrap();
    w.on_stage(name, &t, layout).unwrap();
    w.commit().unwrap();
}

fn write_two_stages(
    dir: &Path,
    name1: &str,
    vals1: &[f32],
    name2: &str,
    vals2: &[f32],
    layout: &str,
) {
    let metadata = StageDumpMetadata {
        source: "test".into(),
        revision: None,
        model: "test".into(),
        case_id: "gate".into(),
        seed: None,
    };
    let mut w = StageDumpWriter::new(dir, metadata).unwrap();
    let t1 = Tensor::from_slice(vals1, vals1.len(), &Device::Cpu).unwrap();
    let t2 = Tensor::from_slice(vals2, vals2.len(), &Device::Cpu).unwrap();
    w.on_stage(name1, &t1, layout).unwrap();
    w.on_stage(name2, &t2, layout).unwrap();
    w.commit().unwrap();
}

/// Identity comparison passes.
#[test]
fn identity_passes() {
    let dir = temp_dir("identity");
    write_stage(&dir, "talker-input-embed", &[1.0, 2.0, 3.0, 4.0], "BTH");
    let manifest = dir.join("manifest.json");
    let (ok, _) = run_comparator(&[
        manifest.to_str().unwrap(),
        manifest.to_str().unwrap(),
        "--min-cosine",
        "0.999",
    ]);
    assert!(ok, "identity comparison must pass");
}

/// Cosine below threshold fails.
#[test]
fn cosine_below_threshold_fails() {
    let ref_dir = temp_dir("cos-ref");
    let cand_dir = temp_dir("cos-cand");
    write_stage(&ref_dir, "talker-input-embed", &[1.0, 0.0, 0.0, 0.0], "BTH");
    write_stage(
        &cand_dir,
        "talker-input-embed",
        &[0.0, 1.0, 0.0, 0.0],
        "BTH",
    );
    let (ok, out) = run_comparator(&[
        ref_dir.join("manifest.json").to_str().unwrap(),
        cand_dir.join("manifest.json").to_str().unwrap(),
        "--min-cosine",
        "0.999",
    ]);
    assert!(!ok, "orthogonal vectors must fail cosine check");
    assert!(
        out.contains("\"passed\": false"),
        "report must show failure"
    );
}

/// max-abs above threshold fails.
#[test]
fn max_abs_above_threshold_fails() {
    let ref_dir = temp_dir("abs-ref");
    let cand_dir = temp_dir("abs-cand");
    write_stage(&ref_dir, "talker-input-embed", &[1.0, 2.0, 3.0], "BTH");
    write_stage(&cand_dir, "talker-input-embed", &[1.0, 2.0, 3.002], "BTH");
    let (ok, _) = run_comparator(&[
        ref_dir.join("manifest.json").to_str().unwrap(),
        cand_dir.join("manifest.json").to_str().unwrap(),
        "--min-cosine",
        "0.999",
        "--max-abs",
        "0.001",
    ]);
    assert!(!ok, "max_abs 0.002 > 0.001 threshold must fail");
}

/// max-abs within threshold passes.
#[test]
fn max_abs_within_threshold_passes() {
    let ref_dir = temp_dir("abs-ok-ref");
    let cand_dir = temp_dir("abs-ok-cand");
    write_stage(&ref_dir, "talker-input-embed", &[1.0, 2.0, 3.0], "BTH");
    write_stage(&cand_dir, "talker-input-embed", &[1.0, 2.0, 3.0005], "BTH");
    let (ok, _) = run_comparator(&[
        ref_dir.join("manifest.json").to_str().unwrap(),
        cand_dir.join("manifest.json").to_str().unwrap(),
        "--min-cosine",
        "0.999",
        "--max-abs",
        "0.001",
    ]);
    assert!(ok, "max_abs 0.0005 <= 0.001 must pass");
}

/// Missing candidate stage fails.
#[test]
fn missing_candidate_stage_fails() {
    let ref_dir = temp_dir("miss-ref");
    let cand_dir = temp_dir("miss-cand");
    write_two_stages(
        &ref_dir,
        "talker-input-embed",
        &[1.0, 2.0],
        "talker-hidden-prefill-final",
        &[3.0, 4.0],
        "BTH",
    );
    write_stage(&cand_dir, "talker-input-embed", &[1.0, 2.0], "BTH");
    let (ok, out) = run_comparator(&[
        ref_dir.join("manifest.json").to_str().unwrap(),
        cand_dir.join("manifest.json").to_str().unwrap(),
        "--min-cosine",
        "0.999",
    ]);
    assert!(!ok, "missing candidate stage must fail");
    assert!(
        out.contains("missing"),
        "report must mention missing stage, got: {out}"
    );
}

/// Logit top-1 mismatch fails with --logit-top1-exact.
#[test]
fn logit_top1_mismatch_fails() {
    let ref_dir = temp_dir("logit-ref");
    let cand_dir = temp_dir("logit-cand");
    // ref: token 2 is highest; cand: token 0 is highest
    write_stage(
        &ref_dir,
        "talker-logits-prefill",
        &[0.1, 0.2, 0.9, 0.3],
        "BV",
    );
    write_stage(
        &cand_dir,
        "talker-logits-prefill",
        &[0.9, 0.2, 0.1, 0.3],
        "BV",
    );
    let (ok, out) = run_comparator(&[
        ref_dir.join("manifest.json").to_str().unwrap(),
        cand_dir.join("manifest.json").to_str().unwrap(),
        "--min-cosine",
        "0.0",
        "--logit-top1-exact",
    ]);
    assert!(!ok, "top-1 mismatch must fail with --logit-top1-exact");
    assert!(
        out.contains("top1_match"),
        "report must contain logit metrics"
    );
}

/// Logit top-1 match passes with --logit-top1-exact.
#[test]
fn logit_top1_match_passes() {
    let ref_dir = temp_dir("logit-ok-ref");
    let cand_dir = temp_dir("logit-ok-cand");
    write_stage(
        &ref_dir,
        "talker-logits-prefill",
        &[0.1, 0.2, 0.9, 0.3],
        "BV",
    );
    write_stage(
        &cand_dir,
        "talker-logits-prefill",
        &[0.1, 0.2, 0.85, 0.3],
        "BV",
    );
    let (ok, _) = run_comparator(&[
        ref_dir.join("manifest.json").to_str().unwrap(),
        cand_dir.join("manifest.json").to_str().unwrap(),
        "--min-cosine",
        "0.0",
        "--logit-top1-exact",
    ]);
    assert!(ok, "top-1 match must pass");
}

/// Logit top-5 overlap below threshold fails.
#[test]
fn logit_top5_overlap_below_threshold_fails() {
    let ref_dir = temp_dir("top5-ref");
    let cand_dir = temp_dir("top5-cand");
    // ref top-5: {4,3,2,1,0}; cand top-5: {9,8,7,6,5} — zero overlap
    let mut ref_vals = vec![0.0f32; 10];
    ref_vals[4] = 1.0;
    ref_vals[3] = 0.9;
    ref_vals[2] = 0.8;
    ref_vals[1] = 0.7;
    ref_vals[0] = 0.6;
    let mut cand_vals = vec![0.0f32; 10];
    cand_vals[9] = 1.0;
    cand_vals[8] = 0.9;
    cand_vals[7] = 0.8;
    cand_vals[6] = 0.7;
    cand_vals[5] = 0.6;
    write_stage(&ref_dir, "talker-logits-prefill", &ref_vals, "BV");
    write_stage(&cand_dir, "talker-logits-prefill", &cand_vals, "BV");
    let (ok, _) = run_comparator(&[
        ref_dir.join("manifest.json").to_str().unwrap(),
        cand_dir.join("manifest.json").to_str().unwrap(),
        "--min-cosine",
        "0.0",
        "--min-logit-top5-overlap",
        "0.8",
    ]);
    assert!(!ok, "zero top-5 overlap must fail with threshold 0.8");
}

/// Expected stage count mismatch fails.
#[test]
fn expected_stage_count_mismatch_fails() {
    let dir = temp_dir("count");
    write_stage(&dir, "talker-input-embed", &[1.0, 2.0], "BTH");
    let manifest = dir.join("manifest.json");
    let (ok, out) = run_comparator(&[
        manifest.to_str().unwrap(),
        manifest.to_str().unwrap(),
        "--min-cosine",
        "0.999",
        "--expected-stages",
        "723",
    ]);
    assert!(!ok, "stage count 1 != 723 must fail");
    assert!(
        out.contains("stage count"),
        "report must mention stage count"
    );
}

/// SHA-256 mismatch fails (existing behavior preserved).
#[test]
fn sha256_mismatch_fails() {
    let ref_dir = temp_dir("sha-ref");
    let cand_dir = temp_dir("sha-cand");
    write_stage(&ref_dir, "talker-input-embed", &[1.0, 2.0], "BTH");
    write_stage(&cand_dir, "talker-input-embed", &[1.0, 2.0], "BTH");
    // Corrupt the candidate manifest sha256
    let manifest_path = cand_dir.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["stages"][0]["sha256"] = serde_json::Value::String("0".repeat(64));
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let (ok, _) = run_comparator(&[
        ref_dir.join("manifest.json").to_str().unwrap(),
        manifest_path.to_str().unwrap(),
        "--min-cosine",
        "0.999",
    ]);
    assert!(!ok, "sha256 mismatch must fail");
}

/// Mapping-based comparison works for qwentts.cpp anchor stages.
#[test]
fn mapping_comparison_works() {
    let ref_dir = temp_dir("map-ref");
    let cand_dir = temp_dir("map-cand");
    let map_path = temp_dir("map-json");

    // Write a qwentts.cpp-style reference manifest manually (underscore suffix
    // names are not valid Candle structured identifiers, so we bypass the writer).
    fs::create_dir_all(&ref_dir).unwrap();
    let ref_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let ref_file = ref_dir.join("talker-input-embed_0012_0000.f32.bin");
    let ref_bytes: Vec<u8> = ref_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(&ref_file, &ref_bytes).unwrap();
    let ref_sha = format!("{:x}", Sha256::digest(&ref_bytes));
    let ref_manifest = serde_json::json!({
        "schema_version": 1,
        "source": "qwentts.cpp",
        "revision": null,
        "model": "test",
        "case_id": "gate",
        "seed": null,
        "stages": [{
            "name": "talker-input-embed_0012",
            "dtype": "f32",
            "shape": [4],
            "file": "talker-input-embed_0012_0000.f32.bin",
            "sha256": ref_sha,
            "layout": "C",
            "byte_order": "little-endian"
        }]
    });
    fs::write(
        ref_dir.join("manifest.json"),
        serde_json::to_string_pretty(&ref_manifest).unwrap(),
    )
    .unwrap();

    // Candle-style candidate (same values, different name)
    write_stage(
        &cand_dir,
        "talker-input-embed",
        &[1.0, 2.0, 3.0, 4.0],
        "BTH",
    );

    fs::create_dir_all(&map_path).unwrap();
    let map_file = map_path.join("map.json");
    fs::write(
        &map_file,
        serde_json::json!({
            "mappings": {"talker-input-embed_0012": "talker-input-embed"},
            "logit_stages": [],
            "unmapped": {}
        })
        .to_string(),
    )
    .unwrap();

    let (ok, out) = run_comparator(&[
        ref_dir.join("manifest.json").to_str().unwrap(),
        cand_dir.join("manifest.json").to_str().unwrap(),
        "--min-cosine",
        "0.999",
        "--mapping",
        map_file.to_str().unwrap(),
    ]);
    assert!(
        ok,
        "mapping comparison with identical values must pass: {out}"
    );
}

/// Mapping with missing candidate stage fails.
#[test]
fn mapping_missing_candidate_fails() {
    let ref_dir = temp_dir("mapmiss-ref");
    let cand_dir = temp_dir("mapmiss-cand");
    let map_path = temp_dir("mapmiss-json");

    // Write a qwentts.cpp-style reference manifest manually
    fs::create_dir_all(&ref_dir).unwrap();
    let ref_vals: Vec<f32> = vec![1.0, 2.0];
    let ref_file = ref_dir.join("talker-input-embed_0012_0000.f32.bin");
    let ref_bytes: Vec<u8> = ref_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(&ref_file, &ref_bytes).unwrap();
    let ref_sha = format!("{:x}", Sha256::digest(&ref_bytes));
    let ref_manifest = serde_json::json!({
        "schema_version": 1,
        "source": "qwentts.cpp",
        "revision": null,
        "model": "test",
        "case_id": "gate",
        "seed": null,
        "stages": [{
            "name": "talker-input-embed_0012",
            "dtype": "f32",
            "shape": [2],
            "file": "talker-input-embed_0012_0000.f32.bin",
            "sha256": ref_sha,
            "layout": "C",
            "byte_order": "little-endian"
        }]
    });
    fs::write(
        ref_dir.join("manifest.json"),
        serde_json::to_string_pretty(&ref_manifest).unwrap(),
    )
    .unwrap();

    write_stage(&cand_dir, "talker-hidden-prefill-final", &[1.0, 2.0], "BTH");

    fs::create_dir_all(&map_path).unwrap();
    let map_file = map_path.join("map.json");
    fs::write(
        &map_file,
        serde_json::json!({
            "mappings": {"talker-input-embed_0012": "talker-input-embed"},
            "logit_stages": [],
            "unmapped": {}
        })
        .to_string(),
    )
    .unwrap();

    let (ok, out) = run_comparator(&[
        ref_dir.join("manifest.json").to_str().unwrap(),
        cand_dir.join("manifest.json").to_str().unwrap(),
        "--min-cosine",
        "0.999",
        "--mapping",
        map_file.to_str().unwrap(),
    ]);
    assert!(!ok, "mapping with missing candidate stage must fail");
    assert!(
        out.contains("missing"),
        "report must mention missing stage, got: {out}"
    );
}
