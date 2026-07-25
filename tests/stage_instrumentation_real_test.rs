#![cfg(feature = "stage-dump")]
use candle_core::Device;
use qwen3tts::StageDumpObserver;
use qwen3tts::alignment_stage_dump::{StageDumpManifest, StageDumpMetadata, StageDumpWriter};
use qwen3tts::talker::sampling::{Sampler, SamplingOptions};
use qwen3tts::talker::{InputBuilder, TalkerConfig, TalkerWeightLoader};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Deserialize)]
struct S {
    do_sample: bool,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repetition_penalty: f64,
}
#[derive(Deserialize)]
struct F {
    snapshot_revision: String,
    official_qwen_revision: String,
    qwentts_cpp_revision: String,
    prompt_ids: Vec<Vec<u32>>,
    language: String,
    seed: u64,
    talker: S,
    subtalker: S,
}
fn opts(s: &S) -> SamplingOptions {
    SamplingOptions {
        temperature: s.temperature,
        top_k: s.top_k,
        top_p: s.top_p,
        repetition_penalty: s.repetition_penalty,
    }
}
fn model_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("QWEN3_TTS_REAL_MODEL_DIR")
            .expect("FIXTURE_MISSING: QWEN3_TTS_REAL_MODEL_DIR"),
    )
}
fn run<O: StageDumpObserver>(
    talker: &qwen3tts::talker::TalkerForConditionalGeneration,
    o: &mut O,
    f: &F,
    d: &Device,
) -> Vec<Vec<u32>> {
    let b = InputBuilder::new(talker, d);
    let (i, m, t, p) = b
        .build(&f.prompt_ids[0], None, &f.language, None)
        .expect("InputBuilder");
    let mut s = Sampler::new(f.seed);
    let out = talker
        .generate_sampled_with_observer(
            &i,
            Some(&m),
            Some(&t),
            Some(&p),
            2,
            d,
            &mut s,
            opts(&f.talker),
            opts(&f.subtalker),
            f.talker.do_sample,
            f.subtalker.do_sample,
            o,
        )
        .expect("generate");
    out.to_vec2().expect("codes")
}
#[test]
#[ignore = "loads pinned 0.6B model"]
fn pinned_real_stage_manifest_is_complete_and_noop_parity_holds() {
    let d = Device::Cpu;
    let dir = model_dir();
    let expected_dir = PathBuf::from(
        r"C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc",
    );
    assert_eq!(
        dir.canonicalize().unwrap(),
        expected_dir.canonicalize().unwrap()
    );
    for n in ["config.json", "generation_config.json", "model.safetensors"] {
        assert!(
            dir.join(n).exists(),
            "FIXTURE_MISSING: {}",
            dir.join(n).display()
        );
    }
    let f: F = serde_json::from_str(
        &fs::read_to_string("fixtures/alignment/p02_deterministic_token_sequences_real.json")
            .expect("FIXTURE_MISSING: real fixture"),
    )
    .unwrap();
    assert_eq!(
        f.snapshot_revision,
        "5d83992436eae1d760afd27aff78a71d676296fc"
    );
    assert_eq!(
        f.official_qwen_revision,
        "022e286b98fbec7e1e916cb940cdf532cd9f488e"
    );
    assert_eq!(
        f.qwentts_cpp_revision,
        "82cd05b9f3a175612dc89fd6943e610fab096ef5"
    );
    assert!(dir.to_string_lossy().contains(&f.snapshot_revision));
    let talker = TalkerWeightLoader::from_safetensors(dir.join("model.safetensors"), &d)
        .unwrap()
        .build_talker(&TalkerConfig::default())
        .unwrap();
    let mut base = qwen3tts::NoopStageDumpObserver;
    let expected = run(&talker, &mut base, &f, &d);
    let mut out = std::env::temp_dir();
    out.push(format!(
        "qwen3tts-p03-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut w = StageDumpWriter::new(
        &out,
        StageDumpMetadata {
            source: "qwen3tts-rs".into(),
            revision: Some(f.snapshot_revision.clone()),
            model: "Qwen3-TTS-12Hz-0.6B-Base".into(),
            case_id: "p03-t01-real".into(),
            seed: Some(f.seed as i64),
        },
    )
    .unwrap();
    let actual = run(&talker, &mut w, &f, &d);
    assert_eq!(actual, expected);
    w.commit().unwrap();
    let m: StageDumpManifest =
        serde_json::from_str(&fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m.source, "qwen3tts-rs");
    assert_eq!(m.model, "Qwen3-TTS-12Hz-0.6B-Base");
    assert_eq!(m.case_id, "p03-t01-real");
    assert_eq!(m.seed, Some(f.seed as i64));
    assert_eq!(m.revision.as_deref(), Some(f.snapshot_revision.as_str()));
    assert_eq!(m.stages.len(), 723);
    let mut names = std::collections::HashSet::new();
    for s in &m.stages {
        assert!(names.insert(&s.name));
        assert!(!s.shape.is_empty());
        assert_eq!(s.byte_order, "little-endian");
        assert_eq!(s.sha256.len(), 64);
        let bytes = fs::read(out.join(&s.file)).unwrap();
        assert_eq!(
            bytes.len(),
            s.shape.iter().product::<usize>() * std::mem::size_of::<f32>()
        );
        assert_eq!(s.sha256, format!("{:x}", Sha256::digest(&bytes)));
        let expected_layout = if s.name.contains("position-ids") {
            if s.name.starts_with("talker-") {
                "ABT"
            } else {
                "BT"
            }
        } else if s.name.contains("rope-cos") || s.name.contains("rope-sin") {
            "BBTH"
        } else if s.name.starts_with("talker-logits-") {
            "BV"
        } else if s.name.contains('_') {
            "C"
        } else {
            "BTH"
        };
        assert_eq!(s.layout, expected_layout, "layout mismatch for {}", s.name);
    }
    let count = |p: &str| m.stages.iter().filter(|s| s.name == p).count();
    let subs = [
        "input-norm",
        "attention-output",
        "post-attention-norm",
        "mlp-output",
    ];
    for phase in ["prefill", "step1"] {
        for layer in 0..28 {
            for sub in subs {
                assert_eq!(count(&format!("talker-{phase}-l{layer}-{sub}")), 1);
            }
            assert_eq!(count(&format!("talker-hidden-{phase}-l{layer}")), 1);
        }
    }
    for layer in 0..5 {
        for sub in subs {
            assert_eq!(
                count(&format!("code-predictor-prefill-frame0-l{layer}-{sub}")),
                1
            );
        }
        assert_eq!(
            count(&format!("code-predictor-hidden-prefill-frame0-l{layer}")),
            1
        );
    }
    for step in 1..=14 {
        for layer in 0..5 {
            for sub in subs {
                assert_eq!(
                    count(&format!("code-predictor-step{step}-frame0-l{layer}-{sub}")),
                    1
                );
            }
            assert_eq!(
                count(&format!("code-predictor-hidden-step{step}-frame0-l{layer}")),
                1
            );
        }
    }
    for singleton in [
        "talker-input-embed",
        "talker-hidden-prefill-final",
        "talker-hidden-step1",
        "talker-hidden-step1-final",
        "talker-logits-prefill",
        "next-emb-step0",
        "code-predictor-prefill-frame0-position-ids",
        "code-predictor-prefill-frame0-rope-cos",
        "code-predictor-prefill-frame0-rope-sin",
        "code-predictor-hidden-prefill-frame0-final",
    ] {
        assert_eq!(count(singleton), 1, "{singleton}");
    }
    for step in 1..=14 {
        assert_eq!(
            count(&format!("code-predictor-hidden-step{step}-frame0-final")),
            1
        );
    }
    assert_eq!(
        m.stages
            .iter()
            .filter(|s| s.name == "talker-logits-step1")
            .count(),
        1
    );
    let next = m
        .stages
        .iter()
        .find(|s| s.name == "next-emb-step0")
        .unwrap();
    let alias = m
        .stages
        .iter()
        .find(|s| s.name == "talker-step1-codec-embed")
        .unwrap();
    assert_eq!(next.sha256, alias.sha256);
    println!("real_stage_manifest_count={}", m.stages.len());
}
