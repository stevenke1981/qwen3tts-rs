use candle_core::{Device, Result as CandleResult, Tensor};
use qwen3tts::StageDumpObserver;
use qwen3tts::talker::sampling::{Sampler, SamplingOptions};
use qwen3tts::talker::{InputBuilder, TalkerConfig, TalkerWeightLoader};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct SamplingFixture {
    do_sample: bool,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repetition_penalty: f64,
}

#[derive(Debug, Deserialize)]
struct RealFixture {
    snapshot_revision: String,
    official_qwen_revision: String,
    qwentts_cpp_revision: String,
    prompt_ids: Vec<Vec<u32>>,
    language: String,
    seed: u64,
    talker: SamplingFixture,
    subtalker: SamplingFixture,
    max_new_tokens: usize,
    philox_draws: usize,
    frames: Vec<Vec<u32>>,
}

fn sampling_options(fixture: &SamplingFixture) -> SamplingOptions {
    SamplingOptions {
        temperature: fixture.temperature,
        top_k: fixture.top_k,
        top_p: fixture.top_p,
        repetition_penalty: fixture.repetition_penalty,
    }
}

fn model_path() -> PathBuf {
    let value = std::env::var("QWEN3_TTS_REAL_MODEL_DIR")
        .unwrap_or_else(|_| panic!("FIXTURE_MISSING: QWEN3_TTS_REAL_MODEL_DIR"));
    let path = PathBuf::from(value).join("model.safetensors");
    assert!(path.exists(), "FIXTURE_MISSING: {}", path.display());
    path
}

fn assert_real_manifest(path: &str) {
    use sha2::{Digest, Sha256};
    let hash = format!(
        "{:x}",
        Sha256::digest(fs::read(path).expect("read fixture"))
    );
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string("config/fixtures.json").expect("read manifest"))
            .expect("parse manifest");
    let entry = manifest["fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["id"] == "p02-deterministic-token-sequences-real")
        .expect("manifest real fixture");
    assert_eq!(entry["sha256"], hash);
}

#[derive(Default)]
struct FirstLogitsObserver {
    first: Option<Vec<f32>>,
}

impl StageDumpObserver for FirstLogitsObserver {
    fn wants_capture(&self) -> bool {
        true
    }

    fn on_talker_codebook0_logits(
        &mut self,
        frame_index: usize,
        logits: &Tensor,
    ) -> CandleResult<()> {
        if frame_index == 0 && self.first.is_none() {
            self.first = Some(logits.flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(())
    }
}

#[test]
#[ignore = "requires pinned local 0.6B snapshot and candle-llm runtime"]
fn pinned_real_oracle_fixture_is_present_and_fail_closed() {
    let path = "fixtures/alignment/p02_deterministic_token_sequences_real.json";
    let fixture: RealFixture = serde_json::from_str(
        &fs::read_to_string(path).unwrap_or_else(|e| panic!("FIXTURE_MISSING: {path}: {e}")),
    )
    .expect("parse real fixture");
    assert_eq!(
        fixture.snapshot_revision,
        "5d83992436eae1d760afd27aff78a71d676296fc"
    );
    assert_eq!(
        fixture.official_qwen_revision,
        "022e286b98fbec7e1e916cb940cdf532cd9f488e"
    );
    assert_eq!(
        fixture.qwentts_cpp_revision,
        "82cd05b9f3a175612dc89fd6943e610fab096ef5"
    );
    assert_real_manifest(path);
    assert_eq!(fixture.philox_draws, 49);
    assert_eq!(fixture.frames.len(), 3);
    let _ = model_path();
}

#[test]
fn captured_real_first_c0_row_replays_qwentts_vocab_order() {
    let mut logits = vec![-100.0f32; 3072];
    for (idx, value) in [
        (1221, 23.9389),
        (1995, 23.870457),
        (404, 23.013311),
        (29, 21.951431),
        (279, 20.621677),
        (1464, 20.444881),
        (730, 19.324387),
        (765, 17.334648),
        (9, 17.042849),
        (67, 16.678019),
    ] {
        logits[idx] = value;
    }
    let mut sampler = Sampler::new(12345);
    let selected = sampler
        .sample_with_mode(
            &Tensor::from_slice(&logits, logits.len(), &Device::Cpu).expect("captured logits"),
            sampling_options(&SamplingFixture {
                do_sample: true,
                temperature: 0.9,
                top_k: 50,
                top_p: 1.0,
                repetition_penalty: 1.0,
            }),
            true,
            Some(2048),
            None,
            &[],
        )
        .expect("replay captured row");
    assert_eq!(selected, 1995);
    assert_eq!(sampler.subsequence_counter(), 1);
}

#[test]
#[ignore = "requires pinned local 0.6B snapshot and full Candle talker inference"]
fn pinned_real_oracle_runs_production_talker() {
    let path = "fixtures/alignment/p02_deterministic_token_sequences_real.json";
    let fixture: RealFixture = serde_json::from_str(
        &fs::read_to_string(path).unwrap_or_else(|e| panic!("FIXTURE_MISSING: {path}: {e}")),
    )
    .expect("parse real fixture");
    assert_eq!(
        fixture.snapshot_revision,
        "5d83992436eae1d760afd27aff78a71d676296fc"
    );
    let model_path = model_path();
    let device = Device::Cpu;
    let loader = TalkerWeightLoader::from_safetensors(&model_path, &device).expect("load talker");
    let talker = loader
        .build_talker(&TalkerConfig::default())
        .expect("build talker");
    let prompt_ids = fixture.prompt_ids.first().expect("prompt_ids batch");
    let language = match fixture.language.as_str() {
        "chinese" => "Chinese",
        other => other,
    };
    let builder = InputBuilder::new(&talker, &device);
    let (inputs_embeds, attention_mask, trailing_text_hidden, tts_pad_embed) = builder
        .build(prompt_ids, None, language, None)
        .expect("build prompt inputs");
    let mut sampler = Sampler::new(fixture.seed);
    let mut observer = FirstLogitsObserver::default();
    let codes = talker
        .generate_sampled_with_observer(
            &inputs_embeds,
            Some(&attention_mask),
            Some(&trailing_text_hidden),
            Some(&tts_pad_embed),
            fixture.max_new_tokens,
            &device,
            &mut sampler,
            sampling_options(&fixture.talker),
            sampling_options(&fixture.subtalker),
            fixture.talker.do_sample,
            fixture.subtalker.do_sample,
            &mut observer,
        )
        .expect("production sampled generation");
    let first = observer.first.expect("first c0 logits");
    let mut ranked: Vec<(usize, f32)> = first.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("rust_first_c0_top10={:?}", &ranked[..10]);
    let actual = codes.to_vec2::<u32>().expect("generated token ids");
    assert_eq!(actual, fixture.frames, "real oracle token matrix mismatch");
    assert_eq!(sampler.subsequence_counter(), fixture.philox_draws as u64);
}
