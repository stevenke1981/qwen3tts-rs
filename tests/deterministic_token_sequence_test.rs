use candle_core::{Device, Tensor};
use qwen3tts::talker::sampling::{Sampler, SamplingOptions};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    seed: u64,
    frames: Vec<Vec<u32>>,
    c0_history: Vec<u32>,
    cp_history_empty: bool,
    draw_count: u64,
    calls: Vec<Call>,
}
#[derive(Deserialize)]
struct Call {
    frame: usize,
    logits: Vec<f32>,
    options: Options,
    history: Vec<u16>,
    suppress_from: Option<usize>,
    allow_token: Option<usize>,
    expected: u32,
    subsequence: u64,
    codebook: usize,
}
#[derive(Deserialize)]
struct Options {
    do_sample: bool,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repetition_penalty: f64,
}

fn opts(o: &Options) -> SamplingOptions {
    SamplingOptions {
        temperature: o.temperature,
        top_k: o.top_k,
        top_p: o.top_p,
        repetition_penalty: o.repetition_penalty,
    }
}

#[test]
fn production_sampler_replays_independent_literal_oracle() {
    let fixture: Fixture = serde_json::from_str(
        &fs::read_to_string("fixtures/alignment/p02_deterministic_token_sequences.json").unwrap(),
    )
    .unwrap();
    for case in fixture.cases {
        let mut sampler = Sampler::new(case.seed);
        let mut actual = BTreeMap::<usize, Vec<u32>>::new();
        let mut c0_history = Vec::new();
        for call in &case.calls {
            assert_eq!(
                sampler.subsequence_counter(),
                call.subsequence,
                "{} codebook {}",
                case.id,
                call.codebook
            );
            let t = Tensor::from_slice(&call.logits, call.logits.len(), &Device::Cpu).unwrap();
            let token = sampler
                .sample_with_mode(
                    &t,
                    opts(&call.options),
                    call.options.do_sample,
                    call.suppress_from,
                    call.allow_token,
                    &call.history,
                )
                .unwrap();
            assert_eq!(
                token, call.expected,
                "{} frame/codebook {}/{}",
                case.id, call.codebook, call.subsequence
            );
            if call.frame < case.frames.len() {
                actual.entry(call.frame).or_default().push(token);
            }
            if call.codebook == 0 && call.frame < case.frames.len() {
                c0_history.push(token);
            }
        }
        let matrix: Vec<Vec<u32>> = actual.into_values().collect();
        assert_eq!(matrix, case.frames, "{} literal frame matrix", case.id);
        assert_eq!(c0_history, case.c0_history, "{} c0 history", case.id);
        assert!(case.cp_history_empty);
        assert_eq!(
            sampler.subsequence_counter(),
            case.draw_count,
            "{} draw count",
            case.id
        );
        assert_eq!(sampler.diagnostics().sequence_len, case.calls.len());
    }
}

#[test]
fn deterministic_seed_rerun_and_change_contract() {
    let logits = Tensor::from_slice(&[0.0_f32, 1.0, 2.0], 3, &Device::Cpu).unwrap();
    let o = SamplingOptions {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };
    let run = |seed| {
        let mut s = Sampler::new(seed);
        (0..12)
            .map(|_| {
                s.sample_with_mode(&logits, o, true, None, None, &[])
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(7), run(7));
    assert_ne!(run(7), run(8));
}
