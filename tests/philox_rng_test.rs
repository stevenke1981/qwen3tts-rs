use candle_core::{Device, Tensor};
use qwen3tts::talker::sampling::{Sampler, SamplingOptions};
use serde::Deserialize;
use std::fs;

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;
const CURAND_2POW32_INV: f32 = 2.328_306_436_538_696_3e-10f32;

#[derive(Debug, Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    seed: String,
    #[allow(dead_code)]
    seed_hex: String,
    ctr_lo: u32,
    #[allow(dead_code)]
    subsequence_start: String,
    samples: Vec<Sample>,
}

#[derive(Debug, Deserialize)]
struct Sample {
    uniform_bits: String,
    #[allow(dead_code)]
    uniform: f64,
    words: Vec<u32>,
}

fn load_fixture() -> Fixture {
    let text = fs::read_to_string("fixtures/alignment/p02_philox_vectors.json").unwrap();
    serde_json::from_str(&text).unwrap()
}

fn find_case<'a>(fixture: &'a Fixture, name: &str) -> &'a Case {
    fixture
        .cases
        .iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("missing case: {name}"))
}

fn case_seed(case: &Case) -> u64 {
    u64::from_str_radix(case.seed.as_str(), 10).unwrap()
}

fn parse_hex_u32(bits: &str) -> u32 {
    u32::from_str_radix(bits.trim_start_matches("0x"), 16).unwrap()
}

fn mulhilo32(a: u32, b: u32) -> (u32, u32) {
    let prod = (a as u64) * (b as u64);
    ((prod >> 32) as u32, prod as u32)
}

fn philox_round(state: [u32; 4], k0: u32, k1: u32) -> [u32; 4] {
    let (hi0, lo0) = mulhilo32(PHILOX_M0, state[0]);
    let (hi1, lo1) = mulhilo32(PHILOX_M1, state[2]);
    [hi1 ^ state[1] ^ k0, lo1, hi0 ^ state[3] ^ k1, lo0]
}

fn philox4x32_10(seed: u64, subsequence: u64, ctr_lo: u32) -> [u32; 4] {
    let seed_lo = (seed & 0xFFFF_FFFF) as u32;
    let seed_hi = (seed >> 32) as u32;
    let mut state = [
        ctr_lo,
        0,
        (subsequence & 0xFFFF_FFFF) as u32,
        (subsequence >> 32) as u32,
    ];
    state = philox_round(state, seed_lo, seed_hi);
    let mut k0 = seed_lo;
    let mut k1 = seed_hi;
    for _ in 0..9 {
        k0 = k0.wrapping_add(PHILOX_W0);
        k1 = k1.wrapping_add(PHILOX_W1);
        state = philox_round(state, k0, k1);
    }
    state
}

fn philox_uniform_bits(seed: u64, subsequence: u64, ctr_lo: u32) -> u32 {
    let words = philox4x32_10(seed, subsequence, ctr_lo);
    (((words[0] as f32) + 0.5) * CURAND_2POW32_INV).to_bits()
}

fn token_for_two_token_equal_logit_uniform(r: f64) -> u32 {
    if r <= 0.5 { 0 } else { 1 }
}

#[test]
fn philox_zero_seed_zero_counter_matches_reference_words_and_uniform_bits() {
    let fixture = load_fixture();
    let case = find_case(&fixture, "zero_seed_zero_counter");
    let sample = &case.samples[0];
    let words = philox4x32_10(0, 0, 0);
    assert_eq!(
        words,
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
        "zero key/zero counter words must match pinned reference"
    );
    assert_eq!(words, sample.words.as_slice());
    assert_eq!(
        sample.uniform_bits, "0x3ecc4fd2",
        "pinned zero key/zero counter uniform bits must match"
    );
    assert_eq!(
        parse_hex_u32(&sample.uniform_bits),
        philox_uniform_bits(0, 0, 0)
    );
}

#[test]
fn sampler_stochastic_sampling_consumes_one_philox_step() {
    let fixture = load_fixture();
    let case = find_case(&fixture, "seed1_subseq0_ctr0");
    let seed = case_seed(case);
    let logits = Tensor::from_slice(&[0.0_f32, 0.0_f32], 2, &Device::Cpu).unwrap();
    let stoch = SamplingOptions {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };

    let expected_u0 = f32::from_bits(philox_uniform_bits(seed, 0, case.ctr_lo));
    let expected_u1 = f32::from_bits(philox_uniform_bits(seed, 1, case.ctr_lo));
    let expected = [
        token_for_two_token_equal_logit_uniform(f64::from(expected_u0)),
        token_for_two_token_equal_logit_uniform(f64::from(expected_u1)),
    ];

    let mut sampler = Sampler::new(seed);
    let got = vec![
        sampler.sample(&logits, stoch, None, None, &[]).unwrap(),
        sampler.sample(&logits, stoch, None, None, &[]).unwrap(),
    ];
    assert_eq!(got, expected);

    let mut reseeded = Sampler::new(seed);
    let replay = vec![
        reseeded.sample(&logits, stoch, None, None, &[]).unwrap(),
        reseeded.sample(&logits, stoch, None, None, &[]).unwrap(),
    ];
    assert_eq!(
        replay, expected,
        "same seed must produce the same subsequence replay"
    );
}

#[test]
fn sampler_greedy_does_not_consume_philox_subsequence() {
    let fixture = load_fixture();
    let case = find_case(&fixture, "seed1_subseq0_ctr0");
    let seed = case_seed(case);
    let logits = Tensor::from_slice(&[0.0_f32, 0.0_f32], 2, &Device::Cpu).unwrap();
    let stoch = SamplingOptions {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };

    let mut sampler = Sampler::new(seed);
    let _ = sampler
        .sample(&logits, SamplingOptions::greedy(), None, None, &[])
        .unwrap();
    let got = vec![
        sampler.sample(&logits, stoch, None, None, &[]).unwrap(),
        sampler.sample(&logits, stoch, None, None, &[]).unwrap(),
    ];

    let expected_u0 = f32::from_bits(philox_uniform_bits(seed, 0, case.ctr_lo));
    let expected_u1 = f32::from_bits(philox_uniform_bits(seed, 1, case.ctr_lo));
    let expected = [
        token_for_two_token_equal_logit_uniform(f64::from(expected_u0)),
        token_for_two_token_equal_logit_uniform(f64::from(expected_u1)),
    ];

    let mut sampler_ref = Sampler::new(seed);
    let replay = vec![
        sampler_ref.sample(&logits, stoch, None, None, &[]).unwrap(),
        sampler_ref.sample(&logits, stoch, None, None, &[]).unwrap(),
    ];

    assert_eq!(got, expected);
    assert_eq!(got, replay, "greedy path must not advance subsequence");
}
