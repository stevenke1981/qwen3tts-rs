//! Sampling helpers for practical codec generation.

use std::cmp::Ordering;

use candle_core::{Result, Tensor};

const DEFAULT_SAMPLER_CAPACITY: usize = 4096;

// Random123 / cuRAND Philox constants.
const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;
const CURAND_2POW32_INV: f32 = 2.328_306_436_538_696_3e-10f32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingOptions {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f64,
}

impl SamplingOptions {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    seed: u64,
    subseq_counter: u64,
    ctr_lo: u32,
    candidates: Vec<(usize, f32)>,
    probs: Vec<(usize, f64)>,
    repetition_seen: Vec<u32>,
    repetition_seen_epoch: u32,
    penalties: Vec<f32>,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            subseq_counter: 0,
            ctr_lo: 0,
            candidates: Vec::with_capacity(DEFAULT_SAMPLER_CAPACITY),
            probs: Vec::with_capacity(DEFAULT_SAMPLER_CAPACITY),
            repetition_seen: Vec::with_capacity(DEFAULT_SAMPLER_CAPACITY),
            repetition_seen_epoch: 0,
            penalties: Vec::with_capacity(DEFAULT_SAMPLER_CAPACITY),
        }
    }

    pub fn sample(
        &mut self,
        logits: &Tensor,
        options: SamplingOptions,
        suppress_from: Option<usize>,
        allow_suppressed_token: Option<usize>,
        history: &[u16],
    ) -> Result<u32> {
        let do_sample = options.temperature > 0.0;
        self.sample_with_mode(
            logits,
            options,
            do_sample,
            suppress_from,
            allow_suppressed_token,
            history,
        )
    }

    pub fn sample_with_mode(
        &mut self,
        logits: &Tensor,
        options: SamplingOptions,
        do_sample: bool,
        suppress_from: Option<usize>,
        allow_suppressed_token: Option<usize>,
        history: &[u16],
    ) -> Result<u32> {
        let logits = logits.flatten_all()?.to_vec1::<f32>()?;
        if !options.repetition_penalty.is_finite() || options.repetition_penalty <= 0.0 {
            return Err(candle_core::Error::Msg(
                "invalid repetition_penalty: expected finite value > 0.0".to_string(),
            ));
        }
        if do_sample && (!options.temperature.is_finite() || options.temperature <= 0.0) {
            return Err(candle_core::Error::Msg(
                "invalid temperature: expected finite value > 0.0 when sampling is enabled"
                    .to_string(),
            ));
        }
        if !options.top_p.is_finite() || options.top_p <= 0.0 || options.top_p > 1.0 {
            return Err(candle_core::Error::Msg(
                "invalid top_p: expected finite value in (0.0, 1.0]".to_string(),
            ));
        }

        let mut subseq_counter = self.subseq_counter;
        let seed = self.seed;
        let ctr_lo = self.ctr_lo;
        let next = || {
            let u = next_uniform(seed, subseq_counter, ctr_lo);
            subseq_counter = subseq_counter.wrapping_add(1);
            u
        };

        let sampled = sample_logits_with_scratch(
            &logits,
            if do_sample {
                options
            } else {
                SamplingOptions {
                    temperature: -1.0,
                    top_k: options.top_k,
                    top_p: options.top_p,
                    repetition_penalty: 1.0,
                }
            },
            suppress_from,
            allow_suppressed_token,
            history,
            &mut self.penalties,
            &mut self.repetition_seen,
            &mut self.repetition_seen_epoch,
            next,
            &mut self.candidates,
            &mut self.probs,
        );

        if do_sample {
            self.subseq_counter = subseq_counter;
        }
        sampled
    }
}

#[inline]
fn mulhilo32(a: u32, b: u32) -> (u32, u32) {
    let prod = (a as u64) * (b as u64);
    ((prod >> 32) as u32, prod as u32)
}

#[inline]
fn philox_round(state: [u32; 4], k0: u32, k1: u32) -> [u32; 4] {
    let (hi0, lo0) = mulhilo32(PHILOX_M0, state[0]);
    let (hi1, lo1) = mulhilo32(PHILOX_M1, state[2]);
    [hi1 ^ state[1] ^ k0, lo1, hi0 ^ state[3] ^ k1, lo0]
}

#[inline]
fn philox4x32_10(ctr: [u32; 4], seed_lo: u32, seed_hi: u32) -> [u32; 4] {
    let mut k0 = seed_lo;
    let mut k1 = seed_hi;
    let mut state = philox_round(ctr, k0, k1);
    for _ in 0..9 {
        k0 = k0.wrapping_add(PHILOX_W0);
        k1 = k1.wrapping_add(PHILOX_W1);
        state = philox_round(state, k0, k1);
    }
    state
}

fn next_uniform(seed: u64, subsequence: u64, ctr_lo: u32) -> f64 {
    let seed_lo = (seed & 0xFFFF_FFFF) as u32;
    let seed_hi = (seed >> 32) as u32;
    let ctr = [
        ctr_lo,
        0,
        (subsequence & 0xFFFF_FFFF) as u32,
        (subsequence >> 32) as u32,
    ];
    let words = philox4x32_10(ctr, seed_lo, seed_hi);
    let u = ((words[0] as f32) + 0.5f32) * CURAND_2POW32_INV;
    f64::from(u)
}

#[inline]
fn apply_temperature_in_place(candidates: &mut Vec<(usize, f32)>, temperature: f32) {
    let inv_temp = 1.0f32 / temperature;
    for (_, logit) in candidates.iter_mut() {
        *logit *= inv_temp;
    }
}

#[cfg(test)]
fn sample_logits(
    logits: &[f32],
    options: SamplingOptions,
    suppress_from: Option<usize>,
    allow_suppressed_token: Option<usize>,
    history: &[u16],
    mut rand01: impl FnMut() -> f64,
) -> u32 {
    let mut candidates = Vec::with_capacity(logits.len());
    let mut probs = Vec::new();
    let mut repetition_seen = Vec::new();
    let mut repetition_seen_epoch = 0u32;
    let mut penalties = Vec::new();
    sample_logits_with_scratch(
        logits,
        options,
        suppress_from,
        allow_suppressed_token,
        history,
        &mut penalties,
        &mut repetition_seen,
        &mut repetition_seen_epoch,
        &mut rand01,
        &mut candidates,
        &mut probs,
    )
    .expect("sample_logits helper should not fail")
}

#[cfg(test)]
fn run_reference_sampler(
    logits: &[f32],
    options: SamplingOptions,
    suppress_from: Option<usize>,
    allow_suppressed_token: Option<usize>,
    history: &[u16],
    mut rand01: impl FnMut() -> f64,
) -> u32 {
    let mut repetition_seen = Vec::new();
    let mut repetition_seen_epoch = 0u32;
    let mut penalties = logits.to_vec();

    let mut candidates: Vec<(usize, f32)> = penalties
        .iter()
        .copied()
        .enumerate()
        .filter(|(idx, v)| {
            v.is_finite()
                && suppress_from
                    .map(|start| *idx < start || Some(*idx) == allow_suppressed_token)
                    .unwrap_or(true)
        })
        .collect();

    if candidates.is_empty() {
        return 0;
    }

    candidates.sort_by(cmp_logit_desc);

    if options.temperature <= 0.0 {
        return candidates[0].0 as u32;
    }

    if options.repetition_penalty != 1.0 {
        apply_repetition_penalty(
            &mut penalties,
            history,
            options.repetition_penalty,
            &mut repetition_seen,
            &mut repetition_seen_epoch,
        )
        .unwrap();
    }

    candidates.clear();
    candidates.extend(penalties.iter().copied().enumerate().filter(|(idx, v)| {
        v.is_finite()
            && suppress_from
                .map(|start| *idx < start || Some(*idx) == allow_suppressed_token)
                .unwrap_or(true)
    }));
    if candidates.is_empty() {
        return 0;
    }

    candidates.sort_by(cmp_logit_desc);

    apply_temperature_in_place(&mut candidates, options.temperature as f32);

    if options.top_k > 0 && candidates.len() > options.top_k {
        candidates.sort_by(cmp_logit_desc);
        candidates.truncate(options.top_k);
    }
    candidates.sort_by(cmp_logit_desc);

    if options.top_p < 1.0 {
        let max_logit = candidates[0].1 as f64;
        let mut probs: Vec<(usize, f64)> = candidates
            .iter()
            .map(|(idx, logit)| (*idx, (f64::from(*logit) - max_logit).exp()))
            .collect();
        let total: f64 = probs.iter().map(|(_, p)| *p).sum();
        if total <= 0.0 || !total.is_finite() {
            return probs[0].0 as u32;
        }
        for (_, p) in probs.iter_mut() {
            *p /= total;
        }
        let mut cumulative = 0.0;
        let keep = probs
            .iter()
            .position(|(_, p)| {
                cumulative += *p;
                cumulative >= options.top_p
            })
            .map(|idx| idx + 1)
            .unwrap_or(probs.len())
            .max(1);
        probs.truncate(keep);
        let renorm: f64 = probs.iter().map(|(_, p)| *p).sum();
        if renorm > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= renorm;
            }
        }
        let r = rand01();
        let mut cumulative = 0.0;
        for (idx, p) in probs {
            cumulative += p;
            if r <= cumulative {
                return idx as u32;
            }
        }
        return candidates.last().map(|(idx, _)| *idx as u32).unwrap_or(0);
    }

    let max_logit = candidates[0].1 as f64;
    let mut probs: Vec<(usize, f64)> = candidates
        .into_iter()
        .map(|(idx, logit)| (idx, (f64::from(logit) - max_logit).exp()))
        .collect();
    let total: f64 = probs.iter().map(|(_, p)| *p).sum();
    if total <= 0.0 || !total.is_finite() {
        return probs[0].0 as u32;
    }
    for (_, p) in probs.iter_mut() {
        *p /= total;
    }

    let r = rand01();
    let mut cumulative = 0.0;
    for (idx, p) in probs {
        cumulative += p;
        if r <= cumulative {
            return idx as u32;
        }
    }

    0
}

fn sample_logits_with_scratch(
    logits: &[f32],
    options: SamplingOptions,
    suppress_from: Option<usize>,
    allow_suppressed_token: Option<usize>,
    history: &[u16],
    penalties: &mut Vec<f32>,
    repetition_seen: &mut Vec<u32>,
    repetition_seen_epoch: &mut u32,
    mut rand01: impl FnMut() -> f64,
    mut candidates: &mut Vec<(usize, f32)>,
    probs: &mut Vec<(usize, f64)>,
) -> Result<u32> {
    candidates.clear();
    candidates.extend(logits.iter().copied().enumerate().filter(|(idx, v)| {
        v.is_finite()
            && suppress_from
                .map(|start| *idx < start || Some(*idx) == allow_suppressed_token)
                .unwrap_or(true)
    }));

    if candidates.is_empty() {
        return Ok(0);
    }

    candidates.sort_unstable_by(cmp_logit_desc);

    if options.temperature <= 0.0 {
        return Ok(candidates[0].0 as u32);
    }

    if penalties.len() < logits.len() {
        penalties.resize(logits.len(), 0.0);
    }
    let penalties = &mut penalties[..logits.len()];
    penalties.copy_from_slice(logits);

    if options.repetition_penalty != 1.0 {
        apply_repetition_penalty(
            penalties,
            history,
            options.repetition_penalty,
            repetition_seen,
            repetition_seen_epoch,
        )?;
    }

    candidates.clear();
    candidates.extend(penalties.iter().copied().enumerate().filter(|(idx, v)| {
        v.is_finite()
            && suppress_from
                .map(|start| *idx < start || Some(*idx) == allow_suppressed_token)
                .unwrap_or(true)
    }));

    if candidates.is_empty() {
        return Ok(0);
    }
    candidates.sort_unstable_by(cmp_logit_desc);

    apply_temperature_in_place(&mut candidates, options.temperature as f32);
    candidates.sort_unstable_by(cmp_logit_desc);

    if options.top_k > 0 && candidates.len() > options.top_k {
        let top_k = options.top_k;
        candidates.select_nth_unstable_by(top_k - 1, cmp_logit_desc);
        candidates.truncate(top_k);
    }
    candidates.sort_unstable_by(cmp_logit_desc);

    let max_logit = candidates[0].1 as f64;
    probs.clear();
    probs.extend(
        candidates
            .iter()
            .map(|(idx, logit)| (*idx, (f64::from(*logit) - max_logit).exp())),
    );
    let total: f64 = probs.iter().map(|(_, p)| *p).sum();
    if total <= 0.0 || !total.is_finite() {
        return Ok(probs[0].0 as u32);
    }
    for (_, p) in probs.iter_mut() {
        *p /= total;
    }

    if options.top_p < 1.0 {
        let mut cumulative = 0.0;
        let keep = probs
            .iter()
            .position(|(_, p)| {
                cumulative += *p;
                cumulative >= options.top_p
            })
            .map(|idx| idx + 1)
            .unwrap_or(probs.len())
            .max(1);
        probs.truncate(keep);
        let renorm: f64 = probs.iter().map(|(_, p)| *p).sum();
        if renorm > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= renorm;
            }
        }
    }

    let r = rand01();
    let mut cumulative = 0.0;
    for (idx, p) in probs.iter() {
        cumulative += *p;
        if r <= cumulative {
            return Ok(*idx as u32);
        }
    }
    if probs.is_empty() {
        Ok(0)
    } else {
        Ok(probs[probs.len() - 1].0 as u32)
    }
}

fn apply_repetition_penalty(
    logits: &mut [f32],
    history: &[u16],
    penalty: f64,
    seen: &mut Vec<u32>,
    seen_epoch: &mut u32,
) -> Result<()> {
    if penalty == 1.0 || history.is_empty() {
        return Ok(());
    }

    if seen.len() < logits.len() {
        seen.resize(logits.len(), 0);
    }
    if *seen_epoch == u32::MAX {
        seen.fill(0);
        *seen_epoch = 0;
    }
    *seen_epoch = seen_epoch.wrapping_add(1);
    let epoch = *seen_epoch;

    let penalty = penalty as f32;
    for &token in history {
        let idx = usize::from(token);
        if idx >= logits.len() {
            continue;
        }
        if seen[idx] == epoch {
            continue;
        }
        seen[idx] = epoch;

        let score = logits[idx];
        logits[idx] = if score < 0.0 {
            score * penalty
        } else {
            score / penalty
        };
    }

    Ok(())
}

fn cmp_logit_desc(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.0.cmp(&b.0))
}

#[cfg(test)]
mod tests {
    use super::{
        Sampler, SamplingOptions, apply_repetition_penalty, apply_temperature_in_place,
        next_uniform, philox4x32_10, run_reference_sampler, sample_logits,
    };
    use candle_core::{Device, Tensor};

    #[test]
    fn greedy_returns_highest_finite_logit() {
        let token = sample_logits(
            &[0.0, 3.0, f32::NEG_INFINITY, 2.0],
            SamplingOptions::greedy(),
            None,
            None,
            &[],
            || 0.5,
        );
        assert_eq!(token, 1);
    }

    #[test]
    fn suppression_blocks_control_tokens_but_allows_eos() {
        let mut logits = vec![0.0; 2050];
        logits[2] = 1.0;
        logits[2048] = 10.0;
        logits[2049] = 9.0;

        let blocked = sample_logits(
            &logits,
            SamplingOptions::greedy(),
            Some(2048),
            Some(2049),
            &[],
            || 0.0,
        );
        assert_eq!(blocked, 2049);
    }

    #[test]
    fn top_k_sampling_limits_candidates() {
        let token = sample_logits(
            &[10.0, 9.0, 8.0],
            SamplingOptions {
                temperature: 1.0,
                top_k: 1,
                top_p: 1.0,
                repetition_penalty: 1.0,
            },
            None,
            None,
            &[],
            || 0.99,
        );
        assert_eq!(token, 0);
    }

    #[test]
    fn optimized_sampler_matches_reference_top_k_top_p() {
        let logits = [
            0.0,
            4.0,
            f32::NEG_INFINITY,
            2.5,
            4.0,
            -1.0,
            3.2,
            1.0,
            f32::NAN,
            0.5,
        ];
        let options = SamplingOptions {
            temperature: 0.9,
            top_k: 4,
            top_p: 0.8,
            repetition_penalty: 1.0,
        };
        for r in [0.0, 0.15, 0.5, 0.95] {
            let actual = sample_logits(&logits, options, Some(8), None, &[], || r);
            let expected = run_reference_sampler(&logits, options, Some(8), None, &[], || r);
            assert_eq!(actual, expected, "r={r}");
        }
    }

    #[test]
    fn philox_round_trip_matches_reference_non_zero_cases() {
        const CASES: &[(u64, u64, u32, [u32; 4], u32)] = &[
            (
                0x0000_0000_0000_0001_u64,
                0_u64,
                0_u32,
                [3_823_634_032, 3_842_641_596, 2_515_673_792, 3_054_873_127],
                0x3f63_e806u32,
            ),
            (
                0x1234_5678_9a_bc_de_f0_u64,
                1_u64,
                7_u32,
                [477_801_367, 731_680_216, 3_144_223_409, 1_653_668_364],
                0x3de3_d55du32,
            ),
            (
                0xdead_beef_dead_beef_u64,
                42_u64,
                0_u32,
                [3_419_006_238, 2_188_664_071, 2_672_270_330, 2_497_621_687],
                0x3f4b_c9e5u32,
            ),
        ];

        for &(seed, subseq, ctr_lo, expected_words, expected_uniform_bits) in CASES {
            let seed_lo = (seed & 0xFFFF_FFFF) as u32;
            let seed_hi = (seed >> 32) as u32;
            let words = philox4x32_10(
                [
                    ctr_lo,
                    0,
                    (subseq & 0xFFFF_FFFF) as u32,
                    (subseq >> 32) as u32,
                ],
                seed_lo,
                seed_hi,
            );
            assert_eq!(
                words, expected_words,
                "seed={seed}, subseq={subseq}, ctr_lo={ctr_lo}"
            );
            assert_eq!(
                (next_uniform(seed, subseq, ctr_lo) as f32).to_bits(),
                expected_uniform_bits
            );
        }

        let zero_seed_zero_subseq_words = philox4x32_10([0, 0, 0, 0], 0, 0);
        assert_eq!(
            zero_seed_zero_subseq_words,
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!((next_uniform(0, 0, 0) as f32).to_bits(), 0x3ecc_4fd2u32);
    }

    fn token_for_equal_logit_two_token(r: f64) -> u32 {
        if r <= 0.5 { 0 } else { 1 }
    }

    #[test]
    fn sampler_subseq_counter_rolls_over_when_sequential_sampling() {
        let mut sampler = Sampler::new(0x0123_4567_89ab_cdef_u64);
        sampler.subseq_counter = u64::MAX;
        let logits = Tensor::from_slice(&[0.0_f32, 0.0_f32], 2, &Device::Cpu).unwrap();
        let options = SamplingOptions {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };

        let first = sampler
            .sample(&logits, options, None, None, &[])
            .expect("first draw");
        let second = sampler
            .sample(&logits, options, None, None, &[])
            .expect("second draw");

        assert_eq!(
            first,
            token_for_equal_logit_two_token(
                next_uniform(0x0123_4567_89ab_cdef_u64, u64::MAX, 0) as f64
            )
        );
        assert_eq!(
            second,
            token_for_equal_logit_two_token(next_uniform(0x0123_4567_89ab_cdef_u64, 0, 0) as f64)
        );
        assert_eq!(sampler.subseq_counter, 1);
    }

    #[test]
    fn sampler_does_not_consume_subseq_for_all_suppressed_tokens_even_with_temperature() {
        let mut sampler = Sampler::new(42);
        let logits = Tensor::from_slice(&[0.0_f32, 1.0, -1.0], 3, &Device::Cpu).unwrap();
        let options = SamplingOptions {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let original_subseq = sampler.subseq_counter;

        let out = sampler
            .sample(&logits, options, Some(0), None, &[])
            .unwrap();
        assert_eq!(out, 0);
        assert_eq!(sampler.subseq_counter, original_subseq);
    }

    #[test]
    fn sampler_does_not_consume_subseq_for_all_nonfinite_logits() {
        let mut sampler = Sampler::new(1337);
        let logits = Tensor::from_slice(
            &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            3,
            &Device::Cpu,
        )
        .unwrap();
        let options = SamplingOptions {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let original_subseq = sampler.subseq_counter;

        let out = sampler.sample(&logits, options, None, None, &[]).unwrap();
        assert_eq!(out, 0);
        assert_eq!(sampler.subseq_counter, original_subseq);
    }

    #[test]
    fn run_reference_sampler_reorders_temperature_before_top_k() {
        let logits = [1.0_f32, 2.0_f32, 3.0_f32, f32::NEG_INFINITY];
        let options = SamplingOptions {
            temperature: 0.5,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let token = run_reference_sampler(&logits, options, Some(4), None, &[], || 0.999);
        assert_eq!(token, 2);
    }

    #[test]
    fn apply_repetition_penalty_produces_expected_bits() {
        let mut penalties = vec![1.0_f32, -2.0, 0.0, -0.0, 7.5];
        let mut seen = Vec::new();
        let mut seen_epoch = 0u32;
        let history = [0u16, 3, 1, 3, 1, 99];

        apply_repetition_penalty(&mut penalties, &history, 1.5, &mut seen, &mut seen_epoch)
            .unwrap();

        let expected_after_penalty = [1.0_f32 / 1.5, -3.0_f32, 0.0, -0.0, 7.5_f32];
        for (idx, expected) in expected_after_penalty.iter().enumerate() {
            assert_eq!(
                penalties[idx].to_bits(),
                expected.to_bits(),
                "after_penalty idx={idx}"
            );
        }

        let mut scaled = penalties
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, logit)| (idx, logit))
            .collect::<Vec<_>>();
        apply_temperature_in_place(&mut scaled, 0.5);
        let expected_after_temperature: Vec<u32> =
            scaled.iter().map(|(_, logit)| logit.to_bits()).collect();
        let actual_after_temperature: Vec<u32> =
            penalties.iter().map(|v| (v / 0.5).to_bits()).collect();
        assert_eq!(expected_after_temperature, actual_after_temperature);
        assert_eq!(
            penalties[0].to_bits(),
            (1.0_f32 / 1.5_f32).to_bits(),
            "positive token 0 must be divided by penalty"
        );
        assert!(penalties[1].to_bits() == (-3.0f32).to_bits());
        assert_eq!(penalties[2].to_bits(), 0.0f32.to_bits());
        assert_eq!(penalties[3].to_bits(), (-0.0f32).to_bits());
        assert_eq!(seen_epoch, 1);
    }
}
