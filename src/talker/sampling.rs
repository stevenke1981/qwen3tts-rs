//! Sampling helpers for practical codec generation.

use std::cmp::Ordering;

use candle_core::{Result, Tensor};

const DEFAULT_SAMPLER_CAPACITY: usize = 4096;

#[derive(Debug, Clone, Copy)]
pub struct SamplingOptions {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
}

impl SamplingOptions {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    state: u64,
    candidates: Vec<(usize, f32)>,
    probs: Vec<(usize, f64)>,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
            candidates: Vec::with_capacity(DEFAULT_SAMPLER_CAPACITY),
            probs: Vec::with_capacity(DEFAULT_SAMPLER_CAPACITY),
        }
    }

    pub fn sample(
        &mut self,
        logits: &Tensor,
        options: SamplingOptions,
        suppress_from: Option<usize>,
        allow_suppressed_token: Option<usize>,
    ) -> Result<u32> {
        let logits = logits.flatten_all()?.to_vec1::<f32>()?;
        let Self {
            state,
            candidates,
            probs,
        } = self;
        Ok(sample_logits_with_scratch(
            &logits,
            options,
            suppress_from,
            allow_suppressed_token,
            || next_f64_state(state),
            candidates,
            probs,
        ))
    }
}

fn next_f64_state(state: &mut u64) -> f64 {
    // SplitMix64: small, deterministic, and good enough for top-k sampling.
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

#[cfg(test)]
fn sample_logits(
    logits: &[f32],
    options: SamplingOptions,
    suppress_from: Option<usize>,
    allow_suppressed_token: Option<usize>,
    mut rand01: impl FnMut() -> f64,
) -> u32 {
    let mut candidates = Vec::with_capacity(logits.len());
    let mut probs = Vec::new();
    sample_logits_with_scratch(
        logits,
        options,
        suppress_from,
        allow_suppressed_token,
        &mut rand01,
        &mut candidates,
        &mut probs,
    )
}

fn sample_logits_with_scratch(
    logits: &[f32],
    options: SamplingOptions,
    suppress_from: Option<usize>,
    allow_suppressed_token: Option<usize>,
    mut rand01: impl FnMut() -> f64,
    candidates: &mut Vec<(usize, f32)>,
    probs: &mut Vec<(usize, f64)>,
) -> u32 {
    candidates.clear();
    candidates.extend(logits.iter().copied().enumerate().filter(|(idx, v)| {
        v.is_finite()
            && suppress_from
                .map(|start| *idx < start || Some(*idx) == allow_suppressed_token)
                .unwrap_or(true)
    }));

    if candidates.is_empty() {
        return 0;
    }

    if options.top_k > 0 && candidates.len() > options.top_k {
        let top_k = options.top_k;
        candidates.select_nth_unstable_by(top_k, cmp_logit_desc);
        candidates.truncate(top_k);
    }
    candidates.sort_unstable_by(cmp_logit_desc);

    if options.temperature <= 0.0 {
        return candidates[0].0 as u32;
    }

    let max_logit = candidates[0].1 as f64;
    let inv_temp = 1.0 / options.temperature.max(1e-6);
    probs.clear();
    probs.extend(
        candidates
            .iter()
            .map(|(idx, logit)| (*idx, ((*logit as f64 - max_logit) * inv_temp).exp())),
    );
    let total: f64 = probs.iter().map(|(_, p)| *p).sum();
    if total <= 0.0 || !total.is_finite() {
        return probs[0].0 as u32;
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
            return *idx as u32;
        }
    }
    0
}

fn cmp_logit_desc(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.0.cmp(&b.0))
}

#[cfg(test)]
mod tests {
    use super::{SamplingOptions, sample_logits};

    #[test]
    fn greedy_returns_highest_finite_logit() {
        let token = sample_logits(
            &[0.0, 3.0, f32::NEG_INFINITY, 2.0],
            SamplingOptions::greedy(),
            None,
            None,
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
            },
            None,
            None,
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
        };
        for r in [0.0, 0.15, 0.5, 0.95] {
            let actual = sample_logits(&logits, options, Some(8), None, || r);
            let expected = sample_logits_reference(&logits, options, Some(8), None, || r);
            assert_eq!(actual, expected, "r={r}");
        }
    }

    fn sample_logits_reference(
        logits: &[f32],
        options: SamplingOptions,
        suppress_from: Option<usize>,
        allow_suppressed_token: Option<usize>,
        mut rand01: impl FnMut() -> f64,
    ) -> u32 {
        let mut candidates: Vec<(usize, f32)> = logits
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

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if options.temperature <= 0.0 {
            return candidates[0].0 as u32;
        }

        if options.top_k > 0 && candidates.len() > options.top_k {
            candidates.truncate(options.top_k);
        }

        let max_logit = candidates[0].1 as f64;
        let inv_temp = 1.0 / options.temperature.max(1e-6);
        let mut probs: Vec<(usize, f64)> = candidates
            .into_iter()
            .map(|(idx, logit)| (idx, ((logit as f64 - max_logit) * inv_temp).exp()))
            .collect();
        let total: f64 = probs.iter().map(|(_, p)| *p).sum();
        if total <= 0.0 || !total.is_finite() {
            return probs[0].0 as u32;
        }
        for (_, p) in &mut probs {
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
                for (_, p) in &mut probs {
                    *p /= renorm;
                }
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
        0
    }
}
