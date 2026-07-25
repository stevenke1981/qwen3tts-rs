#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub seed: u64,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

#[derive(Debug, Clone)]
pub struct GenerationSampling {
    pub talker: SamplingConfig,
    pub code_predictor: SamplingConfig,
}

/// Required operation order:
/// 1. mask invalid/control tokens
/// 2. apply repetition penalty
/// 3. divide by temperature
/// 4. top-k
/// 5. top-p
/// 6. Philox multinomial
pub trait LogitSampler {
    type Error;
    fn sample(
        &mut self,
        logits: &mut [f32],
        history: &[u32],
        config: SamplingConfig,
    ) -> Result<u32, Self::Error>;
}
