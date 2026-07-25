#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q4Km,
}

pub trait QuantizedTensor: Send + Sync {
    fn format(&self) -> QuantFormat;
    fn shape(&self) -> &[usize];
    fn resident_bytes(&self) -> usize;
    fn is_fully_materialized_f32(&self) -> bool;
}

pub trait QuantLinear {
    type Error;
    fn forward(&self, input: &candle_core::Tensor)
        -> Result<candle_core::Tensor, Self::Error>;
}

pub trait QuantEmbedding {
    type Error;
    fn lookup(&self, ids: &candle_core::Tensor)
        -> Result<candle_core::Tensor, Self::Error>;
}
