use candle_core::{Result, Tensor};

/// Snake 激活函數（無學習參數版本）
pub fn snake(x: &Tensor) -> Result<Tensor> {
    let sin2 = x.sin()?.sqr()?;
    Ok((x + sin2)?)
}

/// 有參數的 SnakeBeta（x + alpha * sin²(beta*x) / beta）
#[allow(dead_code)]
pub fn snake_beta_v2(x: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let beta_x = (x * beta)?;
    let sin2 = beta_x.sin()?.sqr()?;
    let sin2_scaled = (sin2 / beta)?;
    Ok((x + sin2_scaled)?)
}

/// SnakeBeta 模組包裝
#[allow(dead_code)]
pub struct SnakeBeta {
    beta: Tensor,
}

#[allow(dead_code)]
impl SnakeBeta {
    pub fn new(weights: &crate::weights::WeightLoader, prefix: &str) -> crate::Result<Self> {
        let beta = weights.get(&format!("{prefix}.beta"))?.clone();
        Ok(Self { beta })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        snake_beta_v2(x, &self.beta)
    }
}
