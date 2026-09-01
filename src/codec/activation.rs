use candle_core::Tensor;
use crate::Result;

/// Snake 激活函數（無學習參數版本）: x + sin²(x)
pub fn snake(x: &Tensor) -> Result<Tensor> {
    let sin2 = x.sin()?.sqr()?;
    Ok((x + sin2)?)
}

/// SnakeBeta 激活函數: x + (sin²(exp(alpha) * x)) / (exp(beta) + 1e-9)
/// 其中 alpha 與 beta 先經 .exp() 變換
#[allow(dead_code)]
pub fn snake_beta(x: &Tensor, alpha: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let alpha = alpha.reshape((1, alpha.elem_count(), 1))?.exp()?;
    let beta = beta.reshape((1, beta.elem_count(), 1))?.exp()?;
    let alpha_x = x.broadcast_mul(&alpha)?;
    let sin2 = alpha_x.sin()?.sqr()?;
    let denom = (beta + 1e-9f64)?;
    Ok(x.broadcast_add(&sin2.broadcast_div(&denom)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_snake_activation_formula() {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.0f32, std::f32::consts::FRAC_PI_2, std::f32::consts::PI], &device).unwrap();
        let y = snake(&x).unwrap();
        let y_vec = y.to_vec1::<f32>().unwrap();
        // snake(0) = 0 + sin²(0) = 0
        assert!((y_vec[0] - 0.0).abs() < 1e-6);
        // snake(pi/2) = pi/2 + sin²(pi/2) = pi/2 + 1
        assert!((y_vec[1] - (std::f32::consts::FRAC_PI_2 + 1.0)).abs() < 1e-5);
        // snake(pi) = pi + sin²(pi) = pi + 0 = pi
        assert!((y_vec[2] - std::f32::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn test_snake_beta_matches_decoder_blocks() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0f32, (1, 8, 16), &device).unwrap();
        let alpha = Tensor::zeros((8,), DType::F32, &device).unwrap(); // exp(0) = 1
        let beta = Tensor::zeros((8,), DType::F32, &device).unwrap();  // exp(0) = 1

        let y1 = snake_beta(&x, &alpha, &beta).unwrap();
        let y2 = crate::codec::decoder_blocks::snake_beta(&x, &alpha, &beta).unwrap();

        let diff = (y1 - y2).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-6, "activation snake_beta must match decoder_blocks snake_beta");
    }
}


