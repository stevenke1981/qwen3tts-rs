//! # 多碼本 Embedding Lookup (12Hz 核心)
//!
//! 16 層 × 2048 離散碼本的平行查表實作。
//!
//! ## 記憶體佈局
//! 使用扁平化連續 Tensor 儲存所有碼本權重：
//! - 索引計算：`idx = layer * codebook_size + token`
//! - 禁止使用巢狀 Vec 或動態索引
//!
//! ## 並行策略
//! 16 層 Embedding Lookup 透過 `rayon::par_iter` 並行執行。
//!
//! ## 量化注意事項
//! - 禁止使用通用 LLM 量化配置（如 Q4_K_M / Q5_K_M）
//! - 必須使用碼本專屬校準數據集進行 INT8 量化
//! - 量化後須通過 cosine ≥ 0.995 對齊測試

use candle_core::{Device, Tensor};
use rayon::prelude::*;

use crate::Error;

/// 扁平化多碼本查表器
///
/// 儲存兩種可能形狀的權重張量：
/// - 3D: `(num_layers, codebook_size, embedding_dim)` — 原始格式
/// - 2D: `(num_layers * codebook_size, embedding_dim)` — 扁平化格式
///
/// 內部統一使用 2D 扁平化佈局以實現零分支索引。
pub struct CodebookLookup {
    /// 權重張量，扁平化為 2D: (num_layers * codebook_size, embedding_dim)
    flat_weights: Tensor,

    /// 層數
    num_layers: usize,

    /// 每層碼本大小
    codebook_size: usize,

    /// 嵌入維度
    embedding_dim: usize,

    /// 裝置
    device: Device,
}

impl CodebookLookup {
    /// 建立新的碼本查表器
    ///
    /// # 參數
    /// - `weights`: 形狀為 `(num_layers, codebook_size, embedding_dim)` 或
    ///              `(num_layers * codebook_size, embedding_dim)` 的權重張量
    pub fn new(weights: Tensor) -> crate::Result<Self> {
        let shape = weights.dims();
        let device = weights.device().clone();

        let (num_layers, codebook_size, embedding_dim, flat_weights) = match shape.len() {
            2 => {
                // 已扁平化
                let flat = shape[0];
                let emb = shape[1];
                // 預設 16 層 × 2048
                let layers = 16;
                let cb_size = flat / layers;
                if flat % layers != 0 {
                    return Err(Error::Config(format!(
                        "Flat codebook dim 0 ({flat}) must be divisible by num_layers ({layers})",
                    )));
                }
                (layers, cb_size, emb, weights)
            }
            3 => {
                let layers = shape[0];
                let cb_size = shape[1];
                let emb = shape[2];
                // 扁平化: (L, C, E) -> (L*C, E)
                let flat = weights.reshape((layers * cb_size, emb))?;
                (layers, cb_size, emb, flat)
            }
            _ => {
                return Err(Error::Config(format!(
                    "Codebook weights must be 2D or 3D, got shape {:?}",
                    shape
                )));
            }
        };

        Ok(Self {
            num_layers,
            codebook_size,
            embedding_dim,
            flat_weights,
            device,
        })
    }

    /// 從 safetensors 張量字典載入碼本權重
    pub fn from_safetensors(
        tensors: &std::collections::HashMap<String, Tensor>,
        key: &str,
    ) -> crate::Result<Self> {
        let weights = tensors
            .get(key)
            .ok_or_else(|| Error::Weight(format!("Missing codebook tensor: {key}")))?;
        Self::new(weights.clone())
    }

    /// 查表：取得單層單個 Token 的嵌入向量
    ///
    /// 使用扁平化索引：`idx = layer * codebook_size + token`
    #[inline]
    pub fn lookup(&self, layer: usize, token: u16) -> crate::Result<Tensor> {
        let token_usize = token as usize;
        if token_usize >= self.codebook_size {
            return Err(Error::InvalidToken { token, layer });
        }
        let idx = layer * self.codebook_size + token_usize;
        let embedding = self.flat_weights.narrow(0, idx, 1)?;
        Ok(embedding)
    }

    /// 批次查表：並行查詢所有層的 Token
    ///
    /// # 參數
    /// - `tokens`: 形狀為 `(num_layers,)` 的 Token 陣列
    ///
    /// # 回傳值
    /// 形狀為 `(num_layers, 1, embedding_dim)` 的堆疊嵌入張量
    ///
    /// # 實現細節
    /// - 16 層查表透過 `rayon::par_iter` 並行執行
    /// - 容錯：單層失敗時以上層均值填充並記錄 warn
    pub fn batch_lookup(&self, tokens: &[u16]) -> crate::Result<Tensor> {
        let n = tokens.len();
        if n != self.num_layers {
            return Err(Error::Config(format!(
                "Expected {} tokens for {} layers, got {n}",
                self.num_layers, self.num_layers
            )));
        }

        // 並行查表：每層一個獨立任務
        let embeddings: Vec<crate::Result<Tensor>> = (0..n)
            .into_par_iter()
            .map(|layer| {
                let token = tokens[layer];
                self.lookup(layer, token).or_else(|e| {
                    // 容錯降級：單層失敗時回退到均值
                    log::warn!(
                        "Codebook lookup failed at layer {layer}, token={token}: {e}. Using mean embedding."
                    );
                    self.mean_embedding(layer)
                })
            })
            .collect();

        // 收集結果，若全部失敗則回傳 error
        let valid: Vec<Tensor> = embeddings.into_iter().filter_map(|r| r.ok()).collect();
        if valid.is_empty() {
            return Err(Error::Decode {
                layer: 0,
                reason: "All codebook layers failed".to_string(),
            });
        }

        // 堆疊為 (num_layers, 1, embedding_dim)
        Tensor::stack(&valid, 0).map_err(Into::into)
    }

    /// 取得指定層的均值嵌入向量
    #[inline]
    fn mean_embedding(&self, layer: usize) -> crate::Result<Tensor> {
        let start = layer * self.codebook_size;
        let layer_weights = self.flat_weights.narrow(0, start, self.codebook_size)?;
        let mean = layer_weights.mean(0).map_err(crate::Error::from)?;
        mean.reshape((1, self.embedding_dim))
            .map_err(crate::Error::from)
    }

    // ------------------------------------------------------------------
    // 存取子
    // ------------------------------------------------------------------

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 回傳底層權重張量的參考
    pub fn weights(&self) -> &Tensor {
        &self.flat_weights
    }
}

// ---------------------------------------------------------------------------
// 平行碼本處理器（高層封裝）
// ---------------------------------------------------------------------------

/// 平行碼本處理器
///
/// 封裝 `CodebookLookup` 並提供更高層的批次介面。
pub struct ParallelCodebook {
    inner: CodebookLookup,
}

impl ParallelCodebook {
    /// 從 CodebookLookup 實例建立
    pub fn new(lookup: CodebookLookup) -> Self {
        Self { inner: lookup }
    }

    /// 平行解碼：將多層 Token 序列轉換為嵌入張量
    ///
    /// # 參數
    /// - `tokens`: 形狀為 `(num_layers,)` 的 Token 陣列
    ///
    /// # 回傳值
    /// 形狀為 `(num_layers, embedding_dim)` 的嵌入張量（已 squeeze 維度 1）
    pub fn decode(&self, tokens: &[u16]) -> crate::Result<Tensor> {
        let stacked = self.inner.batch_lookup(tokens)?;
        // squeeze 維度 1: (num_layers, 1, embedding_dim) -> (num_layers, embedding_dim)
        stacked.squeeze(1).map_err(Into::into)
    }

    pub fn inner(&self) -> &CodebookLookup {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// 單元測試
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    fn test_device() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_codebook_creation() {
        let device = test_device();
        // (num_layers=16, codebook_size=2048, embedding_dim=512)
        let weights = Tensor::zeros((16, 2048, 512), candle_core::DType::F32, &device).unwrap();
        let lookup = CodebookLookup::new(weights).unwrap();
        assert_eq!(lookup.num_layers(), 16);
        assert_eq!(lookup.codebook_size(), 2048);
        assert_eq!(lookup.embedding_dim(), 512);
    }

    #[test]
    fn test_single_lookup() {
        let device = test_device();
        let weights = Tensor::ones((16, 2048, 512), candle_core::DType::F32, &device).unwrap();
        let lookup = CodebookLookup::new(weights).unwrap();
        let emb = lookup.lookup(0, 42).unwrap();
        assert_eq!(emb.dims(), &[1, 512]);
    }

    #[test]
    fn test_batch_lookup() {
        let device = test_device();
        let weights = Tensor::ones((16, 2048, 512), candle_core::DType::F32, &device).unwrap();
        let lookup = CodebookLookup::new(weights).unwrap();
        let tokens: Vec<u16> = (0..16).map(|i| i as u16 * 100).collect();
        let result = lookup.batch_lookup(&tokens).unwrap();
        assert_eq!(result.dims(), &[16, 1, 512]);
    }

    #[test]
    fn test_invalid_token() {
        let device = test_device();
        let weights = Tensor::zeros((16, 2048, 512), candle_core::DType::F32, &device).unwrap();
        let lookup = CodebookLookup::new(weights).unwrap();
        let result = lookup.lookup(0, 3000);
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::InvalidToken { token, layer } => {
                assert_eq!(token, 3000);
                assert_eq!(layer, 0);
            }
            e => panic!("Expected InvalidToken, got {e}"),
        }
    }

    #[test]
    fn test_wrong_dims() {
        let device = test_device();
        // 1D tensor is invalid for codebook
        let weights = Tensor::zeros((16,), candle_core::DType::F32, &device).unwrap();
        let result = CodebookLookup::new(weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_parallel_codebook() {
        let device = test_device();
        let weights = Tensor::ones((16, 2048, 512), candle_core::DType::F32, &device).unwrap();
        let lookup = CodebookLookup::new(weights).unwrap();
        let pc = ParallelCodebook::new(lookup);
        let tokens: Vec<u16> = (0..16).collect();
        let emb = pc.decode(&tokens).unwrap();
        assert_eq!(emb.dims(), &[16, 512]);
    }
}
