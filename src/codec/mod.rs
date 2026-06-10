//! # 解碼器核心模組
//!
//! 包含 12Hz 即時模式與 25Hz 高品質模式的解碼器元件。

mod activation;
mod causal_conv;
mod codebook;
mod decoder_blocks;
mod flow_matching;
mod transformer;

pub use activation::snake;
pub use causal_conv::{CausalConv1d, CausalConvConfig, CausalConvState};
pub use codebook::{CodebookLookup, ParallelCodebook};
pub use decoder_blocks::{snake_beta, ConvNeXtBlock, DecoderBlock, ResidualUnit, UpsampleBlock};
pub use flow_matching::{
    DiTBackbone, DiTConfig, DiTTimestepEmbedding, FlowMatchingDecoder, OdeSolver, OdeSolverConfig,
    OdeSolverType,
};
pub use transformer::{PreTransformer, PreTransformerConfig, RMSNorm, TransformerBlock};
