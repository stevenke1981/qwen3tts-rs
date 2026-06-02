//! # Talker LLM 配置
//!
//! 對應 `Qwen3TTSTalkerConfig` 與 `Qwen3TTSTalkerCodePredictorConfig`。

/// Talker 主模型配置（28 層 Qwen2 風格）
#[derive(Debug, Clone)]
pub struct TalkerConfig {
    /// 隱藏維度
    pub hidden_size: usize,
    /// 中間層維度（SwiGLU FFN）
    pub intermediate_size: usize,
    /// 注意力頭數
    pub num_attention_heads: usize,
    /// KV 頭數（GQA）
    pub num_key_value_heads: usize,
    /// 每頭維度（與 hidden_size/num_heads 解耦）
    pub head_dim: usize,
    /// 層數
    pub num_hidden_layers: usize,
    /// 文字嵌入維度（text_embedding 的輸出維度）
    pub text_hidden_size: usize,
    /// 文字詞彙量
    pub text_vocab_size: usize,
    /// 合併碼本詞彙量（codec_embedding / codec_head）
    pub vocab_size: usize,
    /// 碼本組數（= 16）
    pub num_code_groups: usize,
    /// 最大位置編碼
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon
    pub rms_norm_eps: f64,
    /// RoPE theta
    pub rope_theta: f64,
    /// 3D RoPE section 分割
    pub mrope_section: Vec<usize>,
    /// RoPE interleaved 模式
    pub rope_interleaved: bool,
    /// 隱藏層激活函數
    pub hidden_act: String,
    /// 注意力 bias
    pub attention_bias: bool,
    /// 注意力 dropout
    pub attention_dropout: f64,
    /// sliding window（None = 使用因果遮罩）
    pub sliding_window: Option<usize>,

    // --- 特殊 Token ID ---
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    pub codec_pad_id: u32,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,

    /// 語言 ID 對照表
    pub codec_language_id: Vec<(String, u32)>,
    /// 說話者 ID 對照表（可選）
    pub spk_id: Vec<(String, u32)>,
    /// 說話者是否為方言
    pub spk_is_dialect: Vec<(String, bool)>,

    /// 子碼本預測器配置
    pub code_predictor: CodePredictorConfig,
}

/// 子碼本預測器配置（5 層）
#[derive(Debug, Clone)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,      // = 2048
    pub num_code_groups: usize, // = 16
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub layer_types: Vec<String>,
}

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 3072,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            num_hidden_layers: 28,
            text_hidden_size: 2048,
            text_vocab_size: 151936,
            vocab_size: 3072,
            num_code_groups: 16,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            mrope_section: vec![24, 20, 20],
            rope_interleaved: true,
            hidden_act: "silu".into(),
            attention_bias: false,
            attention_dropout: 0.0,
            sliding_window: None,
            codec_bos_id: 2149,
            codec_eos_token_id: 2150,
            codec_think_id: 2154,
            codec_nothink_id: 2155,
            codec_think_bos_id: 2156,
            codec_think_eos_id: 2157,
            codec_pad_id: 2148,
            tts_bos_token_id: 151672,
            tts_eos_token_id: 151673,
            tts_pad_token_id: 151671,
            codec_language_id: vec![
                ("chinese".into(), 2055),
                ("english".into(), 2050),
                ("german".into(), 2053),
                ("italian".into(), 2070),
                ("portuguese".into(), 2071),
                ("spanish".into(), 2054),
                ("japanese".into(), 2058),
                ("korean".into(), 2064),
                ("french".into(), 2061),
                ("russian".into(), 2069),
            ],
            spk_id: vec![],
            spk_is_dialect: vec![],
            code_predictor: CodePredictorConfig {
                hidden_size: 1024,
                intermediate_size: 3072,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                head_dim: 128,
                num_hidden_layers: 5,
                vocab_size: 2048,
                num_code_groups: 16,
                max_position_embeddings: 65536,
                rms_norm_eps: 1e-6,
                rope_theta: 1_000_000.0,
                hidden_act: "silu".into(),
                attention_bias: false,
                attention_dropout: 0.0,
                layer_types: vec![
                    "full_attention".into(),
                    "full_attention".into(),
                    "full_attention".into(),
                    "full_attention".into(),
                    "full_attention".into(),
                ],
            },
        }
    }
}
