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
    /// `assistant_token_id` from `config.json`
    pub assistant_token_id: u32,
    /// `im_start_token_id` from `config.json`
    pub im_start_token_id: u32,
    /// `im_end_token_id` from `config.json`
    pub im_end_token_id: u32,
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
    pub spk_is_dialect: Vec<(String, Option<String>)>,

    /// 子碼本預測器配置
    pub code_predictor: CodePredictorConfig,
}

impl TalkerConfig {
    /// 取得可用語言清單（保留 metadata 載入順序）。
    pub fn supported_languages(&self) -> Vec<&str> {
        self.codec_language_id
            .iter()
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// 取得可用 speaker 清單（保留 metadata 載入順序）。
    pub fn supported_speakers(&self) -> Vec<&str> {
        self.spk_id.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// 依語言名稱（不分大小寫）回傳對應 codec 語言 token。
    pub fn language_id_for(&self, language: &str) -> Option<u32> {
        let normalized = normalize_lookup_key(language);
        self.codec_language_id
            .iter()
            .find(|(name, _)| normalize_lookup_key(name) == normalized)
            .map(|(_, id)| *id)
    }

    /// 依 speaker 名稱（不分大小寫）回傳對應 speaker token。
    pub fn speaker_id_for(&self, speaker: &str) -> Option<u32> {
        let normalized = normalize_lookup_key(speaker);
        self.spk_id
            .iter()
            .find(|(name, _)| normalize_lookup_key(name) == normalized)
            .map(|(_, id)| *id)
    }

    /// 依 speaker 名稱（不分大小寫）回傳方言對應語言鍵。
    pub fn speaker_dialect_for(&self, speaker: &str) -> Option<&str> {
        let normalized = normalize_lookup_key(speaker);
        self.spk_is_dialect
            .iter()
            .find(|(name, _)| normalize_lookup_key(name) == normalized)
            .and_then(|(_, dialect)| dialect.as_deref())
    }
}

fn normalize_lookup_key(value: &str) -> String {
    let normalized: String = value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();

    match normalized.as_str() {
        "zh" | "zhcn" | "zhtw" => "chinese".to_string(),
        "en" => "english".to_string(),
        "es" => "spanish".to_string(),
        "fr" => "french".to_string(),
        "de" => "german".to_string(),
        "it" => "italian".to_string(),
        "ja" => "japanese".to_string(),
        "ko" => "korean".to_string(),
        "ru" => "russian".to_string(),
        "pt" => "portuguese".to_string(),
        other => other.to_string(),
    }
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
            assistant_token_id: 77091,
            im_start_token_id: 151644,
            im_end_token_id: 151645,
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

#[cfg(test)]
mod tests {
    use super::{TalkerConfig, normalize_lookup_key};

    fn custom_voice_like_config() -> TalkerConfig {
        let mut config = TalkerConfig::default();
        config.codec_language_id = vec![
            ("chinese".into(), 2055),
            ("english".into(), 2050),
            ("beijing_dialect".into(), 9001),
        ];
        config.spk_id = vec![("Dylan".into(), 2878)];
        config.spk_is_dialect = vec![("Dylan".into(), Some("beijing_dialect".into()))];
        config
    }

    #[test]
    fn supports_default_languages_and_speakers_deterministically() {
        let config = TalkerConfig::default();
        assert_eq!(
            config.supported_languages(),
            vec![
                "chinese",
                "english",
                "german",
                "italian",
                "portuguese",
                "spanish",
                "japanese",
                "korean",
                "french",
                "russian",
            ]
        );
        assert_eq!(config.supported_speakers(), Vec::<&str>::new());
    }

    #[test]
    fn language_lookup_is_case_and_alias_aware() {
        let config = TalkerConfig::default();
        assert_eq!(config.language_id_for("zh-CN"), Some(2055));
        assert_eq!(config.language_id_for("En"), Some(2050));
        assert_eq!(config.language_id_for("  FR  "), Some(2061));
        assert_eq!(config.language_id_for("kOrEaN"), Some(2064));
        assert_eq!(config.language_id_for("unknown"), None);
    }

    #[test]
    fn speaker_lookup_is_case_insensitive() {
        let config = custom_voice_like_config();
        assert_eq!(config.speaker_id_for("dylan"), Some(2878));
        assert_eq!(config.speaker_id_for("DYLAn"), Some(2878));
    }

    #[test]
    fn dialect_lookup_uses_case_insensitive_speaker() {
        let config = custom_voice_like_config();
        assert_eq!(config.speaker_dialect_for("dylan"), Some("beijing_dialect"));
        assert_eq!(config.speaker_dialect_for("DYLAn"), Some("beijing_dialect"));
        assert_eq!(config.speaker_dialect_for("eric"), None);
    }

    #[test]
    fn normalize_lookup_key_maps_aliases() {
        assert_eq!(normalize_lookup_key("zh"), "chinese");
        assert_eq!(normalize_lookup_key("zh-CN"), "chinese");
        assert_eq!(normalize_lookup_key("en"), "english");
        assert_eq!(normalize_lookup_key("pt"), "portuguese");
        assert_eq!(normalize_lookup_key("es"), "spanish");
        assert_eq!(normalize_lookup_key("ja"), "japanese");
    }
}
