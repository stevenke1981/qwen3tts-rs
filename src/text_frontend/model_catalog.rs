//! Qwen3-TTS model capability catalog.
//!
//! This module provides:
//! - Static model catalog for discovery/UI display
//! - Parsed `config.json` metadata schema parsing and validation
//! - Capability derivation from parsed metadata for runtime validation

use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use crate::talker::sampling::SamplingOptions;

/// Supported language tags used by examples and user-facing docs.
pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "auto", "en", "zh", "zh-cn", "zh-tw", "ja", "ko", "fr", "de", "es", "it",
];

const GENERATION_CONFIG_FILE: &str = "generation_config.json";

fn default_talker_do_sample() -> bool {
    true
}

fn default_talker_temperature() -> f64 {
    0.9
}

fn default_talker_top_k() -> usize {
    50
}

fn default_talker_top_p() -> f64 {
    1.0
}

fn default_talker_repetition_penalty() -> f64 {
    1.05
}

fn default_subtalker_do_sample() -> bool {
    true
}

fn default_subtalker_temperature() -> f64 {
    0.9
}

fn default_subtalker_top_k() -> usize {
    50
}

fn default_subtalker_top_p() -> f64 {
    1.0
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BranchSamplingConfig {
    /// 是否啟用隨機取樣。
    pub do_sample: bool,
    /// 取樣參數。
    pub options: SamplingOptions,
}

/// Sampling configuration used by the Talker (codebook 0) and Subtalker (codebooks 1-15).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationSamplingConfig {
    /// Talker sampling (codebook 0 branch).
    pub talker: BranchSamplingConfig,
    /// Subtalker sampling (codebooks 1-15).
    pub subtalker: BranchSamplingConfig,
}

/// Resolved sampling plan used by the production Candle generation path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EffectiveSamplingPlan {
    pub talker: BranchSamplingConfig,
    pub subtalker: BranchSamplingConfig,
}

impl EffectiveSamplingPlan {
    pub fn uses_explicit_sampler(self) -> bool {
        self.talker.do_sample || self.subtalker.do_sample
    }
}

/// Apply public synthesis overrides to Talker only and resolve both branch
/// flags used by `CandleLLM::synthesize`.
pub fn resolve_effective_sampling_plan(
    config: GenerationSamplingConfig,
    temperature: f64,
    top_k: u32,
    top_p: f64,
) -> crate::Result<EffectiveSamplingPlan> {
    if !temperature.is_finite() {
        return Err(crate::Error::Config(
            "SynthesisOptions.temperature must be finite".into(),
        ));
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(crate::Error::Config(
            "SynthesisOptions.top_p must be finite and in (0.0, 1.0]".into(),
        ));
    }
    let mut talker = config.talker.options;
    talker.temperature = temperature;
    talker.top_k = top_k as usize;
    talker.top_p = top_p;
    Ok(EffectiveSamplingPlan {
        talker: BranchSamplingConfig {
            do_sample: config.talker.do_sample && temperature > 0.0,
            options: talker,
        },
        subtalker: config.subtalker,
    })
}

impl GenerationSamplingConfig {
    /// Default configuration matching repository convention when `generation_config.json`
    /// is missing or intentionally not loaded.
    pub fn default() -> Self {
        Self {
            talker: BranchSamplingConfig {
                do_sample: default_talker_do_sample(),
                options: SamplingOptions {
                    temperature: default_talker_temperature(),
                    top_k: default_talker_top_k(),
                    top_p: default_talker_top_p(),
                    repetition_penalty: default_talker_repetition_penalty(),
                },
            },
            subtalker: BranchSamplingConfig {
                do_sample: default_subtalker_do_sample(),
                options: SamplingOptions {
                    temperature: default_subtalker_temperature(),
                    top_k: default_subtalker_top_k(),
                    top_p: default_subtalker_top_p(),
                    repetition_penalty: 1.0,
                },
            },
        }
    }

    /// Parse `generation_config.json` from an explicit path.
    pub fn from_path<P: AsRef<Path>>(generation_config_path: P) -> crate::Result<Self> {
        let config_path = generation_config_path.as_ref();
        let payload = fs::read_to_string(config_path).map_err(|err| {
            crate::Error::Config(format!(
                "unable to read generation_config.json at {}: {err}",
                config_path.display()
            ))
        })?;
        let parsed: ParsedGenerationConfig = serde_json::from_str(&payload).map_err(|err| {
            crate::Error::Config(format!(
                "invalid generation_config.json at {}: {err}",
                config_path.display()
            ))
        })?;
        Self::from_parsed(parsed)
    }

    /// Parse `generation_config.json` from a model directory.
    ///
    /// Missing `generation_config.json` is treated as defaults, preserving
    /// backward-compatibility with older model snapshots.
    pub fn from_model_dir<P: AsRef<Path>>(model_dir: P) -> crate::Result<Self> {
        let config_path = model_dir.as_ref().join(GENERATION_CONFIG_FILE);
        if !config_path.exists() {
            return Ok(Self::default());
        }
        Self::from_path(config_path)
    }

    fn from_parsed(config: ParsedGenerationConfig) -> crate::Result<Self> {
        let do_sample = config.do_sample.unwrap_or_else(default_talker_do_sample);
        let subtalker_do_sample = config
            .subtalker_dosample
            .unwrap_or_else(default_subtalker_do_sample);

        let talker = SamplingOptions {
            temperature: config
                .temperature
                .unwrap_or_else(default_talker_temperature),
            top_k: config.top_k.unwrap_or_else(default_talker_top_k),
            top_p: config.top_p.unwrap_or_else(default_talker_top_p),
            repetition_penalty: config
                .repetition_penalty
                .unwrap_or_else(default_talker_repetition_penalty),
        };
        let subtalker = SamplingOptions {
            temperature: config
                .subtalker_temperature
                .unwrap_or_else(default_subtalker_temperature),
            top_k: config
                .subtalker_top_k
                .unwrap_or_else(default_subtalker_top_k),
            top_p: config
                .subtalker_top_p
                .unwrap_or_else(default_subtalker_top_p),
            repetition_penalty: 1.0,
        };

        validate_generation_sampling("talker", do_sample, &talker)?;
        validate_generation_sampling("subtalker", subtalker_do_sample, &subtalker)?;
        Ok(Self {
            talker: BranchSamplingConfig {
                do_sample,
                options: talker,
            },
            subtalker: BranchSamplingConfig {
                do_sample: subtalker_do_sample,
                options: subtalker,
            },
        })
    }
}

#[derive(Debug, Deserialize)]
struct ParsedGenerationConfig {
    do_sample: Option<bool>,
    temperature: Option<f64>,
    top_k: Option<usize>,
    top_p: Option<f64>,
    repetition_penalty: Option<f64>,

    subtalker_dosample: Option<bool>,
    subtalker_temperature: Option<f64>,
    subtalker_top_k: Option<usize>,
    subtalker_top_p: Option<f64>,
}

fn validate_generation_sampling(
    branch: &str,
    do_sample: bool,
    sampling: &SamplingOptions,
) -> crate::Result<()> {
    if !sampling.top_p.is_finite() || sampling.top_p <= 0.0 || sampling.top_p > 1.0 {
        return Err(crate::Error::Config(format!(
            "generation_config[{branch}].top_p must be finite and in (0.0, 1.0]"
        )));
    }

    if !sampling.repetition_penalty.is_finite() || sampling.repetition_penalty <= 0.0 {
        return Err(crate::Error::Config(format!(
            "generation_config[{branch}].repetition_penalty must be finite and > 0.0"
        )));
    }

    if !sampling.temperature.is_finite() {
        return Err(crate::Error::Config(format!(
            "generation_config[{branch}].temperature must be finite"
        )));
    }

    if do_sample && sampling.temperature <= 0.0 {
        return Err(crate::Error::Config(format!(
            "generation_config[{branch}].temperature must be > 0.0 when do_sample=true"
        )));
    }

    Ok(())
}

/// Generation mode requested by the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationMode {
    Auto,
    CustomVoice,
    VoiceDesign,
    VoiceClone,
}

impl GenerationMode {
    pub fn parse(value: &str) -> crate::Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "custom-voice" | "custom_voice" | "custom" => Ok(Self::CustomVoice),
            "voice-design" | "voice_design" | "design" => Ok(Self::VoiceDesign),
            "voice-clone" | "voice_clone" | "clone" => Ok(Self::VoiceClone),
            other => Err(crate::Error::Config(format!(
                "unknown generation mode: {other}; use auto/custom-voice/voice-design/voice-clone"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::CustomVoice => "custom-voice",
            Self::VoiceDesign => "voice-design",
            Self::VoiceClone => "voice-clone",
        }
    }
}

/// Instruction-control capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionControl {
    Full,
    None,
}

impl InstructionControl {
    pub fn label(self) -> &'static str {
        match self {
            Self::Full => "yes",
            Self::None => "-",
        }
    }
}

/// Static model capability entry (for display / discovery only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCapability {
    pub model_id: &'static str,
    pub parameters: &'static str,
    pub main_function: &'static str,
    pub languages: usize,
    pub streaming: bool,
    pub instruction_control: InstructionControl,
    pub recommended_scenario: &'static str,
    pub supports_speaker_presets: bool,
    pub supports_voice_design: bool,
    pub supports_voice_clone: bool,
}

const MODEL_TABLE: &[ModelCapability] = &[
    ModelCapability {
        model_id: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        parameters: "1.7B",
        main_function: "text-described voice design",
        languages: 10,
        streaming: true,
        instruction_control: InstructionControl::Full,
        recommended_scenario: "custom voice creation",
        supports_speaker_presets: false,
        supports_voice_design: true,
        supports_voice_clone: false,
    },
    ModelCapability {
        model_id: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        parameters: "1.7B",
        main_function: "9 preset voices plus instruction style control",
        languages: 10,
        streaming: true,
        instruction_control: InstructionControl::Full,
        recommended_scenario: "high quality narration and multi-character speech",
        supports_speaker_presets: true,
        supports_voice_design: false,
        supports_voice_clone: false,
    },
    ModelCapability {
        model_id: "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        parameters: "1.7B",
        main_function: "3-second voice cloning and fine-tuning base",
        languages: 10,
        streaming: true,
        instruction_control: InstructionControl::None,
        recommended_scenario: "voice cloning and fine-tuning",
        supports_speaker_presets: false,
        supports_voice_design: false,
        supports_voice_clone: true,
    },
    ModelCapability {
        model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        parameters: "0.6B",
        main_function: "9 preset voices without instruction control",
        languages: 10,
        streaming: true,
        instruction_control: InstructionControl::None,
        recommended_scenario: "lightweight deployment",
        supports_speaker_presets: true,
        supports_voice_design: false,
        supports_voice_clone: false,
    },
    ModelCapability {
        model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        parameters: "0.6B",
        main_function: "3-second voice cloning and fine-tuning base",
        languages: 10,
        streaming: true,
        instruction_control: InstructionControl::None,
        recommended_scenario: "resource-constrained environments",
        supports_speaker_presets: false,
        supports_voice_design: false,
        supports_voice_clone: true,
    },
];

/// Parsed metadata-supported runtime model families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsModelType {
    Base,
    CustomVoice,
    VoiceDesign,
}

impl TtsModelType {
    fn parse(value: &str) -> crate::Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "base" => Ok(Self::Base),
            "customvoice" | "custom_voice" => Ok(Self::CustomVoice),
            "voicedesign" | "voice_design" => Ok(Self::VoiceDesign),
            other => Err(crate::Error::Config(format!(
                "unknown tts_model_type: {other}"
            ))),
        }
    }
}

/// Owned metadata parsed from an official Qwen3-TTS model `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMetadata {
    pub model_type: String,
    pub tokenizer_type: String,
    pub assistant_token_id: u32,
    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
    pub tts_model_size: String,
    pub tts_model_type: TtsModelType,
    pub talker_config_model_type: String,
    pub talker_code_predictor_config_model_type: String,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    pub codec_pad_id: u32,
    /// `talker_config.vocab_size` (combined codec + speaker + special token vocab).
    pub talker_vocab_size: usize,
    /// `talker_config.text_vocab_size` (main text/tokenizer vocab).
    pub talker_text_vocab_size: usize,
    /// `talker_config.num_code_groups` (typically 16).
    pub talker_num_code_groups: usize,
    /// `talker_config.codec_language_id`
    pub codec_language_id: Vec<(String, u32)>,
    /// `talker_config.spk_id`
    pub spk_id: Vec<(String, u32)>,
    /// `talker_config.spk_is_dialect` (`false` or dialect-key string)
    pub spk_is_dialect: Vec<(String, Option<String>)>,
    pub talker_num_hidden_layers: usize,
    pub talker_num_attention_heads: usize,
    pub talker_hidden_size: usize,
    pub talker_head_dim: usize,
    pub talker_code_predictor_num_hidden_layers: usize,
    pub talker_code_predictor_num_attention_heads: usize,
    pub talker_code_predictor_hidden_size: usize,
    /// `talker_config.rope_scaling.interleaved`
    pub rope_scaling_interleaved: bool,
    /// `talker_config.rope_scaling.mrope_section`
    pub rope_scaling_mrope_section: Vec<usize>,
    /// `talker_config.rope_scaling.rope_theta`
    pub rope_scaling_rope_theta: f64,
}

impl ModelMetadata {
    /// Parse metadata from an explicit `config.json` path.
    pub fn from_config_path<P: AsRef<Path>>(config_path: P) -> crate::Result<Self> {
        let config_path = config_path.as_ref();
        let config_data = fs::read_to_string(config_path).map_err(|err| {
            crate::Error::Config(format!(
                "unable to read config.json at {}: {err}",
                config_path.display()
            ))
        })?;
        let parsed: ParsedConfig = serde_json::from_str(&config_data).map_err(|err| {
            crate::Error::Config(format!(
                "invalid config.json at {}: {err}",
                config_path.display()
            ))
        })?;
        Self::from_parsed(parsed).map_err(|err| {
            crate::Error::Config(format!(
                "invalid model metadata in {}: {err}",
                config_path.display()
            ))
        })
    }

    /// Parse metadata from a model snapshot directory.
    pub fn from_model_dir<P: AsRef<Path>>(model_dir: P) -> crate::Result<Self> {
        let dir = model_dir.as_ref();
        let config_path = dir.join("config.json");
        if !config_path.exists() {
            return Err(crate::Error::Config(format!(
                "missing config.json in {}",
                dir.display()
            )));
        }
        Self::from_config_path(config_path)
    }

    /// Metadata-backed generation mode.
    pub fn runtime_generation_mode(&self) -> GenerationMode {
        match self.tts_model_type {
            TtsModelType::VoiceDesign => GenerationMode::VoiceDesign,
            TtsModelType::CustomVoice => GenerationMode::CustomVoice,
            TtsModelType::Base => GenerationMode::VoiceClone,
        }
    }

    pub fn supports_speaker_presets(&self) -> bool {
        matches!(self.tts_model_type, TtsModelType::CustomVoice)
    }

    pub fn supports_voice_design(&self) -> bool {
        matches!(self.tts_model_type, TtsModelType::VoiceDesign)
    }

    pub fn supports_voice_clone(&self) -> bool {
        matches!(self.tts_model_type, TtsModelType::Base)
    }

    pub fn supports_instruction_control(&self) -> bool {
        matches!(
            (self.tts_model_size.as_str(), self.tts_model_type),
            ("1b7", TtsModelType::CustomVoice | TtsModelType::VoiceDesign)
        )
    }

    fn from_parsed(config: ParsedConfig) -> crate::Result<Self> {
        if config.model_type != "qwen3_tts" {
            return Err(crate::Error::Config(format!(
                "unsupported model_type: {} (expected qwen3_tts)",
                config.model_type
            )));
        }
        if config.tokenizer_type != "qwen3_tts_tokenizer_12hz" {
            return Err(crate::Error::Config(format!(
                "unsupported tokenizer_type: {} (expected qwen3_tts_tokenizer_12hz)",
                config.tokenizer_type
            )));
        }
        if !matches!(config.tts_model_size.as_str(), "0b6" | "1b7") {
            return Err(crate::Error::Config(format!(
                "unsupported tts_model_size: {} (expected 0b6 or 1b7)",
                config.tts_model_size
            )));
        }
        let tts_model_type = TtsModelType::parse(&config.tts_model_type)?;

        let talker = &config.talker_config;
        if talker.model_type != "qwen3_tts_talker" {
            return Err(crate::Error::Config(format!(
                "unsupported talker_config.model_type: {} (expected qwen3_tts_talker)",
                talker.model_type
            )));
        }
        if talker.code_predictor_config.model_type.as_str() != "qwen3_tts_talker_code_predictor" {
            return Err(crate::Error::Config(format!(
                "unsupported talker_config.code_predictor_config.model_type: {} (expected qwen3_tts_talker_code_predictor)",
                talker.code_predictor_config.model_type
            )));
        }

        validate_positive("talker_config.num_hidden_layers", talker.num_hidden_layers)?;
        validate_positive("talker_config.hidden_size", talker.hidden_size)?;
        validate_positive("talker_config.head_dim", talker.head_dim)?;
        validate_positive(
            "talker_config.num_attention_heads",
            talker.num_attention_heads,
        )?;
        validate_positive(
            "talker_config.code_predictor_config.num_hidden_layers",
            talker.code_predictor_config.num_hidden_layers,
        )?;
        validate_positive(
            "talker_config.code_predictor_config.hidden_size",
            talker.code_predictor_config.hidden_size,
        )?;
        validate_positive(
            "talker_config.code_predictor_config.num_attention_heads",
            talker.code_predictor_config.num_attention_heads,
        )?;

        let talker_rope = &talker.rope_scaling;
        if talker_rope.mrope_section.len() != 3 {
            return Err(crate::Error::Config(
                "talker_config.rope_scaling.mrope_section must contain exactly 3 values".into(),
            ));
        }
        if talker_rope.mrope_section.iter().any(|value| *value == 0) {
            return Err(crate::Error::Config(
                "talker_config.rope_scaling.mrope_section must contain only positive values".into(),
            ));
        }
        let mrope_sum: usize = talker_rope.mrope_section.iter().sum();
        if mrope_sum * 2 != talker.head_dim {
            return Err(crate::Error::Config(format!(
                "talker_config.rope_scaling.mrope_section sum {} * 2 must equal talker_config.head_dim {}",
                mrope_sum, talker.head_dim
            )));
        }
        if !talker_rope.rope_theta.is_finite() || talker_rope.rope_theta <= 0.0 {
            return Err(crate::Error::Config(
                "talker_config.rope_scaling.rope_theta must be finite and > 0".into(),
            ));
        }

        if talker.text_vocab_size == 0 {
            return Err(crate::Error::Config(
                "talker_config.text_vocab_size must be > 0".into(),
            ));
        }
        if talker.vocab_size == 0 {
            return Err(crate::Error::Config(
                "talker_config.vocab_size must be > 0".into(),
            ));
        }
        if talker.num_code_groups == 0 {
            return Err(crate::Error::Config(
                "talker_config.num_code_groups must be > 0".into(),
            ));
        }
        if talker.num_code_groups < 16 {
            return Err(crate::Error::Config(
                "talker_config.num_code_groups must be at least 16".into(),
            ));
        }

        let assistant_token_id = validate_token_within_vocab(
            "assistant_token_id",
            config.assistant_token_id,
            talker.text_vocab_size,
        )?;
        let im_start_token_id = validate_token_within_vocab(
            "im_start_token_id",
            config.im_start_token_id,
            talker.text_vocab_size,
        )?;
        let im_end_token_id = validate_token_within_vocab(
            "im_end_token_id",
            config.im_end_token_id,
            talker.text_vocab_size,
        )?;
        let tts_bos_token_id = validate_token_within_vocab(
            "tts_bos_token_id",
            config.tts_bos_token_id,
            talker.text_vocab_size,
        )?;
        let tts_eos_token_id = validate_token_within_vocab(
            "tts_eos_token_id",
            config.tts_eos_token_id,
            talker.text_vocab_size,
        )?;
        let tts_pad_token_id = validate_token_within_vocab(
            "tts_pad_token_id",
            config.tts_pad_token_id,
            talker.text_vocab_size,
        )?;

        let codec_bos_id = validate_token_within_vocab(
            "talker_config.codec_bos_id",
            talker.codec_bos_id,
            talker.vocab_size,
        )?;
        let codec_eos_token_id = validate_token_within_vocab(
            "talker_config.codec_eos_token_id",
            talker.codec_eos_token_id,
            talker.vocab_size,
        )?;
        let codec_think_id = validate_token_within_vocab(
            "talker_config.codec_think_id",
            talker.codec_think_id,
            talker.vocab_size,
        )?;
        let codec_nothink_id = validate_token_within_vocab(
            "talker_config.codec_nothink_id",
            talker.codec_nothink_id,
            talker.vocab_size,
        )?;
        let codec_think_bos_id = validate_token_within_vocab(
            "talker_config.codec_think_bos_id",
            talker.codec_think_bos_id,
            talker.vocab_size,
        )?;
        let codec_think_eos_id = validate_token_within_vocab(
            "talker_config.codec_think_eos_id",
            talker.codec_think_eos_id,
            talker.vocab_size,
        )?;
        let codec_pad_id = validate_token_within_vocab(
            "talker_config.codec_pad_id",
            talker.codec_pad_id,
            talker.vocab_size,
        )?;

        let codec_language_id = parse_string_to_u32_map(
            "talker_config.codec_language_id",
            &talker.codec_language_id,
            |id| {
                validate_u32_in_vocab_range(
                    "talker_config.codec_language_id",
                    id,
                    talker.vocab_size,
                )
            },
        )?;
        let spk_id = parse_string_to_u32_map("talker_config.spk_id", &talker.spk_id, |id| {
            validate_u32_in_vocab_range("talker_config.spk_id", id, talker.vocab_size)
        })?;
        let spk_is_dialect = parse_speaker_dialect_map(
            "talker_config.spk_is_dialect",
            &talker.spk_is_dialect,
            &codec_language_id,
            if talker.spk_id.is_empty() {
                None
            } else {
                Some(&spk_id)
            },
        )?;

        if codec_language_id.is_empty() {
            return Err(crate::Error::Config(
                "talker_config.codec_language_id must contain at least one entry".into(),
            ));
        }

        if !spk_is_dialect.is_empty() && spk_is_dialect.len() != spk_id.len() {
            return Err(crate::Error::Config(
                "talker_config.spk_is_dialect must include each speaker from spk_id".into(),
            ));
        }
        if !spk_id.is_empty() && spk_is_dialect.is_empty() {
            return Err(crate::Error::Config(
                "talker_config.spk_is_dialect cannot be empty when spk_id exists".into(),
            ));
        }

        Ok(Self {
            model_type: config.model_type,
            tokenizer_type: config.tokenizer_type,
            assistant_token_id,
            im_start_token_id,
            im_end_token_id,
            tts_model_size: config.tts_model_size,
            tts_model_type,
            tts_bos_token_id,
            tts_eos_token_id,
            tts_pad_token_id,
            codec_bos_id,
            codec_eos_token_id,
            codec_think_id,
            codec_nothink_id,
            codec_think_bos_id,
            codec_think_eos_id,
            codec_pad_id,
            talker_vocab_size: talker.vocab_size,
            talker_text_vocab_size: talker.text_vocab_size,
            talker_num_code_groups: talker.num_code_groups,
            codec_language_id: codec_language_id.into_iter().collect(),
            spk_id: spk_id.into_iter().collect(),
            spk_is_dialect: spk_is_dialect.into_iter().collect(),
            talker_config_model_type: talker.model_type.clone(),
            talker_code_predictor_config_model_type: talker
                .code_predictor_config
                .model_type
                .clone(),
            talker_num_hidden_layers: talker.num_hidden_layers,
            talker_num_attention_heads: talker.num_attention_heads,
            talker_hidden_size: talker.hidden_size,
            talker_head_dim: talker.head_dim,
            talker_code_predictor_num_hidden_layers: talker.code_predictor_config.num_hidden_layers,
            talker_code_predictor_num_attention_heads: talker
                .code_predictor_config
                .num_attention_heads,
            talker_code_predictor_hidden_size: talker.code_predictor_config.hidden_size,
            rope_scaling_interleaved: talker_rope.interleaved,
            rope_scaling_mrope_section: talker_rope.mrope_section.clone(),
            rope_scaling_rope_theta: talker_rope.rope_theta,
        })
    }
}

fn validate_positive(name: &str, value: usize) -> crate::Result<()> {
    if value == 0 {
        return Err(crate::Error::Config(format!("{name} must be > 0")));
    }
    Ok(())
}

fn validate_token_within_vocab(name: &str, token_id: u32, vocab_size: usize) -> crate::Result<u32> {
    if token_id == 0 {
        return Err(crate::Error::Config(format!("{name} must be > 0")));
    }
    let max = u32::try_from(vocab_size).map_err(|_| {
        crate::Error::Config(format!(
            "invalid vocab_size {} for {name} (does not fit u32)",
            vocab_size
        ))
    })?;
    if token_id >= max {
        return Err(crate::Error::Config(format!(
            "{name}={} exceeds metadata vocab_size={vocab_size}",
            token_id
        )));
    }
    Ok(token_id)
}

fn validate_u32_in_vocab_range(
    name: &str,
    token_id: &u32,
    vocab_size: usize,
) -> crate::Result<u32> {
    validate_token_within_vocab(name, *token_id, vocab_size)
}

fn parse_string_to_u32_map(
    context: &str,
    values: &BTreeMap<String, u32>,
    validate_value: impl Fn(&u32) -> crate::Result<u32>,
) -> crate::Result<BTreeMap<String, u32>> {
    if values.is_empty() {
        return Ok(BTreeMap::new());
    }

    let mut normalized = BTreeSet::new();
    let mut parsed = BTreeMap::new();
    for (key, token_id) in values {
        if key.is_empty() {
            return Err(crate::Error::Config(format!(
                "{context} contains empty key"
            )));
        }
        let normalized_key = normalize_map_key(key);
        if !normalized.insert(normalized_key.clone()) {
            return Err(crate::Error::Config(format!(
                "{context} has duplicate case-folded key: {key}"
            )));
        }
        parsed.insert(key.clone(), validate_value(token_id)?);
    }
    Ok(parsed)
}

fn parse_speaker_dialect_map(
    context: &str,
    entries: &BTreeMap<String, Value>,
    codec_language_id: &BTreeMap<String, u32>,
    spk_id: Option<&BTreeMap<String, u32>>,
) -> crate::Result<BTreeMap<String, Option<String>>> {
    let mut seen = BTreeSet::new();
    let mut parsed = BTreeMap::new();
    let speaker_key_set = spk_id.map(|speakers| {
        speakers
            .keys()
            .map(|value| (normalize_map_key(value), value.clone()))
            .collect::<BTreeSet<_>>()
    });
    let language_key_set = codec_language_id
        .keys()
        .map(|key| normalize_map_key(key))
        .collect::<BTreeSet<_>>();
    for (speaker_key, value) in entries {
        if speaker_key.is_empty() {
            return Err(crate::Error::Config(format!(
                "{context} contains empty key"
            )));
        }
        let normalized_key = normalize_map_key(speaker_key);
        if !seen.insert(normalized_key.clone()) {
            return Err(crate::Error::Config(format!(
                "{context} has duplicate case-folded key: {speaker_key}"
            )));
        }
        if let Some(speakers) = &speaker_key_set {
            let mut found = false;
            for (_normalized_speaker, original_speaker) in speakers {
                if normalize_map_key(original_speaker) == normalized_key {
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(crate::Error::Config(format!(
                    "{context} contains unknown speaker key: {speaker_key}"
                )));
            }
        }

        let dialect = match value {
            Value::Bool(false) => None,
            Value::Bool(true) => {
                return Err(crate::Error::Config(format!(
                    "{context}.{speaker_key} = true is forbidden; use false or dialect key string"
                )));
            }
            Value::String(dialect) if dialect.is_empty() => {
                return Err(crate::Error::Config(format!(
                    "{context}.{speaker_key} dialect value cannot be empty"
                )));
            }
            Value::String(dialect) => {
                let dialect_key = normalize_map_key(dialect);
                if !language_key_set.contains(&dialect_key) {
                    return Err(crate::Error::Config(format!(
                        "{context}.{speaker_key} references unknown dialect key: {dialect}"
                    )));
                }
                Some(
                    codec_language_id
                        .iter()
                        .find(|(language_key, _)| normalize_map_key(language_key) == dialect_key)
                        .map(|(language_key, _)| language_key.clone())
                        .unwrap_or_else(|| dialect.clone()),
                )
            }
            _ => {
                return Err(crate::Error::Config(format!(
                    "{context}.{speaker_key} must be false or a dialect string"
                )));
            }
        };
        parsed.insert(speaker_key.clone(), dialect);
    }
    Ok(parsed)
}

fn normalize_map_key(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Parsed view of official config JSON.
#[derive(Deserialize)]
struct ParsedConfig {
    model_type: String,
    tokenizer_type: String,
    tts_model_size: String,
    tts_model_type: String,
    assistant_token_id: u32,
    im_start_token_id: u32,
    im_end_token_id: u32,
    tts_bos_token_id: u32,
    tts_eos_token_id: u32,
    tts_pad_token_id: u32,
    talker_config: ParsedTalkerConfig,
}

#[derive(Deserialize)]
struct ParsedTalkerConfig {
    #[serde(rename = "model_type")]
    model_type: String,
    #[serde(rename = "num_hidden_layers")]
    num_hidden_layers: usize,
    hidden_size: usize,
    head_dim: usize,
    #[serde(rename = "num_attention_heads")]
    num_attention_heads: usize,
    #[serde(rename = "code_predictor_config")]
    code_predictor_config: ParsedCodePredictorConfig,
    #[serde(rename = "num_code_groups")]
    num_code_groups: usize,
    #[serde(rename = "text_vocab_size")]
    text_vocab_size: usize,
    #[serde(rename = "vocab_size")]
    vocab_size: usize,
    #[serde(rename = "codec_bos_id")]
    codec_bos_id: u32,
    #[serde(rename = "codec_eos_token_id")]
    codec_eos_token_id: u32,
    #[serde(rename = "codec_think_id")]
    codec_think_id: u32,
    #[serde(rename = "codec_nothink_id")]
    codec_nothink_id: u32,
    #[serde(rename = "codec_think_bos_id")]
    codec_think_bos_id: u32,
    #[serde(rename = "codec_think_eos_id")]
    codec_think_eos_id: u32,
    #[serde(rename = "codec_pad_id")]
    codec_pad_id: u32,
    #[serde(rename = "codec_language_id", default)]
    codec_language_id: BTreeMap<String, u32>,
    #[serde(rename = "spk_id", default)]
    spk_id: BTreeMap<String, u32>,
    #[serde(rename = "spk_is_dialect", default)]
    spk_is_dialect: BTreeMap<String, Value>,
    rope_scaling: ParsedRopeScaling,
}

#[derive(Deserialize)]
struct ParsedCodePredictorConfig {
    #[serde(rename = "model_type")]
    model_type: String,
    #[serde(rename = "num_hidden_layers")]
    num_hidden_layers: usize,
    hidden_size: usize,
    #[serde(rename = "num_attention_heads")]
    num_attention_heads: usize,
}

#[derive(Deserialize)]
struct ParsedRopeScaling {
    interleaved: bool,
    mrope_section: Vec<usize>,
    rope_theta: f64,
}

/// Static model table for discovery and CLI display only.
pub fn model_table() -> &'static [ModelCapability] {
    MODEL_TABLE
}

/// Static lookup by model IDs or paths. Use only for non-runtime discovery display.
pub fn model_capability(model_id_or_path: &str) -> Option<&'static ModelCapability> {
    let normalized = normalize_model_key(model_id_or_path);
    MODEL_TABLE.iter().find(|model| {
        normalized.contains(&normalize_model_key(model.model_id))
            || normalize_model_key(model.model_id).contains(&normalized)
    })
}

/// Inference mode derived from parsed metadata.
pub fn infer_mode(metadata: &ModelMetadata) -> GenerationMode {
    metadata.runtime_generation_mode()
}

/// Resolve requested generation mode against model metadata.
pub fn resolve_generation_mode(
    metadata: &ModelMetadata,
    requested_mode: GenerationMode,
) -> GenerationMode {
    match requested_mode {
        GenerationMode::Auto => metadata.runtime_generation_mode(),
        other => other,
    }
}

/// Validate requested generation mode against parsed model metadata.
pub fn validate_generation_request(
    metadata: &ModelMetadata,
    requested_mode: GenerationMode,
    speaker: Option<&str>,
    instruct: Option<&str>,
    reference_audio: Option<&str>,
) -> crate::Result<()> {
    let has_speaker = speaker.map(str::trim).is_some_and(|s| !s.is_empty());
    let has_instruct = instruct.map(str::trim).is_some_and(|s| !s.is_empty());
    let has_reference_audio = reference_audio
        .map(str::trim)
        .is_some_and(|s| !s.is_empty());

    let resolved_mode = resolve_generation_mode(metadata, requested_mode);
    match resolved_mode {
        GenerationMode::Auto => {
            return Err(crate::Error::Config(
                "generation mode was not resolved before validation".into(),
            ));
        }
        GenerationMode::CustomVoice => {
            if !metadata.supports_speaker_presets() {
                return Err(crate::Error::Config(
                    "model does not support CustomVoice speaker presets; use a CustomVoice model"
                        .into(),
                ));
            }
            if !has_speaker {
                return Err(crate::Error::Config(
                    "custom-voice mode requires --speaker, e.g. --speaker Vivian".into(),
                ));
            }
            if has_reference_audio {
                return Err(crate::Error::Config(
                    "voice cloning does not apply to custom-voice mode; remove --reference-audio"
                        .into(),
                ));
            }
            if has_instruct && !metadata.supports_instruction_control() {
                return Err(crate::Error::Config(
                    "model does not support --instruct; use 1.7B-CustomVoice or 1.7B-VoiceDesign"
                        .into(),
                ));
            }
            Ok(())
        }
        GenerationMode::VoiceDesign => {
            if !metadata.supports_voice_design() {
                return Err(crate::Error::Config(
                    "model does not support VoiceDesign; use a 1.7B-VoiceDesign model".into(),
                ));
            }
            if has_speaker {
                return Err(crate::Error::Config(
                    "voice-design mode does not accept --speaker".into(),
                ));
            }
            if has_reference_audio {
                return Err(crate::Error::Config(
                    "voice-design mode does not accept --reference-audio".into(),
                ));
            }
            if !has_instruct {
                return Err(crate::Error::Config(
                    "voice-design mode requires --instruct or --instruct-file".into(),
                ));
            }
            Ok(())
        }
        GenerationMode::VoiceClone => {
            if !metadata.supports_voice_clone() {
                return Err(crate::Error::Config(
                    "model does not support voice clone; use a Base model".into(),
                ));
            }
            if has_speaker {
                return Err(crate::Error::Config(
                    "voice-clone mode does not accept --speaker".into(),
                ));
            }
            if has_instruct {
                return Err(crate::Error::Config(
                    "voice-clone mode does not accept --instruct".into(),
                ));
            }
            if !has_reference_audio {
                return Err(crate::Error::Config(
                    "voice-clone mode requires --reference-audio with at least 3 seconds of audio"
                        .into(),
                ));
            }
            Ok(())
        }
    }
}

fn normalize_model_key(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        GenerationMode, InstructionControl, ModelMetadata, TtsModelType, model_capability,
        model_table, resolve_generation_mode, validate_generation_request,
    };
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::SystemTime;

    static TEST_CONFIG_COUNTER: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug)]
    struct FakeModelConfig {
        assistant_token_id: u32,
        im_start_token_id: u32,
        im_end_token_id: u32,
        tts_bos_token_id: u32,
        tts_eos_token_id: u32,
        tts_pad_token_id: u32,
        model_type: &'static str,
        tokenizer_type: &'static str,
        tts_model_size: &'static str,
        tts_model_type: &'static str,
        talker_num_hidden_layers: usize,
        talker_head_dim: usize,
        talker_hidden_size: usize,
        talker_num_attention_heads: usize,
        talker_vocab_size: usize,
        talker_text_vocab_size: usize,
        codec_bos_id: u32,
        codec_eos_token_id: u32,
        codec_think_id: u32,
        codec_nothink_id: u32,
        codec_think_bos_id: u32,
        codec_think_eos_id: u32,
        codec_pad_id: u32,
        cp_num_hidden_layers: usize,
        cp_hidden_size: usize,
        cp_num_attention_heads: usize,
        num_code_groups: usize,
        rope_theta: f64,
        codec_language_id: &'static str,
        spk_id: &'static str,
        spk_is_dialect: &'static str,
        interleaved: bool,
        mrope_section: Vec<usize>,
        talker_model_type: &'static str,
        cp_model_type: &'static str,
    }

    impl FakeModelConfig {
        fn config_json(&self) -> String {
            let mrope_section: String = self
                .mrope_section
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                r#"{{
  "assistant_token_id": {},
  "im_start_token_id": {},
  "im_end_token_id": {},
  "tts_bos_token_id": {},
  "tts_eos_token_id": {},
  "tts_pad_token_id": {},
  "model_type": "{}",
  "tokenizer_type": "{}",
  "tts_model_size": "{}",
  "tts_model_type": "{}",
  "talker_config": {{
    "model_type": "{}",
    "num_hidden_layers": {},
    "hidden_size": {},
    "head_dim": {},
    "num_attention_heads": {},
    "vocab_size": {},
    "text_vocab_size": {},
    "num_code_groups": {},
    "codec_bos_id": {},
    "codec_eos_token_id": {},
    "codec_think_id": {},
    "codec_nothink_id": {},
    "codec_think_bos_id": {},
    "codec_think_eos_id": {},
    "codec_pad_id": {},
    "codec_language_id": {},
    "spk_id": {},
    "spk_is_dialect": {},
    "code_predictor_config": {{
      "model_type": "{}",
      "num_hidden_layers": {},
      "hidden_size": {},
      "num_attention_heads": {}
    }},
    "rope_scaling": {{
      "interleaved": {},
      "rope_theta": {},
      "mrope_section": [{}]
    }}
  }}
}}"#,
                self.assistant_token_id,
                self.im_start_token_id,
                self.im_end_token_id,
                self.tts_bos_token_id,
                self.tts_eos_token_id,
                self.tts_pad_token_id,
                self.model_type,
                self.tokenizer_type,
                self.tts_model_size,
                self.tts_model_type,
                self.talker_model_type,
                self.talker_num_hidden_layers,
                self.talker_hidden_size,
                self.talker_head_dim,
                self.talker_num_attention_heads,
                self.talker_vocab_size,
                self.talker_text_vocab_size,
                self.num_code_groups,
                self.codec_bos_id,
                self.codec_eos_token_id,
                self.codec_think_id,
                self.codec_nothink_id,
                self.codec_think_bos_id,
                self.codec_think_eos_id,
                self.codec_pad_id,
                self.codec_language_id,
                self.spk_id,
                self.spk_is_dialect,
                self.cp_model_type,
                self.cp_num_hidden_layers,
                self.cp_hidden_size,
                self.cp_num_attention_heads,
                if self.interleaved { "true" } else { "false" },
                self.rope_theta,
                mrope_section
            )
        }
    }

    fn write_temp_config(config: &FakeModelConfig, suffix: &str) -> PathBuf {
        let counter = TEST_CONFIG_COUNTER.fetch_add(1, Ordering::Relaxed);
        let base = std::env::temp_dir().join(format!(
            "qwen3tts-metadata-test-{}-{}-{}",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos(),
            counter,
            suffix
        ));
        std::fs::create_dir_all(&base).unwrap();
        let config_path = base.join("config.json");
        std::fs::write(&config_path, config.config_json()).unwrap();
        config_path
    }

    fn write_temp_config_json(json: &str, suffix: &str) -> PathBuf {
        let counter = TEST_CONFIG_COUNTER.fetch_add(1, Ordering::Relaxed);
        let base = std::env::temp_dir().join(format!(
            "qwen3tts-metadata-test-custom-{}-{}-{}",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos(),
            counter,
            suffix
        ));
        std::fs::create_dir_all(&base).unwrap();
        let path = base.join("config.json");
        std::fs::write(&path, json).unwrap();
        path
    }

    fn base_config() -> FakeModelConfig {
        FakeModelConfig {
            assistant_token_id: 77091,
            im_start_token_id: 151644,
            im_end_token_id: 151645,
            tts_bos_token_id: 151672,
            tts_eos_token_id: 151673,
            tts_pad_token_id: 151671,
            model_type: "qwen3_tts",
            tokenizer_type: "qwen3_tts_tokenizer_12hz",
            tts_model_size: "0b6",
            tts_model_type: "base",
            talker_num_hidden_layers: 28,
            talker_head_dim: 128,
            talker_hidden_size: 1024,
            talker_num_attention_heads: 16,
            talker_vocab_size: 3072,
            talker_text_vocab_size: 151936,
            codec_bos_id: 2149,
            codec_eos_token_id: 2150,
            codec_think_id: 2154,
            codec_nothink_id: 2155,
            codec_think_bos_id: 2156,
            codec_think_eos_id: 2157,
            codec_pad_id: 2148,
            num_code_groups: 16,
            codec_language_id: r#"{"chinese":2055,"english":2050,"german":2053,"italian":2070,"portuguese":2071,"spanish":2054,"japanese":2058,"korean":2064,"french":2061,"russian":2069}"#,
            spk_id: "{}",
            spk_is_dialect: "{}",
            cp_num_hidden_layers: 5,
            cp_hidden_size: 1024,
            cp_num_attention_heads: 16,
            rope_theta: 1_000_000.0,
            interleaved: true,
            mrope_section: vec![24, 20, 20],
            talker_model_type: "qwen3_tts_talker",
            cp_model_type: "qwen3_tts_talker_code_predictor",
        }
    }

    fn custom_voice_1_7_config() -> FakeModelConfig {
        FakeModelConfig {
            tts_model_size: "1b7",
            tts_model_type: "custom_voice",
            codec_language_id: r#"{"chinese":2055,"english":2050,"german":2053,"italian":2070,"portuguese":2071,"spanish":2054,"japanese":2058,"korean":2064,"french":2061,"russian":2069,"beijing_dialect":2051,"sichuan_dialect":2052}"#,
            spk_id: r#"{"serena": 3066,"vivian": 3065,"uncle_fu": 3010,"ryan": 3061,"aiden": 2861,"ono_anna": 2873,"sohee": 2864,"eric": 2875,"dylan": 2878}"#,
            spk_is_dialect: r#"{"dylan":"beijing_dialect","eric":"sichuan_dialect","serena":false,"vivian":false,"uncle_fu":false,"ryan":false,"aiden":false,"ono_anna":false,"sohee":false}"#,
            ..base_config()
        }
    }

    fn voicedesign_1_7_config() -> FakeModelConfig {
        FakeModelConfig {
            tts_model_type: "voice_design",
            ..custom_voice_1_7_config()
        }
    }

    fn parse_metadata(config: &FakeModelConfig) -> ModelMetadata {
        let path = write_temp_config(config, "valid");
        ModelMetadata::from_config_path(&path).unwrap()
    }

    #[test]
    fn synthetic_base_and_custom_voice_metadata_cover_required_tables() {
        let base = parse_metadata(&base_config());
        assert_eq!(base.codec_language_id.len(), 10);
        assert_eq!(base.spk_id.len(), 0);
        assert_eq!(base.spk_is_dialect.len(), 0);

        let custom_voice = parse_metadata(&custom_voice_1_7_config());
        assert_eq!(custom_voice.codec_language_id.len(), 12);
        assert_eq!(custom_voice.spk_id.len(), 9);
        assert_eq!(custom_voice.spk_is_dialect.len(), 9);
        assert_eq!(
            custom_voice
                .codec_language_id
                .iter()
                .find(|(name, _)| name == "chinese"),
            Some(&("chinese".to_string(), 2055))
        );
        assert_eq!(
            custom_voice
                .spk_is_dialect
                .iter()
                .find(|(speaker, _)| speaker.eq_ignore_ascii_case("dylan"))
                .and_then(|(_, dialect)| dialect.as_deref()),
            Some("beijing_dialect")
        );
        assert_eq!(
            custom_voice
                .spk_is_dialect
                .iter()
                .find(|(speaker, _)| speaker.eq_ignore_ascii_case("eric"))
                .and_then(|(_, dialect)| dialect.as_deref()),
            Some("sichuan_dialect")
        );
    }

    #[test]
    fn malformed_and_inconsistent_dialect_maps_fail_closed() {
        let with_true = custom_voice_1_7_config();
        let mut custom = with_true.config_json();
        custom = custom.replace("\"eric\":\"sichuan_dialect\"", "\"eric\":true");
        let path = write_temp_config_json(&custom, "true_dialect");
        assert!(matches!(
            ModelMetadata::from_config_path(&path),
            Err(crate::Error::Config(_))
        ));

        let mut with_unknown = custom_voice_1_7_config().config_json();
        with_unknown = with_unknown.replace(
            "\"eric\":\"sichuan_dialect\"",
            "\"eric\":\"unknown_dialect\"",
        );
        let path = write_temp_config_json(&with_unknown, "unknown_dialect");
        assert!(matches!(
            ModelMetadata::from_config_path(&path),
            Err(crate::Error::Config(_))
        ));
    }

    #[test]
    fn duplicate_casefolded_and_empty_keys_are_rejected() {
        let replaced = custom_voice_1_7_config()
            .config_json()
            .replace("\"chinese\":2055", "\"chinese\":2055,\"Chinese\":2056");
        let path = write_temp_config_json(&replaced, "casefold");
        assert!(matches!(
            ModelMetadata::from_config_path(&path),
            Err(crate::Error::Config(_))
        ));

        let replaced = custom_voice_1_7_config()
            .config_json()
            .replace("\"dylan\": 2878", "\"dylan\": 2878, \"Dylan\": 2879");
        let path = write_temp_config_json(&replaced, "casefold_speaker");
        assert!(matches!(
            ModelMetadata::from_config_path(&path),
            Err(crate::Error::Config(_))
        ));

        let mut bad = base_config();
        bad.codec_language_id = r#"{"":2055,"chinese":2055}"#;
        let path = write_temp_config(&bad, "empty_language_key");
        assert!(matches!(
            ModelMetadata::from_config_path(&path),
            Err(crate::Error::Config(_))
        ));
    }

    #[test]
    fn model_table_contains_five_qwen3_tts_variants() {
        let table = model_table();
        assert_eq!(table.len(), 5);
        assert!(table.iter().all(|model| model.languages == 10));
        assert!(table.iter().all(|model| model.streaming));
        assert!(table.iter().any(|model| {
            model.model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
                && model.parameters == "1.7B"
                && model.instruction_control == InstructionControl::Full
        }));
        assert!(table.iter().any(|model| {
            model.model_id == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
                && model.parameters == "0.6B"
                && model.instruction_control == InstructionControl::None
        }));
    }

    #[test]
    fn detects_model_family_from_id_or_path() {
        let capability =
            model_capability("C:/hf/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/x")
                .expect("custom voice model");
        assert_eq!(capability.parameters, "1.7B");
        assert!(capability.supports_speaker_presets);
        assert_eq!(capability.instruction_control, InstructionControl::Full);
    }

    #[test]
    fn metadata_from_config_infers_capability_independent_of_path() {
        let a = write_temp_config(&custom_voice_1_7_config(), "path_a");
        let b = write_temp_config(&custom_voice_1_7_config(), "path_b");

        let metadata_a = ModelMetadata::from_config_path(&a).unwrap();
        let metadata_b = ModelMetadata::from_config_path(&b).unwrap();
        assert_eq!(metadata_a, metadata_b);
        assert!(metadata_a.tts_model_type == TtsModelType::CustomVoice);
        assert_eq!(
            metadata_a.runtime_generation_mode(),
            GenerationMode::CustomVoice
        );
    }

    #[test]
    fn rename_model_directory_does_not_change_capability() {
        let original = write_temp_config(&voicedesign_1_7_config(), "alpha");
        let renamed = write_temp_config(&voicedesign_1_7_config(), "beta");
        let left = ModelMetadata::from_config_path(original).unwrap();
        let right = ModelMetadata::from_config_path(renamed).unwrap();
        assert_eq!(
            left.runtime_generation_mode(),
            right.runtime_generation_mode()
        );
    }

    #[test]
    fn validates_custom_voice_and_voicedesign_modes() {
        let metadata = parse_metadata(&custom_voice_1_7_config());
        validate_generation_request(
            &metadata,
            GenerationMode::CustomVoice,
            Some("Vivian"),
            None,
            None,
        )
        .unwrap();

        let metadata = parse_metadata(&voicedesign_1_7_config());
        validate_generation_request(
            &metadata,
            GenerationMode::VoiceDesign,
            None,
            Some("年輕女性，溫柔親切"),
            None,
        )
        .unwrap();
    }

    #[test]
    fn rejects_unsupported_or_incomplete_modes() {
        let metadata_06 = parse_metadata(&base_config());
        let err = validate_generation_request(
            &metadata_06,
            GenerationMode::CustomVoice,
            Some("Vivian"),
            Some("用開心語氣"),
            None,
        );
        assert!(err.is_err());

        let metadata_17cv = parse_metadata(&custom_voice_1_7_config());
        let err = validate_generation_request(
            &metadata_17cv,
            GenerationMode::VoiceDesign,
            None,
            None,
            None,
        );
        assert!(err.is_err());

        let err =
            validate_generation_request(&metadata_06, GenerationMode::VoiceClone, None, None, None);
        assert!(err.is_err());

        let err = validate_generation_request(
            &metadata_06,
            GenerationMode::VoiceClone,
            Some("Vivian"),
            None,
            Some("reference.wav"),
        );
        assert!(err.is_err());

        let err = validate_generation_request(
            &metadata_06,
            GenerationMode::VoiceClone,
            None,
            Some("指令文本"),
            Some("reference.wav"),
        );
        assert!(err.is_err());
    }

    #[test]
    fn validates_voice_clone_when_reference_audio_is_present() {
        validate_generation_request(
            &parse_metadata(&base_config()),
            GenerationMode::VoiceClone,
            None,
            None,
            Some("reference.wav"),
        )
        .unwrap();

        let err = validate_generation_request(
            &parse_metadata(&base_config()),
            GenerationMode::VoiceClone,
            Some("Vivian"),
            None,
            Some("reference.wav"),
        );
        assert!(err.is_err());
    }

    #[test]
    fn auto_mode_uses_metadata_runtime_mode() {
        let base = parse_metadata(&base_config());
        let metadata = parse_metadata(&voicedesign_1_7_config());
        assert_eq!(
            resolve_generation_mode(&base, GenerationMode::Auto),
            GenerationMode::VoiceClone
        );
        assert_eq!(
            resolve_generation_mode(&metadata, GenerationMode::Auto),
            GenerationMode::VoiceDesign
        );
        let metadata = parse_metadata(&custom_voice_1_7_config());
        assert_eq!(
            resolve_generation_mode(&metadata, GenerationMode::Auto),
            GenerationMode::CustomVoice
        );
        assert_eq!(
            resolve_generation_mode(&metadata, GenerationMode::CustomVoice),
            GenerationMode::CustomVoice
        );
        assert_eq!(
            resolve_generation_mode(&metadata, GenerationMode::VoiceDesign),
            GenerationMode::VoiceDesign
        );
    }

    #[test]
    fn auto_mode_validation_is_backed_by_metadata_runtime_mode() {
        let custom_voice = parse_metadata(&custom_voice_1_7_config());
        let base = parse_metadata(&base_config());
        let voicedesign = parse_metadata(&voicedesign_1_7_config());

        let err = validate_generation_request(
            &custom_voice,
            GenerationMode::Auto,
            Some("Vivian"),
            None,
            None,
        );
        assert!(err.is_ok());

        let err = validate_generation_request(
            &voicedesign,
            GenerationMode::Auto,
            None,
            Some("年輕女性，溫柔親切"),
            None,
        );
        assert!(err.is_ok());

        let err = validate_generation_request(
            &base,
            GenerationMode::Auto,
            None,
            None,
            Some("reference.wav"),
        );
        assert!(err.is_ok());

        let err = validate_generation_request(&base, GenerationMode::Auto, None, None, None);
        assert!(err.is_err());

        let err = validate_generation_request(
            &base,
            GenerationMode::Auto,
            Some("Vivian"),
            Some("指令文本"),
            Some("reference.wav"),
        );
        assert!(err.is_err());
    }

    #[test]
    fn validates_missing_or_invalid_config_paths() {
        let missing_dir = std::env::temp_dir().join(format!(
            "qwen3tts-metadata-missing-{}",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::fs::create_dir_all(&missing_dir).unwrap();
        assert!(matches!(
            ModelMetadata::from_model_dir(&missing_dir),
            Err(crate::Error::Config(_))
        ));

        let malformed = missing_dir.join("config.json");
        std::fs::write(&malformed, "{not-json").unwrap();
        assert!(matches!(
            ModelMetadata::from_config_path(&malformed),
            Err(crate::Error::Config(_))
        ));
    }

    #[test]
    fn detects_unknown_metadata_and_conflicting_model_types() {
        let mut cfg = base_config();
        cfg.model_type = "wrong_type";
        let path = write_temp_config(&cfg, "unknown_model");
        assert!(matches!(
            ModelMetadata::from_config_path(&path),
            Err(crate::Error::Config(_))
        ));

        let mut nested_cfg = base_config();
        nested_cfg.talker_model_type = "wrong_talker";
        let path = write_temp_config(&nested_cfg, "wrong_talker");
        assert!(matches!(
            ModelMetadata::from_config_path(&path),
            Err(crate::Error::Config(_))
        ));

        let mut predictor_cfg = base_config();
        predictor_cfg.cp_model_type = "wrong_predictor";
        let path = write_temp_config(&predictor_cfg, "wrong_predictor");
        assert!(matches!(
            ModelMetadata::from_config_path(&path),
            Err(crate::Error::Config(_))
        ));
    }

    #[test]
    fn rejects_zero_layer_head_dims_and_invalid_mrope_shape() {
        let mut cfg = base_config();
        cfg.talker_num_hidden_layers = 0;
        let path = write_temp_config(&cfg, "zero_layers");
        assert!(matches!(
            ModelMetadata::from_config_path(&path),
            Err(crate::Error::Config(_))
        ));

        let mut cfg = base_config();
        cfg.mrope_section = vec![24, 20, 0];
        let path = write_temp_config(&cfg, "invalid_mrope");
        assert!(matches!(
            ModelMetadata::from_config_path(&path),
            Err(crate::Error::Config(_))
        ));

        let mut cfg = base_config();
        cfg.mrope_section = vec![12, 12];
        let path = write_temp_config(&cfg, "invalid_mrope_len");
        assert!(matches!(
            ModelMetadata::from_config_path(&path),
            Err(crate::Error::Config(_))
        ));

        let mut cfg = base_config();
        cfg.talker_head_dim = 64;
        cfg.mrope_section = vec![20, 20, 20];
        let path = write_temp_config(&cfg, "invalid_mrope_sum");
        assert!(matches!(
            ModelMetadata::from_config_path(&path),
            Err(crate::Error::Config(_))
        ));

        let mut cfg = base_config();
        cfg.mrope_section = vec![];
        let path = write_temp_config(&cfg, "empty_mrope");
        assert!(matches!(
            ModelMetadata::from_config_path(&path),
            Err(crate::Error::Config(_))
        ));
    }

    #[test]
    fn malformed_json_and_missing_required_fields_return_config_errors() {
        let malformed = std::env::temp_dir().join("qwen3tts-metadata-malformed.json");
        std::fs::write(&malformed, "{").unwrap();
        assert!(matches!(
            ModelMetadata::from_config_path(&malformed),
            Err(crate::Error::Config(_))
        ));

        let missing_fields = std::env::temp_dir().join("qwen3tts-metadata-missing-fields.json");
        std::fs::write(&missing_fields, r#"{"model_type":"qwen3_tts"}"#).unwrap();
        assert!(matches!(
            ModelMetadata::from_config_path(&missing_fields),
            Err(crate::Error::Config(_))
        ));
    }
}
