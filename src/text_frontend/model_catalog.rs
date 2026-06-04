//! Qwen3-TTS model capability catalog.
//!
//! This module keeps user-facing model selection rules in one place so the CLI
//! can explain capabilities and reject combinations that the native Rust path
//! cannot satisfy yet.

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

/// Static model capability entry.
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

/// 10 languages reported for Qwen3-TTS generation.
pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "Chinese",
    "English",
    "French",
    "German",
    "Italian",
    "Spanish",
    "Portuguese",
    "Japanese",
    "Korean",
    "Russian",
];

pub fn model_table() -> &'static [ModelCapability] {
    MODEL_TABLE
}

pub fn model_capability(model_id_or_path: &str) -> Option<&'static ModelCapability> {
    let normalized = normalize_model_key(model_id_or_path);
    MODEL_TABLE.iter().find(|model| {
        normalized.contains(&normalize_model_key(model.model_id))
            || normalize_model_key(model.model_id).contains(&normalized)
    })
}

pub fn infer_mode(model_id_or_path: &str) -> GenerationMode {
    model_capability(model_id_or_path)
        .map(|capability| {
            if capability.supports_voice_design {
                GenerationMode::VoiceDesign
            } else if capability.supports_speaker_presets {
                GenerationMode::CustomVoice
            } else if capability.supports_voice_clone {
                GenerationMode::VoiceClone
            } else {
                GenerationMode::Auto
            }
        })
        .unwrap_or(GenerationMode::Auto)
}

/// Validate a requested generation mode against the model family.
pub fn validate_generation_request(
    model_id_or_path: &str,
    requested_mode: GenerationMode,
    speaker: Option<&str>,
    instruct: Option<&str>,
    reference_audio: Option<&str>,
) -> crate::Result<()> {
    let Some(capability) = model_capability(model_id_or_path) else {
        return Ok(());
    };
    let mode = requested_mode;

    match mode {
        GenerationMode::Auto => Ok(()),
        GenerationMode::CustomVoice => {
            if !capability.supports_speaker_presets {
                return Err(crate::Error::Config(format!(
                    "{} does not support CustomVoice speaker presets; use a CustomVoice model",
                    capability.model_id
                )));
            }
            if speaker.is_none() {
                return Err(crate::Error::Config(
                    "custom-voice mode requires --speaker, e.g. --speaker Vivian".into(),
                ));
            }
            if instruct.is_some() && capability.instruction_control == InstructionControl::None {
                return Err(crate::Error::Config(format!(
                    "{} does not support --instruct; use 1.7B-CustomVoice or 1.7B-VoiceDesign",
                    capability.model_id
                )));
            }
            Ok(())
        }
        GenerationMode::VoiceDesign => {
            if !capability.supports_voice_design {
                return Err(crate::Error::Config(format!(
                    "{} does not support VoiceDesign; use Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    capability.model_id
                )));
            }
            if instruct.map(str::trim).unwrap_or("").is_empty() {
                return Err(crate::Error::Config(
                    "voice-design mode requires --instruct or --instruct-file".into(),
                ));
            }
            Ok(())
        }
        GenerationMode::VoiceClone => {
            if !capability.supports_voice_clone {
                return Err(crate::Error::Config(format!(
                    "{} does not support voice clone; use a Base model",
                    capability.model_id
                )));
            }
            if reference_audio.is_none() {
                return Err(crate::Error::Config(
                    "voice-clone mode requires --reference-audio with at least 3 seconds of audio"
                        .into(),
                ));
            }
            Err(crate::Error::Config(
                "voice-clone reference-audio conditioning is not implemented in the native Rust path yet"
                    .into(),
            ))
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
        GenerationMode, InstructionControl, model_capability, model_table,
        validate_generation_request,
    };

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
    fn validates_custom_voice_and_voicedesign_modes() {
        validate_generation_request(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            GenerationMode::CustomVoice,
            Some("Vivian"),
            Some("用開心語氣"),
            None,
        )
        .unwrap();

        validate_generation_request(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            GenerationMode::VoiceDesign,
            None,
            Some("年輕女性，溫柔親切"),
            None,
        )
        .unwrap();
    }

    #[test]
    fn rejects_unsupported_or_incomplete_modes() {
        assert!(
            validate_generation_request(
                "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                GenerationMode::CustomVoice,
                Some("Vivian"),
                Some("用開心語氣"),
                None,
            )
            .is_err()
        );

        assert!(
            validate_generation_request(
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                GenerationMode::VoiceDesign,
                None,
                None,
                None,
            )
            .is_err()
        );

        assert!(
            validate_generation_request(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                GenerationMode::VoiceClone,
                None,
                None,
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn auto_mode_does_not_require_reference_audio_for_base_models() {
        validate_generation_request(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            GenerationMode::Auto,
            None,
            None,
            None,
        )
        .unwrap();
    }
}
