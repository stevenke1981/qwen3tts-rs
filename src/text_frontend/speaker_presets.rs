//! Built-in CustomVoice speaker presets.
//!
//! Qwen CustomVoice models expose these names as true speaker ids. Base and
//! VoiceDesign snapshots commonly have an empty speaker-id map, so the same
//! names are also translated into natural-language instructions as a fallback.

/// A known Qwen CustomVoice speaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeakerPreset {
    /// Canonical speaker name accepted by upstream CustomVoice models.
    pub name: &'static str,
    /// Short natural-language voice description for VoiceDesign fallback.
    pub description: &'static str,
    /// Preferred language for the speaker preset.
    pub native_language: &'static str,
}

const SPEAKER_PRESETS: &[SpeakerPreset] = &[
    SpeakerPreset {
        name: "Vivian",
        description: "Bright young female voice",
        native_language: "Chinese",
    },
    SpeakerPreset {
        name: "Serena",
        description: "Warm and gentle young female voice",
        native_language: "Chinese",
    },
    SpeakerPreset {
        name: "Uncle_Fu",
        description: "Mature male voice with a mellow, seasoned timbre",
        native_language: "Chinese",
    },
    SpeakerPreset {
        name: "Dylan",
        description: "Youthful Beijing male voice",
        native_language: "Chinese (Beijing)",
    },
    SpeakerPreset {
        name: "Eric",
        description: "Lively Chengdu male voice",
        native_language: "Chinese (Sichuan)",
    },
    SpeakerPreset {
        name: "Ryan",
        description: "Dynamic male voice with rhythmic delivery",
        native_language: "English",
    },
    SpeakerPreset {
        name: "Aiden",
        description: "Sunny American male voice",
        native_language: "English",
    },
    SpeakerPreset {
        name: "Ono_Anna",
        description: "Playful Japanese female voice",
        native_language: "Japanese",
    },
    SpeakerPreset {
        name: "Sohee",
        description: "Warm Korean female voice",
        native_language: "Korean",
    },
];

/// Return the canonical names of all built-in speaker presets.
pub fn speaker_names() -> Vec<&'static str> {
    SPEAKER_PRESETS.iter().map(|preset| preset.name).collect()
}

/// Return the canonical upstream speaker id if `name` is a known preset.
pub fn canonical_name(name: &str) -> Option<&'static str> {
    lookup(name).map(|preset| preset.name)
}

/// Look up a built-in speaker by name, accepting case and separator variants.
pub fn lookup(name: &str) -> Option<&'static SpeakerPreset> {
    let key = normalize_name(name);
    SPEAKER_PRESETS
        .iter()
        .find(|preset| normalize_name(preset.name) == key)
}

/// Merge a built-in speaker preset into the user-provided instruction.
///
/// The original `speaker` option is still passed through to the model. When the
/// loaded model has a real speaker-id map, the id path remains active; when it
/// does not, this instruction gives VoiceDesign/Base snapshots a useful voice
/// condition instead of silently ignoring `--speaker`.
pub fn effective_instruct(instruct: Option<String>, speaker: Option<&str>) -> Option<String> {
    let preset = speaker.and_then(lookup);
    match (preset, instruct) {
        (Some(preset), Some(user)) if !user.trim().is_empty() => Some(format!(
            "Use the built-in {name} speaker style: {description}. Native language: {native_language}. {user}",
            name = preset.name,
            description = preset.description,
            native_language = preset.native_language,
            user = user.trim()
        )),
        (Some(preset), _) => Some(format!(
            "Use the built-in {name} speaker style: {description}. Native language: {native_language}.",
            name = preset.name,
            description = preset.description,
            native_language = preset.native_language
        )),
        (None, Some(user)) if !user.trim().is_empty() => Some(user.trim().to_string()),
        (None, _) => None,
    }
}

fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{canonical_name, effective_instruct, lookup, speaker_names};

    #[test]
    fn lookup_accepts_case_and_separator_variants() {
        let preset = lookup("uncle-fu").expect("uncle-fu preset");

        assert_eq!(preset.name, "Uncle_Fu");
        assert_eq!(lookup("UNCLE_FU").unwrap().name, "Uncle_Fu");
        assert_eq!(lookup("Vivian").unwrap().name, "Vivian");
        assert_eq!(canonical_name("uncle-fu"), Some("Uncle_Fu"));
    }

    #[test]
    fn effective_instruct_merges_speaker_preset_and_user_instruction() {
        let value = effective_instruct(Some("說話速度自然，台灣口語".to_string()), Some("vivian"))
            .expect("merged instruct");

        assert!(value.contains("Vivian"));
        assert!(value.contains("Bright young female voice"));
        assert!(value.contains("說話速度自然，台灣口語"));
    }

    #[test]
    fn unknown_speaker_keeps_original_instruction() {
        assert_eq!(
            effective_instruct(Some("warm".to_string()), Some("unknown")),
            Some("warm".to_string())
        );
        assert_eq!(effective_instruct(None, Some("unknown")), None);
    }

    #[test]
    fn speaker_list_includes_qwen_custom_voice_names() {
        let names = speaker_names();

        assert!(names.contains(&"Vivian"));
        assert!(names.contains(&"Uncle_Fu"));
        assert!(names.contains(&"Dylan"));
        assert_eq!(names.len(), 9);
    }
}
