//! Native Rust/Candle voice-clone planning helpers.
//!
//! The best-quality Qwen3-TTS clone path is in-context learning (ICL): a short
//! reference audio clip, its transcript, a speaker embedding, and reference
//! codec tokens are all provided to the talker before generation.

use std::path::PathBuf;

use candle_core::Tensor;

use crate::talker::VoiceClonePrompt;
use crate::text_frontend::SynthesisOptions;
use crate::{Error, Result};

#[path = "voice_clone_speaker.rs"]
pub mod speaker_encoder;
#[path = "voice_clone_tokenizer.rs"]
pub mod speech_tokenizer;

/// Native voice-clone conditioning mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceCloneMode {
    /// Full ICL mode: reference audio + reference transcript + speaker embedding.
    InContextLearning,
}

/// Validated native Candle voice-clone plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeVoiceClonePlan {
    pub reference_audio: PathBuf,
    pub reference_text: Option<String>,
    pub mode: VoiceCloneMode,
}

/// Validated reference codec tokens for native ICL voice clone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeReferenceCodes {
    frames: Vec<[u16; 16]>,
}

impl NativeReferenceCodes {
    pub const NUM_CODEBOOKS: usize = 16;
    pub const CODEBOOK_SIZE: u16 = 2048;

    pub fn from_frames(frames: Vec<[u16; Self::NUM_CODEBOOKS]>) -> Result<Self> {
        if frames.is_empty() {
            return Err(Error::Config(
                "native voice clone reference codec tokens cannot be empty".into(),
            ));
        }
        for (frame_idx, frame) in frames.iter().enumerate() {
            for (layer, &token) in frame.iter().enumerate() {
                if token >= Self::CODEBOOK_SIZE {
                    let err = Error::InvalidToken { token, layer };
                    return Err(Error::Config(format!(
                        "{err} at reference frame {frame_idx}"
                    )));
                }
            }
        }
        Ok(Self { frames })
    }

    pub fn frames(&self) -> &[[u16; Self::NUM_CODEBOOKS]] {
        &self.frames
    }

    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }
}

/// Fully prepared native voice-clone condition for talker ICL generation.
#[derive(Debug, Clone)]
pub struct NativeVoiceCloneCondition {
    reference_text_token_ids: Vec<u32>,
    reference_codes: NativeReferenceCodes,
    speaker_embedding: Tensor,
}

impl NativeVoiceCloneCondition {
    pub fn new(
        reference_text_token_ids: Vec<u32>,
        reference_codes: NativeReferenceCodes,
        speaker_embedding: Tensor,
    ) -> Self {
        Self {
            reference_text_token_ids,
            reference_codes,
            speaker_embedding,
        }
    }

    pub fn as_talker_prompt(&self) -> VoiceClonePrompt<'_> {
        VoiceClonePrompt {
            reference_text_token_ids: &self.reference_text_token_ids,
            reference_codes: self.reference_codes.frames(),
            speaker_embedding: Some(&self.speaker_embedding),
        }
    }

    pub fn reference_codes(&self) -> &NativeReferenceCodes {
        &self.reference_codes
    }

    pub fn speaker_embedding(&self) -> &Tensor {
        &self.speaker_embedding
    }
}

impl NativeVoiceClonePlan {
    /// Build a native plan from synthesis options.
    ///
    /// `None` means voice clone was not requested. When it is requested, native
    /// Candle currently requires `--reference-text` so the talker can use the
    /// same ICL prompt structure as the official implementation.
    pub fn from_options(options: &SynthesisOptions) -> Result<Option<Self>> {
        let Some(reference_audio) = options.reference_audio.as_deref() else {
            return Ok(None);
        };

        let reference_audio = reference_audio.trim();
        if reference_audio.is_empty() {
            return Err(Error::Config(
                "--reference-audio cannot be empty for native voice clone".into(),
            ));
        }

        let reference_text = options
            .reference_text
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .ok_or_else(|| {
                Error::Config(
                    "Candle/Rust native voice clone requires --reference-text for ICL mode".into(),
                )
            })?;

        Ok(Some(Self {
            reference_audio: PathBuf::from(reference_audio),
            reference_text: Some(reference_text.to_string()),
            mode: VoiceCloneMode::InContextLearning,
        }))
    }

    pub fn requires_reference_codec_tokens(&self) -> bool {
        matches!(self.mode, VoiceCloneMode::InContextLearning)
    }

    pub fn requires_speaker_embedding(&self) -> bool {
        true
    }
}

/// Number of generated PCM samples belonging to the prepended reference audio.
///
/// Mirrors the official decode trim rule:
/// `reference_samples = reference_frames / total_frames * total_samples`.
pub fn reference_prefix_samples(
    total_samples: usize,
    reference_frames: usize,
    total_frames: usize,
) -> usize {
    if total_frames == 0 {
        return 0;
    }
    total_samples.saturating_mul(reference_frames) / total_frames
}
