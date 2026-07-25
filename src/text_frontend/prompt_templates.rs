//! Official prompt formatting helpers for Qwen3-TTS chat-template style.

/// Extract the body segment from official-style assistant prompt token ids.
///
/// This matches the official reference-slice rule used for ICL prompt
/// reconstruction: remove the 3-role-prefix and terminal 2-token tail markers.
///
/// Returns error when the prompt is too short to contain a valid body+suffix.
pub fn reference_text_tokens_from_prompt_ids(prompt_ids: &[u32]) -> crate::Result<&[u32]> {
    if prompt_ids.len() <= 5 {
        return Err(crate::Error::Config(
            "reference prompt token sequence is too short".into(),
        ));
    }
    Ok(&prompt_ids[3..prompt_ids.len().saturating_sub(2)])
}

/// Build `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
pub fn build_assistant_prompt(text: &str) -> String {
    format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

/// Build `<|im_start|>assistant\n{reference_text}<|im_end|>\n`
pub fn build_reference_prompt(reference_text: &str) -> String {
    format!("<|im_start|>assistant\n{reference_text}<|im_end|>\n")
}

/// Build `<|im_start|>user\n{instruction}<|im_end|>\n`
pub fn build_instruction_prompt(instruction: &str) -> String {
    format!("<|im_start|>user\n{instruction}<|im_end|>\n")
}
