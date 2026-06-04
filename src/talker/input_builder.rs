//! # 輸入建構器
//!
//! 從文字 + 語言 + 說話者建構 talker 的 input_embeds。
//! 對應 `Qwen3TTSForConditionalGeneration.generate()` 的輸入構建部分。

use candle_core::{DType, Device, Result, Tensor};

use super::primitives::embedding_lookup;
use super::talker::TalkerForConditionalGeneration;

/// Native voice-clone conditioning already extracted from reference audio.
pub struct VoiceClonePrompt<'b> {
    /// Token IDs for the reference transcript content, excluding chat-template role/tail tokens.
    pub reference_text_token_ids: &'b [u32],
    /// Reference codec frames, one `[16]` frame per 12Hz step.
    pub reference_codes: &'b [[u16; 16]],
    /// Speaker embedding from the native speaker encoder.
    pub speaker_embedding: Option<&'b Tensor>,
}

/// 輸入建構器
pub struct InputBuilder<'a> {
    talker: &'a TalkerForConditionalGeneration,
    device: Device,
}

impl<'a> InputBuilder<'a> {
    pub fn new(talker: &'a TalkerForConditionalGeneration, device: &Device) -> Self {
        Self {
            talker,
            device: device.clone(),
        }
    }

    /// 將文字 token IDs 轉換為 talker input embeddings
    ///
    /// # 參數
    /// - `text_token_ids`: tokenizer 輸出的 token ID 列表
    ///
    /// # 回傳
    /// - `(inputs_embeds, attention_mask, trailing_text_hidden, tts_pad_embed)`
    pub fn build(
        &self,
        text_token_ids: &[u32],
        instruct_token_ids: Option<&[u32]>,
        language: &str,
        speaker: Option<&str>,
    ) -> Result<(
        Tensor, // inputs_embeds: [1, seq_len, hidden]
        Tensor, // attention_mask: [1, seq_len]
        Tensor, // trailing_text_hidden: [1, text_len, hidden]
        Tensor, // tts_pad_embed: [1, 1, hidden]
    )> {
        self.build_inner(text_token_ids, instruct_token_ids, language, speaker, None)
    }

    /// Build talker inputs for native voice-clone ICL mode.
    pub fn build_voice_clone(
        &self,
        text_token_ids: &[u32],
        instruct_token_ids: Option<&[u32]>,
        language: &str,
        speaker: Option<&str>,
        voice_clone: &VoiceClonePrompt<'_>,
    ) -> Result<(
        Tensor, // inputs_embeds: [1, seq_len, hidden]
        Tensor, // attention_mask: [1, seq_len]
        Tensor, // trailing_text_hidden: [1, text_len, hidden]
        Tensor, // tts_pad_embed: [1, 1, hidden]
    )> {
        self.build_inner(
            text_token_ids,
            instruct_token_ids,
            language,
            speaker,
            Some(voice_clone),
        )
    }

    fn build_inner(
        &self,
        text_token_ids: &[u32],
        instruct_token_ids: Option<&[u32]>,
        language: &str,
        speaker: Option<&str>,
        voice_clone: Option<&VoiceClonePrompt<'_>>,
    ) -> Result<(
        Tensor, // inputs_embeds: [1, seq_len, hidden]
        Tensor, // attention_mask: [1, seq_len]
        Tensor, // trailing_text_hidden: [1, text_len, hidden]
        Tensor, // tts_pad_embed: [1, 1, hidden]
    )> {
        let config = &self.talker.config;

        // ── Step 1: 建立 text embedding ──
        let input_tensor =
            Tensor::from_slice(text_token_ids, (1, text_token_ids.len()), &self.device)?;
        let text_seq_len = text_token_ids.len();

        // 特殊 token 嵌入
        let bos = config.tts_bos_token_id;
        let eos = config.tts_eos_token_id;
        let pad = config.tts_pad_token_id;

        let special_ids = Tensor::new(&[[bos, eos, pad]], &self.device)?;
        let special_embeds = self.talker.embed_text(&special_ids)?; // [1, 3, hidden]
        let tts_bos_embed = special_embeds.narrow(1, 0, 1)?;
        let tts_eos_embed = special_embeds.narrow(1, 1, 1)?;
        let tts_pad_embed = special_embeds.narrow(1, 2, 1)?;

        // Codec conditioning token list
        let lang_lower = language.to_lowercase();
        let language_id = if lang_lower == "auto" {
            None
        } else {
            config
                .codec_language_id
                .iter()
                .find(|(name, _)| name == &lang_lower)
                .map(|(_, id)| *id)
        };

        let codec_prefill = if let Some(lid) = language_id {
            vec![
                config.codec_think_id,
                config.codec_think_bos_id,
                lid,
                config.codec_think_eos_id,
            ]
        } else {
            vec![
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
            ]
        };

        let codec_prefill_t =
            Tensor::from_slice(&codec_prefill, (1, codec_prefill.len()), &self.device)?;

        // codec_pad + codec_bos 作為開始標記
        let codec_pad_bos_t =
            Tensor::new(&[[config.codec_pad_id, config.codec_bos_id]], &self.device)?;

        let mut codec_emb = self.talker.embed_codec(&codec_prefill_t)?;
        let codec_pad_bos_emb = self.talker.embed_codec(&codec_pad_bos_t)?;
        codec_emb = Tensor::cat(&[codec_emb, codec_pad_bos_emb], 1)?;

        // Speaker embedding (optional)
        if let Some(spk_emb) = voice_clone.and_then(|prompt| prompt.speaker_embedding) {
            let spk_emb = normalize_speaker_embedding(spk_emb, config.hidden_size)?;
            // Insert between codec_prefill and codec_pad_bos.
            let c_pre = codec_emb.narrow(1, 0, codec_prefill.len())?;
            let c_post = codec_emb.narrow(1, codec_prefill.len(), 2)?;
            codec_emb = Tensor::cat(&[c_pre, spk_emb, c_post], 1)?;
        } else if let Some(spk) = speaker {
            if let Some((_, spk_id)) = config
                .spk_id
                .iter()
                .find(|(name, _)| name == &spk.to_lowercase())
            {
                let spk_t = Tensor::new(&[[*spk_id as u32]], &self.device)?;
                let spk_emb = self.talker.embed_codec(&spk_t)?; // [1, 1, hidden]
                // Insert between codec_prefill and codec_pad_bos
                let c_pre = codec_emb.narrow(1, 0, codec_prefill.len())?;
                let c_post = codec_emb.narrow(1, codec_prefill.len(), 2)?;
                codec_emb = Tensor::cat(&[c_pre, spk_emb, c_post], 1)?;
            }
        }

        // ── Role embedding: <|im_start|>assistant\n ──
        let role_tokens = input_tensor.narrow(1, 0, 3)?;
        let role_emb = self.talker.embed_text(&role_tokens)?;

        // ── Codec input embed: tts_pad * (codec_len-2) + tts_bos ──
        let codec_len = codec_emb.dim(1)?;
        let codec_input = {
            let pads = tts_pad_embed.expand((1, codec_len - 2, config.hidden_size))?;
            let with_bos = Tensor::cat(&[pads, tts_bos_embed], 1)?;
            // codec_emb[:-1] (all but last)
            let codec_prefix = codec_emb.narrow(1, 0, codec_len - 1)?;
            (with_bos + codec_prefix)?
        };

        // ── Full input: optional instruct prompt + role + codec_input ──
        let mut input_embeds = if let Some(ids) = instruct_token_ids {
            if ids.is_empty() {
                Tensor::cat(&[role_emb, codec_input], 1)?
            } else {
                let instruct_tensor = Tensor::from_slice(ids, (1, ids.len()), &self.device)?;
                let instruct_emb = self.talker.embed_text(&instruct_tensor)?;
                Tensor::cat(&[instruct_emb, role_emb, codec_input], 1)?
            }
        } else {
            Tensor::cat(&[role_emb, codec_input], 1)?
        };

        let trailing_text_hidden = if let Some(prompt) = voice_clone {
            let text_content_len = text_seq_len - 3 - 5;
            let text_content_ids = input_tensor.narrow(1, 3, text_content_len)?;
            let (icl_input_embed, trailing_text_hidden) = self.build_icl_prompt(
                prompt.reference_text_token_ids,
                &text_content_ids,
                prompt.reference_codes,
                &tts_pad_embed,
                &tts_eos_embed,
            )?;
            input_embeds = Tensor::cat(&[input_embeds, icl_input_embed], 1)?;
            trailing_text_hidden
        } else {
            // 文字部分
            // text_emb + codec_last
            let text_emb = self.talker.embed_text(&input_tensor.narrow(1, 3, 1)?)?; // first text token
            let codec_last = codec_emb.narrow(1, codec_len - 1, 1)?;
            let first_text_with_codec = (text_emb + codec_last)?;
            input_embeds = Tensor::cat(&[input_embeds, first_text_with_codec], 1)?;

            // 剩餘文字 (trailing_text_hidden)
            let trailing_ids = input_tensor.narrow(1, 4, text_seq_len - 4 - 5)?;
            let trailing_emb = self.talker.embed_text(&trailing_ids)?;
            Tensor::cat(&[trailing_emb, tts_eos_embed], 1)?
        };

        // Attention mask
        let seq_len = input_embeds.dim(1)?;
        let attention_mask = Tensor::ones(&[1, seq_len], DType::I64, &self.device)?;

        Ok((
            input_embeds,
            attention_mask,
            trailing_text_hidden,
            tts_pad_embed.clone(),
        ))
    }

    fn build_icl_prompt(
        &self,
        reference_text_token_ids: &[u32],
        text_content_ids: &Tensor,
        reference_codes: &[[u16; 16]],
        tts_pad_embed: &Tensor,
        tts_eos_embed: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let config = &self.talker.config;
        if reference_codes.is_empty() {
            candle_core::bail!("voice clone reference codec tokens cannot be empty");
        }

        let text_len = text_content_ids.dim(1)?;
        let mut joined_text_ids = Vec::with_capacity(reference_text_token_ids.len() + text_len);
        joined_text_ids.extend_from_slice(reference_text_token_ids);
        joined_text_ids.extend_from_slice(&text_content_ids.to_vec2::<u32>()?[0]);
        let joined_text_t =
            Tensor::from_slice(&joined_text_ids, (1, joined_text_ids.len()), &self.device)?;
        let text_embed = self.talker.embed_text(&joined_text_t)?;
        let text_embed = Tensor::cat(&[text_embed, tts_eos_embed.clone()], 1)?;

        let codec_bos = Tensor::new(&[[config.codec_bos_id]], &self.device)?;
        let codec_bos_embed = self.talker.embed_codec(&codec_bos)?;
        let codec_frames_embed = self.embed_reference_code_frames(reference_codes)?;
        let codec_embed = Tensor::cat(&[codec_bos_embed, codec_frames_embed], 1)?;

        let text_len = text_embed.dim(1)?;
        let codec_len = codec_embed.dim(1)?;
        if text_len > codec_len {
            let icl_input_embed = (text_embed.narrow(1, 0, codec_len)? + codec_embed)?;
            let trailing_text_hidden = text_embed.narrow(1, codec_len, text_len - codec_len)?;
            Ok((icl_input_embed, trailing_text_hidden))
        } else {
            let pad_len = codec_len - text_len;
            let text_embed = if pad_len == 0 {
                text_embed
            } else {
                let pads = tts_pad_embed.expand((1, pad_len, config.hidden_size))?;
                Tensor::cat(&[text_embed, pads], 1)?
            };
            Ok(((text_embed + codec_embed)?, tts_pad_embed.clone()))
        }
    }

    fn embed_reference_code_frames(&self, reference_codes: &[[u16; 16]]) -> Result<Tensor> {
        let frames = reference_codes.len();
        let codebook0: Vec<u32> = reference_codes
            .iter()
            .map(|frame| frame[0] as u32)
            .collect();
        let codebook0_t = Tensor::from_slice(&codebook0, (1, frames), &self.device)?;
        let mut sum_emb = self.talker.embed_codec(&codebook0_t)?;

        for codebook in 1..self.talker.config.num_code_groups {
            let ids: Vec<u32> = reference_codes
                .iter()
                .map(|frame| frame[codebook] as u32)
                .collect();
            let ids_t = Tensor::from_slice(&ids, (1, frames), &self.device)?;
            let emb = embedding_lookup(
                &self.talker.code_predictor.codec_embeddings[codebook - 1],
                &ids_t,
            )?;
            sum_emb = (sum_emb + emb)?;
        }

        Ok(sum_emb)
    }
}

fn normalize_speaker_embedding(speaker_embedding: &Tensor, hidden_size: usize) -> Result<Tensor> {
    match speaker_embedding.dims() {
        [hidden] if *hidden == hidden_size => speaker_embedding.reshape((1, 1, hidden_size)),
        [hidden] if *hidden > hidden_size => speaker_embedding
            .narrow(0, 0, hidden_size)?
            .reshape((1, 1, hidden_size)),
        [1, hidden] if *hidden == hidden_size => speaker_embedding.reshape((1, 1, hidden_size)),
        [1, hidden] if *hidden > hidden_size => speaker_embedding
            .narrow(1, 0, hidden_size)?
            .reshape((1, 1, hidden_size)),
        [1, 1, hidden] if *hidden == hidden_size => Ok(speaker_embedding.clone()),
        [1, 1, hidden] if *hidden > hidden_size => speaker_embedding.narrow(2, 0, hidden_size),
        shape => candle_core::bail!(
            "speaker embedding shape must be [{hidden_size}], [1,{hidden_size}], or [1,1,{hidden_size}], got {shape:?}"
        ),
    }
}
