//! # 輸入建構器
//!
//! 從文字 + 語言 + 說話者建構 talker 的 input_embeds。
//! 對應 `Qwen3TTSForConditionalGeneration.generate()` 的輸入構建部分。

use candle_core::{DType, Device, Result, Tensor};

use super::talker::TalkerForConditionalGeneration;

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
        language: &str,
        speaker: Option<&str>,
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
        if let Some(spk) = speaker {
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

        // ── Full input: role + codec_input ──
        let mut input_embeds = Tensor::cat(&[role_emb, codec_input], 1)?;

        // 文字部分
        // text_emb + codec_last
        let text_emb = self.talker.embed_text(&input_tensor.narrow(1, 3, 1)?)?; // first text token
        let codec_last = codec_emb.narrow(1, codec_len - 1, 1)?;
        let first_text_with_codec = (text_emb + codec_last)?;
        input_embeds = Tensor::cat(&[input_embeds, first_text_with_codec], 1)?;

        // 剩餘文字 (trailing_text_hidden)
        let trailing_ids = input_tensor.narrow(1, 4, text_seq_len - 4 - 5)?;
        let trailing_emb = self.talker.embed_text(&trailing_ids)?;
        let trailing_text_hidden = Tensor::cat(&[trailing_emb, tts_eos_embed], 1)?;

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
}
