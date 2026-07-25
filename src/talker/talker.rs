//! # TalkerForConditionalGeneration — 完整生成引擎
//!
//! 對應 `Qwen3TTSTalkerForConditionalGeneration` (PyTorch)。
//!
//! 生成流程：
//! 1. Prefill: 處理所有輸入文字 + codec conditioning tokens
//! 2. Generation loop:
//!    a. Talker 預測 codebook 0 token (從 codec_head)
//!    b. Code Predictor 預測 codebooks 1-15
//!    c. 16 個 embedding sum → 作為下一步輸入
//!    d. 加上 trailing_text_hidden 或 tts_pad_embed

use candle_core::{DType, Device, Error, Result, Tensor};

use super::code_predictor::CodePredictor;
use super::config::TalkerConfig;
use super::model::TalkerModel;
use super::primitives::{
    MultimodalRotaryEmbedding, create_causal_mask, embedding_lookup, linear, linear_with_bias,
};
use super::sampling::{Sampler, SamplingOptions};
use crate::alignment_stage_dump::{NoopStageDumpObserver, StageDumpObserver};

#[inline]
fn record_c0_history_non_eos(c0_history: &mut Vec<u16>, eos_token_id: u16, c0_val: u16) {
    if c0_val != eos_token_id {
        c0_history.push(c0_val);
    }
}

/// Talker 條件生成模型
#[derive(Debug, Clone)]
pub struct TalkerForConditionalGeneration {
    pub model: TalkerModel,
    pub text_embedding: Tensor,
    pub text_proj_fc1_w: Tensor,
    pub text_proj_fc1_b: Tensor,
    pub text_proj_fc2_w: Tensor,
    pub text_proj_fc2_b: Tensor,
    pub codec_embedding: Tensor,
    pub codec_head: Tensor,
    pub code_predictor: CodePredictor,
    pub rope: MultimodalRotaryEmbedding,
    pub config: TalkerConfig,
}

impl TalkerForConditionalGeneration {
    // ─── Text Projection ────────────────────────────────────────────

    /// text_embedding → text_projection (2048→1024)
    pub fn embed_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        // input_ids: [batch, seq_len]
        let emb = embedding_lookup(&self.text_embedding, input_ids)?; // [batch, seq_len, 2048]
        let h = linear_with_bias(&emb, &self.text_proj_fc1_w, &self.text_proj_fc1_b)?;
        let h = h.silu()?;
        linear_with_bias(&h, &self.text_proj_fc2_w, &self.text_proj_fc2_b)
    }

    /// codec_embedding lookup (combined vocab, codebook 0)
    pub fn embed_codec(&self, input_ids: &Tensor) -> Result<Tensor> {
        // input_ids: [batch, seq_len]
        embedding_lookup(&self.codec_embedding, input_ids)
    }

    /// codec_head 輸出 logits
    pub fn codec_head_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        linear(hidden, &self.codec_head)
    }

    /// 計算 3D position_ids 從 attention_mask
    pub fn compute_position_ids(&self, attention_mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let device = attention_mask.device();
        // attention_mask: [batch, seq_len], 1 for valid, 0 for pad
        let attention_mask_f = attention_mask.to_dtype(DType::F32)?;
        let cumsum = attention_mask_f.cumsum(1)?;
        let one = Tensor::new(&[1.0f32], device)?;
        let cumsum = cumsum.broadcast_sub(&one)?;
        let position_ids = cumsum.broadcast_mul(&attention_mask_f)?;
        let one_minus_attention =
            Tensor::new(&[1.0f32], device)?.broadcast_sub(&attention_mask_f)?;
        let position_ids = position_ids.broadcast_add(&one_minus_attention)?;

        // [3, batch, seq_len]
        let pos_3d =
            position_ids
                .unsqueeze(0)?
                .expand((3, position_ids.dim(0)?, position_ids.dim(1)?))?;

        let max_pos = position_ids.max(1)?.unsqueeze(1)?; // [batch, 1]
        let valid_len = attention_mask_f.sum(1)?.unsqueeze(1)?;
        let delta = max_pos.broadcast_add(&one)?;
        let delta = delta.broadcast_sub(&valid_len)?;

        Ok((pos_3d.to_dtype(DType::U32)?, delta.to_dtype(DType::U32)?))
    }

    // ─── 生成主程式 ──────────────────────────────────────────────

    /// 從 input_embeds 生成 codec tokens
    ///
    /// # 參數
    /// - `inputs_embeds`: [batch, seq_len, hidden_size] — 已組合好的輸入
    /// - `attention_mask`: [batch, seq_len] — 1=有效, 0=填充
    /// - `trailing_text_hidden`: [1, text_seq_len, hidden] — 剩餘文字隱藏狀態
    /// - `tts_pad_embed`: [1, 1, hidden] — padding 嵌入
    /// - `max_new_tokens`: 最大生成步數
    ///
    /// # 回傳
    /// - codes: [generated_seq, 16] — 全部生成的 codec tokens
    pub fn generate(
        &self,
        inputs_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
        trailing_text_hidden: Option<&Tensor>,
        tts_pad_embed: Option<&Tensor>,
        max_new_tokens: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mut observer = NoopStageDumpObserver::default();
        self.generate_with_observer(
            inputs_embeds,
            attention_mask,
            trailing_text_hidden,
            tts_pad_embed,
            max_new_tokens,
            device,
            &mut observer,
        )
    }

    pub fn generate_with_observer<O: StageDumpObserver>(
        &self,
        inputs_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
        trailing_text_hidden: Option<&Tensor>,
        tts_pad_embed: Option<&Tensor>,
        max_new_tokens: usize,
        device: &Device,
        observer: &mut O,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = inputs_embeds.dims3()?;
        assert_eq!(batch, 1, "Only batch=1 supported");
        let capture = observer.wants_capture();

        // ── Prefill: 處理所有輸入 ──
        let mask = attention_mask
            .cloned()
            .unwrap_or_else(|| Tensor::ones(&[batch, seq_len], DType::I64, device).unwrap());

        let (position_ids, rope_delta) = self.compute_position_ids(&mask)?;

        // Create causal mask for prefill
        let causal_mask = create_causal_mask(seq_len, device)?;

        // 3D RoPE
        let (cos, sin) = self.rope.forward(inputs_embeds, &position_ids)?;

        // 28 layers
        let mut kv_caches = vec![None; self.config.num_hidden_layers];
        let hidden = self.model.forward(
            inputs_embeds,
            &cos,
            &sin,
            Some(&causal_mask),
            &mut kv_caches,
        )?;

        // hidden: [batch, seq_len, hidden]
        // 取最後一個位置
        let mut last_hidden = hidden.narrow(1, seq_len - 1, 1)?; // [batch, 1, hidden]

        // ── Generation Loop ──
        let mut flat_codes: Vec<u32> =
            Vec::with_capacity(max_new_tokens * self.config.num_code_groups);
        let mut num_frames = 0usize;

        let tts_pad = tts_pad_embed.cloned().unwrap_or_else(|| {
            Tensor::zeros(&[1, 1, self.config.hidden_size], DType::F32, device).unwrap()
        });

        let trailing = trailing_text_hidden.cloned().unwrap_or_else(|| {
            Tensor::zeros(&[batch, 1, self.config.hidden_size], DType::F32, device).unwrap()
        });
        let trailing_len = trailing.dim(1)?;
        let mut gen_step: usize = 0;

        for _step in 0..max_new_tokens {
            // Step A: 從 last_hidden 預測 codebook 0
            let logits = self.codec_head_logits(&last_hidden)?; // [batch, 1, vocab]
            let logits = logits.squeeze(1)?; // [batch, vocab]
            if capture {
                observer.on_talker_codebook0_logits(num_frames, &logits)?;
            }

            // Argmax (簡單版本)
            let c0_token = logits.argmax(1)?; // [batch]
            let c0_val = c0_token.to_vec1::<u32>()?[0] as u16;
            let c0_t = Tensor::new(&[c0_val as u32], device)?;
            let c0_2d = c0_t.reshape((1, 1))?;
            let c0_emb = self.embed_codec(&c0_2d)?; // [batch, 1, hidden]

            // Step B: Code predictor 生成 codebooks 1-15
            let mut cp_kv_caches = vec![None; self.config.code_predictor.num_hidden_layers];
            let (codes_1_15, _) = self.code_predictor.generate_with_observer(
                &last_hidden,
                &c0_emb,
                &mut cp_kv_caches,
                device,
                num_frames,
                observer,
            )?;

            flat_codes.push(c0_val as u32);
            flat_codes.extend(codes_1_15.squeeze(0)?.to_vec1::<u32>()?);
            num_frames += 1;

            // Step C: 檢查 EOS
            if c0_val as u32 == self.config.codec_eos_token_id {
                break;
            }

            // Step D: 構建下一步的輸入
            // 16 個 codec embeddings sum
            let mut sum_emb = c0_emb;

            for i in 0..(self.config.num_code_groups - 1) {
                let ci_token = codes_1_15.narrow(1, i, 1)?;
                let ci_emb = embedding_lookup(&self.code_predictor.codec_embeddings[i], &ci_token)?;
                sum_emb = (sum_emb + ci_emb)?;
            }

            // Step E: 添加文字隱藏狀態或 padding
            let text_add = if gen_step < trailing_len {
                trailing.narrow(1, gen_step, 1)?
            } else {
                tts_pad.clone()
            };

            let next_input = (sum_emb + text_add)?;

            // Run the next generated frame through the talker transformer.
            let cache_position_start = u32::try_from(seq_len + gen_step).map_err(|_| {
                Error::Msg(format!(
                    "cache_position_start overflow for seq_len {seq_len} gen_step {gen_step}"
                ))
            })?;
            let positions = MultimodalRotaryEmbedding::cached_positions_from_delta(
                cache_position_start,
                &rope_delta,
                1,
                device,
            )?;
            let (cos, sin) = self.rope.forward_single_position(&positions)?;
            let hidden = self
                .model
                .forward(&next_input, &cos, &sin, None, &mut kv_caches)?;
            last_hidden = hidden;
            gen_step += 1;
        }

        // 轉換為 [num_frames, 16]
        let result = Tensor::from_slice(&flat_codes, (num_frames, 16), device)?;
        if capture {
            observer.on_talker_final_codes(&result)?;
        }
        Ok(result)
    }

    pub fn generate_sampled(
        &self,
        inputs_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
        trailing_text_hidden: Option<&Tensor>,
        tts_pad_embed: Option<&Tensor>,
        max_new_tokens: usize,
        device: &Device,
        sampler: &mut Sampler,
        sampling: SamplingOptions,
        do_sample: bool,
        subtalker_sampling: SamplingOptions,
        subtalker_do_sample: bool,
    ) -> Result<Tensor> {
        let mut observer = NoopStageDumpObserver::default();
        self.generate_sampled_with_observer(
            inputs_embeds,
            attention_mask,
            trailing_text_hidden,
            tts_pad_embed,
            max_new_tokens,
            device,
            sampler,
            sampling,
            subtalker_sampling,
            do_sample,
            subtalker_do_sample,
            &mut observer,
        )
    }

    pub fn generate_sampled_with_observer<O: StageDumpObserver>(
        &self,
        inputs_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
        trailing_text_hidden: Option<&Tensor>,
        tts_pad_embed: Option<&Tensor>,
        max_new_tokens: usize,
        device: &Device,
        sampler: &mut Sampler,
        sampling: SamplingOptions,
        subtalker_sampling: SamplingOptions,
        talker_do_sample: bool,
        subtalker_do_sample: bool,
        observer: &mut O,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = inputs_embeds.dims3()?;
        assert_eq!(batch, 1, "Only batch=1 supported");
        let capture = observer.wants_capture();

        let mask = attention_mask
            .cloned()
            .unwrap_or_else(|| Tensor::ones(&[batch, seq_len], DType::I64, device).unwrap());
        let (position_ids, rope_delta) = self.compute_position_ids(&mask)?;
        let causal_mask = create_causal_mask(seq_len, device)?;
        let (cos, sin) = self.rope.forward(inputs_embeds, &position_ids)?;

        let mut kv_caches = vec![None; self.config.num_hidden_layers];
        let hidden = self.model.forward(
            inputs_embeds,
            &cos,
            &sin,
            Some(&causal_mask),
            &mut kv_caches,
        )?;
        let mut last_hidden = hidden.narrow(1, seq_len - 1, 1)?;

        let mut flat_codes: Vec<u32> =
            Vec::with_capacity(max_new_tokens * self.config.num_code_groups);
        let mut num_frames = 0usize;
        let mut c0_history: Vec<u16> = Vec::with_capacity(max_new_tokens);
        let tts_pad = tts_pad_embed.cloned().unwrap_or_else(|| {
            Tensor::zeros(&[1, 1, self.config.hidden_size], DType::F32, device).unwrap()
        });
        let trailing = trailing_text_hidden.cloned().unwrap_or_else(|| {
            Tensor::zeros(&[batch, 1, self.config.hidden_size], DType::F32, device).unwrap()
        });
        let trailing_len = trailing.dim(1)?;
        let mut gen_step: usize = 0;

        for _step in 0..max_new_tokens {
            let logits = self.codec_head_logits(&last_hidden)?.squeeze(1)?;
            if capture {
                observer.on_talker_codebook0_logits(num_frames, &logits)?;
            }
            let c0_val = sampler.sample_with_mode(
                &logits,
                sampling,
                talker_do_sample,
                Some(2048),
                Some(self.config.codec_eos_token_id as usize),
                &c0_history,
            )? as u16;
            let c0_t = Tensor::new(&[c0_val as u32], device)?;
            let c0_2d = c0_t.reshape((1, 1))?;
            let c0_emb = self.embed_codec(&c0_2d)?;

            let mut cp_kv_caches = vec![None; self.config.code_predictor.num_hidden_layers];
            let (codes_1_15, _) = self.code_predictor.generate_sampled_with_observer(
                &last_hidden,
                &c0_emb,
                &mut cp_kv_caches,
                device,
                sampler,
                subtalker_sampling,
                subtalker_do_sample,
                num_frames,
                observer,
            )?;

            flat_codes.push(c0_val as u32);
            flat_codes.extend(codes_1_15.squeeze(0)?.to_vec1::<u32>()?);
            num_frames += 1;

            record_c0_history_non_eos(
                &mut c0_history,
                self.config.codec_eos_token_id as u16,
                c0_val,
            );

            if c0_val as u32 == self.config.codec_eos_token_id {
                break;
            }

            let mut sum_emb = c0_emb;
            for i in 0..(self.config.num_code_groups - 1) {
                let ci_token = codes_1_15.narrow(1, i, 1)?;
                let ci_emb = embedding_lookup(&self.code_predictor.codec_embeddings[i], &ci_token)?;
                sum_emb = (sum_emb + ci_emb)?;
            }

            let text_add = if gen_step < trailing_len {
                trailing.narrow(1, gen_step, 1)?
            } else {
                tts_pad.clone()
            };
            let next_input = (sum_emb + text_add)?;

            let cache_position_start = u32::try_from(seq_len + gen_step).map_err(|_| {
                Error::Msg(format!(
                    "cache_position_start overflow for seq_len {seq_len} gen_step {gen_step}"
                ))
            })?;
            let positions = MultimodalRotaryEmbedding::cached_positions_from_delta(
                cache_position_start,
                &rope_delta,
                1,
                device,
            )?;
            let (cos, sin) = self.rope.forward_single_position(&positions)?;
            let hidden = self
                .model
                .forward(&next_input, &cos, &sin, None, &mut kv_caches)?;
            last_hidden = hidden;
            gen_step += 1;
        }

        let result = Tensor::from_slice(&flat_codes, (num_frames, 16), device)?;
        if capture {
            observer.on_talker_final_codes(&result)?;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::record_c0_history_non_eos;
    use crate::talker::config::TalkerConfig;

    fn test_config() -> TalkerConfig {
        TalkerConfig {
            codec_eos_token_id: 777,
            num_code_groups: 16,
            ..Default::default()
        }
    }

    #[test]
    fn record_c0_history_skips_eos_and_only_pushes_non_eos_tokens() {
        let mut c0_history = Vec::new();
        let config = test_config();
        let eos = config.codec_eos_token_id as u16;

        record_c0_history_non_eos(&mut c0_history, eos, 12);
        record_c0_history_non_eos(&mut c0_history, eos, 77);
        record_c0_history_non_eos(&mut c0_history, eos, eos);
        record_c0_history_non_eos(&mut c0_history, eos, 12);

        assert_eq!(c0_history, vec![12u16, 77u16, 12u16]);
    }

    #[test]
    fn c0_history_capacity_targets_max_new_tokens() {
        let max_new_tokens = 7usize;
        let mut c0_history: Vec<u16> = Vec::with_capacity(max_new_tokens);
        assert_eq!(c0_history.capacity(), max_new_tokens);

        record_c0_history_non_eos(&mut c0_history, 777u16, 1u16);
        record_c0_history_non_eos(&mut c0_history, 777u16, 2u16);
        record_c0_history_non_eos(&mut c0_history, 777u16, 777u16);
        assert_eq!(c0_history, vec![1u16, 2u16]);
        assert!(c0_history.capacity() >= max_new_tokens);
    }
}
