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
use super::sampling::{Sampler, SamplingOptions, greedy_select};
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

/// Production position-id/delta computation shared by generation and tests.
pub fn compute_position_ids_from_attention_mask(
    attention_mask: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let device = attention_mask.device();
    let mask = attention_mask.to_dtype(DType::F32)?;
    let one = Tensor::new(&[1.0f32], device)?;
    let positions = mask.cumsum(1)?.broadcast_sub(&one)?.broadcast_mul(&mask)?;
    let positions = positions.broadcast_add(&one.broadcast_sub(&mask)?)?;
    let pos_3d = positions
        .unsqueeze(0)?
        .expand((3, positions.dim(0)?, positions.dim(1)?))?;
    let delta = positions
        .max(1)?
        .unsqueeze(1)?
        .broadcast_add(&one)?
        .broadcast_sub(&mask.sum(1)?.unsqueeze(1)?)?;
    Ok((pos_3d.to_dtype(DType::U32)?, delta.to_dtype(DType::U32)?))
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
        compute_position_ids_from_attention_mask(attention_mask)
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
        let vocab_size = self.codec_head.dim(0)?;
        let suppress_from = vocab_size.checked_sub(1024).ok_or_else(|| {
            Error::Msg("Talker vocabulary must contain at least 1024 reserved tokens".into())
        })?;
        let eos = self.config.codec_eos_token_id as usize;
        if eos >= vocab_size || eos < suppress_from {
            return Err(Error::Msg(
                "Talker codec EOS must be inside reserved vocabulary suffix".into(),
            ));
        }
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
        if capture {
            observer.on_stage("talker-prefill-position-ids", &position_ids, "ABT")?;
            observer.on_stage("talker-prefill-rope-cos", &cos, "BBTH")?;
            observer.on_stage("talker-prefill-rope-sin", &sin, "BBTH")?;
        }
        let hidden = self.model.forward_with_observer(
            inputs_embeds,
            &cos,
            &sin,
            Some(&causal_mask),
            &mut kv_caches,
            "prefill",
            observer,
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

        for step in 0..max_new_tokens {
            // Step A: 從 last_hidden 預測 codebook 0
            let logits = self.codec_head_logits(&last_hidden)?; // [batch, 1, vocab]
            let logits = logits.squeeze(1)?; // [batch, vocab]
            if capture {
                observer.on_talker_codebook0_logits(num_frames, &logits)?;
                if step == 0 {
                    observer.on_stage("talker-logits-prefill", &logits, "BV")?;
                } else {
                    observer.on_stage(&format!("talker-logits-step{step}"), &logits, "BV")?;
                }
            }

            // Argmax (簡單版本)
            let allow_eos = (step >= 2).then_some(eos);
            let c0_val = greedy_select(&logits, Some(suppress_from), allow_eos)? as u16;
            if c0_val as usize == eos {
                break;
            }
            if terminal_cap_step(step, max_new_tokens) {
                break;
            }
            let c0_t = Tensor::new(&[c0_val as u32], device)?;
            let c0_2d = c0_t.reshape((1, 1))?;
            let c0_emb = self.embed_codec(&c0_2d)?; // [batch, 1, hidden]
            if capture {
                observer.on_stage(
                    &format!("talker-codec-embed-frame{num_frames}"),
                    &c0_emb,
                    "BTH",
                )?;
            }

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
            if capture {
                observer.on_stage(&format!("next-emb-step{gen_step}"), &next_input, "BTH")?;
            }

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
            let hidden = if capture {
                observer.on_stage(
                    &format!("talker-step{}-codec-embed", step + 1),
                    &next_input,
                    "BTH",
                )?;
                observer.on_stage(&format!("talker-step{}-rope-cos", step + 1), &cos, "BBTH")?;
                observer.on_stage(&format!("talker-step{}-rope-sin", step + 1), &sin, "BBTH")?;
                self.model.forward_with_observer(
                    &next_input,
                    &cos,
                    &sin,
                    None,
                    &mut kv_caches,
                    &format!("step{}", step + 1),
                    observer,
                )?
            } else {
                self.model
                    .forward(&next_input, &cos, &sin, None, &mut kv_caches)?
            };
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

        let vocab_size = self.codec_head.dim(0)?;
        let suppress_from = vocab_size.checked_sub(1024).ok_or_else(|| {
            Error::Msg("Talker vocabulary must contain at least 1024 reserved tokens".into())
        })?;
        let eos = self.config.codec_eos_token_id as usize;
        if eos >= vocab_size || eos < suppress_from {
            return Err(Error::Msg(
                "Talker codec EOS must be inside reserved vocabulary suffix".into(),
            ));
        }

        let mask = attention_mask
            .cloned()
            .unwrap_or_else(|| Tensor::ones(&[batch, seq_len], DType::I64, device).unwrap());
        let (position_ids, rope_delta) = self.compute_position_ids(&mask)?;
        let causal_mask = create_causal_mask(seq_len, device)?;
        let (cos, sin) = self.rope.forward(inputs_embeds, &position_ids)?;

        let mut kv_caches = vec![None; self.config.num_hidden_layers];
        if capture {
            observer.on_stage("talker-prefill-position-ids", &position_ids, "ABT")?;
            observer.on_stage("talker-prefill-rope-cos", &cos, "BBTH")?;
            observer.on_stage("talker-prefill-rope-sin", &sin, "BBTH")?;
        }
        let hidden = self.model.forward_with_observer(
            inputs_embeds,
            &cos,
            &sin,
            Some(&causal_mask),
            &mut kv_caches,
            "prefill",
            observer,
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

        for step in 0..max_new_tokens {
            let logits = self.codec_head_logits(&last_hidden)?.squeeze(1)?;
            if capture {
                observer.on_talker_codebook0_logits(num_frames, &logits)?;
                if step == 0 {
                    observer.on_stage("talker-logits-prefill", &logits, "BV")?;
                } else {
                    observer.on_stage(&format!("talker-logits-step{step}"), &logits, "BV")?;
                }
            }
            let c0_val = sampler.sample_with_mode(
                &logits,
                sampling,
                talker_do_sample,
                Some(suppress_from),
                (step >= 2).then_some(eos),
                &c0_history,
            )? as u16;
            if c0_val as usize == eos {
                break;
            }
            // HuggingFace generation counts the terminal codebook-0 draw in
            // `max_new_tokens`, but drops that incomplete frame (there is no
            // subtalker pass or following Talker state for it).
            if terminal_cap_step(step, max_new_tokens) {
                break;
            }
            let c0_t = Tensor::new(&[c0_val as u32], device)?;
            let c0_2d = c0_t.reshape((1, 1))?;
            let c0_emb = self.embed_codec(&c0_2d)?;
            if capture {
                observer.on_stage(
                    &format!("talker-codec-embed-frame{num_frames}"),
                    &c0_emb,
                    "BTH",
                )?;
            }

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
            if capture {
                observer.on_stage(&format!("next-emb-step{gen_step}"), &next_input, "BTH")?;
            }

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
            let hidden = if capture {
                observer.on_stage(
                    &format!("talker-step{}-codec-embed", step + 1),
                    &next_input,
                    "BTH",
                )?;
                observer.on_stage(&format!("talker-step{}-rope-cos", step + 1), &cos, "BBTH")?;
                observer.on_stage(&format!("talker-step{}-rope-sin", step + 1), &sin, "BBTH")?;
                self.model.forward_with_observer(
                    &next_input,
                    &cos,
                    &sin,
                    None,
                    &mut kv_caches,
                    &format!("step{}", step + 1),
                    observer,
                )?
            } else {
                self.model
                    .forward(&next_input, &cos, &sin, None, &mut kv_caches)?
            };
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

#[inline]
fn terminal_cap_step(step: usize, max_new_tokens: usize) -> bool {
    step.saturating_add(1) >= max_new_tokens
}

#[cfg(test)]
mod tests {
    use super::{record_c0_history_non_eos, terminal_cap_step};
    use crate::talker::code_predictor::CodePredictor;
    use crate::talker::config::TalkerConfig;
    use crate::talker::model::TalkerModel;
    use crate::talker::primitives::{MultimodalRotaryEmbedding, RMSNorm};
    use crate::talker::sampling::{Sampler, SamplingOptions};
    use candle_core::{DType, Device, Tensor};

    fn test_config() -> TalkerConfig {
        TalkerConfig {
            codec_eos_token_id: 777,
            num_code_groups: 16,
            ..Default::default()
        }
    }

    #[test]
    fn terminal_cap_step_covers_n2_n3_n4_and_natural_eos_is_distinct() {
        for max in [2, 3, 4] {
            assert!(!terminal_cap_step(max - 2, max));
            assert!(terminal_cap_step(max - 1, max));
        }
        assert!(terminal_cap_step(0, 0));
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

    #[test]
    fn sampled_generation_records_c0_then_fifteen_cp_calls_with_history_transition() {
        let device = Device::Cpu;
        let mut config = TalkerConfig::default();
        config.hidden_size = 6;
        config.text_hidden_size = 6;
        config.head_dim = 6;
        config.num_attention_heads = 1;
        config.num_key_value_heads = 1;
        config.num_hidden_layers = 0;
        config.vocab_size = 1028;
        config.num_code_groups = 16;
        config.mrope_section = vec![1, 1, 1];
        config.codec_eos_token_id = 1024;
        let mut cp_config = config.code_predictor.clone();
        cp_config.hidden_size = 6;
        cp_config.head_dim = 6;
        cp_config.num_attention_heads = 1;
        cp_config.num_key_value_heads = 1;
        cp_config.num_hidden_layers = 0;
        cp_config.vocab_size = 8;
        cp_config.num_code_groups = 16;
        config.code_predictor = cp_config.clone();
        let norm = RMSNorm::new(Tensor::ones((6,), DType::F32, &device).unwrap(), 1e-6);
        let rope = MultimodalRotaryEmbedding::new(&config, &device).unwrap();
        let cp_norm = RMSNorm::new(Tensor::ones((6,), DType::F32, &device).unwrap(), 1e-6);
        let predictor = CodePredictor {
            codec_embeddings: (0..15)
                .map(|_| Tensor::zeros((8, 6), DType::F32, &device).unwrap())
                .collect(),
            lm_heads: (0..15)
                .map(|_| Tensor::zeros((8, 6), DType::F32, &device).unwrap())
                .collect(),
            layers: Vec::new(),
            norm: cp_norm,
            small_to_mtp_proj: None,
            config: cp_config,
        };
        let dummy = Tensor::zeros((1,), DType::F32, &device).unwrap();
        let talker = super::TalkerForConditionalGeneration {
            model: TalkerModel::new(Vec::new(), norm),
            text_embedding: dummy.clone(),
            text_proj_fc1_w: dummy.clone(),
            text_proj_fc1_b: dummy.clone(),
            text_proj_fc2_w: dummy.clone(),
            text_proj_fc2_b: dummy.clone(),
            codec_embedding: Tensor::zeros((1028, 6), DType::F32, &device).unwrap(),
            codec_head: Tensor::zeros((1028, 6), DType::F32, &device).unwrap(),
            code_predictor: predictor,
            rope,
            config,
        };
        let inputs = Tensor::zeros((1, 1, 6), DType::F32, &device).unwrap();
        let options = SamplingOptions {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.5,
        };
        for &(talker_sample, cp_sample) in
            &[(false, false), (false, true), (true, false), (true, true)]
        {
            let mut sampler = Sampler::new(3);
            let output = talker
                .generate_sampled(
                    &inputs,
                    None,
                    None,
                    None,
                    3,
                    &device,
                    &mut sampler,
                    options,
                    talker_sample,
                    options,
                    cp_sample,
                )
                .unwrap();
            let diag = sampler.diagnostics();
            let expected_sampled = 3 * u32::from(talker_sample) + 30 * u32::from(cp_sample);
            assert_eq!(diag.call_count, 33);
            assert_eq!(diag.sampled_calls, expected_sampled);
            assert_eq!(diag.greedy_calls, 33 - expected_sampled);
            assert_eq!(sampler.subsequence_counter(), u64::from(expected_sampled));
            assert_eq!(diag.history_nonempty_calls, 2);
            assert_eq!(diag.sequence_len, 33);
            for frame in 0..2 {
                assert_eq!(diag.sequence[frame * 16], u8::from(talker_sample));
                assert!(
                    diag.sequence[frame * 16 + 1..frame * 16 + 16]
                        .iter()
                        .all(|&v| v == u8::from(cp_sample))
                );
            }
            assert_eq!(diag.history_nonempty_sequence[0], 0);
            assert_eq!(diag.history_nonempty_sequence[16], 1);
            for index in 0..32 {
                if index != 16 {
                    assert_eq!(diag.history_nonempty_sequence[index], 0);
                }
            }
            let expected = match (talker_sample, cp_sample) {
                (false, false) => vec![vec![0; 16], vec![0; 16]],
                (false, true) => vec![
                    vec![0, 6, 1, 1, 0, 3, 3, 7, 1, 6, 1, 2, 5, 3, 2, 2],
                    vec![0, 5, 3, 2, 2, 7, 0, 5, 5, 6, 7, 7, 4, 4, 7, 3],
                ],
                (true, false) => vec![
                    vec![3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                (true, true) => vec![
                    vec![3, 1, 1, 0, 3, 3, 7, 1, 6, 1, 2, 5, 3, 2, 2, 5],
                    vec![1, 2, 2, 7, 0, 5, 5, 6, 7, 7, 4, 4, 7, 3, 4, 4],
                ],
            };
            assert_eq!(output.to_vec2::<u32>().unwrap(), expected);
        }

        let mut eos_talker = talker.clone();
        let mut head = vec![0.0_f32; 1028 * 6];
        for value in &mut head[..6] {
            *value = 0.1;
        }
        for value in &mut head[1024 * 6..1025 * 6] {
            *value = 1.0;
        }
        eos_talker.codec_head = Tensor::from_slice(&head, (1028, 6), &device).unwrap();
        let mut eos_sampler = Sampler::new(3);
        let eos_pad = Tensor::ones((1, 1, 6), DType::F32, &device).unwrap();
        let eos_output = eos_talker
            .generate_sampled(
                &inputs,
                None,
                None,
                Some(&eos_pad),
                4,
                &device,
                &mut eos_sampler,
                options,
                true,
                options,
                true,
            )
            .unwrap();
        assert_eq!(eos_output.dims(), &[2, 16]);
        let eos_diag = eos_sampler.diagnostics();
        assert_eq!(eos_diag.call_count, 33);
        assert_eq!(eos_diag.sampled_calls, 33);
        assert_eq!(eos_sampler.subsequence_counter(), 33);
        assert_eq!(eos_diag.history_nonempty_calls, 2);
        assert_eq!(eos_diag.history_nonempty_sequence[16], 1);
        assert_eq!(eos_output.to_vec2::<u32>().unwrap()[0][0], 3);
        assert_eq!(eos_output.to_vec2::<u32>().unwrap()[1][0], 1);

        let greedy_output = eos_talker
            .generate(&inputs, None, None, Some(&eos_pad), 4, &device)
            .unwrap();
        assert_eq!(greedy_output.dims(), &[2, 16]);
        assert_eq!(greedy_output.to_vec2::<u32>().unwrap()[0][0], 0);
        assert_eq!(greedy_output.to_vec2::<u32>().unwrap()[1][0], 0);
    }
}
