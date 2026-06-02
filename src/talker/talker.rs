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

use candle_core::{DType, Device, Result, Tensor};

use super::code_predictor::CodePredictor;
use super::config::TalkerConfig;
use super::model::TalkerModel;
use super::primitives::{
    create_causal_mask, embedding_lookup, linear, linear_with_bias, MultimodalRotaryEmbedding,
};

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
        let zero = Tensor::new(&[0.0f32], device)?;
        let cumsum = cumsum.broadcast_maximum(&zero)?;
        let cumsum = cumsum.broadcast_mul(&attention_mask_f)?;

        // [3, batch, seq_len]
        let pos_3d = cumsum
            .unsqueeze(0)?
            .expand((3, cumsum.dim(0)?, cumsum.dim(1)?))?;

        let max_pos = pos_3d.max(2)?.max(1)?;

        let seq_lens = attention_mask.sum(1)?;
        let max_pos_plus_one = max_pos.broadcast_add(&one)?;
        let delta = max_pos_plus_one.broadcast_sub(&seq_lens.to_dtype(DType::F32)?)?;

        Ok((pos_3d.to_dtype(DType::U32)?, delta))
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
        let (batch, seq_len, _hidden) = inputs_embeds.dims3()?;
        assert_eq!(batch, 1, "Only batch=1 supported");

        // ── Prefill: 處理所有輸入 ──
        let mask = attention_mask
            .cloned()
            .unwrap_or_else(|| Tensor::ones(&[batch, seq_len], DType::I64, device).unwrap());

        let (position_ids, _rope_delta) = self.compute_position_ids(&mask)?;

        // Create causal mask for prefill
        let causal_mask = create_causal_mask(seq_len, device)?;

        // 3D RoPE
        let (cos, sin) = self.rope.forward(inputs_embeds, &position_ids)?;

        // 28 layers
        let mut kv_caches = vec![None; self.config.num_hidden_layers];
        let (hidden, new_caches) = self.model.forward(
            inputs_embeds,
            &cos,
            &sin,
            Some(&causal_mask),
            &mut kv_caches,
        )?;
        kv_caches = new_caches;

        // hidden: [batch, seq_len, hidden]
        // 取最後一個位置
        let mut last_hidden = hidden.narrow(1, seq_len - 1, 1)?; // [batch, 1, hidden]

        // ── Generation Loop ──
        let mut all_codes: Vec<Vec<u16>> = Vec::new();

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

            // Argmax (簡單版本)
            let c0_token = logits.argmax(1)?; // [batch]
            let c0_val = c0_token.to_vec1::<u32>()?[0] as u16;
            let c0_t = Tensor::new(&[c0_val as u32], device)?;
            let c0_2d = c0_t.reshape((1, 1))?;
            let c0_emb = self.embed_codec(&c0_2d)?; // [batch, 1, hidden]

            // Step B: Code predictor 生成 codebooks 1-15
            let mut cp_kv_caches = vec![None; self.config.code_predictor.num_hidden_layers];
            let (codes_1_15, _) =
                self.code_predictor
                    .generate(&last_hidden, &c0_emb, &mut cp_kv_caches, device)?;

            // 組合 [c0, c1, ..., c15]
            let full_codes = Tensor::cat(&[c0_2d.clone(), codes_1_15.clone()], 1)?; // [batch, 16]

            // 儲存結果
            let codes_flat = full_codes.squeeze(0)?.to_vec1::<u32>()?;
            let frame: Vec<u16> = codes_flat.iter().map(|&x| x as u16).collect();
            all_codes.push(frame);

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
            let position_ids = self.generation_position_ids(seq_len + gen_step, batch, device)?;
            let (cos, sin) = self.rope.forward(&next_input, &position_ids)?;
            let (hidden, new_caches) =
                self.model
                    .forward(&next_input, &cos, &sin, None, &mut kv_caches)?;
            kv_caches = new_caches;
            last_hidden = hidden;
            gen_step += 1;
        }

        // 轉換為 [num_frames, 16]
        let n = all_codes.len();
        let flat_u32: Vec<u32> = all_codes
            .iter()
            .flat_map(|frame| frame.iter().map(|&x| x as u32))
            .collect();
        let result = Tensor::from_slice(&flat_u32, (n, 16), device)?;
        Ok(result)
    }

    fn generation_position_ids(
        &self,
        position: usize,
        batch: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let pos = position as u32;
        let mut data = Vec::with_capacity(3 * batch);
        for _ in 0..3 {
            for _ in 0..batch {
                data.push(pos);
            }
        }
        Tensor::from_slice(&data, (3, batch, 1), device)
    }
}
