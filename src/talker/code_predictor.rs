//! # Code Predictor — 子碼本預測器
//!
//! 5 層 Transformer，預測 codebooks 1-15。
//! 每步生成一個 token，使用對應的 codec_embedding 和 lm_head。

use candle_core::{Device, Result, Tensor};

use super::config::CodePredictorConfig;
use super::decoder_layer::StandardDecoderLayer;
use super::primitives::{RMSNorm, create_causal_mask, embedding_lookup, linear, linear_with_bias};
use super::sampling::{Sampler, SamplingOptions};
use crate::alignment_stage_dump::{NoopStageDumpObserver, StageDumpObserver};

/// 子碼本預測器
#[derive(Debug, Clone)]
pub struct CodePredictor {
    /// 15 個 codec 嵌入 (0..14)
    pub codec_embeddings: Vec<Tensor>,
    /// 15 個 lm_heads (0..14)
    pub lm_heads: Vec<Tensor>,
    /// 5 層解碼器
    pub layers: Vec<StandardDecoderLayer>,
    /// 最終 norm
    pub norm: RMSNorm,
    /// Optional projection from talker/codebook embedding width to code predictor width.
    pub small_to_mtp_proj: Option<(Tensor, Tensor)>,
    /// 配置
    pub config: CodePredictorConfig,
}

impl CodePredictor {
    /// Return codebook-1 logits after prefill `[talker_hidden, codebook_0_embed]`.
    pub fn first_step_logits(
        &self,
        talker_hidden: &Tensor,
        codebook_0_embed: &Tensor,
        kv_caches: &mut [Option<(Tensor, Tensor)>],
        device: &Device,
    ) -> Result<Tensor> {
        let mut observer = NoopStageDumpObserver::default();
        self.first_step_logits_with_observer(
            talker_hidden,
            codebook_0_embed,
            kv_caches,
            device,
            0,
            &mut observer,
        )
    }

    fn first_step_logits_with_observer<O: StageDumpObserver>(
        &self,
        talker_hidden: &Tensor,
        codebook_0_embed: &Tensor,
        kv_caches: &mut [Option<(Tensor, Tensor)>],
        device: &Device,
        frame_index: usize,
        observer: &mut O,
    ) -> Result<Tensor> {
        let capture = observer.wants_capture();
        let prefill = Tensor::cat(&[talker_hidden.clone(), codebook_0_embed.clone()], 1)?;
        let prefill = self.project_input(&prefill)?;
        if capture {
            observer.on_stage(
                &format!("code-predictor-prefill-frame{frame_index}-input-embed"),
                &prefill,
                "BTH",
            )?;
        }
        let (cos, sin) = self.compute_rope_for_positions(&[0, 1], device)?;
        if capture {
            let positions = Tensor::from_slice(&[0u32, 1], (1, 2), device)?;
            observer.on_stage(
                &format!("code-predictor-prefill-frame{frame_index}-position-ids"),
                &positions,
                "BT",
            )?;
            observer.on_stage(
                &format!("code-predictor-prefill-frame{frame_index}-rope-cos"),
                &cos,
                "BBTH",
            )?;
            observer.on_stage(
                &format!("code-predictor-prefill-frame{frame_index}-rope-sin"),
                &sin,
                "BBTH",
            )?;
        }
        let causal_mask = create_causal_mask(2, device)?;
        let h = self.forward_layers(
            &prefill,
            &cos,
            &sin,
            Some(&causal_mask),
            kv_caches,
            "prefill",
            frame_index,
            observer,
        )?;
        let last_hidden = h.narrow(1, 1, 1)?;
        linear(&last_hidden, &self.lm_heads[0])?.squeeze(1)
    }

    /// 生成 codebooks 1-15
    ///
    /// # 參數
    /// - `talker_hidden`: [batch, 1, hidden_size] — 來自 talker 的最後隱藏狀態
    /// - `codebook_0_embed`: [batch, 1, hidden_size] — codebook 0 的主 codec embedding
    /// - `kv_caches`: 5 層的 KV cache（可選，用於後續步數）
    ///
    /// # 回傳
    /// - `code_ids`: [batch, 15] — codebooks 1-15
    /// - `updated_caches`: 更新後的 KV caches
    pub fn generate(
        &self,
        talker_hidden: &Tensor,
        codebook_0_embed: &Tensor,
        kv_caches: &mut [Option<(Tensor, Tensor)>],
        device: &Device,
    ) -> Result<(Tensor, Vec<Option<(Tensor, Tensor)>>)> {
        let mut observer = NoopStageDumpObserver::default();
        self.generate_with_observer(
            talker_hidden,
            codebook_0_embed,
            kv_caches,
            device,
            0,
            &mut observer,
        )
    }

    pub fn generate_with_observer<O: StageDumpObserver>(
        &self,
        talker_hidden: &Tensor,
        codebook_0_embed: &Tensor,
        kv_caches: &mut [Option<(Tensor, Tensor)>],
        device: &Device,
        frame_index: usize,
        observer: &mut O,
    ) -> Result<(Tensor, Vec<Option<(Tensor, Tensor)>>)> {
        let mut generated_ids: Vec<u32> = Vec::with_capacity(self.config.num_code_groups - 1);
        let capture = observer.wants_capture();

        let logits = self.first_step_logits_with_observer(
            talker_hidden,
            codebook_0_embed,
            kv_caches,
            device,
            frame_index,
            observer,
        )?;
        if capture {
            observer.on_code_predictor_step_logits(frame_index, 0, &logits)?;
        }
        let next_token = logits.argmax(1)?;
        let mut next_val = next_token.to_vec1::<u32>()?[0];
        generated_ids.push(next_val);

        let step_positions: Vec<u32> = (2..self.config.num_code_groups as u32).collect();
        let (step_cos, step_sin) = self.compute_rope_for_positions(&step_positions, device)?;
        for step in 1..(self.config.num_code_groups - 1) {
            let emb_weight = &self.codec_embeddings[step - 1];
            let next_input_ids = Tensor::from_slice(&[next_val], (1, 1), device)?;
            let next_input = embedding_lookup(emb_weight, &next_input_ids)?;
            let next_input = self.project_input(&next_input)?;
            if capture {
                observer.on_stage(
                    &format!("code-predictor-step{step}-frame{frame_index}-codec-embed"),
                    &next_input,
                    "BTH",
                )?;
            }
            let cos = step_cos.narrow(2, step - 1, 1)?;
            let sin = step_sin.narrow(2, step - 1, 1)?;
            let layer_phase = if capture {
                Some(format!("step{step}"))
            } else {
                None
            };
            let h = self.forward_layers(
                &next_input,
                &cos,
                &sin,
                None,
                kv_caches,
                layer_phase.as_deref().unwrap_or("step"),
                frame_index,
                observer,
            )?;
            let logits = linear(&h, &self.lm_heads[step])?.squeeze(1)?;
            if capture {
                observer.on_code_predictor_step_logits(frame_index, step, &logits)?;
            }
            let next_token = logits.argmax(1)?;
            next_val = next_token.to_vec1::<u32>()?[0];
            generated_ids.push(next_val);
        }

        let code_tensor = Tensor::from_slice(&generated_ids, (1, generated_ids.len()), device)?;
        if capture {
            observer.on_code_predictor_final_codes(frame_index, &code_tensor)?;
        }
        Ok((code_tensor, kv_caches.to_vec()))
    }

    pub fn generate_sampled(
        &self,
        talker_hidden: &Tensor,
        codebook_0_embed: &Tensor,
        kv_caches: &mut [Option<(Tensor, Tensor)>],
        device: &Device,
        sampler: &mut Sampler,
        sampling: SamplingOptions,
        do_sample: bool,
    ) -> Result<(Tensor, Vec<Option<(Tensor, Tensor)>>)> {
        let mut observer = NoopStageDumpObserver::default();
        self.generate_sampled_with_observer(
            talker_hidden,
            codebook_0_embed,
            kv_caches,
            device,
            sampler,
            sampling,
            do_sample,
            0,
            &mut observer,
        )
    }

    pub fn generate_sampled_with_observer<O: StageDumpObserver>(
        &self,
        talker_hidden: &Tensor,
        codebook_0_embed: &Tensor,
        kv_caches: &mut [Option<(Tensor, Tensor)>],
        device: &Device,
        sampler: &mut Sampler,
        sampling: SamplingOptions,
        do_sample: bool,
        frame_index: usize,
        observer: &mut O,
    ) -> Result<(Tensor, Vec<Option<(Tensor, Tensor)>>)> {
        let mut generated_ids: Vec<u32> = Vec::with_capacity(self.config.num_code_groups - 1);
        let capture = observer.wants_capture();

        let logits = self.first_step_logits_with_observer(
            talker_hidden,
            codebook_0_embed,
            kv_caches,
            device,
            frame_index,
            observer,
        )?;
        if capture {
            observer.on_code_predictor_step_logits(frame_index, 0, &logits)?;
        }
        let sampling = if do_sample {
            sampling
        } else {
            SamplingOptions {
                temperature: -1.0,
                ..sampling
            }
        };
        let mut next_val = sampler.sample(&logits, sampling, None, None, &[])?;
        generated_ids.push(next_val);

        let step_positions: Vec<u32> = (2..self.config.num_code_groups as u32).collect();
        let (step_cos, step_sin) = self.compute_rope_for_positions(&step_positions, device)?;
        for step in 1..(self.config.num_code_groups - 1) {
            let emb_weight = &self.codec_embeddings[step - 1];
            let next_input_ids = Tensor::from_slice(&[next_val], (1, 1), device)?;
            let next_input = embedding_lookup(emb_weight, &next_input_ids)?;
            let next_input = self.project_input(&next_input)?;
            if capture {
                observer.on_stage(
                    &format!("code-predictor-step{step}-frame{frame_index}-codec-embed"),
                    &next_input,
                    "BTH",
                )?;
            }
            let cos = step_cos.narrow(2, step - 1, 1)?;
            let sin = step_sin.narrow(2, step - 1, 1)?;
            let layer_phase = if capture {
                Some(format!("step{step}"))
            } else {
                None
            };
            let h = self.forward_layers(
                &next_input,
                &cos,
                &sin,
                None,
                kv_caches,
                layer_phase.as_deref().unwrap_or("step"),
                frame_index,
                observer,
            )?;
            let logits = linear(&h, &self.lm_heads[step])?.squeeze(1)?;
            if capture {
                observer.on_code_predictor_step_logits(frame_index, step, &logits)?;
            }
            next_val = sampler.sample(&logits, sampling, None, None, &[])?;
            generated_ids.push(next_val);
        }

        let code_tensor = Tensor::from_slice(&generated_ids, (1, generated_ids.len()), device)?;
        if capture {
            observer.on_code_predictor_final_codes(frame_index, &code_tensor)?;
        }
        Ok((code_tensor, kv_caches.to_vec()))
    }

    fn project_input(&self, input: &Tensor) -> Result<Tensor> {
        if let Some((weight, bias)) = &self.small_to_mtp_proj {
            linear_with_bias(input, weight, bias)
        } else {
            Ok(input.clone())
        }
    }

    fn forward_layers(
        &self,
        input: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_caches: &mut [Option<(Tensor, Tensor)>],
        phase: &str,
        frame_index: usize,
        observer: &mut impl StageDumpObserver,
    ) -> Result<Tensor> {
        let mut h = input.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let cache = if layer_idx < kv_caches.len() {
                kv_caches[layer_idx].as_ref().map(|(k, v)| (k, v))
            } else {
                None
            };
            let (next_h, updated_cache) = if observer.wants_capture() {
                let layer_phase = format!("{phase}-frame{frame_index}");
                layer.forward_with_observer(
                    &h,
                    cos,
                    sin,
                    attention_mask,
                    cache,
                    layer_idx,
                    &layer_phase,
                    observer,
                )?
            } else {
                layer.forward(&h, cos, sin, attention_mask, cache)?
            };
            h = next_h;
            if layer_idx < kv_caches.len() {
                kv_caches[layer_idx] = Some(updated_cache);
            }
        }
        let h = self.norm.forward(&h)?;
        if observer.wants_capture() {
            observer.on_stage(
                &format!("code-predictor-hidden-{phase}-frame{frame_index}-final"),
                &h,
                "BTH",
            )?;
        }
        Ok(h)
    }

    fn compute_rope_for_positions(
        &self,
        positions: &[u32],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let half = self.config.head_dim / 2;
        let mut emb = Vec::with_capacity(positions.len() * self.config.head_dim);
        for &pos in positions {
            for i in 0..half {
                emb.push(pos as f32 / self.config.rope_theta.powf(i as f64 / half as f64) as f32);
            }
            for i in 0..half {
                emb.push(pos as f32 / self.config.rope_theta.powf(i as f64 / half as f64) as f32);
            }
        }
        let emb = Tensor::from_slice(&emb, (1, 1, positions.len(), self.config.head_dim), device)?;
        Ok((emb.cos()?, emb.sin()?))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;
    use crate::talker::decoder_layer::StandardDecoderLayer;
    use crate::talker::primitives::SwiGLUMLP;
    use crate::talker::talker_attention::StandardAttention;

    fn ones1(a: usize) -> Tensor {
        Tensor::ones(a, DType::F32, &Device::Cpu).unwrap()
    }

    fn zeros2(a: usize, b: usize) -> Tensor {
        Tensor::zeros((a, b), DType::F32, &Device::Cpu).unwrap()
    }

    fn zeros3(a: usize, b: usize, c: usize) -> Tensor {
        Tensor::zeros((a, b, c), DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn generate_returns_sub_codebooks_and_updates_cache() {
        let device = Device::Cpu;
        let config = CodePredictorConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            num_hidden_layers: 1,
            vocab_size: 8,
            num_code_groups: 16,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            hidden_act: "silu".into(),
            attention_bias: false,
            attention_dropout: 0.0,
            layer_types: vec!["full_attention".into()],
        };

        let attn = StandardAttention::new(
            zeros2(4, 4),
            zeros2(2, 4),
            zeros2(2, 4),
            zeros2(4, 4),
            ones1(2),
            ones1(2),
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rms_norm_eps,
        );
        let layer = StandardDecoderLayer::new(
            super::RMSNorm::new(ones1(4), config.rms_norm_eps),
            attn,
            super::RMSNorm::new(ones1(4), config.rms_norm_eps),
            SwiGLUMLP::new(zeros2(8, 4), zeros2(8, 4), zeros2(4, 8)),
        );
        let predictor = CodePredictor {
            codec_embeddings: (0..15).map(|_| zeros2(8, 4)).collect(),
            lm_heads: (0..15).map(|_| zeros2(8, 4)).collect(),
            layers: vec![layer],
            norm: super::RMSNorm::new(ones1(4), config.rms_norm_eps),
            small_to_mtp_proj: None,
            config,
        };

        let talker_hidden = zeros3(1, 1, 4);
        let c0_embed = zeros3(1, 1, 4);
        let mut caches = vec![None];

        let (codes, updated) = predictor
            .generate(&talker_hidden, &c0_embed, &mut caches, &device)
            .unwrap();

        assert_eq!(codes.dims(), &[1, 15]);
        assert_eq!(updated.len(), 1);
        let (k, v) = updated[0].as_ref().expect("cache should be populated");
        assert_eq!(k.dim(2).unwrap(), 16);
        assert_eq!(v.dim(2).unwrap(), 16);

        let mut sampled_sampler = Sampler::new(42);
        let sampled_options = SamplingOptions {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        predictor
            .generate_sampled(
                &talker_hidden,
                &c0_embed,
                &mut vec![None],
                &device,
                &mut sampled_sampler,
                sampled_options,
                true,
            )
            .unwrap();
        let diagnostics = sampled_sampler.diagnostics();
        assert_eq!(diagnostics.call_count, 15);
        assert_eq!(diagnostics.sampled_calls, 15);
        assert_eq!(diagnostics.history_nonempty_calls, 0);
        assert_eq!(diagnostics.sequence_len, 15);
        assert!(diagnostics.sequence[..15].iter().all(|&mode| mode == 1));

        let mut reserved_predictor = predictor.clone();
        reserved_predictor.config.vocab_size = 2048;
        reserved_predictor.codec_embeddings = (0..15).map(|_| zeros2(2048, 4)).collect();
        reserved_predictor.lm_heads = (0..15)
            .map(|index| {
                let mut weights = vec![0.0_f32; 2048 * 4];
                if index == 0 {
                    for value in &mut weights[1500 * 4..1501 * 4] {
                        *value = 1.0;
                    }
                }
                Tensor::from_slice(&weights, (2048, 4), &device).unwrap()
            })
            .collect();
        let mut reserved_sampler = Sampler::new(11);
        let (reserved_codes, _) = reserved_predictor
            .generate_sampled(
                &Tensor::ones((1, 1, 4), DType::F32, &device).unwrap(),
                &Tensor::ones((1, 1, 4), DType::F32, &device).unwrap(),
                &mut vec![None],
                &device,
                &mut reserved_sampler,
                sampled_options,
                false,
            )
            .unwrap();
        assert_eq!(reserved_codes.to_vec2::<u32>().unwrap()[0][0], 1500);
    }
}
