//! # Talker 權重載入器
//!
//! 從原始的 Qwen3-TTS model.safetensors 載入權重，
//! 並分發到各個元件。

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};

use crate::Error;
use crate::Result;

use super::config::TalkerConfig;
use super::decoder_layer::{StandardDecoderLayer, TalkerDecoderLayer};
use super::primitives::{MultimodalRotaryEmbedding, RMSNorm, SwiGLUMLP};
use super::talker_attention::{StandardAttention, TalkerAttention};

/// Talker 權重載入器
pub struct TalkerWeightLoader {
    tensors: HashMap<String, Tensor>,
    device: Device,
}

impl TalkerWeightLoader {
    /// 從 model.safetensors 載入 talker 權重
    pub fn from_safetensors(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let data = std::fs::read(path.as_ref())?;
        let sf = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| Error::Weight(format!("Failed to deserialize: {e}")))?;

        let mut tensors = HashMap::new();
        for (name, view) in sf.tensors() {
            let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
            let dtype = view.dtype();
            let raw_data = view.data().to_vec();

            let tensor = match dtype {
                safetensors::Dtype::BF16 => {
                    // Convert BF16 bytes to f32
                    let n = raw_data.len() / 2;
                    let mut floats = Vec::with_capacity(n);
                    for chunk in raw_data.chunks_exact(2) {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        floats.push(bf16_to_f32(bits));
                    }
                    Tensor::from_slice(&floats, shape.as_slice(), device)?
                }
                safetensors::Dtype::F32 => {
                    let n = raw_data.len() / 4;
                    let mut floats = Vec::with_capacity(n);
                    for chunk in raw_data.chunks_exact(4) {
                        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                    Tensor::from_slice(&floats, shape.as_slice(), device)?
                }
                _ => {
                    return Err(Error::Weight(format!(
                        "Unsupported dtype {:?} for tensor {}",
                        dtype, name
                    )));
                }
            };
            tensors.insert(name.to_string(), tensor);
        }

        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// 取得張量
    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| Error::Weight(format!("Tensor '{name}' not found")))
    }

    /// 建立完整的 TalkerForConditionalGeneration
    pub fn build_talker(
        &self,
        config: &TalkerConfig,
    ) -> Result<super::talker::TalkerForConditionalGeneration> {
        use super::talker::TalkerForConditionalGeneration;

        // ── Text Embedding + Projection ──
        let text_embedding = self.get("talker.model.text_embedding.weight")?;
        let text_proj_fc1_w = self.get("talker.text_projection.linear_fc1.weight")?;
        let text_proj_fc1_b = self.get("talker.text_projection.linear_fc1.bias")?;
        let text_proj_fc2_w = self.get("talker.text_projection.linear_fc2.weight")?;
        let text_proj_fc2_b = self.get("talker.text_projection.linear_fc2.bias")?;

        // ── Codec Embedding + Head ──
        let codec_embedding = self.get("talker.model.codec_embedding.weight")?;
        let codec_head = self.get("talker.codec_head.weight")?;

        // ── Talker Model ──
        let model = self.build_talker_model(config)?;

        // ── Code Predictor ──
        let code_predictor = self.build_code_predictor(config)?;

        // ── RoPE ──
        let rope = MultimodalRotaryEmbedding::new(config, &self.device)?;

        Ok(TalkerForConditionalGeneration {
            model,
            text_embedding,
            text_proj_fc1_w,
            text_proj_fc1_b,
            text_proj_fc2_w,
            text_proj_fc2_b,
            codec_embedding,
            codec_head,
            code_predictor,
            rope,
            config: config.clone(),
        })
    }

    /// 建立 TalkerModel（28 層解碼器）
    fn build_talker_model(&self, config: &TalkerConfig) -> Result<super::model::TalkerModel> {
        use super::model::TalkerModel;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("talker.model.layers.{i}");

            let input_ln = self.get(&format!("{prefix}.input_layernorm.weight"))?;
            let post_attn_ln = self.get(&format!("{prefix}.post_attention_layernorm.weight"))?;

            // Attention
            let q_proj = self.get(&format!("{prefix}.self_attn.q_proj.weight"))?;
            let k_proj = self.get(&format!("{prefix}.self_attn.k_proj.weight"))?;
            let v_proj = self.get(&format!("{prefix}.self_attn.v_proj.weight"))?;
            let o_proj = self.get(&format!("{prefix}.self_attn.o_proj.weight"))?;
            let q_norm = self.get(&format!("{prefix}.self_attn.q_norm.weight"))?;
            let k_norm = self.get(&format!("{prefix}.self_attn.k_norm.weight"))?;

            let attn = TalkerAttention::new(
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps,
            );

            // MLP
            let gate_proj = self.get(&format!("{prefix}.mlp.gate_proj.weight"))?;
            let up_proj = self.get(&format!("{prefix}.mlp.up_proj.weight"))?;
            let down_proj = self.get(&format!("{prefix}.mlp.down_proj.weight"))?;
            let mlp = SwiGLUMLP::new(gate_proj, up_proj, down_proj);

            let layer = TalkerDecoderLayer::new(
                RMSNorm::new(input_ln, config.rms_norm_eps),
                attn,
                RMSNorm::new(post_attn_ln, config.rms_norm_eps),
                mlp,
            );
            layers.push(layer);
        }

        let norm = self.get("talker.model.norm.weight")?;

        Ok(TalkerModel {
            layers,
            norm: RMSNorm::new(norm, config.rms_norm_eps),
        })
    }

    /// 建立 Code Predictor（5 層）
    fn build_code_predictor(
        &self,
        config: &TalkerConfig,
    ) -> Result<super::code_predictor::CodePredictor> {
        use super::code_predictor::CodePredictor;

        let cp = &config.code_predictor;

        // 15 codec embeddings (code predictor 部分)
        let mut codec_embeds = Vec::with_capacity(cp.num_code_groups - 1);
        for i in 0..(cp.num_code_groups - 1) {
            let emb = self.get(&format!(
                "talker.code_predictor.model.codec_embedding.{i}.weight"
            ))?;
            codec_embeds.push(emb);
        }

        // 15 lm_heads
        let mut lm_heads = Vec::with_capacity(cp.num_code_groups - 1);
        for i in 0..(cp.num_code_groups - 1) {
            let head = self.get(&format!("talker.code_predictor.lm_head.{i}.weight"))?;
            lm_heads.push(head);
        }

        // 5 decoder layers
        let mut layers = Vec::with_capacity(cp.num_hidden_layers);
        for i in 0..cp.num_hidden_layers {
            let prefix = format!("talker.code_predictor.model.layers.{i}");

            let input_ln = self.get(&format!("{prefix}.input_layernorm.weight"))?;
            let post_attn_ln = self.get(&format!("{prefix}.post_attention_layernorm.weight"))?;

            let q_proj = self.get(&format!("{prefix}.self_attn.q_proj.weight"))?;
            let k_proj = self.get(&format!("{prefix}.self_attn.k_proj.weight"))?;
            let v_proj = self.get(&format!("{prefix}.self_attn.v_proj.weight"))?;
            let o_proj = self.get(&format!("{prefix}.self_attn.o_proj.weight"))?;

            // Code predictor has QK_Norm too
            let q_norm = self.get(&format!("{prefix}.self_attn.q_norm.weight"))?;
            let k_norm = self.get(&format!("{prefix}.self_attn.k_norm.weight"))?;

            let attn = StandardAttention::new(
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                cp.num_attention_heads,
                cp.num_key_value_heads,
                cp.head_dim,
                cp.rms_norm_eps,
            );

            let gate_proj = self.get(&format!("{prefix}.mlp.gate_proj.weight"))?;
            let up_proj = self.get(&format!("{prefix}.mlp.up_proj.weight"))?;
            let down_proj = self.get(&format!("{prefix}.mlp.down_proj.weight"))?;
            let mlp = SwiGLUMLP::new(gate_proj, up_proj, down_proj);

            let layer = StandardDecoderLayer::new(
                RMSNorm::new(input_ln, cp.rms_norm_eps),
                attn,
                RMSNorm::new(post_attn_ln, cp.rms_norm_eps),
                mlp,
            );
            layers.push(layer);
        }

        let norm = self.get("talker.code_predictor.model.norm.weight")?;

        // Code predictor also has a projection from talker hidden to its hidden
        // Since both are 1024, this is identity

        Ok(CodePredictor {
            codec_embeddings: codec_embeds,
            lm_heads,
            layers,
            norm: RMSNorm::new(norm, cp.rms_norm_eps),
            config: cp.clone(),
        })
    }
}

// BF16 → F32 conversion
fn bf16_to_f32(bits: u16) -> f32 {
    let extended = (bits as u32) << 16;
    f32::from_bits(extended)
}
