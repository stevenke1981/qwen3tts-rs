//! # Talker 權重載入器
//!
//! 從原始的 Qwen3-TTS model.safetensors 載入權重，
//! 並分發到各個元件。

use std::collections::HashMap;
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use rayon::prelude::*;

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
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        Self::from_bytes(&mmap, device)
    }

    /// 從記憶體中的 safetensors 位元組載入
    pub fn from_bytes(data: &[u8], device: &Device) -> Result<Self> {
        let sf = safetensors::SafeTensors::deserialize(data)
            .map_err(|e| Error::Weight(format!("Failed to deserialize: {e}")))?;

        let mut tensors = HashMap::new();
        for (name, view) in sf.tensors() {
            let shape = view.shape().to_vec();
            let dtype = view.dtype();
            let data = view.data();

            let tensor = match dtype {
                safetensors::Dtype::BF16 => {
                    let floats = bf16_bytes_to_f32_vec(data)?;
                    Tensor::from_slice(&floats, shape.as_slice(), device)?
                }
                safetensors::Dtype::F32 => {
                    let floats = f32_bytes_to_f32_vec(data)?;
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

    /// 從 GGUF 檔案載入 talker 權重
    ///
    /// 使用 candle-core 內建 `gguf_file` 模組讀取 qwentts.cpp 格式的 GGUF 檔案。
    /// 所有量化權重會自動 dequantize 為 F32，並透過 `gguf_key_to_safetensors`
    /// 將 GGUF tensor 名稱對應到與 `from_safetensors` 一致的鍵名，
    /// 因此 `build_talker()` / `build_talker_model()` / `build_code_predictor()`
    /// 可以無修改共用。
    pub fn from_gguf(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path)
            .map_err(|e| Error::Weight(format!("Failed to open GGUF file {path:?}: {e}")))?;
        let content = gguf_file::Content::read(&mut file).map_err(|e| {
            Error::Weight(format!("Failed to read GGUF content from {path:?}: {e}"))
        })?;

        let mut tensors = HashMap::new();
        for (gguf_name, info) in &content.tensor_infos {
            let qtensor = info
                .read(&mut file, content.tensor_data_offset, device)
                .map_err(|e| {
                    Error::Weight(format!("Failed to read GGUF tensor '{gguf_name}': {e}"))
                })?;
            let f32_tensor = qtensor.dequantize(device).map_err(|e| {
                Error::Weight(format!(
                    "Failed to dequantize GGUF tensor '{gguf_name}': {e}"
                ))
            })?;
            let sf_name = gguf_key_to_safetensors(gguf_name);
            if tensors.insert(sf_name.clone(), f32_tensor).is_some() {
                log::warn!(
                    "Duplicate tensor key after GGUF→safetensors mapping: {sf_name} (was {gguf_name})"
                );
            }
        }

        log::info!("Loaded {} tensors from GGUF file {path:?}", tensors.len());

        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// 從已載入的權重 shape 推斷 `TalkerConfig`
    ///
    /// 用於支援不同 hidden_size / intermediate_size 的模型變體
    /// （0.6B 與 1.7B 共用同一套權重命名規則，僅維度不同）。
    pub fn infer_config(&self) -> Result<TalkerConfig> {
        // Talker 主幹
        let codec_emb = self.get("talker.model.codec_embedding.weight")?;
        let (_codec_vocab, hidden_size) = codec_emb.dims2()?;
        let text_emb = self.get("talker.model.text_embedding.weight")?;
        let (_text_vocab, text_hidden_size) = text_emb.dims2()?;

        // 取 layer 0 的 mlp.gate_proj 形狀以推斷 intermediate_size
        let mlp_gate = self.get("talker.model.layers.0.mlp.gate_proj.weight")?;
        let (intermediate_size, _in_dim) = mlp_gate.dims2()?;

        // num_hidden_layers: 透過 self_attn.q_proj.{i} 的數量
        let num_hidden_layers = (0..64)
            .take_while(|i| {
                self.tensors
                    .contains_key(&format!("talker.model.layers.{i}.self_attn.q_proj.weight"))
            })
            .count();

        // code_predictor: 透過 lm_head.{0..14}
        let cp_hidden_size = self
            .get("talker.code_predictor.lm_head.0.weight")
            .and_then(|t| t.dim(1).map_err(Error::from))
            .unwrap_or(1024); // fallback for 0.6B-like
        let cp_intermediate_size = self
            .get("talker.code_predictor.model.layers.0.mlp.gate_proj.weight")
            .ok()
            .and_then(|t| t.dims2().ok())
            .map(|(d, _)| d)
            .unwrap_or(3072);
        let cp_num_layers = (0..16)
            .take_while(|i| {
                self.tensors.contains_key(&format!(
                    "talker.code_predictor.model.layers.{i}.self_attn.q_proj.weight"
                ))
            })
            .count()
            .max(5);

        // 從 default 開始，覆寫推斷出的維度
        let mut cfg = TalkerConfig::default();
        cfg.hidden_size = hidden_size;
        cfg.intermediate_size = intermediate_size;
        cfg.text_hidden_size = text_hidden_size;
        cfg.num_hidden_layers = num_hidden_layers;
        cfg.code_predictor.hidden_size = cp_hidden_size;
        cfg.code_predictor.intermediate_size = cp_intermediate_size;
        cfg.code_predictor.num_hidden_layers = cp_num_layers;
        Ok(cfg)
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

        // 1.7B projects 2048-wide talker/codebook embeddings into the
        // 1024-wide code predictor transformer. 0.6B already uses 1024-wide
        // embeddings here and has no projection tensor.
        let small_to_mtp_proj = match (
            self.tensors
                .get("talker.code_predictor.small_to_mtp_projection.weight"),
            self.tensors
                .get("talker.code_predictor.small_to_mtp_projection.bias"),
        ) {
            (Some(w), Some(b)) => Some((w.clone(), b.clone())),
            _ => None,
        };

        Ok(CodePredictor {
            codec_embeddings: codec_embeds,
            lm_heads,
            layers,
            norm: RMSNorm::new(norm, cp.rms_norm_eps),
            small_to_mtp_proj,
            config: cp.clone(),
        })
    }
}

/// 將 GGUF tensor 名稱轉換為 safetensors 風格的鍵名
///
/// GGUF 檔案使用 `blk.{i}.{component}.{param}` 的命名慣例（與 llama.cpp 生態一致），
/// 而現有 `TalkerWeightLoader` 使用形如 `talker.model.layers.{i}.{component}.{param}` 的命名。
/// 此函式實作完整的轉換對照表。
///
/// # 已知限制
/// - GGUF 可能不包含 `codec_embedding.{i}.weight` 與 `lm_head.{i}.weight`（i=0..14）
///   這些 tensor 如果遺漏，`build_code_predictor()` 會回傳 `Tensor not found` 錯誤。
///   需要依實際 GGUF 檔案內容調整。
fn gguf_key_to_safetensors(gguf_key: &str) -> String {
    // ── Talker 主模型層 ──
    // talker.blk.{i}.attn_q.weight  → talker.model.layers.{i}.self_attn.q_proj.weight
    // 實際 GGUF 使用 flat 命名：attn_output.weight、attn_q_norm.weight、ffn_gate.weight 等
    if let Some(rest) = gguf_key.strip_prefix("talker.blk.") {
        let mapped = rest
            .replace("attn_output.weight", "self_attn.o_proj.weight")
            .replace("attn_q.weight", "self_attn.q_proj.weight")
            .replace("attn_k.weight", "self_attn.k_proj.weight")
            .replace("attn_v.weight", "self_attn.v_proj.weight")
            .replace("attn_q_norm.weight", "self_attn.q_norm.weight")
            .replace("attn_k_norm.weight", "self_attn.k_norm.weight")
            .replace("attn_norm.weight", "input_layernorm.weight")
            .replace("ffn_norm.weight", "post_attention_layernorm.weight")
            .replace("ffn_gate.weight", "mlp.gate_proj.weight")
            .replace("ffn_up.weight", "mlp.up_proj.weight")
            .replace("ffn_down.weight", "mlp.down_proj.weight");
        return format!("talker.model.layers.{mapped}");
    }

    // ── Code Predictor 層 ──
    // code_pred.blk.{i}.attn_q.weight  → talker.code_predictor.model.layers.{i}.self_attn.q_proj.weight
    if let Some(rest) = gguf_key.strip_prefix("code_pred.blk.") {
        let mapped = rest
            .replace("attn_output.weight", "self_attn.o_proj.weight")
            .replace("attn_q.weight", "self_attn.q_proj.weight")
            .replace("attn_k.weight", "self_attn.k_proj.weight")
            .replace("attn_v.weight", "self_attn.v_proj.weight")
            .replace("attn_q_norm.weight", "self_attn.q_norm.weight")
            .replace("attn_k_norm.weight", "self_attn.k_norm.weight")
            .replace("attn_norm.weight", "input_layernorm.weight")
            .replace("ffn_norm.weight", "post_attention_layernorm.weight")
            .replace("ffn_gate.weight", "mlp.gate_proj.weight")
            .replace("ffn_up.weight", "mlp.up_proj.weight")
            .replace("ffn_down.weight", "mlp.down_proj.weight");
        return format!("talker.code_predictor.model.layers.{mapped}");
    }

    // ── 簡單對照的頂層 tensor ──
    match gguf_key {
        "talker.text_embd.weight" => "talker.model.text_embedding.weight".to_string(),
        "talker.codec_embd.weight" => "talker.model.codec_embedding.weight".to_string(),
        "talker.text_proj.fc1.weight" => "talker.text_projection.linear_fc1.weight".to_string(),
        "talker.text_proj.fc1.bias" => "talker.text_projection.linear_fc1.bias".to_string(),
        "talker.text_proj.fc2.weight" => "talker.text_projection.linear_fc2.weight".to_string(),
        "talker.text_proj.fc2.bias" => "talker.text_projection.linear_fc2.bias".to_string(),
        "talker.codec_head.weight" => "talker.codec_head.weight".to_string(),
        "talker.output_norm.weight" => "talker.model.norm.weight".to_string(),
        "code_pred.output_norm.weight" => "talker.code_predictor.model.norm.weight".to_string(),
        "code_pred.mtp_proj.weight" => {
            "talker.code_predictor.small_to_mtp_projection.weight".to_string()
        }
        "code_pred.mtp_proj.bias" => {
            "talker.code_predictor.small_to_mtp_projection.bias".to_string()
        }

        // ── Code Predictor 的子碼本嵌入表 ──
        // code_pred.codec_embd.{i}.weight → talker.code_predictor.model.codec_embedding.{i}.weight
        // 注意：GGUF 使用 codec_embd（縮寫），safetensors 使用 codec_embedding（全稱）
        _ if gguf_key.starts_with("code_pred.codec_embd.") => gguf_key.replacen(
            "code_pred.codec_embd.",
            "talker.code_predictor.model.codec_embedding.",
            1,
        ),

        // ── Code Predictor lm_head ──
        // code_pred.lm_head.{i}.weight → talker.code_predictor.lm_head.{i}.weight
        _ if gguf_key.starts_with("talker.") => gguf_key.to_string(),
        _ if gguf_key.starts_with("code_pred.") => {
            let rest = gguf_key.strip_prefix("code_pred.").unwrap();
            format!("talker.code_predictor.{rest}")
        }

        _ => {
            log::warn!("Unknown GGUF tensor key: {gguf_key}, passing through as-is");
            gguf_key.to_string()
        }
    }
}

/// 將 BF16 位元組轉為 f32 向量（若 N >= 16384 則使用 Rayon 並行轉換）
pub fn bf16_bytes_to_f32_vec(data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(2) {
        return Err(Error::Weight(format!(
            "Invalid BF16 tensor byte length {}",
            data.len()
        )));
    }
    let n = data.len() / 2;
    let mut floats = vec![0.0f32; n];
    if n >= 16384 {
        floats.par_chunks_mut(4096).enumerate().for_each(|(chunk_idx, chunk)| {
            let base = chunk_idx * 4096 * 2;
            for (i, out) in chunk.iter_mut().enumerate() {
                let offset = base + i * 2;
                let bits = u16::from_le_bytes([data[offset], data[offset + 1]]) as u32;
                *out = f32::from_bits(bits << 16);
            }
        });
    } else {
        for (i, out) in floats.iter_mut().enumerate() {
            let offset = i * 2;
            let bits = u16::from_le_bytes([data[offset], data[offset + 1]]) as u32;
            *out = f32::from_bits(bits << 16);
        }
    }
    Ok(floats)
}

/// 將 F32 位元組轉為 f32 向量（若 N >= 16384 則使用 Rayon 並行轉換）
pub fn f32_bytes_to_f32_vec(data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(4) {
        return Err(Error::Weight(format!(
            "Invalid F32 tensor byte length {}",
            data.len()
        )));
    }
    let n = data.len() / 4;
    let mut floats = vec![0.0f32; n];
    if n >= 16384 {
        floats.par_chunks_mut(4096).enumerate().for_each(|(chunk_idx, chunk)| {
            let base = chunk_idx * 4096 * 4;
            for (i, out) in chunk.iter_mut().enumerate() {
                let offset = base + i * 4;
                *out = f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
            }
        });
    } else {
        for (i, out) in floats.iter_mut().enumerate() {
            let offset = i * 4;
            *out = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
        }
    }
    Ok(floats)
}

/// 單個 BF16 bits → F32 轉換
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    let extended = (bits as u32) << 16;
    f32::from_bits(extended)
}

#[cfg(test)]
mod tests {
    use super::gguf_key_to_safetensors;

    #[test]
    fn gguf_talker_layer_keys() {
        assert_eq!(
            gguf_key_to_safetensors("talker.blk.0.attn_q.weight"),
            "talker.model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("talker.blk.27.attn_k.weight"),
            "talker.model.layers.27.self_attn.k_proj.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("talker.blk.5.attn_v.weight"),
            "talker.model.layers.5.self_attn.v_proj.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("talker.blk.10.attn_output.weight"),
            "talker.model.layers.10.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn gguf_talker_layer_norm_keys() {
        assert_eq!(
            gguf_key_to_safetensors("talker.blk.0.attn_norm.weight"),
            "talker.model.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("talker.blk.1.ffn_norm.weight"),
            "talker.model.layers.1.post_attention_layernorm.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("talker.blk.2.attn_q_norm.weight"),
            "talker.model.layers.2.self_attn.q_norm.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("talker.blk.3.attn_k_norm.weight"),
            "talker.model.layers.3.self_attn.k_norm.weight"
        );
    }

    #[test]
    fn gguf_talker_mlp_keys() {
        assert_eq!(
            gguf_key_to_safetensors("talker.blk.3.ffn_gate.weight"),
            "talker.model.layers.3.mlp.gate_proj.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("talker.blk.3.ffn_up.weight"),
            "talker.model.layers.3.mlp.up_proj.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("talker.blk.3.ffn_down.weight"),
            "talker.model.layers.3.mlp.down_proj.weight"
        );
    }

    #[test]
    fn gguf_talker_top_level_keys() {
        assert_eq!(
            gguf_key_to_safetensors("talker.text_embd.weight"),
            "talker.model.text_embedding.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("talker.codec_embd.weight"),
            "talker.model.codec_embedding.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("talker.output_norm.weight"),
            "talker.model.norm.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("talker.codec_head.weight"),
            "talker.codec_head.weight"
        );
    }

    #[test]
    fn gguf_text_proj_keys() {
        assert_eq!(
            gguf_key_to_safetensors("talker.text_proj.fc1.weight"),
            "talker.text_projection.linear_fc1.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("talker.text_proj.fc2.bias"),
            "talker.text_projection.linear_fc2.bias"
        );
    }

    #[test]
    fn gguf_code_pred_layer_keys() {
        assert_eq!(
            gguf_key_to_safetensors("code_pred.blk.0.attn_q.weight"),
            "talker.code_predictor.model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("code_pred.blk.4.ffn_down.weight"),
            "talker.code_predictor.model.layers.4.mlp.down_proj.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("code_pred.blk.2.attn_output.weight"),
            "talker.code_predictor.model.layers.2.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn gguf_code_pred_top_keys() {
        assert_eq!(
            gguf_key_to_safetensors("code_pred.output_norm.weight"),
            "talker.code_predictor.model.norm.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("code_pred.mtp_proj.weight"),
            "talker.code_predictor.small_to_mtp_projection.weight"
        );
    }

    #[test]
    fn gguf_code_pred_codec_embd() {
        // code_pred.codec_embd.{i}.weight → talker.code_predictor.model.codec_embedding.{i}.weight
        assert_eq!(
            gguf_key_to_safetensors("code_pred.codec_embd.0.weight"),
            "talker.code_predictor.model.codec_embedding.0.weight"
        );
        assert_eq!(
            gguf_key_to_safetensors("code_pred.codec_embd.14.weight"),
            "talker.code_predictor.model.codec_embedding.14.weight"
        );
    }

    #[test]
    fn gguf_lm_head_fallback() {
        assert_eq!(
            gguf_key_to_safetensors("code_pred.lm_head.0.weight"),
            "talker.code_predictor.lm_head.0.weight"
        );
    }

    #[test]
    fn test_bf16_bytes_to_f32_vec_small() {
        // 1.0f32 in BF16 is 0x3F80 -> LE [0x80, 0x3F]
        // 2.0f32 in BF16 is 0x4000 -> LE [0x00, 0x40]
        // -0.5f32 in BF16 is 0xBF00 -> LE [0x00, 0xBF]
        // 0.0f32 in BF16 is 0x0000 -> LE [0x00, 0x00]
        let bytes = vec![0x80, 0x3F, 0x00, 0x40, 0x00, 0xBF, 0x00, 0x00];
        let floats = super::bf16_bytes_to_f32_vec(&bytes).unwrap();
        assert_eq!(floats, vec![1.0, 2.0, -0.5, 0.0]);
    }

    #[test]
    fn test_bf16_bytes_to_f32_vec_large_rayon() {
        let count = 20000;
        let mut bytes = Vec::with_capacity(count * 2);
        let mut expected = Vec::with_capacity(count);
        for i in 0..count {
            let bits = (i as u16) ^ 0x3F80;
            bytes.push((bits & 0xFF) as u8);
            bytes.push(((bits >> 8) & 0xFF) as u8);
            let expected_val = f32::from_bits((bits as u32) << 16);
            expected.push(expected_val);
        }

        let floats = super::bf16_bytes_to_f32_vec(&bytes).unwrap();
        assert_eq!(floats.len(), count);
        for (a, b) in floats.iter().zip(expected.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn test_bf16_bytes_invalid_length() {
        let bytes = vec![0x80, 0x3F, 0x00];
        assert!(super::bf16_bytes_to_f32_vec(&bytes).is_err());
    }

    #[test]
    fn test_f32_bytes_to_f32_vec_small() {
        let values = vec![1.0f32, -2.5, 123.456, 0.0];
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let floats = super::f32_bytes_to_f32_vec(&bytes).unwrap();
        assert_eq!(floats, values);
    }

    #[test]
    fn test_f32_bytes_to_f32_vec_large_rayon() {
        let count = 20000;
        let mut bytes = Vec::with_capacity(count * 4);
        let mut expected = Vec::with_capacity(count);
        for i in 0..count {
            let val = (i as f32) * 1.5 - 500.0;
            bytes.extend_from_slice(&val.to_le_bytes());
            expected.push(val);
        }

        let floats = super::f32_bytes_to_f32_vec(&bytes).unwrap();
        assert_eq!(floats.len(), count);
        assert_eq!(floats, expected);
    }

    #[test]
    fn test_f32_bytes_invalid_length() {
        let bytes = vec![0x00, 0x00, 0x80];
        assert!(super::f32_bytes_to_f32_vec(&bytes).is_err());
    }

    #[test]
    fn test_talker_weight_loader_from_safetensors_mmap() {
        use safetensors::tensor::Dtype;
        use safetensors::tensor::View;

        #[derive(Debug)]
        struct TestTensor {
            dtype: Dtype,
            shape: Vec<usize>,
            data: Vec<u8>,
        }
        impl View for TestTensor {
            fn dtype(&self) -> Dtype {
                self.dtype
            }
            fn shape(&self) -> &[usize] {
                &self.shape
            }
            fn data(&self) -> std::borrow::Cow<'_, [u8]> {
                std::borrow::Cow::Borrowed(&self.data)
            }
            fn data_len(&self) -> usize {
                self.data.len()
            }
        }

        let tmp = std::env::temp_dir().join(format!(
            "qwen3tts-talker-loader-test-{}.safetensors",
            std::process::id()
        ));

        let t1 = TestTensor {
            dtype: Dtype::BF16,
            shape: vec![2, 2],
            data: vec![0x80, 0x3F, 0x00, 0x40, 0x00, 0xBF, 0x00, 0x00],
        };
        let t2 = TestTensor {
            dtype: Dtype::F32,
            shape: vec![2],
            data: vec![
                0x00, 0x00, 0x80, 0x3F, // 1.0f32
                0x00, 0x00, 0x00, 0x40, // 2.0f32
            ],
        };

        safetensors::serialize_to_file(
            vec![
                ("talker.model.text_embedding.weight".to_string(), t1),
                ("talker.model.norm.weight".to_string(), t2),
            ],
            None,
            &tmp,
        )
        .unwrap();

        let device = candle_core::Device::Cpu;
        let loader = super::TalkerWeightLoader::from_safetensors(&tmp, &device).unwrap();

        let t1_loaded = loader.get("talker.model.text_embedding.weight").unwrap();
        assert_eq!(t1_loaded.dims(), &[2, 2]);
        assert_eq!(
            t1_loaded.to_vec2::<f32>().unwrap(),
            vec![vec![1.0, 2.0], vec![-0.5, 0.0]]
        );

        let t2_loaded = loader.get("talker.model.norm.weight").unwrap();
        assert_eq!(t2_loaded.dims(), &[2]);
        assert_eq!(t2_loaded.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);

        let _ = std::fs::remove_file(tmp);
    }
}
