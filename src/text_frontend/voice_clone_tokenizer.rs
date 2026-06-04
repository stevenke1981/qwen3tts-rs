//! Native Rust/Candle speech-tokenizer encoder and RVQ path for voice clone.

use std::path::Path;

use candle_core::{Device, Tensor};

use crate::talker::primitives::linear;
use crate::text_frontend::voice_clone::NativeReferenceCodes;
use crate::weights::WeightLoader;
use crate::{Error, Result};

const CODEBOOK_SIZE: usize = 2048;
const CODEBOOK_DIM: usize = 256;
const TRANSFORMER_LAYERS: usize = 8;
const TRANSFORMER_HEADS: usize = 8;
const TRANSFORMER_HEAD_DIM: usize = 64;
const TRANSFORMER_SLIDING_WINDOW: usize = 250;
const TRANSFORMER_ROPE_THETA: f64 = 10_000.0;

/// 12Hz speech tokenizer encoder loaded from `encoder.safetensors`.
pub struct NativeSpeechTokenizerEncoder {
    weights: WeightLoader,
    semantic_codebook: Vec<Vec<f32>>,
    acoustic_codebooks: Vec<Vec<f32>>,
}

impl NativeSpeechTokenizerEncoder {
    pub fn from_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        Self::from_safetensors(dir.as_ref().join("encoder.safetensors"), device)
    }

    pub fn from_safetensors(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let weights = WeightLoader::from_file(path, device)?;
        let semantic_codebook = vec![load_encoder_codebook(
            &weights,
            "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook",
        )?];
        let mut acoustic_codebooks = Vec::with_capacity(15);
        for layer in 0..15 {
            acoustic_codebooks.push(load_encoder_codebook(
                &weights,
                &format!(
                    "encoder.quantizer.acoustic_residual_vector_quantizer.layers.{layer}.codebook"
                ),
            )?);
        }
        Ok(Self {
            weights,
            semantic_codebook,
            acoustic_codebooks,
        })
    }

    /// Encode `[batch, samples]` waveform into native reference codec frames.
    pub fn encode_waveform(&self, waveform: &Tensor) -> Result<NativeReferenceCodes> {
        let (_batch, _samples) = waveform.dims2()?;
        let x = waveform.unsqueeze(1)?;
        let latents = self.forward_latents(&x)?;
        self.encode_latents(&latents)
    }

    /// Forward waveform `[batch, 1, samples]` through the convolutional encoder stack.
    pub fn forward_latents(&self, input: &Tensor) -> Result<Tensor> {
        let mut h = self.conv1d(input, "encoder.encoder.layers.0.conv", 3, 1, 1)?;
        h = self.residual_unit(&h, "encoder.encoder.layers.1")?;
        h = self.conv1d(&elu(&h)?, "encoder.encoder.layers.3.conv", 0, 4, 1)?;
        h = self.residual_unit(&h, "encoder.encoder.layers.4")?;
        h = self.conv1d(&elu(&h)?, "encoder.encoder.layers.6.conv", 0, 5, 1)?;
        h = self.residual_unit(&h, "encoder.encoder.layers.7")?;
        h = self.conv1d(&elu(&h)?, "encoder.encoder.layers.9.conv", 0, 6, 1)?;
        h = self.residual_unit(&h, "encoder.encoder.layers.10")?;
        h = self.conv1d(&elu(&h)?, "encoder.encoder.layers.12.conv", 0, 8, 1)?;
        h = self.conv1d(&elu(&h)?, "encoder.encoder.layers.14.conv", 1, 1, 1)?;
        let h = self.forward_transformer(&h.transpose(1, 2)?.contiguous()?)?;
        self.conv1d_with_pad_mode(
            &h.transpose(1, 2)?.contiguous()?,
            "encoder.downsample.conv",
            0,
            2,
            1,
            PadMode::Replicate,
        )
    }

    /// Quantize latent frames `[batch, 512, frames]` to `[frames][16]` codes.
    pub fn encode_latents(&self, latents: &Tensor) -> Result<NativeReferenceCodes> {
        let semantic = self.project_latents(
            latents,
            "encoder.quantizer.semantic_residual_vector_quantizer.input_proj",
        )?;
        let acoustic = self.project_latents(
            latents,
            "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj",
        )?;
        let semantic_codes = encode_rvq(&semantic, &self.semantic_codebook)?;
        let acoustic_codes = encode_rvq(&acoustic, &self.acoustic_codebooks)?;
        let frames = semantic_codes.len().min(acoustic_codes.len());
        if frames == 0 {
            return Err(Error::Config(
                "speech tokenizer encoder produced zero reference frames".into(),
            ));
        }
        let mut out = Vec::with_capacity(frames);
        for frame_idx in 0..frames {
            let mut frame = [0u16; 16];
            frame[0] = semantic_codes[frame_idx][0];
            for i in 0..15 {
                frame[i + 1] = acoustic_codes[frame_idx][i];
            }
            out.push(frame);
        }
        NativeReferenceCodes::from_frames(out)
    }

    fn residual_unit(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let h = elu(input)?;
        let h = self.conv1d(&h, &format!("{prefix}.block.1.conv"), 1, 1, 1)?;
        let h = elu(&h)?;
        let h = self.conv1d(&h, &format!("{prefix}.block.3.conv"), 0, 1, 1)?;
        crop_or_pad_add(input, &h)
    }

    fn project_latents(&self, latents: &Tensor, prefix: &str) -> Result<Tensor> {
        self.conv1d(latents, prefix, 0, 1, 1)
    }

    fn conv1d(
        &self,
        input: &Tensor,
        name: &str,
        padding: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<Tensor> {
        let weight = self.weights.conv1d_weight(name)?;
        let bias = self.weights.conv1d_bias(name).transpose()?;
        self.conv1d_with_weight(
            input,
            &weight,
            bias,
            padding,
            stride,
            dilation,
            PadMode::Constant,
        )
    }

    fn conv1d_with_pad_mode(
        &self,
        input: &Tensor,
        name: &str,
        padding: usize,
        stride: usize,
        dilation: usize,
        pad_mode: PadMode,
    ) -> Result<Tensor> {
        let weight = self.weights.conv1d_weight(name)?;
        let bias = self.weights.conv1d_bias(name).transpose()?;
        self.conv1d_with_weight(input, &weight, bias, padding, stride, dilation, pad_mode)
    }

    fn conv1d_with_weight(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<Tensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        pad_mode: PadMode,
    ) -> Result<Tensor> {
        let input = causal_pad1d(input, weight.dim(2)?, stride, dilation, padding, pad_mode)?;
        let y = input.conv1d(&weight, 0, stride, dilation, 1)?;
        if let Some(bias) = bias {
            y.broadcast_add(&bias.unsqueeze(0)?.unsqueeze(2)?)
                .map_err(Into::into)
        } else {
            Ok(y)
        }
    }

    fn forward_transformer(&self, input: &Tensor) -> Result<Tensor> {
        let mut h = input.clone();
        for layer_idx in 0..TRANSFORMER_LAYERS {
            h = self.transformer_layer(&h, layer_idx)?;
        }
        Ok(h)
    }

    fn transformer_layer(&self, input: &Tensor, layer_idx: usize) -> Result<Tensor> {
        let prefix = format!("encoder.encoder_transformer.layers.{layer_idx}");
        let normed = self.layer_norm(input, &format!("{prefix}.input_layernorm"))?;
        let attn = self.self_attention(&normed, &format!("{prefix}.self_attn"))?;
        let attn_scale = self
            .weights
            .get(&format!("{prefix}.self_attn_layer_scale.scale"))?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let h = input.broadcast_add(&attn.broadcast_mul(&attn_scale)?)?;

        let normed = self.layer_norm(&h, &format!("{prefix}.post_attention_layernorm"))?;
        let mlp = self.mlp(&normed, &format!("{prefix}.mlp"))?;
        let mlp_scale = self
            .weights
            .get(&format!("{prefix}.mlp_layer_scale.scale"))?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        h.broadcast_add(&mlp.broadcast_mul(&mlp_scale)?)
            .map_err(Into::into)
    }

    fn layer_norm(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let dims = input.dims();
        let weight = self.weights.get(&format!("{prefix}.weight"))?;
        let bias = self.weights.get(&format!("{prefix}.bias"))?;
        let mean = input.mean_keepdim(2)?;
        let centered = (input - mean.expand(dims)?)?;
        let var = centered.sqr()?.mean_keepdim(2)?;
        let normalized = centered.broadcast_div(&(var + 1e-5)?.sqrt()?)?;
        normalized
            .broadcast_mul(&weight.unsqueeze(0)?.unsqueeze(0)?)?
            .broadcast_add(&bias.unsqueeze(0)?.unsqueeze(0)?)
            .map_err(Into::into)
    }

    fn self_attention(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = input.dims3()?;
        let q = linear(input, self.weights.get(&format!("{prefix}.q_proj.weight"))?)?;
        let k = linear(input, self.weights.get(&format!("{prefix}.k_proj.weight"))?)?;
        let v = linear(input, self.weights.get(&format!("{prefix}.v_proj.weight"))?)?;

        let q = q
            .reshape((batch, seq_len, TRANSFORMER_HEADS, TRANSFORMER_HEAD_DIM))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, TRANSFORMER_HEADS, TRANSFORMER_HEAD_DIM))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, TRANSFORMER_HEADS, TRANSFORMER_HEAD_DIM))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        let (cos, sin) = precompute_rope(seq_len, TRANSFORMER_HEAD_DIM, input.device())?;
        let q = apply_rope_half(&q, &cos, &sin)?;
        let k = apply_rope_half(&k, &cos, &sin)?;

        let scale = (TRANSFORMER_HEAD_DIM as f64).sqrt().recip();
        let attn = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let attn = apply_sliding_window_mask(&attn, seq_len, TRANSFORMER_SLIDING_WINDOW)?;
        let attn = candle_nn::ops::softmax(&attn, 3)?;
        let attn = attn.matmul(&v)?;
        let attn = attn.permute((0, 2, 1, 3))?.reshape((
            batch,
            seq_len,
            TRANSFORMER_HEADS * TRANSFORMER_HEAD_DIM,
        ))?;
        linear(&attn, self.weights.get(&format!("{prefix}.o_proj.weight"))?).map_err(Into::into)
    }

    fn mlp(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let h = linear(input, self.weights.get(&format!("{prefix}.fc1.weight"))?)?.gelu()?;
        linear(&h, self.weights.get(&format!("{prefix}.fc2.weight"))?).map_err(Into::into)
    }
}

#[derive(Clone, Copy)]
enum PadMode {
    Constant,
    Replicate,
}

fn elu(input: &Tensor) -> Result<Tensor> {
    let values = input.flatten_all()?.to_vec1::<f32>()?;
    let out: Vec<f32> = values
        .into_iter()
        .map(|x| if x > 0.0 { x } else { x.exp() - 1.0 })
        .collect();
    Tensor::from_slice(&out, input.dims(), input.device()).map_err(Into::into)
}

fn precompute_rope(seq_len: usize, head_dim: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| (1.0 / TRANSFORMER_ROPE_THETA.powf(i as f64 / half as f64)) as f32)
        .collect();
    let inv = Tensor::from_slice(&inv, (half,), device)?.unsqueeze(0)?;
    let pos: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let pos = Tensor::from_slice(&pos, (seq_len,), device)?.unsqueeze(1)?;
    let freqs = pos.matmul(&inv)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

fn apply_rope_half(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = input.dim(3)? / 2;
    let x1 = input.narrow(3, 0, half)?;
    let x2 = input.narrow(3, half, half)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let first = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let second = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
    Tensor::cat(&[first, second], 3).map_err(Into::into)
}

fn apply_sliding_window_mask(attn: &Tensor, seq_len: usize, window: usize) -> Result<Tensor> {
    let mut mask = Vec::with_capacity(seq_len * seq_len);
    for query_idx in 0..seq_len {
        let window_start = (query_idx + 1).saturating_sub(window);
        for key_idx in 0..seq_len {
            let visible = key_idx <= query_idx && key_idx >= window_start;
            mask.push(if visible { 0.0f32 } else { -1.0e9f32 });
        }
    }
    let mask = Tensor::from_slice(&mask, (seq_len, seq_len), attn.device())?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    attn.broadcast_add(&mask).map_err(Into::into)
}

fn causal_pad1d(
    input: &Tensor,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    fallback_padding: usize,
    pad_mode: PadMode,
) -> Result<Tensor> {
    let effective_kernel = (kernel_size - 1) * dilation + 1;
    let left = effective_kernel
        .saturating_sub(stride)
        .max(fallback_padding);
    let len = input.dim(2)?;
    let n_frames = (len as f64 - effective_kernel as f64 + left as f64) / stride as f64 + 1.0;
    let ideal_len =
        ((n_frames.ceil() as usize).saturating_sub(1)) * stride + effective_kernel - left;
    let right = ideal_len.saturating_sub(len);
    if left == 0 && right == 0 {
        return Ok(input.clone());
    }
    let (batch, channels, len) = input.dims3()?;
    let values = input.to_vec3::<f32>()?;
    let mut out = vec![0.0f32; batch * channels * (len + left + right)];
    let out_len = len + left + right;
    for b in 0..batch {
        for c in 0..channels {
            let base = (b * channels + c) * out_len;
            match pad_mode {
                PadMode::Constant => {}
                PadMode::Replicate => {
                    for idx in 0..left {
                        out[base + idx] = values[b][c][0];
                    }
                    for idx in 0..right {
                        out[base + left + len + idx] = values[b][c][len - 1];
                    }
                }
            }
            let dst = base + left;
            out[dst..dst + len].copy_from_slice(&values[b][c]);
        }
    }
    Tensor::from_slice(&out, (batch, channels, out_len), input.device()).map_err(Into::into)
}

fn crop_or_pad_add(input: &Tensor, residual: &Tensor) -> Result<Tensor> {
    let input_len = input.dim(2)?;
    let residual_len = residual.dim(2)?;
    let residual = if residual_len > input_len {
        residual.narrow(2, 0, input_len)?
    } else {
        residual.clone()
    };
    if residual.dim(2)? != input_len {
        return Err(Error::Config(format!(
            "residual length mismatch: input={input_len}, residual={}",
            residual.dim(2)?
        )));
    }
    (input + residual).map_err(Into::into)
}

fn load_encoder_codebook(weights: &WeightLoader, prefix: &str) -> Result<Vec<f32>> {
    let usage = weights
        .get(&format!("{prefix}.cluster_usage"))?
        .to_vec1::<f32>()?;
    let sums = weights
        .get(&format!("{prefix}.embed_sum"))?
        .to_vec2::<f32>()?;
    if usage.len() != CODEBOOK_SIZE || sums.len() != CODEBOOK_SIZE {
        return Err(Error::Weight(format!(
            "{prefix} codebook shape mismatch: usage={} sums={}",
            usage.len(),
            sums.len()
        )));
    }
    let mut codebook = vec![0.0f32; CODEBOOK_SIZE * CODEBOOK_DIM];
    for token in 0..CODEBOOK_SIZE {
        let denom = usage[token].max(1e-5);
        for dim in 0..CODEBOOK_DIM {
            codebook[token * CODEBOOK_DIM + dim] = sums[token][dim] / denom;
        }
    }
    Ok(codebook)
}

fn encode_rvq(projected: &Tensor, codebooks: &[Vec<f32>]) -> Result<Vec<Vec<u16>>> {
    let (batch, dim, frames) = projected.dims3()?;
    if batch != 1 || dim != CODEBOOK_DIM {
        return Err(Error::Config(format!(
            "RVQ expects [1,{CODEBOOK_DIM},frames], got {batch},{dim},{frames}"
        )));
    }
    let values = projected.to_vec3::<f32>()?;
    let mut frame_vectors = vec![vec![0.0f32; CODEBOOK_DIM]; frames];
    for (dim_idx, dim_values) in values[0].iter().enumerate() {
        for frame_idx in 0..frames {
            frame_vectors[frame_idx][dim_idx] = dim_values[frame_idx];
        }
    }

    let mut all_codes = vec![vec![0u16; codebooks.len()]; frames];
    for frame_idx in 0..frames {
        let mut residual = frame_vectors[frame_idx].clone();
        for (layer_idx, codebook) in codebooks.iter().enumerate() {
            let token = nearest_code(&residual, codebook);
            all_codes[frame_idx][layer_idx] = token as u16;
            let base = token * CODEBOOK_DIM;
            for dim_idx in 0..CODEBOOK_DIM {
                residual[dim_idx] -= codebook[base + dim_idx];
            }
        }
    }
    Ok(all_codes)
}

fn nearest_code(vector: &[f32], codebook: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for token in 0..CODEBOOK_SIZE {
        let base = token * CODEBOOK_DIM;
        let mut dist = 0.0f32;
        for dim in 0..CODEBOOK_DIM {
            let d = vector[dim] - codebook[base + dim];
            dist += d * d;
        }
        if dist < best_dist {
            best_dist = dist;
            best = token;
        }
    }
    best
}
