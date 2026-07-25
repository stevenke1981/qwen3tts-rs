//! Native Candle speaker encoder for Qwen3-TTS Base voice clone.

use std::path::Path;

use candle_core::{DType, Device, Tensor};

use crate::weights::WeightLoader;
use crate::{Error, Result};

/// Qwen3-TTS speaker encoder loaded from `speaker_encoder.safetensors`.
pub struct NativeSpeakerEncoder {
    weights: WeightLoader,
}

impl NativeSpeakerEncoder {
    pub fn from_safetensors(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        Ok(Self {
            weights: WeightLoader::from_file(path, device)?,
        })
    }

    /// Forward log-mel features `[batch, frames, 128]` into a speaker embedding.
    pub fn forward_mels(&self, mels: &Tensor) -> Result<Tensor> {
        let (_batch, _frames, mel_dim) = mels.dims3()?;
        if mel_dim != 128 {
            return Err(Error::Config(format!(
                "speaker encoder expects mel dim 128, got {mel_dim}"
            )));
        }

        let x = mels.transpose(1, 2)?;
        let h0 = self.conv_act(&x, "speaker_encoder.blocks.0.conv", 2, 1)?;
        let h1 = self.se_res2_block(&h0, "speaker_encoder.blocks.1", 2)?;
        let h2 = self.se_res2_block(&h1, "speaker_encoder.blocks.2", 3)?;
        let h3 = self.se_res2_block(&h2, "speaker_encoder.blocks.3", 4)?;
        let h = Tensor::cat(&[h1, h2, h3], 1)?;
        let h = self.conv_act(&h, "speaker_encoder.mfa.conv", 0, 1)?;
        let pooled = self.attentive_statistics_pool(&h)?;
        let pooled = pooled.unsqueeze(2)?;
        self.conv1d(&pooled, "speaker_encoder.fc", 0, 1)?
            .squeeze(2)
            .map_err(Into::into)
    }

    fn se_res2_block(&self, input: &Tensor, prefix: &str, res2_dilation: usize) -> Result<Tensor> {
        let residual = input.clone();
        let h = self.conv_act(input, &format!("{prefix}.tdnn1.conv"), 0, 1)?;
        let splits = h.chunk(8, 1)?;
        let mut out_parts = Vec::with_capacity(splits.len());
        out_parts.push(splits[0].clone());
        let mut prev: Option<Tensor> = None;
        for i in 0..7 {
            let current = if let Some(prev) = &prev {
                (&splits[i + 1] + prev)?
            } else {
                splits[i + 1].clone()
            };
            let y = self.conv_act(
                &current,
                &format!("{prefix}.res2net_block.blocks.{i}.conv"),
                res2_dilation,
                res2_dilation,
            )?;
            prev = Some(y.clone());
            out_parts.push(y);
        }
        let h = Tensor::cat(&out_parts, 1)?;
        let h = self.conv_act(&h, &format!("{prefix}.tdnn2.conv"), 0, 1)?;
        let scale = self.se_scale(&h, prefix)?;
        ((h.broadcast_mul(&scale)?) + residual).map_err(Into::into)
    }

    fn se_scale(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let pooled = input.mean_keepdim(2)?;
        let h = relu(&self.conv1d(&pooled, &format!("{prefix}.se_block.conv1"), 0, 1)?)?;
        let h = self.conv1d(&h, &format!("{prefix}.se_block.conv2"), 0, 1)?;
        sigmoid(&h)
    }

    fn attentive_statistics_pool(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, channels, frames) = input.dims3()?;
        let mean = input.mean_keepdim(2)?;
        let mean_expanded = mean.expand((batch, channels, frames))?;
        let centered_for_std = (input - &mean_expanded)?;
        let var = centered_for_std.sqr()?.mean_keepdim(2)?;
        let std = var
            .broadcast_maximum(&Tensor::new(&[1e-12f32], input.device())?)?
            .sqrt()?;
        let std_expanded = std.expand((batch, channels, frames))?;
        let attn_in = Tensor::cat(&[input.clone(), mean_expanded, std_expanded], 1)?;
        let attn = relu(&self.conv1d(&attn_in, "speaker_encoder.asp.tdnn.conv", 0, 1)?)?;
        let attn = self.conv1d(&attn.tanh()?, "speaker_encoder.asp.conv", 0, 1)?;
        let weights = candle_nn::ops::softmax(&attn, 2)?;
        let weighted_mean = input.broadcast_mul(&weights)?.sum_keepdim(2)?;
        let centered = (input - weighted_mean.expand((batch, channels, frames))?)?;
        let weighted_var = centered.sqr()?.broadcast_mul(&weights)?.sum_keepdim(2)?;
        let weighted_std = weighted_var
            .broadcast_maximum(&Tensor::new(&[1e-12f32], input.device())?)?
            .sqrt()?;
        Tensor::cat(&[weighted_mean.squeeze(2)?, weighted_std.squeeze(2)?], 1).map_err(Into::into)
    }

    fn conv_act(
        &self,
        input: &Tensor,
        name: &str,
        padding: usize,
        dilation: usize,
    ) -> Result<Tensor> {
        relu(&self.conv1d(input, name, padding, dilation)?)
    }

    fn conv1d(
        &self,
        input: &Tensor,
        name: &str,
        _padding: usize,
        dilation: usize,
    ) -> Result<Tensor> {
        let weight = self.weights.conv1d_weight(name)?;
        let bias = self.weights.conv1d_bias(name).transpose()?;
        let input = reflect_pad1d_same(input, weight.dim(2)?, dilation)?;
        let y = input.conv1d(&weight, 0, 1, dilation, 1)?;
        if let Some(bias) = bias {
            y.broadcast_add(&bias.unsqueeze(0)?.unsqueeze(2)?)
                .map_err(Into::into)
        } else {
            Ok(y)
        }
    }
}

/// Qwen3-TTS upstream log-mel extraction for voice-clone speaker references.
///
/// Matches `qwen_tts.core.models.modeling_qwen3_tts.mel_spectrogram`:
/// reflect pad by `(n_fft - hop) / 2`, Hann-window STFT, Slaney-normalized
/// mel filterbank, magnitude floor at `1e-9`, then log compression.
pub fn upstream_mel_spectrogram(waveform: &Tensor, device: &Device) -> Result<Tensor> {
    let (batch, samples) = waveform.dims2()?;
    let values = waveform.to_vec2::<f32>()?;
    let n_fft = 1024usize;
    let hop = 256usize;
    let n_mels = 128usize;
    let pad = (n_fft - hop) / 2;
    if samples <= pad {
        return Err(Error::Config(
            "reference WAV is too short for upstream mel extraction".into(),
        ));
    }

    let padded_len = samples + 2 * pad;
    let frames = padded_len.saturating_sub(n_fft) / hop + 1;
    if frames == 0 {
        return Err(Error::Config(
            "reference WAV is too short for speaker encoder mel extraction".into(),
        ));
    }

    let hann: Vec<f64> = (0..n_fft)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos())
        .collect();
    let (cos_table, sin_table) = dft_tables(n_fft);
    let mel = slaney_mel_filterbank(24_000, n_fft, n_mels, 0.0, 12_000.0);
    let fft_bins = n_fft / 2 + 1;

    let mut out = vec![0.0f32; batch * frames * n_mels];
    let mut padded = vec![0.0f32; padded_len];
    let mut spec = vec![0.0f64; fft_bins * frames];
    for batch_idx in 0..batch {
        reflect_pad_waveform(&values[batch_idx], pad, &mut padded);
        for frame_idx in 0..frames {
            let start = frame_idx * hop;
            for freq_idx in 0..fft_bins {
                let table_base = freq_idx * n_fft;
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for n in 0..n_fft {
                    let windowed = padded[start + n] as f64 * hann[n];
                    re += windowed * cos_table[table_base + n];
                    im -= windowed * sin_table[table_base + n];
                }
                spec[freq_idx * frames + frame_idx] = (re * re + im * im + 1e-9).sqrt();
            }
        }

        for mel_idx in 0..n_mels {
            for frame_idx in 0..frames {
                let mut value = 0.0f64;
                for freq_idx in 0..fft_bins {
                    value +=
                        mel[mel_idx * fft_bins + freq_idx] * spec[freq_idx * frames + frame_idx];
                }
                let value = value.max(1e-5).ln() as f32;
                out[(batch_idx * frames + frame_idx) * n_mels + mel_idx] = value;
            }
        }
    }

    Tensor::from_slice(&out, (batch, frames, n_mels), device).map_err(Into::into)
}

fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let one = Tensor::ones(x.dims(), DType::F32, x.device())?;
    let denom = x.neg()?.exp()?.broadcast_add(&one)?;
    one.broadcast_div(&denom).map_err(Into::into)
}

fn relu(x: &Tensor) -> Result<Tensor> {
    x.broadcast_maximum(&Tensor::new(&[0.0f32], x.device())?)
        .map_err(Into::into)
}

fn reflect_pad1d_same(input: &Tensor, kernel_size: usize, dilation: usize) -> Result<Tensor> {
    let total = (kernel_size.saturating_sub(1)) * dilation;
    let left = total / 2;
    let right = total - left;
    if left == 0 && right == 0 {
        return Ok(input.clone());
    }
    let values = input.to_vec3::<f32>()?;
    let batch = values.len();
    let channels = values[0].len();
    let len = values[0][0].len();
    let mut out = Vec::with_capacity(batch * channels * (len + left + right));
    for batch_values in &values {
        for channel_values in batch_values {
            for pad_idx in 0..left {
                let src = reflect_index(left - pad_idx, len);
                out.push(channel_values[src]);
            }
            out.extend_from_slice(channel_values);
            for pad_idx in 0..right {
                let src = reflect_index(len - 2 - pad_idx, len);
                out.push(channel_values[src]);
            }
        }
    }
    Tensor::from_slice(&out, (batch, channels, len + left + right), input.device())
        .map_err(Into::into)
}

fn reflect_index(index: usize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * len - 2;
    let idx = index % period;
    if idx < len { idx } else { period - idx }
}

fn reflect_pad_waveform(input: &[f32], pad: usize, out: &mut [f32]) {
    let len = input.len();
    debug_assert_eq!(out.len(), len + 2 * pad);
    for pad_idx in 0..pad {
        out[pad_idx] = input[reflect_index(pad - pad_idx, len)];
    }
    out[pad..pad + len].copy_from_slice(input);
    for pad_idx in 0..pad {
        out[pad + len + pad_idx] = input[reflect_index(len - 2 - pad_idx, len)];
    }
}

fn dft_tables(n_fft: usize) -> (Vec<f64>, Vec<f64>) {
    let bins = n_fft / 2 + 1;
    let mut cos_table = vec![0.0f64; bins * n_fft];
    let mut sin_table = vec![0.0f64; bins * n_fft];
    for freq_idx in 0..bins {
        for n in 0..n_fft {
            let phase = 2.0 * std::f64::consts::PI * freq_idx as f64 * n as f64 / n_fft as f64;
            cos_table[freq_idx * n_fft + n] = phase.cos();
            sin_table[freq_idx * n_fft + n] = phase.sin();
        }
    }
    (cos_table, sin_table)
}

fn slaney_mel_filterbank(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<f64> {
    let fft_bins = n_fft / 2 + 1;
    let min_mel = hz_to_slaney_mel(fmin);
    let max_mel = hz_to_slaney_mel(fmax);
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| {
            let alpha = i as f64 / (n_mels + 1) as f64;
            slaney_mel_to_hz((min_mel as f64 + alpha * (max_mel - min_mel) as f64) as f32) as f64
        })
        .collect();
    let fft_freqs: Vec<f64> = (0..fft_bins)
        .map(|i| sample_rate as f64 * i as f64 / n_fft as f64)
        .collect();

    let mut weights = vec![0.0f64; n_mels * fft_bins];
    for mel_idx in 0..n_mels {
        let lower_edge = mel_points[mel_idx];
        let center = mel_points[mel_idx + 1];
        let upper_edge = mel_points[mel_idx + 2];
        let lower_width = center - lower_edge;
        let upper_width = upper_edge - center;
        let enorm = 2.0 / (upper_edge - lower_edge);
        for (freq_idx, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - lower_edge) / lower_width;
            let upper = (upper_edge - freq) / upper_width;
            weights[mel_idx * fft_bins + freq_idx] = lower.min(upper).max(0.0) * enorm;
        }
    }
    weights
}

fn hz_to_slaney_mel(freq: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f32.ln() / 27.0;
    if freq >= min_log_hz {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    } else {
        freq / f_sp
    }
}

fn slaney_mel_to_hz(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f32.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        mel * f_sp
    }
}
