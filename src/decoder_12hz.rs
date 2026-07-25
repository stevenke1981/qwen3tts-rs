use candle_core::{Device, Tensor};

use crate::alignment_stage_dump::{NoopStageDumpObserver, StageDumpObserver};
use crate::codec::{
    CausalConv1d, CausalConvConfig, CodebookLookup, DecoderBlock, ParallelCodebook, PreTransformer,
    PreTransformerConfig, UpsampleBlock, snake_beta,
};
use crate::weights::WeightLoader;
use crate::{DecoderConfig, Error, Result, TtsDecoder};

pub struct Decoder12Hz {
    config: DecoderConfig,
    device: Device,

    codebook: ParallelCodebook,

    pre_conv: CausalConv1d,
    pre_transformer: PreTransformer,

    upsample_blocks: Vec<UpsampleBlock>,

    decoder_start: CausalConv1d,
    decoder_blocks: Vec<DecoderBlock>,
    final_conv: CausalConv1d,
    final_snake_a: Tensor,
    final_snake_b: Tensor,

    temperature: f64,

    // Streaming state: accumulated pre_conv outputs (latent_dim per frame)
    pre_conv_buffer: Vec<f32>,
    /// Number of audio samples produced so far (for extracting only new samples)
    output_offset: usize,
}

impl Decoder12Hz {
    pub fn from_safetensors(
        config: DecoderConfig,
        weight_path: impl AsRef<std::path::Path>,
        device: &Device,
    ) -> Result<Self> {
        let loader = WeightLoader::from_dir(weight_path, device)?;

        let codebook_w = loader.codebook_weights()?;
        let codebook = CodebookLookup::new(codebook_w)?;
        let codebook = ParallelCodebook::new(codebook);

        let (pw, pb) = loader.conv1d_pair("pre_conv")?;
        let pre_conv_cfg = CausalConvConfig {
            in_channels: config.embedding_dim,
            out_channels: config.latent_dim,
            kernel_size: 3,
            dilation: 1,
            groups: 1,
        };
        let pre_conv = CausalConv1d::new(pw, pb, pre_conv_cfg, config.ring_buffer_capacity)?;

        let pt_cfg = PreTransformerConfig {
            input_dim: config.latent_dim,
            hidden_dim: config.transformer_dim,
            num_heads: config.transformer_heads,
            num_kv_heads: config.transformer_kv_heads,
            num_layers: config.transformer_layers,
            sliding_window: config.sliding_window,
            ffn_hidden_mult: 4,
            max_seq_len: config.ring_buffer_capacity,
            rope_theta: 10000.0,
            eps: 1e-6,
        };
        let pre_transformer = PreTransformer::from_loader(&loader, &pt_cfg, device)?;

        let mut upsample_blocks = Vec::new();
        for i in 0..2 {
            upsample_blocks.push(UpsampleBlock::from_loader(
                &loader,
                &format!("upsample.{i}"),
            )?);
        }

        let (sw, sb) = loader.conv1d_pair("0.conv")?;
        let ds_cfg = CausalConvConfig::from_weight(&sw, 1, 1);
        let decoder_start = CausalConv1d::new(sw, sb, ds_cfg, config.ring_buffer_capacity)?;

        let mut decoder_blocks = Vec::new();
        for i in 1..=4 {
            decoder_blocks.push(DecoderBlock::from_loader(&loader, &format!("{i}"))?);
        }

        let (fw, fb) = loader.conv1d_pair("6.conv")?;
        let fc_cfg = CausalConvConfig::from_weight(&fw, 1, 1);
        let final_conv = CausalConv1d::new(fw, fb, fc_cfg, config.ring_buffer_capacity)?;

        let fs_a = loader.get("5.alpha")?.clone();
        let fs_b = loader.get("5.beta")?.clone();

        log::info!(
            "Decoder12Hz loaded from safetensors: {} tensors",
            loader.len(),
        );

        let cap = config.ring_buffer_capacity * config.latent_dim;

        log::info!(
            "Decoder12Hz loaded from safetensors: {} tensors",
            loader.len(),
        );
        log::info!(
            "Decoder12Hz streaming buffer capacity = {} frames (no trim – O(n²) for streaming, use batch decode_frames for bulk)",
            config.ring_buffer_capacity,
        );

        Ok(Self {
            config,
            device: device.clone(),
            codebook,
            pre_conv,
            pre_transformer,
            upsample_blocks,
            decoder_start,
            decoder_blocks,
            final_conv,
            final_snake_a: fs_a,
            final_snake_b: fs_b,
            temperature: 1.0,
            pre_conv_buffer: Vec::with_capacity(cap),
            output_offset: 0,
        })
    }

    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }

    /// Decode a complete 12Hz token sequence with the same causal batch path
    /// used by the reference tokenizer decoder.
    pub fn decode_frames(&mut self, frames: &[[u16; 16]]) -> Result<Vec<f32>> {
        let mut observer = NoopStageDumpObserver::default();
        self.decode_frames_with_observer(frames, &mut observer)
    }

    pub fn decode_frames_with_observer<O: StageDumpObserver>(
        &mut self,
        frames: &[[u16; 16]],
        observer: &mut O,
    ) -> Result<Vec<f32>> {
        let capture = observer.wants_capture();
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        if capture {
            let input_codes: Vec<f32> = frames
                .iter()
                .flat_map(|frame| frame.iter().map(|&token| token as f32))
                .collect();
            let input_shape = (frames.len(), self.config.num_codebook_layers);
            let input_codes = Tensor::from_slice(&input_codes, input_shape, &self.device)?;
            observer.on_codec_input_codes(&input_codes)?;
        }

        let num_frames = frames.len();
        let mut frame_embeddings: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
        for frame in frames {
            let embeddings = self.codebook.decode(frame)?;
            let frame_embed = embeddings.sum(0)?;
            frame_embeddings.push(frame_embed.to_vec1()?);
        }

        // Apply speed adjustment if requested (via tensor-vectorized linear interpolation)
        let speed = self.config.speed;
        let frame_embeddings = if speed != 1.0 && num_frames > 1 {
            let new_num_frames = ((num_frames as f64) / speed).round() as usize;
            let new_num_frames = new_num_frames.max(1);
            let mut interpolated = Vec::with_capacity(new_num_frames);
            if new_num_frames == 1 {
                interpolated.push(frame_embeddings[0].clone());
            } else {
                let emb_dim = frame_embeddings[0].len();
                for j in 0..new_num_frames {
                    let pos =
                        (j as f64) * ((num_frames - 1) as f64) / ((new_num_frames - 1) as f64);
                    let left = pos.floor() as usize;
                    let right = pos.ceil() as usize;
                    let weight = pos - left as f64;
                    let mixed = if right == left {
                        frame_embeddings[left].clone()
                    } else {
                        // 使用張量算術進行向量化內插：left + weight × (right - left)
                        // 避免逐元素手動迴圈，GPU 可平行，CPU 亦可 SIMD
                        let left_t = Tensor::from_slice(
                            &frame_embeddings[left],
                            (1, 1, emb_dim),
                            &self.device,
                        )?;
                        let right_t = Tensor::from_slice(
                            &frame_embeddings[right],
                            (1, 1, emb_dim),
                            &self.device,
                        )?;
                        let w = Tensor::from_slice(&[weight as f32], (1, 1, 1), &self.device)?;
                        right_t.sub(&left_t)?.mul(&w)?.add(&left_t)?.to_vec1()?
                    };
                    interpolated.push(mixed);
                }
            }
            interpolated
        } else {
            frame_embeddings
        };

        let num_frames = frame_embeddings.len();
        let mut batch_data: Vec<f32> = Vec::with_capacity(self.config.embedding_dim * num_frames);
        for ch in 0..self.config.embedding_dim {
            for frame in &frame_embeddings {
                batch_data.push(frame[ch]);
            }
        }

        let x = Tensor::from_slice(
            &batch_data,
            (1, self.config.embedding_dim, num_frames),
            &self.device,
        )?;

        let h = self.pre_conv.forward(&x)?.narrow(2, 0, num_frames)?;
        let h = self.pre_transformer.forward(&h)?;

        let mut h = h;
        for ub in &self.upsample_blocks {
            h = ub.forward(&h)?;
        }

        let mut h = self.decoder_start.forward(&h)?;
        for db in &self.decoder_blocks {
            h = db.forward(&h)?;
        }

        let h = snake_beta(&h, &self.final_snake_a, &self.final_snake_b)?;
        let h = self.final_conv.forward(&h)?;
        let pcm: Vec<f32> = h.squeeze(0)?.squeeze(0)?.to_vec1()?;
        if capture {
            observer.on_codec_output_pcm(&Tensor::from_slice(&pcm, pcm.len(), &self.device)?)?;
        }

        Ok(pcm)
    }

    pub fn decode_chunk_with_observer<O: StageDumpObserver>(
        &mut self,
        tokens: &[u16],
        observer: &mut O,
    ) -> Result<Vec<f32>> {
        let capture = observer.wants_capture();
        if tokens.len() != self.config.num_codebook_layers {
            return Err(Error::Config(format!(
                "Expected {} tokens (one per codebook layer), got {}",
                self.config.num_codebook_layers,
                tokens.len()
            )));
        }
        if capture {
            let input_codes: Vec<u32> = tokens.iter().map(|&token| token as u32).collect();
            let input_codes = Tensor::from_slice(
                &input_codes,
                (1, self.config.num_codebook_layers),
                &self.device,
            )?;
            observer.on_codec_input_codes(&input_codes)?;
        }

        let output = self.decode_chunk_inner(tokens)?;
        if capture {
            observer.on_codec_output_pcm(&Tensor::from_slice(
                &output,
                output.len(),
                &self.device,
            )?)?;
        }
        Ok(output)
    }

    fn decode_chunk_inner(&mut self, tokens: &[u16]) -> Result<Vec<f32>> {
        let embeddings = self.codebook.decode(tokens)?;
        let frame_embed = embeddings.sum(0)?;

        // Step 1: pre_conv step — maintains ring buffer state across frames
        let x = self.pre_conv.step_tensor(&frame_embed)?; // (1, latent_dim, 1)

        // Step 2: Append pre_conv output to streaming buffer.
        // NOTE: We accumulate ALL frames because the pipeline is not frame-independent.
        // ConvTranspose1d + dilated convs in downstream layers create inter-frame overlap
        // that prevents simple buffer trimming. Trimming would change the output length,
        // breaking the output_offset tail extraction. For O(n²) mitigation, use batch
        // decode_frames() for bulk decoding or GPU (CUDA) for hardware acceleration.
        // Buffer stores frame-major: [f0_ch0..f0_chC-1, f1_ch0..f1_chC-1, ...]
        let x_vec = x.squeeze(0)?.squeeze(1)?.to_vec1()?; // (latent_dim,)
        self.pre_conv_buffer.extend(&x_vec);
        let total_frames = self.pre_conv_buffer.len() / self.config.latent_dim;

        // Step 3: Build accumulated tensor in channel-major order.
        // Candle's C-order for shape (1, C, T) expects:
        //   [ch0_t0, ch0_t1, ..., ch0_tT-1, ch1_t0, ..., ch1_tT-1, ...]
        // Our buffer is frame-major:
        //   [t0_ch0, t0_ch1, ..., t0_chC-1, t1_ch0, ..., t1_chC-1, ...]
        // So we build with shape (1, T, C) and transpose to (1, C, T).
        let latent_dim = self.config.latent_dim;
        let h_tensor = Tensor::from_slice(
            &self.pre_conv_buffer,
            (1, total_frames, latent_dim),
            &self.device,
        )?
        .transpose(1, 2)?
        .contiguous()?;

        // Step 4: Full pipeline on accumulated history
        let h = self.pre_transformer.forward(&h_tensor)?;

        let mut h = h;
        for ub in &self.upsample_blocks {
            h = ub.forward(&h)?;
        }

        let h = self.decoder_start.forward(&h)?;

        let mut h = h;
        for db in &self.decoder_blocks {
            h = db.forward(&h)?;
        }

        let h = snake_beta(&h, &self.final_snake_a, &self.final_snake_b)?;

        let h = self.final_conv.forward(&h)?; // (1, 1, total_output_samples)

        // Step 5: Extract only the NEW audio samples for this frame
        let all_output: Vec<f32> = h.squeeze(0)?.squeeze(0)?.to_vec1()?;
        let prev_offset = self.output_offset;
        self.output_offset = all_output.len();

        if prev_offset == 0 {
            // First frame: return everything (no baseline to subtract)
            Ok(all_output)
        } else {
            // Subsequent frames: return only the newly produced tail
            Ok(all_output[prev_offset..].to_vec())
        }
    }
}

impl TtsDecoder for Decoder12Hz {
    fn new(_config: DecoderConfig) -> Result<Self> {
        Err(Error::Config(
            "Use Decoder12Hz::from_safetensors(...) instead".into(),
        ))
    }

    fn decode_chunk(&mut self, tokens: &[u16]) -> Result<Vec<f32>> {
        if tokens.len() != self.config.num_codebook_layers {
            return Err(Error::Config(format!(
                "Expected {} tokens (one per codebook layer), got {}",
                self.config.num_codebook_layers,
                tokens.len()
            )));
        }
        let mut observer = NoopStageDumpObserver::default();
        self.decode_chunk_with_observer(tokens, &mut observer)
    }

    fn reset_state(&mut self) {
        self.pre_conv.reset_state();
        self.pre_conv_buffer.clear();
        self.output_offset = 0;
    }
}
