use candle_core::{Device, Tensor};

use crate::alignment_stage_dump::{NoopStageDumpObserver, StageDumpObserver};
use crate::codec::{
    CausalConv1d, CausalConvConfig, CodebookLookup, DecoderBlock, KvRing as DeviceKvCache,
    ParallelCodebook, PreTransformer, PreTransformerConfig, UpsampleBlock, snake_beta,
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

    /// KV device caches for each transformer layer (streaming attention cache)
    kv_caches: Vec<DeviceKvCache>,
    /// Current frame position index for RoPE
    position: usize,
}

impl Decoder12Hz {
    pub fn from_safetensors(
        config: DecoderConfig,
        weight_path: impl AsRef<std::path::Path>,
        device: &Device,
    ) -> Result<Self> {
        let loader = WeightLoader::from_dir(weight_path, device)?;
        Self::from_loader(config, &loader, device)
    }

    /// 從已載入的 WeightLoader 建立 Decoder12Hz
    pub fn from_loader(
        config: DecoderConfig,
        loader: &WeightLoader,
        device: &Device,
    ) -> Result<Self> {
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
        let pre_transformer = PreTransformer::from_loader(loader, &pt_cfg, device)?;

        let mut upsample_blocks = Vec::new();
        for i in 0..2 {
            upsample_blocks.push(UpsampleBlock::from_loader(
                loader,
                &format!("upsample.{i}"),
            )?);
        }

        let (sw, sb) = loader.conv1d_pair("0.conv")?;
        let ds_cfg = CausalConvConfig::from_weight(&sw, 1, 1);
        let decoder_start = CausalConv1d::new(sw, sb, ds_cfg, config.ring_buffer_capacity)?;

        let mut decoder_blocks = Vec::new();
        for i in 1..=4 {
            decoder_blocks.push(DecoderBlock::from_loader(loader, &format!("{i}"))?);
        }

        let (fw, fb) = loader.conv1d_pair("6.conv")?;
        let fc_cfg = CausalConvConfig::from_weight(&fw, 1, 1);
        let final_conv = CausalConv1d::new(fw, fb, fc_cfg, config.ring_buffer_capacity)?;

        let fs_a = loader.get("5.alpha")?.clone();
        let fs_b = loader.get("5.beta")?.clone();

        log::info!(
            "Decoder12Hz loaded: {} tensors",
            loader.len(),
        );

        let kv_caches = PreTransformer::new_kv_caches(&pt_cfg);

        log::info!(
            "Decoder12Hz loaded: {} tensors, {} kv_caches",
            loader.len(),
            kv_caches.len(),
        );
        log::info!(
            "Decoder12Hz pure O(1) device-resident streaming step pipeline initialized"
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
            kv_caches,
            position: 0,
        })
    }

    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }

    /// Decode a complete 12Hz token sequence with the same causal batch path
    /// used by the reference tokenizer decoder.
    pub fn decode_frames(&mut self, frames: &[[u16; 16]]) -> Result<Vec<f32>> {
        let mut observer = NoopStageDumpObserver;
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
                        let w = weight as f32;
                        right_t
                            .sub(&left_t)?
                            .affine(w as f64, 0.0)?
                            .add(&left_t)?
                            .flatten_all()?
                            .to_vec1()?
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
        // 1. Codebook lookup (1 幀 16 tokens -> 512 embedding)
        let embeddings = self.codebook.decode(tokens)?;
        let frame_embed = embeddings.sum(0)?.reshape((1, self.config.embedding_dim, 1))?;

        // 2. Pre-Conv (512 -> 1024, 1 幀 -> 1 幀)
        let mut h = self.pre_conv.step_tensor(&frame_embed)?;

        // 3. Pre-Transformer (1024 -> 1024, 1 幀 -> 1 幀)
        h = self.pre_transformer.step(&h, &mut self.kv_caches, self.position)?;
        self.position += 1;

        // 4. Upsample Blocks (1 幀 -> 2 幀 -> 4 幀)
        for ub in &mut self.upsample_blocks {
            h = ub.step(&h)?;
        }

        // 5. Decoder Start (4 幀 -> 4 幀, 1024 -> 1536)
        h = self.decoder_start.step_tensor(&h)?;

        // 6. Decoder Blocks (4 幀 -> 32 幀 -> 160 幀 -> 640 幀 -> 1920 幀)
        for db in &mut self.decoder_blocks {
            h = db.step(&h)?;
        }

        // 7. Final SnakeBeta + Conv (1920 幀 -> 1920 PCM 取樣點)
        h = snake_beta(&h, &self.final_snake_a, &self.final_snake_b)?;
        h = self.final_conv.step_tensor(&h)?;

        // 8. 抽出當前 Chunk 的 1920 個 PCM 取樣點
        let pcm: Vec<f32> = h.squeeze(0)?.squeeze(0)?.to_vec1()?;
        Ok(pcm)
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
        let mut observer = NoopStageDumpObserver;
        self.decode_chunk_with_observer(tokens, &mut observer)
    }

    fn reset_state(&mut self) {
        self.pre_conv.reset_state();
        PreTransformer::reset_kv_caches(&mut self.kv_caches);
        self.position = 0;
        for ub in &mut self.upsample_blocks {
            ub.reset_state();
        }
        self.decoder_start.reset_state();
        for db in &mut self.decoder_blocks {
            db.reset_state();
        }
        self.final_conv.reset_state();
    }
}
