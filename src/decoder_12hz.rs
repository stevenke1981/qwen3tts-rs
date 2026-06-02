use candle_core::{Device, Tensor};

use crate::codec::{
    CausalConvNet, CodebookLookup, DecoderBlock, ParallelCodebook, PreTransformer,
    PreTransformerConfig, UpsampleBlock, snake_beta,
};
use crate::weights::WeightLoader;
use crate::{DecoderConfig, Error, Result, TtsDecoder};

pub struct Decoder12Hz {
    config: DecoderConfig,
    device: Device,

    codebook: ParallelCodebook,

    pre_conv: crate::codec::CausalConv1d,
    pre_transformer: PreTransformer,

    upsample_blocks: Vec<UpsampleBlock>,

    decoder_start: CausalConvNet,
    decoder_blocks: Vec<DecoderBlock>,
    final_conv: CausalConvNet,
    final_snake_a: Tensor,
    final_snake_b: Tensor,

    temperature: f64,
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
        let pre_conv_cfg = crate::codec::CausalConvConfig {
            in_channels: config.embedding_dim,
            out_channels: config.latent_dim,
            kernel_size: 3,
            dilation: 1,
        };
        let pre_conv =
            crate::codec::CausalConv1d::new(pw, pb, pre_conv_cfg, config.ring_buffer_capacity)?;

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
        let decoder_start = CausalConvNet::new(sw, sb, 1, 1, 1);

        let mut decoder_blocks = Vec::new();
        for i in 1..=4 {
            decoder_blocks.push(DecoderBlock::from_loader(&loader, &format!("{i}"))?);
        }

        let (fw, fb) = loader.conv1d_pair("6.conv")?;
        let final_conv = CausalConvNet::new(fw, fb, 1, 1, 1);

        let fs_a = loader.get("5.alpha")?.clone();
        let fs_b = loader.get("5.beta")?.clone();

        log::info!(
            "Decoder12Hz loaded from safetensors: {} tensors",
            loader.len(),
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
        })
    }

    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }

    /// Decode a complete 12Hz token sequence with the same causal batch path
    /// used by the reference tokenizer decoder.
    pub fn decode_frames(&mut self, frames: &[[u16; 16]]) -> Result<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let num_frames = frames.len();
        let mut frame_embeddings: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
        for frame in frames {
            let embeddings = self.codebook.decode(frame)?;
            let frame_embed = embeddings.sum(0)?;
            frame_embeddings.push(frame_embed.to_vec1()?);
        }

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

        h.squeeze(0)?.squeeze(0)?.to_vec1().map_err(Into::into)
    }

    fn decode_chunk_inner(&mut self, tokens: &[u16]) -> Result<Vec<f32>> {
        let device = &self.device;

        let embeddings = self.codebook.decode(tokens)?;
        let frame_embed = embeddings.sum(0)?;

        let frame_vec: Vec<f32> = frame_embed.to_vec1()?;
        let pre_conv_out = self.pre_conv.step(&frame_vec)?;

        let x = Tensor::from_slice(&pre_conv_out, (1, self.config.latent_dim, 1), device)?;

        let h = self.pre_transformer.forward(&x)?;

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

        let h = self.final_conv.forward(&h)?;

        let output: Vec<f32> = h.squeeze(0)?.squeeze(0)?.to_vec1()?;
        Ok(output)
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

        self.decode_chunk_inner(tokens)
    }

    fn reset_state(&mut self) {
        self.pre_conv.reset_state();
    }
}
