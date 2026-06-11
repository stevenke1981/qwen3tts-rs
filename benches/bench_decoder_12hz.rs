//! # 12Hz 解碼器端到端基準測試
//!
//! 測量 Decoder12Hz::decode_chunk 的首包延遲（p50/p99）。
//! 目標：p99 ≤ 97ms（12Hz 即時模式）。
//!
//! **注意**：需要真實權重檔案才能執行。若權重不存在則自動跳過。

use std::path::Path;
use std::time::Duration;

use candle_core::Device;
use criterion::{Criterion, SamplingMode};

use qwen3tts::{Decoder12Hz, DecoderConfig, DecoderMode, TtsDecoder};

fn main() {
    let mut c = Criterion::default().configure_from_args();
    bench_decoder_12hz_decode_chunk(&mut c);
    c.final_summary();
}

fn bench_decoder_12hz_decode_chunk(c: &mut Criterion) {
    let weight_dir = Path::new("weights/refs2");
    if !weight_dir.join("pre_conv.weight").exists() {
        eprintln!(
            "Skipping Decoder12Hz benchmark: weight directory '{:?}' not found or incomplete",
            weight_dir
        );
        return;
    }

    let device = Device::Cpu;

    // Build config matching the reference decoder
    let config = DecoderConfig {
        mode: DecoderMode::RealTime,
        num_codebook_layers: 16,
        codebook_size: 2048,
        embedding_dim: 512,
        sample_rate: 24000,
        kernel_size: 3,
        conv_channels: 512,
        ring_buffer_capacity: 64,
        speed: 1.0,
        latent_dim: 1024,
        transformer_dim: 512,
        transformer_heads: 16,
        transformer_kv_heads: 16,
        transformer_layers: 8,
        sliding_window: 72,
        upsample_kernel: 8,
        stream_window: 0,
        snake_beta: 1.0,
        dit_hidden_dim: 1024,
        dit_num_heads: 16,
        dit_num_blocks: 12,
        ode_steps: 32,
    };

    // Load via Decoder12Hz::from_safetensors
    let mut decoder = match Decoder12Hz::from_safetensors(config, weight_dir, &device) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping Decoder12Hz benchmark: failed to load weights: {e}");
            return;
        }
    };

    let mut group = c.benchmark_group("decoder_12hz::decode_chunk");
    group.sampling_mode(SamplingMode::Auto);
    group.sample_size(30); // Fewer samples — each decode_chunk is expensive
    group.measurement_time(Duration::from_secs(10));

    // Generate random valid tokens (one frame = 16 tokens for 16 codebook layers)
    let tokens: Vec<u16> = (0..16).map(|i| ((i * 42 + 17) % 2048) as u16).collect();

    group.bench_function("single_frame", |b| {
        b.iter(|| {
            decoder.reset_state();
            let _ = decoder.decode_chunk(&tokens);
        });
    });

    // Multi-frame benchmark: decode 10 frames sequentially (streaming)
    group.bench_function("streaming_10_frames", |b| {
        b.iter(|| {
            decoder.reset_state();
            for _ in 0..10 {
                let _ = decoder.decode_chunk(&tokens);
            }
        });
    });

    group.finish();
}
