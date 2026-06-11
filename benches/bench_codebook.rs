//! # 碼本查表基準測試
//!
//! 測量 ParallelCodebook::decode 的並行查表延遲（p50/p99）。
//! 16 層 Embedding Lookup 透過 rayon::par_iter 執行，是 12Hz 解碼器的瓶頸之一。

use std::time::Duration;

use candle_core::{Device, Tensor};
use criterion::{Criterion, SamplingMode};

use qwen3tts::codec::{CodebookLookup, ParallelCodebook};

fn main() {
    let mut c = Criterion::default().configure_from_args();
    bench_parallel_codebook_decode(&mut c);
    c.final_summary();
}

fn bench_parallel_codebook_decode(c: &mut Criterion) {
    let device = Device::Cpu;
    let num_layers = 16;
    let codebook_size = 2048;
    let embedding_dim = 512;

    let weights = Tensor::rand(
        -0.1f32,
        0.1,
        (num_layers, codebook_size, embedding_dim),
        &device,
    )
    .unwrap();
    let lookup = CodebookLookup::new(weights).unwrap();
    let pc = ParallelCodebook::new(lookup);

    let mut group = c.benchmark_group("codebook::decode");
    group.sampling_mode(SamplingMode::Auto);
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(5));

    // Random tokens within valid range
    let tokens: Vec<u16> = (0..16).map(|i| (i * 137 + 42) as u16 % 2048).collect();

    group.bench_function("16_layers_parallel", |b| {
        b.iter(|| {
            let _ = pc.decode(&tokens);
        });
    });

    // Also bench single CodebookLookup::batch_lookup (returns stacked with dim 1)
    let lookup2 = CodebookLookup::new(
        Tensor::rand(
            -0.1f32,
            0.1,
            (num_layers, codebook_size, embedding_dim),
            &device,
        )
        .unwrap(),
    )
    .unwrap();

    group.bench_function("batch_lookup_16_layers", |b| {
        b.iter(|| {
            let _ = lookup2.batch_lookup(&tokens);
        });
    });

    group.finish();
}
