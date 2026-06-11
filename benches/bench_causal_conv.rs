//! # 因果卷積基準測試
//!
//! 測量 CausalConv1d::step / step_tensor 的單幀延遲（p50/p99）。
//! 這是 12Hz 即時解碼器熱路徑的核心操作，直接影響首包延遲（目標 ≤97ms）。

use std::time::Duration;

use candle_core::{Device, Tensor};
use criterion::{Criterion, SamplingMode};

use qwen3tts::codec::{CausalConv1d, CausalConvConfig};

fn main() {
    let mut c = Criterion::default().configure_from_args();
    bench_causal_conv_step(&mut c);
    bench_causal_conv_forward(&mut c);
    c.final_summary();
}

fn bench_causal_conv_step(c: &mut Criterion) {
    let device = Device::Cpu;
    let in_channels = 512;
    let out_channels = 1024;
    let kernel_size = 3;
    let state_capacity = 64;

    let weight = Tensor::rand(
        -0.1f32,
        0.1,
        (out_channels, in_channels, kernel_size),
        &device,
    )
    .unwrap();
    let bias = Some(Tensor::rand(-0.05f32, 0.05, out_channels, &device).unwrap());

    let config = CausalConvConfig {
        in_channels,
        out_channels,
        kernel_size,
        dilation: 1,
        groups: 1,
    };

    let mut conv = CausalConv1d::new(weight, bias, config.clone(), state_capacity).unwrap();
    let frame: Vec<f32> = (0..in_channels).map(|i| (i as f32) * 0.01).collect();

    let mut group = c.benchmark_group("causal_conv::step");
    group.sampling_mode(SamplingMode::Auto);
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(5));

    // Warmup: 3 steps to fill ring buffer partially
    for _ in 0..3 {
        let _ = conv.step(&frame);
    }
    conv.reset_state();

    group.bench_function("step_f32_slice", |b| {
        b.iter(|| {
            let _ = conv.step(&frame);
        });
    });

    // Reset and benchmark step_tensor
    let frame_t = Tensor::from_slice(&frame, (in_channels,), &device).unwrap();
    for _ in 0..3 {
        let _ = conv.step_tensor(&frame_t);
    }
    conv.reset_state();

    // Re-create conv for fresh state so step_tensor starts from empty buffer
    let weight = Tensor::rand(
        -0.1f32,
        0.1,
        (out_channels, in_channels, kernel_size),
        &device,
    )
    .unwrap();
    let bias = Some(Tensor::rand(-0.05f32, 0.05, out_channels, &device).unwrap());
    let mut conv2 = CausalConv1d::new(weight, bias, config, state_capacity).unwrap();

    group.bench_function("step_tensor", |b| {
        b.iter(|| {
            let _ = conv2.step_tensor(&frame_t);
        });
    });

    group.finish();
}

fn bench_causal_conv_forward(c: &mut Criterion) {
    let device = Device::Cpu;
    let in_channels = 512;
    let out_channels = 1024;
    let kernel_size = 3;

    let weight = Tensor::rand(
        -0.1f32,
        0.1,
        (out_channels, in_channels, kernel_size),
        &device,
    )
    .unwrap();
    let bias = Some(Tensor::rand(-0.05f32, 0.05, out_channels, &device).unwrap());

    let config = CausalConvConfig {
        in_channels,
        out_channels,
        kernel_size,
        dilation: 1,
        groups: 1,
    };

    let conv = CausalConv1d::new(weight, bias, config, 64).unwrap();

    let mut group = c.benchmark_group("causal_conv::forward");
    group.sampling_mode(SamplingMode::Auto);
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(5));

    for &seq_len in &[8usize, 32, 128, 512] {
        let input = Tensor::rand(-1.0f32, 1.0, (1, in_channels, seq_len), &device).unwrap();
        group.bench_with_input(
            criterion::BenchmarkId::new("seq_len", seq_len),
            &input,
            |b, inp| b.iter(|| conv.forward(inp)),
        );
    }

    group.finish();
}
