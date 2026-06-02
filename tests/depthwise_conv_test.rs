use candle_core::{Device, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig};

fn read_npy(path: &str) -> Vec<f32> {
    let data = std::fs::read(path).unwrap();
    let magic = &data[..6];
    assert_eq!(magic, b"\x93NUMPY", "Bad npy magic in {path}");
    let major = data[6];
    let (header_len_size, header_len) = if major == 1 {
        (2, u16::from_le_bytes([data[8], data[9]]) as usize)
    } else {
        (
            4,
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize,
        )
    };
    let header_start = 8 + header_len_size;
    let data_start = header_start + header_len;
    let raw = &data[data_start..];
    let n = raw.len() / 4;
    let mut floats = Vec::with_capacity(n);
    for chunk in raw.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    floats
}

#[test]
fn depthwise_conv_test() {
    let device = Device::Cpu;

    // Load test data: [C=2, T=4], [2, 1, 3], [2], [2, 4]
    let x_data = read_npy("test_dw2_input.npy");
    let w_data = read_npy("test_dw2_weight.npy");
    let b_data = read_npy("test_dw2_bias.npy");
    let pt_out = read_npy("test_dw2_output.npy");

    let x = Tensor::from_slice(&x_data, (1, 2, 4), &device).unwrap();
    let w = Tensor::from_slice(&w_data, (2, 1, 3), &device).unwrap();
    let b = Tensor::from_slice(&b_data, (2,), &device).unwrap();

    // Candle depthwise conv with groups=2
    let conv = Conv1d::new(
        w.clone(),
        Some(b.clone()),
        Conv1dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 2,
            cudnn_fwd_algo: None,
        },
    );
    let out = conv.forward(&x).unwrap();
    let out_v: Vec<f32> = out
        .squeeze(0)
        .unwrap()
        .to_vec2()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();

    // Manual: groups=1 on each channel separately
    let w0 = w.narrow(0, 0, 1).unwrap(); // [1, 1, 3]
    let w1 = w.narrow(0, 1, 1).unwrap();
    let b0 = b.narrow(0, 0, 1).unwrap(); // [1]
    let b1 = b.narrow(0, 1, 1).unwrap();

    let conv0 = Conv1d::new(
        w0,
        Some(b0),
        Conv1dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        },
    );
    let conv1 = Conv1d::new(
        w1,
        Some(b1),
        Conv1dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        },
    );

    let x0 = x.narrow(1, 0, 1).unwrap(); // [1, 1, 4]
    let x1 = x.narrow(1, 1, 1).unwrap();

    let out0 = conv0.forward(&x0).unwrap();
    let out1 = conv1.forward(&x1).unwrap();

    let out0_v: Vec<f32> = out0
        .squeeze(0)
        .unwrap()
        .to_vec2()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    let out1_v: Vec<f32> = out1
        .squeeze(0)
        .unwrap()
        .to_vec2()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    let combined: Vec<f32> = [out0_v, out1_v].concat();

    let cos = |a: &[f32], b: &[f32]| -> f64 {
        let n = a.len().min(b.len());
        let dot: f64 = a[..n]
            .iter()
            .zip(b[..n].iter())
            .map(|(x, y)| *x as f64 * *y as f64)
            .sum();
        let na: f64 = a[..n]
            .iter()
            .map(|x| *x as f64 * *x as f64)
            .sum::<f64>()
            .sqrt();
        let nb: f64 = b[..n]
            .iter()
            .map(|x| *x as f64 * *x as f64)
            .sum::<f64>()
            .sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    };

    println!(
        "Depthwise (groups=2) vs PT: cos={:.8}",
        cos(&out_v, &pt_out)
    );
    println!(
        "Groups=1 (manual) vs PT:   cos={:.8}",
        cos(&combined, &pt_out)
    );
    println!(
        "Depthwise vs manual:       cos={:.8}",
        cos(&out_v, &combined)
    );

    println!("\nPT output:         {:?}", &pt_out[..8]);
    println!("Candle groups=2:   {:?}", &out_v[..8]);
    println!("Candle groups=1:   {:?}", &combined[..8]);

    // Test that groups=2 matches groups=1 approach
    let max_diff: f64 = out_v
        .iter()
        .zip(combined.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max);
    println!("Max diff groups=2 vs manual: {:.10}", max_diff);
    assert!(
        max_diff < 1e-5,
        "Depthwise conv (groups=2) differs from manual groups=1 per channel: max_diff={:.10}",
        max_diff
    );
}

#[test]
fn depthwise_conv_1024_groups() {
    let device = Device::Cpu;

    // Load test data: [C=1024, T=6], [1024, 1, 7], [1024], [1024, 6]
    let x_data = read_npy("test_dw1024_input.npy");
    let w_data = read_npy("test_dw1024_weight.npy");
    let b_data = read_npy("test_dw1024_bias.npy");
    let pt_out = read_npy("test_dw1024_output.npy");

    let x = Tensor::from_slice(&x_data, (1, 1024, 6), &device).unwrap();
    let w = Tensor::from_slice(&w_data, (1024, 1, 7), &device).unwrap();
    let b = Tensor::from_slice(&b_data, (1024,), &device).unwrap();

    // Candle depthwise conv with groups=1024
    let conv = Conv1d::new(
        w.clone(),
        Some(b.clone()),
        Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: 1024,
            cudnn_fwd_algo: None,
        },
    );
    let out = conv.forward(&x).unwrap();
    let out_v: Vec<f32> = out
        .squeeze(0)
        .unwrap()
        .to_vec2()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();

    let cos = |a: &[f32], b: &[f32]| -> f64 {
        let n = a.len().min(b.len());
        let dot: f64 = a[..n]
            .iter()
            .zip(b[..n].iter())
            .map(|(x, y)| *x as f64 * *y as f64)
            .sum();
        let na: f64 = a[..n]
            .iter()
            .map(|x| *x as f64 * *x as f64)
            .sum::<f64>()
            .sqrt();
        let nb: f64 = b[..n]
            .iter()
            .map(|x| *x as f64 * *x as f64)
            .sum::<f64>()
            .sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    };

    println!("\n--- 1024 groups test ---");
    println!(
        "Candle groups=1024 vs PyTorch: cos={:.8}",
        cos(&out_v, &pt_out)
    );

    // Also compare per-channel: first few channels
    let pt_out_v = &pt_out;
    for ch in 0..3.min(1024) {
        let rust_ch: Vec<f32> = out_v[ch * 6..(ch + 1) * 6].to_vec();
        let pt_ch: Vec<f32> = pt_out_v[ch * 6..(ch + 1) * 6].to_vec();
        let ch_cos = cos(&rust_ch, &pt_ch);
        let max_abs_diff: f64 = rust_ch
            .iter()
            .zip(pt_ch.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0f64, f64::max);
        println!("  Channel {ch}: cos={ch_cos:.8}, max_diff={max_abs_diff:.10}");
        if ch == 0 {
            println!("    Rust: {:?}", &rust_ch);
            println!("    PT:   {:?}", &pt_ch);
        }
    }

    // Overall max diff
    let max_diff: f64 = out_v
        .iter()
        .zip(pt_out.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max);
    let mse: f64 = out_v
        .iter()
        .zip(pt_out.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        / out_v.len() as f64;
    println!("  Max diff: {max_diff:.10}, MSE: {mse:.10}");
    assert!(
        cos(&out_v, &pt_out) > 0.999,
        "Depthwise conv 1024 groups diverges: cos={:.8}, max_diff={:.10}",
        cos(&out_v, &pt_out),
        max_diff
    );
}
