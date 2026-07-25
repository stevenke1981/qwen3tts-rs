//! Per-layer comparison test: Rust Decoder12Hz vs PyTorch reference.
//!
//! Runs all 3 frames through the decoder, captures intermediate outputs
//! at each layer, and compares against saved PyTorch .npy references.
//!
//! Run: cargo test --test debug_per_layer_compare -- --nocapture
//! This test is CPU-only (~2GB RAM, ~30s).

use std::path::Path;

use candle_core::Module;
use qwen3tts::{
    Decoder12Hz, DecoderConfig, TtsDecoder,
    codec::{CausalConv1d, CodebookLookup, DecoderBlock, PreTransformer},
};

// ---------------------------------------------------------------------------
// Simple .npy reader (no dependencies)
// ---------------------------------------------------------------------------

fn read_npy(path: &str) -> Vec<f32> {
    let data = std::fs::read(path).expect(&format!("Failed to read {path}"));
    // .npy header: "\x93NUMPY" magic + version + header_len + header_text
    // Skip magic (6 bytes) + version (2 bytes) + header_len (2/4 bytes)
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
    let _header = &data[header_start..header_start + header_len];
    let data_start = header_start + header_len;

    // Remaining bytes are raw float32 LE
    let raw = &data[data_start..];
    let n = raw.len() / 4;
    let mut floats = Vec::with_capacity(n);
    for chunk in raw.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    floats
}

fn cosine_sim_f32(a: &[f32], b: &[f32]) -> f64 {
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
        return 0.0;
    }
    dot / (na * nb)
}

fn mse_f32(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    a[..n]
        .iter()
        .zip(b[..n].iter())
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum::<f64>()
        / n as f64
}

fn amplitude_ratio(a: &[f32], b: &[f32]) -> f64 {
    let max_a = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max) as f64;
    let max_b = b.iter().map(|x| x.abs()).fold(0.0f32, f32::max) as f64;
    if max_b == 0.0 {
        return 0.0;
    }
    max_a / max_b
}

fn range_str(data: &[f32]) -> String {
    if data.is_empty() {
        return "[]".into();
    }
    let min = data.iter().cloned().fold(f32::MAX, f32::min);
    let max = data.iter().cloned().fold(f32::MIN, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    format!("[{:.4}, {:.4}] mean={:.4}", min, max, mean)
}

fn header(label: &str) {
    println!("\n{}", "=".repeat(72));
    println!("  {label}");
    println!("{}", "=".repeat(72));
}

fn compare(label: &str, rust: &[f32], pytorch: &[f32]) {
    let n = rust.len().min(pytorch.len());
    let cos = cosine_sim_f32(rust, pytorch);
    let mse = mse_f32(rust, pytorch);
    let ratio = amplitude_ratio(rust, pytorch);
    let rstr = range_str(rust);
    let pstr = range_str(pytorch);
    println!(
        "  {label:<30} | rust={rstr:<40} | pt={pstr:<40} | cos={cos:.8} | mse={mse:.10} | amp_ratio={ratio:.4}x | n={n}"
    );
}

// ---------------------------------------------------------------------------
// Main diagnostic test
// ---------------------------------------------------------------------------

#[test]
fn debug_per_layer_compare() {
    let weight_dir = Path::new("weights/tokenizer");
    if !weight_dir.join("codebook.safetensors").exists() {
        eprintln!("Skipping: real weights not found at {weight_dir:?}");
        return;
    }

    let config = DecoderConfig::realtime();
    let device = candle_core::Device::Cpu;

    // Hardcoded tokens from pt_full_trace.py (3 frames × 16 tokens)
    let frames: Vec<[u16; 16]> = vec![
        [
            1221, 1052, 1114, 1364, 1468, 1760, 974, 1318, 746, 391, 161, 1013, 663, 837, 216, 1929,
        ],
        [
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
        ],
        [
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        ],
    ];
    assert_eq!(frames.len(), 3, "Expected 3 frames");
    println!(
        "Loaded {} frames from hardcoded tokens (pt_full_trace.py)",
        frames.len()
    );
    for (i, f) in frames.iter().enumerate() {
        println!("  Frame {i}: [{}, {}, {}, {}, ...]", f[0], f[1], f[2], f[3]);
    }

    // -------------------------------------------------------------------------
    // 1. Codebook comparison
    // -------------------------------------------------------------------------
    header("1. Codebook Lookup (quantizer output)");

    // Build codebook lookup directly
    let cw = qwen3tts::weights::WeightLoader::from_dir(weight_dir, &device)
        .expect("load weights")
        .codebook_weights()
        .expect("codebook weights");
    let codebook = CodebookLookup::new(cw).expect("codebook");
    let pt_quant = read_npy("weights/pt_quant_out.npy");
    // pt_quant_out.npy is shape [512, 3] — C-order: channel-major, time minor
    // Layout: ch0_t0, ch0_t1, ch0_t2, ch1_t0, ch1_t1, ch1_t2, ...
    // To extract frame i: take every 3rd value starting from position i

    // Rust: process each frame, sum across layers
    let mut rust_quant = Vec::new(); // each entry: [512] per frame
    for frame in &frames {
        let stacked = codebook.batch_lookup(frame).expect("batch lookup");
        // stacked: [16, 1, 512]
        let summed = stacked.squeeze(1).expect("squeeze").sum(0).expect("sum");
        // summed: [512]
        let v: Vec<f32> = summed.to_vec1().expect("vec1");
        rust_quant.push(v);
    }

    // PyTorch reference: pt_quant_out.npy is (512, 3) C-order
    let num_channels = rust_quant[0].len(); // 512
    for i in 0..3 {
        let rust_col = &rust_quant[i];
        // PyTorch column i: every 3rd value starting from i
        let pt_col: Vec<f32> = (0..num_channels).map(|ch| pt_quant[ch * 3 + i]).collect();
        compare(&format!("  Frame {i}"), rust_col, &pt_col);
    }

    // Also compare full [512,3] — need Rust data in same C-order [ch, frame]
    let rq = &rust_quant; // borrow to avoid move into closure
    let rust_corder: Vec<f32> = (0..512)
        .flat_map(|ch| (0..3).map(move |f| rq[f][ch]))
        .collect();
    compare("Full [512,3]", &rust_corder, &pt_quant);

    // -------------------------------------------------------------------------
    // 2. Pre-conv comparison (batch mode via CausalConv1d::forward)
    // -------------------------------------------------------------------------
    header("2. Pre-Convolution (causal conv1d)");

    // Build pre_conv directly
    use qwen3tts::codec::CausalConvConfig;
    let loader = qwen3tts::weights::WeightLoader::from_dir(weight_dir, &device).expect("weights");
    let (pw, pb) = loader.conv1d_pair("pre_conv").expect("pre_conv weights");
    let pre_conv_cfg = CausalConvConfig {
        in_channels: config.embedding_dim,
        out_channels: config.latent_dim,
        kernel_size: 3,
        dilation: 1,
        groups: 1,
    };
    let mut pre_conv = CausalConv1d::new(pw, pb, pre_conv_cfg, 64).expect("pre_conv");

    let pt_pre_conv = read_npy("weights/pt_pre_conv_out.npy"); // [1024, 3]

    // Build batch input [1, 512, 3] from codebook outputs
    let batch_size = rust_quant.len(); // 3
    let mut batch_data = Vec::with_capacity(512 * batch_size);
    for ch in 0..512 {
        for frame in 0..batch_size {
            batch_data.push(rust_quant[frame][ch]);
        }
    }
    let batch_input =
        candle_core::Tensor::from_slice(&batch_data, (1, 512, 3), &device).expect("batch tensor");
    let rust_pre_conv = pre_conv.forward(&batch_input).expect("pre_conv forward");
    // Rust output with pad=2: (1, 1024, 5). Take only first 3 positions (causal outputs).
    let causal_out = rust_pre_conv.narrow(2, 0, 3).expect("narrow to causal");
    let rust_pre_conv_v: Vec<f32> = causal_out
        .squeeze(0)
        .expect("squeeze")
        .to_vec2()
        .expect("vec2")
        .into_iter()
        .flatten()
        .collect();
    compare("Pre-conv output", &rust_pre_conv_v, &pt_pre_conv);

    // -------------------------------------------------------------------------
    // 2a. Weight value verification for Upsample.0.cn (ConvNeXtBlock)
    // -------------------------------------------------------------------------
    header("2a. Weight value verification (Upsample.0.cn)");

    let dw_w_t = loader
        .get("upsample.0.1.dwconv.conv.weight")
        .expect("dwconv weight");
    let dw_w_flat = dw_w_t.flatten_all().expect("flatten");
    let dw_w: Vec<f32> = dw_w_flat.to_vec1().expect("v1");
    println!(
        "  dwconv weight[0:7]: [{:.10}, {:.10}, {:.10}, {:.10}, {:.10}, {:.10}, {:.10}]",
        dw_w[0], dw_w[1], dw_w[2], dw_w[3], dw_w[4], dw_w[5], dw_w[6]
    );
    println!(
        "  dwconv weight[7:14]: [{:.10}, {:.10}, {:.10}, {:.10}, {:.10}, {:.10}, {:.10}]",
        dw_w[7], dw_w[8], dw_w[9], dw_w[10], dw_w[11], dw_w[12], dw_w[13]
    );
    let dw_min = dw_w.iter().cloned().fold(f32::MAX, f32::min);
    let dw_max = dw_w.iter().cloned().fold(f32::MIN, f32::max);
    let dw_mean = dw_w.iter().sum::<f32>() / dw_w.len() as f32;
    println!(
        "  dwconv weight stats: min={:.10}, max={:.10}, mean={:.10}",
        dw_min, dw_max, dw_mean
    );

    let dw_b: Vec<f32> = loader
        .get("upsample.0.1.dwconv.conv.bias")
        .expect("dwconv bias")
        .to_vec1()
        .expect("bias v1");
    let b_min = dw_b.iter().cloned().fold(f32::MAX, f32::min);
    let b_max = dw_b.iter().cloned().fold(f32::MIN, f32::max);
    let b_mean = dw_b.iter().sum::<f32>() / dw_b.len() as f32;
    println!("  dwconv bias[0]: {:.10}", dw_b[0]);
    println!("  dwconv bias[1]: {:.10}", dw_b[1]);
    println!(
        "  dwconv bias stats: min={:.10}, max={:.10}, mean={:.10}",
        b_min, b_max, b_mean
    );

    // -------------------------------------------------------------------------
    // 3. Pre-transformer comparison (batch mode)
    // -------------------------------------------------------------------------
    header("3. Pre-Transformer");

    use qwen3tts::codec::PreTransformerConfig;
    let pt_pre_trans = read_npy("weights/pt_pre_trans_out.npy"); // [3, 1024]

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
    let pre_transformer =
        PreTransformer::from_loader(&loader, &pt_cfg, &device).expect("pre_transformer");

    // pre_transformer in Rust takes [B, 1024, T] and internally does transpose(1,2)
    // IMPORTANT: narrow to 3 causal time steps (pre_conv produces 5 with pad=2)
    let pre_trans_input = pre_conv
        .forward(&batch_input)
        .expect("pre_conv for trans")
        .narrow(2, 0, 3)
        .expect("narrow to causal");
    let rust_pre_trans = pre_transformer
        .forward(&pre_trans_input)
        .expect("pre_transformer forward");
    // Output: [1, 1024, 3]. PyTorch ref is [3, 1024] (after transpose in pt_full_trace.py)
    // So transpose Rust to [1, 3, 1024] → squeeze → [3, 1024] to match
    let rust_pre_trans_v: Vec<f32> = rust_pre_trans
        .transpose(1, 2)
        .expect("transpose for comparison")
        .squeeze(0)
        .expect("squeeze")
        .to_vec2()
        .expect("vec2")
        .into_iter()
        .flatten()
        .collect();
    compare("Pre-trans output", &rust_pre_trans_v, &pt_pre_trans);

    // Compare individual layers from refs/
    header("  3a. Pre-Transformer per-layer comparison");
    for layer_idx in 0..8 {
        let ref_path = format!("weights/refs/pre_transformer.layer_{layer_idx}.npy");
        if !Path::new(&ref_path).exists() {
            println!("  Skipping layer {layer_idx}: ref not found");
            continue;
        }
        let pt_layer = read_npy(&ref_path);
        // The refs for individual layers are [3, 512] (hidden dim, after input_proj)
        // We need to get the Rust intermediate output for this layer.
        // Unfortunately, PreTransformer doesn't expose individual layers.
        // We'll skip per-layer comparison for now and compare the full output instead.
        println!(
            "  Layer {layer_idx}: ref exists, shape inferred from size: {} floats",
            pt_layer.len()
        );
    }

    // -------------------------------------------------------------------------
    // 4. Upsample comparison
    // -------------------------------------------------------------------------
    header("4. Upsample Blocks");

    let pt_upsample = read_npy("weights/pt_upsample_out.npy"); // [1024, 12]

    // Build upsample blocks
    use qwen3tts::codec::UpsampleBlock;
    let mut upsample_blocks = Vec::new();
    for i in 0..2 {
        upsample_blocks.push(
            UpsampleBlock::from_loader(&loader, &format!("upsample.{i}")).expect("upsample block"),
        );
    }

    // Pre-transformer output needs permute from (1, 1024, 3) before upsample
    // In PyTorch: pre_transformer → permute(0, 2, 1) → return → upsample
    // In Rust: PreTransformer::forward already does transpose back, so output is [1, 1024, T]
    // So rust_pre_trans is [1, 1024, 3] which matches upsample input format
    let mut h = rust_pre_trans.clone();

    // Per-upsample-block and per-sub-block intermediates
    let mut upsample_intermediates: Vec<Vec<f32>> = Vec::new();
    for (bi, ub) in upsample_blocks.iter().enumerate() {
        // Sub-block 0: ConvTranspose1d (ct)
        let h_ct = ub.ct.forward(&h).expect("upsample ct forward");
        let ct_v: Vec<f32> = h_ct
            .squeeze(0)
            .expect("squeeze")
            .to_vec2()
            .expect("vec2")
            .into_iter()
            .flatten()
            .collect();
        let ct_ref = format!("weights/pt_upsample_block.{bi}.0.npy");
        if Path::new(&ct_ref).exists() {
            compare(&format!("Upsample.{bi}.ct"), &ct_v, &read_npy(&ct_ref));
        }

        // Sub-block 1: ConvNeXtBlock (cn)
        let h_cn = ub.cn.forward(&h_ct).expect("upsample cn forward");
        let cn_v: Vec<f32> = h_cn
            .squeeze(0)
            .expect("squeeze")
            .to_vec2()
            .expect("vec2")
            .into_iter()
            .flatten()
            .collect();
        let cn_ref = format!("weights/pt_upsample_block.{bi}.1.npy");
        if Path::new(&cn_ref).exists() {
            compare(&format!("Upsample.{bi}.cn"), &cn_v, &read_npy(&cn_ref));
        }

        upsample_intermediates.push(cn_v);
        h = h_cn;
    }

    // Expected shape: [1, 1024, 12] (3 * 2 * 2 = 12)
    let h_dims = h.dims();
    println!("  Upsample output shape: {:?}", h_dims);
    let rust_upsample_v: Vec<f32> = upsample_intermediates[1].clone(); // second block = final upsample output
    compare(
        "Upsample output (vs full ref)",
        &rust_upsample_v,
        &pt_upsample,
    );

    // -------------------------------------------------------------------------
    // 4a. ConvNeXtBlock sub-step comparison (Upsample.0.cn)
    // -------------------------------------------------------------------------
    header("4a. ConvNeXtBlock sub-step comparison (Upsample.0.cn)");

    // Re-run upsample.0.ct on pre_trans output for ConvNeXtBlock test
    let cn0_input = upsample_blocks[0]
        .ct
        .forward(&rust_pre_trans)
        .expect("upsample.0.ct forward");
    let cn0_input_v: Vec<f32> = cn0_input
        .squeeze(0)
        .expect("squeeze")
        .to_vec2()
        .expect("vec2")
        .into_iter()
        .flatten()
        .collect();
    let pt_cn_input = read_npy("weights/pt_cn0_input.npy");
    compare("CN input (ct out)", &cn0_input_v, &pt_cn_input);

    // Sub-step 1: dwconv
    let cn0 = &upsample_blocks[0].cn;
    let cn_dwconv = cn0.dwconv.forward(&cn0_input).expect("dwconv");
    let cn_dwconv_v: Vec<f32> = cn_dwconv
        .squeeze(0)
        .expect("squeeze")
        .to_vec2()
        .expect("vec2")
        .into_iter()
        .flatten()
        .collect();
    compare(
        "CN dwconv",
        &cn_dwconv_v,
        &read_npy("weights/pt_cn0_dwconv.npy"),
    );

    // Sub-step 2: transpose(1,2) + norm
    let cn_trans = cn_dwconv.transpose(1, 2).expect("transpose");
    let cn_norm = cn0.norm.forward(&cn_trans).expect("norm");
    let cn_norm_v: Vec<f32> = cn_norm
        .squeeze(0)
        .expect("squeeze")
        .to_vec2()
        .expect("vec2")
        .into_iter()
        .flatten()
        .collect();
    compare("CN norm", &cn_norm_v, &read_npy("weights/pt_cn0_norm.npy"));

    // Manual LayerNorm to verify Candle's implementation
    let n_dim = cn_trans.dim(2).expect("dim 2");
    let norm_mean = cn_trans.mean_keepdim(2).expect("mean");
    let norm_centered = cn_trans.broadcast_sub(&norm_mean).expect("centered");
    let norm_var = (&norm_centered * &norm_centered)
        .expect("squared")
        .sum_keepdim(2)
        .expect("sum")
        .broadcast_div(&candle_core::Tensor::new(&[n_dim as f32], &device).expect("n"))
        .expect("div");
    let norm_std = (norm_var + 1e-6).expect("add eps").sqrt().expect("sqrt");
    let norm_normalized = norm_centered.broadcast_div(&norm_std).expect("div std");
    let manual_norm = norm_normalized
        .broadcast_mul(cn0.norm.weight())
        .expect("mul weight")
        .broadcast_add(cn0.norm.bias().expect("norm bias"))
        .expect("add bias");
    let manual_norm_v: Vec<f32> = manual_norm
        .squeeze(0)
        .expect("squeeze")
        .to_vec2()
        .expect("vec2")
        .into_iter()
        .flatten()
        .collect();
    compare(
        "CN norm (manual)",
        &manual_norm_v,
        &read_npy("weights/pt_cn0_norm.npy"),
    );

    // Sub-step 3: pwconv1
    let cn_pw1 = cn0.pwconv1.forward(&cn_norm).expect("pwconv1");
    let cn_pw1_v: Vec<f32> = cn_pw1
        .squeeze(0)
        .expect("squeeze")
        .to_vec2()
        .expect("vec2")
        .into_iter()
        .flatten()
        .collect();
    compare(
        "CN pwconv1",
        &cn_pw1_v,
        &read_npy("weights/pt_cn0_pwconv1.npy"),
    );

    // Sub-step 4: GeLU activation
    let cn_act = candle_nn::Activation::Gelu.forward(&cn_pw1).expect("gelu");
    let cn_act_v: Vec<f32> = cn_act
        .squeeze(0)
        .expect("squeeze")
        .to_vec2()
        .expect("vec2")
        .into_iter()
        .flatten()
        .collect();
    compare(
        "CN act (GELU)",
        &cn_act_v,
        &read_npy("weights/pt_cn0_act.npy"),
    );

    // Sub-step 5: pwconv2
    let cn_pw2 = cn0.pwconv2.forward(&cn_act).expect("pwconv2");
    let cn_pw2_v: Vec<f32> = cn_pw2
        .squeeze(0)
        .expect("squeeze")
        .to_vec2()
        .expect("vec2")
        .into_iter()
        .flatten()
        .collect();
    compare(
        "CN pwconv2",
        &cn_pw2_v,
        &read_npy("weights/pt_cn0_pwconv2.npy"),
    );

    // Sub-step 6: transpose(1,2) + gamma * h + residual
    let cn_out_t = cn_pw2.transpose(1, 2).expect("transpose");
    let g = cn0
        .gamma
        .reshape((1, cn0.gamma.elem_count(), 1))
        .expect("gamma reshape");
    let cn_out = g
        .broadcast_mul(&cn_out_t)
        .expect("gamma mul")
        .broadcast_add(&cn0_input)
        .expect("residual add");
    let cn_out_v: Vec<f32> = cn_out
        .squeeze(0)
        .expect("squeeze")
        .to_vec2()
        .expect("vec2")
        .into_iter()
        .flatten()
        .collect();
    compare(
        "CN output",
        &cn_out_v,
        &read_npy("weights/pt_cn0_output.npy"),
    );

    // Isolate test: feed PyTorch's exact ct output into Rust ConvNeXtBlock
    header("4b. ConvNeXtBlock isolation test (Rust CN on PyTorch input)");
    let pt_cn_input_data = read_npy("weights/pt_cn0_input.npy"); // [1024, 6 — C, T]
    let pt_cn_input_t = candle_core::Tensor::from_slice(&pt_cn_input_data, (1, 1024, 6), &device)
        .expect("pt cn input tensor");

    // Sub-step comparison on PyTorch input
    let cn_dwconv_pt = cn0.dwconv.forward(&pt_cn_input_t).expect("dwconv");
    let dwconv_rust_v: Vec<f32> = cn_dwconv_pt
        .squeeze(0)
        .expect("sq")
        .to_vec2()
        .expect("v2")
        .into_iter()
        .flatten()
        .collect();
    let dwconv_pt_v = read_npy("weights/pt_cn0_dwconv.npy");
    compare("CN (PT inp) dwconv", &dwconv_rust_v, &dwconv_pt_v);

    // Channel-by-channel diagnostic
    const CH: usize = 1024;
    const T: usize = 6;
    let mut per_ch_cos = Vec::with_capacity(CH);
    for ch in 0..1024.min(CH) {
        let rust_ch: Vec<f32> = dwconv_rust_v[ch * T..(ch + 1) * T].to_vec();
        let pt_ch: Vec<f32> = dwconv_pt_v[ch * T..(ch + 1) * T].to_vec();
        let dot: f64 = rust_ch
            .iter()
            .zip(pt_ch.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let na: f64 = rust_ch
            .iter()
            .map(|a| *a as f64 * *a as f64)
            .sum::<f64>()
            .sqrt();
        let nb: f64 = pt_ch
            .iter()
            .map(|b| *b as f64 * *b as f64)
            .sum::<f64>()
            .sqrt();
        let cos_val = if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        };
        per_ch_cos.push(cos_val);
    }
    // Find worst channels
    let worst: Vec<(usize, f64)> = {
        let mut w: Vec<_> = per_ch_cos.iter().copied().enumerate().collect();
        w.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        w.into_iter().take(10).collect()
    };
    println!("  Worst 10 channels (cos):");
    for (idx, cos) in &worst {
        let ch_slice = &dwconv_rust_v[idx * T..(idx + 1) * T];
        let pt_slice = &dwconv_pt_v[idx * T..(idx + 1) * T];
        let max_abs_diff: f64 = ch_slice
            .iter()
            .zip(pt_slice.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0f64, f64::max);
        println!(
            "    ch={idx:4} cos={cos:.6} max_diff={max_abs_diff:.8}  rust={ch_slice:.6?}  pt={pt_slice:.6?}"
        );
    }
    let best: Vec<(usize, f64)> = {
        let mut w: Vec<_> = per_ch_cos.iter().copied().enumerate().collect();
        w.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        w.into_iter().take(5).collect()
    };
    println!("  Best 5 channels (cos):");
    for (idx, cos) in &best {
        println!("    ch={idx:4} cos={cos:.6}");
    }

    // Also test: build fresh Conv1d from raw safetensors weight (not through ConvNeXtBlock)
    if let Ok(raw_dw_w) = loader.get("upsample.0.1.dwconv.conv.weight") {
        let raw_dw_w = raw_dw_w.clone();
        let raw_dw_b = loader
            .get("upsample.0.1.dwconv.conv.bias")
            .expect("raw dw b")
            .clone();
        let dw_cfg = candle_nn::Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: 1024,
            cudnn_fwd_algo: None,
        };
        let fresh_dwconv = candle_nn::Conv1d::new(raw_dw_w, Some(raw_dw_b), dw_cfg);
        let fresh_out = fresh_dwconv.forward(&pt_cn_input_t).expect("fresh dwconv");
        let fresh_v: Vec<f32> = fresh_out
            .squeeze(0)
            .expect("sq")
            .to_vec2()
            .expect("v2")
            .into_iter()
            .flatten()
            .collect();
        compare("CN (PT inp) fresh dwconv", &fresh_v, &dwconv_pt_v);
    } else {
        println!("  Skipping fresh dwconv test: raw weight not accessible");
    }

    let cn_trans_pt = cn_dwconv_pt.transpose(1, 2).expect("trans");
    let cn_norm_pt = cn0.norm.forward(&cn_trans_pt).expect("norm");
    compare(
        "CN (PT inp) norm",
        &cn_norm_pt
            .squeeze(0)
            .expect("sq")
            .to_vec2()
            .expect("v2")
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>(),
        &read_npy("weights/pt_cn0_norm.npy"),
    );

    let cn_pw1_pt = cn0.pwconv1.forward(&cn_norm_pt).expect("pwconv1");
    compare(
        "CN (PT inp) pwconv1",
        &cn_pw1_pt
            .squeeze(0)
            .expect("sq")
            .to_vec2()
            .expect("v2")
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>(),
        &read_npy("weights/pt_cn0_pwconv1.npy"),
    );

    let cn_act_pt = candle_nn::Activation::Gelu
        .forward(&cn_pw1_pt)
        .expect("gelu");
    compare(
        "CN (PT inp) GELU",
        &cn_act_pt
            .squeeze(0)
            .expect("sq")
            .to_vec2()
            .expect("v2")
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>(),
        &read_npy("weights/pt_cn0_act.npy"),
    );

    let cn_pw2_pt = cn0.pwconv2.forward(&cn_act_pt).expect("pwconv2");
    compare(
        "CN (PT inp) pwconv2",
        &cn_pw2_pt
            .squeeze(0)
            .expect("sq")
            .to_vec2()
            .expect("v2")
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>(),
        &read_npy("weights/pt_cn0_pwconv2.npy"),
    );

    let cn_out_t_pt = cn_pw2_pt.transpose(1, 2).expect("trans");
    let g = cn0
        .gamma
        .reshape((1, cn0.gamma.elem_count(), 1))
        .expect("gamma");
    let cn_out_pt = g
        .broadcast_mul(&cn_out_t_pt)
        .expect("mul")
        .broadcast_add(&pt_cn_input_t)
        .expect("add");
    compare(
        "CN (PT inp) output",
        &cn_out_pt
            .squeeze(0)
            .expect("sq")
            .to_vec2()
            .expect("v2")
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>(),
        &read_npy("weights/pt_cn0_output.npy"),
    );

    // -------------------------------------------------------------------------
    // 5. Decoder Blocks (start conv + 4 blocks + snake + final conv)
    // -------------------------------------------------------------------------
    header("5. Decoder Blocks (start conv + 4 blocks + snake + final conv)");

    // Build decoder start conv
    let (sw, sb) = loader.conv1d_pair("0.conv").expect("decoder start conv");
    let decoder_start = candle_nn::Conv1d::new(
        sw,
        sb,
        candle_nn::Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        },
    );

    // Build decoder blocks
    let mut decoder_blocks = Vec::new();
    for i in 1..=4 {
        decoder_blocks
            .push(DecoderBlock::from_loader(&loader, &format!("{i}")).expect("decoder block"));
    }

    // Build final conv
    let (fw, fb) = loader.conv1d_pair("6.conv").expect("final conv");
    let final_conv = candle_nn::Conv1d::new(
        fw,
        fb,
        candle_nn::Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        },
    );

    // SnakeBeta params
    let fs_a = loader.get("5.alpha").expect("5.alpha").clone();
    let fs_b = loader.get("5.beta").expect("5.beta").clone();

    // Run decoder blocks
    h = decoder_start.forward(&h).expect("decoder start");
    println!("  After decoder.0 (start conv): {:?}", h.dims());

    for (i, db) in decoder_blocks.iter().enumerate() {
        h = db.forward(&h).expect(&format!("decoder block {}", i + 1));
        println!("  After decoder.{}: {:?}", i + 1, h.dims());
    }

    // SnakeBeta + final conv
    h = qwen3tts::codec::snake_beta(&h, &fs_a, &fs_b).expect("snake beta");
    h = final_conv.forward(&h).expect("final conv");

    println!("  After final conv: {:?}", h.dims());
    // Expected: [1, 1, 5760] (for 3 frames)
    let rust_final_v: Vec<f32> = h
        .squeeze(0)
        .expect("squeeze")
        .squeeze(0)
        .expect("squeeze")
        .to_vec1()
        .expect("vec1");

    let pt_final = read_npy("weights/pt_decoder_out.npy"); // [1, 5760]
    compare("FINAL output (full 3 frames)", &rust_final_v, &pt_final);

    // Also compare against output.npy from refs/
    let pt_output_refs = read_npy("weights/refs/output.npy");
    compare("FINAL vs refs/output.npy", &rust_final_v, &pt_output_refs);

    // -------------------------------------------------------------------------
    // 6. Per-decoder-block comparison
    // -------------------------------------------------------------------------
    header("6. Per-Decoder-Block comparison");

    // Re-run decoder blocks, saving each intermediate
    h = rust_pre_trans.clone();
    for ub in &upsample_blocks {
        h = ub.forward(&h).expect("upsample");
    }
    h = decoder_start.forward(&h).expect("decoder start");

    for block_idx in 0..=6 {
        let ref_path = format!("weights/pt_decoder_block.{block_idx}.npy");
        if !Path::new(&ref_path).exists() {
            println!("  Skipping block {block_idx}: ref not found");
            continue;
        }
        let pt_block = read_npy(&ref_path);

        match block_idx {
            0 => {
                // decoder.0: start conv - already computed
                let rust_v: Vec<f32> = h
                    .clone()
                    .squeeze(0)
                    .expect("squeeze")
                    .to_vec2()
                    .expect("vec2")
                    .into_iter()
                    .flatten()
                    .collect();
                compare(&format!("Decoder.{block_idx}"), &rust_v, &pt_block);
            }
            1..=4 => {
                h = decoder_blocks[block_idx - 1]
                    .forward(&h)
                    .expect("decoder block");
                let rust_v: Vec<f32> = h
                    .clone()
                    .squeeze(0)
                    .expect("squeeze")
                    .to_vec2()
                    .expect("vec2")
                    .into_iter()
                    .flatten()
                    .collect();
                compare(&format!("Decoder.{block_idx}"), &rust_v, &pt_block);
            }
            5 => {
                // SnakeBeta
                h = qwen3tts::codec::snake_beta(&h, &fs_a, &fs_b).expect("snake beta");
                let rust_v: Vec<f32> = h
                    .clone()
                    .squeeze(0)
                    .expect("squeeze")
                    .to_vec2()
                    .expect("vec2")
                    .into_iter()
                    .flatten()
                    .collect();
                compare(&format!("Decoder.{block_idx} (snake)"), &rust_v, &pt_block);
            }
            6 => {
                // Final conv
                h = final_conv.forward(&h).expect("final conv");
                let rust_v: Vec<f32> = h
                    .squeeze(0)
                    .expect("squeeze")
                    .squeeze(0)
                    .expect("squeeze")
                    .to_vec1()
                    .expect("vec1");
                compare(&format!("Decoder.{block_idx} (final)"), &rust_v, &pt_block);
            }
            _ => unreachable!(),
        }
    }

    // -------------------------------------------------------------------------
    // 7. Full pipeline summary (streaming mode)
    // -------------------------------------------------------------------------
    header("7. FULL PIPELINE SUMMARY (Streaming Mode)");

    let mut streaming_decoder =
        Decoder12Hz::from_safetensors(DecoderConfig::realtime(), weight_dir, &device)
            .expect("build decoder");
    streaming_decoder.reset_state();

    let mut all_pcm = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        let pcm = streaming_decoder.decode_chunk(frame).expect("decode frame");
        let rng = range_str(&pcm);
        println!("  Frame {i}: len={}, range={rng}", pcm.len());
        all_pcm.extend_from_slice(&pcm);
    }

    println!("\n  Total PCM samples: {}", all_pcm.len());
    compare("Streaming full output", &all_pcm, &pt_final);

    // -------------------------------------------------------------------------
    // 8. Summary: what changed from which layer
    // -------------------------------------------------------------------------
    header("8. DIAGNOSTIC SUMMARY");

    // Compute amplitude ratio at each stage
    let mut layers: Vec<(&str, Vec<f32>, Vec<f32>)> = Vec::new();

    // Codebook
    layers.push(("codebook", rust_corder.clone(), pt_quant));
    // Pre-conv
    layers.push(("pre_conv", rust_pre_conv_v.clone(), pt_pre_conv));
    // Pre-transformer
    layers.push(("pre_transformer", rust_pre_trans_v.clone(), pt_pre_trans));
    // Upsample
    layers.push(("upsample", rust_upsample_v.clone(), pt_upsample));
    // Final
    layers.push(("final", rust_final_v.clone(), pt_final));

    for (name, rust_v, pt_v) in &layers {
        let ratio = amplitude_ratio(rust_v, pt_v);
        let cos = cosine_sim_f32(rust_v, pt_v);
        let mse = mse_f32(rust_v, pt_v);
        let marker = if cos < 0.99 {
            " ⚠️ DIVERGED"
        } else if cos < 0.999 {
            " ⚠️ LOW"
        } else {
            " ✅ OK"
        };
        println!("  {name:<20} | amp_ratio={ratio:8.4}x | cos={cos:.8} | mse={mse:.10}{marker}");
    }

    println!("\n✅ Per-layer comparison done!");
}
