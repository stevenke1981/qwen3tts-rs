//! # GGUF Talker 權重探測器
//!
//! 讀取 qwentts.cpp 發布的 GGUF talker 權重檔案，列出所有 tensor 名稱、shape、dtype，
//! 比對預期 tensor 清單，並測試 dequantize 管線。
//!
//! ## 用法
//! ```bash
//! cargo run --example gguf_talker_probe -- <path/to/model.gguf>
//! ```
//!
//! ## 背景
//! 此程式為 WO-4 的一部分，目的：
//! - 驗證 candle-core 內建 gguf_file 模組能否正常讀取 qwentts.cpp 的 GGUF 檔案
//! - 建立 tensor 命名對照表（GGUF ↔ safetensors）供 WO-5 正式整合使用

use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

use candle_core::Device;
use candle_core::quantized::gguf_file;

/// 預期在 GGUF 中應出現的 tensor 名稱（選取代表性樣本，基於真實 GGUF 實測）
///
/// 完整模式應涵蓋：
/// - talker: blk.0 到 blk.27（28 層 × 11 tensor）
/// - code_pred: blk.0 到 blk.4（5 層 × 11 tensor）
/// - 頂層 tensor + 子碼本嵌入 (15 × 2)
const EXPECTED_KEYS: &[&str] = &[
    // ── 頂層 Talker ──
    "talker.text_embd.weight",
    "talker.codec_embd.weight",
    "talker.text_proj.fc1.weight",
    "talker.text_proj.fc1.bias",
    "talker.text_proj.fc2.weight",
    "talker.text_proj.fc2.bias",
    "talker.codec_head.weight",
    "talker.output_norm.weight",
    // ── Talker 第 0 層（11 種 tensor，驗證全部類型）──
    "talker.blk.0.attn_q.weight",
    "talker.blk.0.attn_k.weight",
    "talker.blk.0.attn_v.weight",
    "talker.blk.0.attn_output.weight",
    "talker.blk.0.attn_q_norm.weight",
    "talker.blk.0.attn_k_norm.weight",
    "talker.blk.0.attn_norm.weight",
    "talker.blk.0.ffn_norm.weight",
    "talker.blk.0.ffn_gate.weight",
    "talker.blk.0.ffn_up.weight",
    "talker.blk.0.ffn_down.weight",
    // ── Talker 第 1 層（確認多層存在）──
    "talker.blk.1.attn_q.weight",
    "talker.blk.1.attn_k.weight",
    "talker.blk.1.attn_output.weight",
    "talker.blk.1.attn_norm.weight",
    "talker.blk.1.ffn_gate.weight",
    "talker.blk.1.ffn_down.weight",
    // ── Talker 最後一層（確認 28 層完整）──
    "talker.blk.27.attn_q.weight",
    "talker.blk.27.attn_output.weight",
    "talker.blk.27.attn_norm.weight",
    "talker.blk.27.ffn_gate.weight",
    // ── Code Predictor 第 0 層（11 種 tensor）──
    "code_pred.blk.0.attn_q.weight",
    "code_pred.blk.0.attn_k.weight",
    "code_pred.blk.0.attn_v.weight",
    "code_pred.blk.0.attn_output.weight",
    "code_pred.blk.0.attn_q_norm.weight",
    "code_pred.blk.0.attn_k_norm.weight",
    "code_pred.blk.0.attn_norm.weight",
    "code_pred.blk.0.ffn_norm.weight",
    "code_pred.blk.0.ffn_gate.weight",
    "code_pred.blk.0.ffn_up.weight",
    "code_pred.blk.0.ffn_down.weight",
    // ── Code Predictor 頂層 ──
    "code_pred.output_norm.weight",
    // ── 子碼本嵌入（15 組）──
    "code_pred.codec_embd.0.weight",
    "code_pred.codec_embd.14.weight",
    "code_pred.lm_head.0.weight",
    "code_pred.lm_head.14.weight",
];

/// 將 GGUF 的 tensor 命名對照到現有 safetensors 命名慣例的說明
/// （完整對照表請見 docs/gguf_tensor_mapping.md）
#[allow(dead_code)]
const GGUF_TO_SAFETENSORS_MAP: &[(&str, &str, &str)] = &[
    // (GGUF key, safetensors key, 備註)
    (
        "talker.text_embd.weight",
        "talker.model.text_embedding.weight",
        "文字嵌入",
    ),
    (
        "talker.codec_embd.weight",
        "talker.model.codec_embedding.weight",
        "碼本 0 嵌入",
    ),
    (
        "talker.text_proj.fc1.{weight,bias}",
        "talker.text_projection.linear_fc1.{weight,bias}",
        "文字投影 fc1",
    ),
    (
        "talker.text_proj.fc2.{weight,bias}",
        "talker.text_projection.linear_fc2.{weight,bias}",
        "文字投影 fc2",
    ),
    (
        "talker.codec_head.weight",
        "talker.codec_head.weight",
        "碼本輸出頭（名稱相同）",
    ),
    (
        "talker.output_norm.weight",
        "talker.model.norm.weight",
        "最終層 RMSNorm",
    ),
    (
        "talker.blk.{i}.attn_q.weight",
        "talker.model.layers.{i}.self_attn.q_proj.weight",
        "Q 投影",
    ),
    (
        "talker.blk.{i}.attn_k.weight",
        "talker.model.layers.{i}.self_attn.k_proj.weight",
        "K 投影",
    ),
    (
        "talker.blk.{i}.attn_v.weight",
        "talker.model.layers.{i}.self_attn.v_proj.weight",
        "V 投影",
    ),
    (
        "talker.blk.{i}.attn_o.weight",
        "talker.model.layers.{i}.self_attn.o_proj.weight",
        "O 投影",
    ),
    (
        "talker.blk.{i}.attn_q.q_norm",
        "talker.model.layers.{i}.self_attn.q_norm.weight",
        "QK-Norm (Q)",
    ),
    (
        "talker.blk.{i}.attn_k.k_norm",
        "talker.model.layers.{i}.self_attn.k_norm.weight",
        "QK-Norm (K)",
    ),
    (
        "talker.blk.{i}.attn_norm",
        "talker.model.layers.{i}.input_layernorm.weight",
        "輸入層 RMSNorm",
    ),
    (
        "talker.blk.{i}.ffn_norm",
        "talker.model.layers.{i}.post_attention_layernorm.weight",
        "FFN 前 RMSNorm",
    ),
    (
        "talker.blk.{i}.ffn.gate_proj.weight",
        "talker.model.layers.{i}.mlp.gate_proj.weight",
        "SwiGLU Gate",
    ),
    (
        "talker.blk.{i}.ffn.up_proj.weight",
        "talker.model.layers.{i}.mlp.up_proj.weight",
        "SwiGLU Up",
    ),
    (
        "talker.blk.{i}.ffn.down_proj.weight",
        "talker.model.layers.{i}.mlp.down_proj.weight",
        "SwiGLU Down",
    ),
    (
        "code_pred.blk.{i}.attn_q.weight",
        "talker.code_predictor.model.layers.{i}.self_attn.q_proj.weight",
        "Code Predictor Q",
    ),
    (
        "code_pred.blk.{i}.attn_k.weight",
        "talker.code_predictor.model.layers.{i}.self_attn.k_proj.weight",
        "Code Predictor K",
    ),
    (
        "code_pred.blk.{i}.attn_v.weight",
        "talker.code_predictor.model.layers.{i}.self_attn.v_proj.weight",
        "Code Predictor V",
    ),
    (
        "code_pred.blk.{i}.attn_o.weight",
        "talker.code_predictor.model.layers.{i}.self_attn.o_proj.weight",
        "Code Predictor O",
    ),
    (
        "code_pred.blk.{i}.attn_q.q_norm",
        "talker.code_predictor.model.layers.{i}.self_attn.q_norm.weight",
        "CP QK-Norm (Q)",
    ),
    (
        "code_pred.blk.{i}.attn_k.k_norm",
        "talker.code_predictor.model.layers.{i}.self_attn.k_norm.weight",
        "CP QK-Norm (K)",
    ),
    (
        "code_pred.blk.{i}.attn_norm",
        "talker.code_predictor.model.layers.{i}.input_layernorm.weight",
        "CP 輸入層 RMSNorm",
    ),
    (
        "code_pred.blk.{i}.ffn_norm",
        "talker.code_predictor.model.layers.{i}.post_attention_layernorm.weight",
        "CP FFN 前 RMSNorm",
    ),
    (
        "code_pred.blk.{i}.ffn.gate_proj.weight",
        "talker.code_predictor.model.layers.{i}.mlp.gate_proj.weight",
        "CP SwiGLU Gate",
    ),
    (
        "code_pred.blk.{i}.ffn.up_proj.weight",
        "talker.code_predictor.model.layers.{i}.mlp.up_proj.weight",
        "CP SwiGLU Up",
    ),
    (
        "code_pred.blk.{i}.ffn.down_proj.weight",
        "talker.code_predictor.model.layers.{i}.mlp.down_proj.weight",
        "CP SwiGLU Down",
    ),
    (
        "code_pred.output_norm.weight",
        "talker.code_predictor.model.norm.weight",
        "CP 最終 RMSNorm",
    ),
    (
        "code_pred.mtp_proj.weight",
        "talker.code_predictor.small_to_mtp_projection.weight",
        "MTP 投影（僅 1.7B）",
    ),
    (
        "code_pred.mtp_proj.bias",
        "talker.code_predictor.small_to_mtp_projection.bias",
        "MTP 投影 bias（僅 1.7B）",
    ),
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("用法: cargo run --example gguf_talker_probe -- <GGUF 檔案路徑>");
        std::process::exit(1);
    }
    let path = &args[1];
    let path_ref = Path::new(path);
    if !path_ref.exists() {
        eprintln!("錯誤: 檔案不存在: {path}");
        std::process::exit(1);
    }

    println!("=== GGUF Talker 權重探測器 ===");
    println!("檔案: {path}");
    println!();

    // ── 開啟 GGUF ──
    let mut file = match File::open(path_ref) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("錯誤: 無法開啟檔案: {e}");
            std::process::exit(1);
        }
    };

    let content = match gguf_file::Content::read(&mut file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("錯誤: 無法讀取 GGUF 內容: {e}");
            std::process::exit(1);
        }
    };

    // ── 印出 metadata ──
    println!("--- Metadata ---");
    println!("  tensor_data_offset: {}", content.tensor_data_offset);
    println!("  metadata 項目數: {}", content.metadata.len());
    for (key, value) in &content.metadata {
        let value_str = match value {
            candle_core::quantized::gguf_file::Value::String(s) => s.clone(),
            candle_core::quantized::gguf_file::Value::U64(v) => v.to_string(),
            candle_core::quantized::gguf_file::Value::I64(v) => v.to_string(),
            candle_core::quantized::gguf_file::Value::F32(v) => format!("{v:.6}"),
            candle_core::quantized::gguf_file::Value::F64(v) => format!("{v:.6}"),
            candle_core::quantized::gguf_file::Value::U32(v) => v.to_string(),
            candle_core::quantized::gguf_file::Value::I32(v) => v.to_string(),
            candle_core::quantized::gguf_file::Value::U16(v) => v.to_string(),
            candle_core::quantized::gguf_file::Value::I16(v) => v.to_string(),
            candle_core::quantized::gguf_file::Value::U8(v) => v.to_string(),
            candle_core::quantized::gguf_file::Value::I8(v) => v.to_string(),
            candle_core::quantized::gguf_file::Value::Bool(v) => v.to_string(),
            candle_core::quantized::gguf_file::Value::Array(arr) => {
                format!("[{} 項目]", arr.len())
            }
        };
        println!("  {key}: {value_str}");
    }
    println!();

    // ── 列出所有 tensor ──
    println!("--- Tensor 清單 ---");
    println!("  tensor 總數: {}", content.tensor_infos.len());
    println!();

    // 按字母排序輸出
    let mut sorted_tensors: Vec<(&String, &candle_core::quantized::gguf_file::TensorInfo)> =
        content.tensor_infos.iter().collect();
    sorted_tensors.sort_by(|a, b| a.0.cmp(b.0));

    for (name, info) in &sorted_tensors {
        println!(
            "  {:<55} shape={:<20} dtype={:?}",
            name,
            format!("{:?}", info.shape.dims()),
            info.ggml_dtype,
        );
    }
    println!();

    // ── 比對預期 tensor 清單 ──
    println!("--- 預期 tensor 比對 ---");
    let actual_keys: HashSet<&str> = content.tensor_infos.keys().map(|s| s.as_str()).collect();

    let mut found_count = 0u32;
    let mut missing_count = 0u32;
    for expected in EXPECTED_KEYS {
        if actual_keys.contains(expected) {
            println!("  ✅ {expected}");
            found_count += 1;
        } else {
            println!("  ❌ {expected}  — 找不到");
            missing_count += 1;
        }
    }
    println!();
    println!("  比對結果: {found_count} 找到, {missing_count} 缺少");
    println!();

    // ── 推斷 talker 層數與 code_pred 層數 ──
    println!("--- 推斷模型結構 ---");
    let talker_layers: Vec<&str> = actual_keys
        .iter()
        .filter(|k| k.starts_with("talker.blk.") && k.ends_with(".attn_q.weight"))
        .map(|k| *k)
        .collect();
    let cp_layers: Vec<&str> = actual_keys
        .iter()
        .filter(|k| k.starts_with("code_pred.blk.") && k.ends_with(".attn_q.weight"))
        .map(|k| *k)
        .collect();
    println!("  Talker 層數: {}", talker_layers.len());
    println!("  Code Predictor 層數: {}", cp_layers.len());
    println!();

    // ── 挑選 talker.text_embd.weight 做 dequantize 測試 ──
    println!("--- Dequantize 測試 ---");
    let test_tensor_name = "talker.text_embd.weight";
    if let Some(info) = content.tensor_infos.get(test_tensor_name) {
        println!("  tensor: {test_tensor_name}");
        println!("  shape: {:?}", info.shape.dims());
        println!("  dtype: {:?}", info.ggml_dtype);

        match info.read(&mut file, content.tensor_data_offset, &Device::Cpu) {
            Ok(qtensor) => {
                println!("  QTensor shape: {:?}", qtensor.shape().dims());
                match qtensor.dequantize(&Device::Cpu) {
                    Ok(f32_tensor) => {
                        let f32_shape: Vec<usize> = f32_tensor.shape().dims().to_vec();
                        println!("  F32 dequantized shape: {f32_shape:?}");
                        let f32_flat: Vec<f32> =
                            f32_tensor.flatten_all().unwrap().to_vec1().unwrap();
                        let n = f32_flat.len();
                        println!("  總元素數: {n}");
                        println!("  前十個值: {:.6?}", &f32_flat[..10.min(n)]);
                        // 數值合理性檢查
                        let all_zero = f32_flat.iter().all(|&v| v.abs() < 1e-10);
                        let has_nan = f32_flat.iter().any(|v| v.is_nan());
                        if all_zero {
                            println!("  ⚠️  警告: 所有數值皆為零（可能讀取管線有問題）");
                        } else if has_nan {
                            println!("  ⚠️  警告: 包含 NaN（可能讀取管線有問題）");
                        } else {
                            println!("  ✅ 數值正常（非全零、非 NaN）");
                        }
                    }
                    Err(e) => {
                        eprintln!("  ❌ dequantize 失敗: {e}");
                    }
                }
            }
            Err(e) => {
                eprintln!("  ❌ 無法讀取 tensor: {e}");
            }
        }
    } else {
        println!("  ⚠️  tensor {test_tensor_name} 不在 GGUF 檔案中");
        // 嘗試用其他 tensor
        if let Some((fallback_name, _)) = content.tensor_infos.iter().next() {
            println!("  改用第一個可用 tensor: {fallback_name}");
            // （此處可延伸，但此為 probe，點到為止）
        }
    }
    println!();

    // ── Shape 轉置說明 ──
    println!("--- Shape 慣例對照 ---");
    println!("  candle-core 的 gguf_file::Content::read() 會自動 reverse GGUF 的維度順序，");
    println!("  因此讀取後的 shape 已符合 Candle/safetensors 慣例，不需額外轉置。");
    println!("  上述印出的 shape 直接對應到 WeightLoader 從 safetensors 讀到的 shape。");
    println!();

    // ── 匯總 ──
    println!("=== 探測完成 ===");
}
