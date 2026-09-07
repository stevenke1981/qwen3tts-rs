//! # convert-gguf: Qwen3-TTS safetensors → GGUF 轉換工具
//!
//! 將 HuggingFace safetensors 權重轉換為 qwentts.cpp 相容的 GGUF F32 檔案。
//!
//! ## 用法
//!
//! ```bash
//! # 轉換 Talker 模型
//! convert-gguf talker <model-dir> <output.gguf>
//!
//! # 範例
//! convert-gguf talker ~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/<sha> qwen-talker-0.6b-base-F32.gguf
//! ```

use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};

use safetensors::SafeTensors;

// ---------------------------------------------------------------------------
// BF16 → F32 conversion
// ---------------------------------------------------------------------------

fn bf16_bytes_to_f32_bytes(bf16_data: &[u8]) -> Vec<u8> {
    let count = bf16_data.len() / 2;
    let mut f32_data = Vec::with_capacity(count * 4);
    for i in 0..count {
        let bf16_bits = u16::from_le_bytes([bf16_data[i * 2], bf16_data[i * 2 + 1]]);
        // BF16 → F32: shift left by 16 bits (BF16 is the upper 16 bits of F32)
        let f32_bits = (bf16_bits as u32) << 16;
        f32_data.extend_from_slice(&f32_bits.to_le_bytes());
    }
    f32_data
}

// ---------------------------------------------------------------------------
// GGUF constants
// ---------------------------------------------------------------------------

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const GGML_TYPE_F32: u32 = 0;
const DEFAULT_ALIGNMENT: usize = 32;

// GGUF metadata value types
const GGUF_META_UINT32: u32 = 4;
const GGUF_META_FLOAT32: u32 = 6;
const GGUF_META_BOOL: u32 = 7;
const GGUF_META_STRING: u32 = 8;
const GGUF_META_ARRAY: u32 = 9;
const GGUF_META_UINT64: u32 = 10;

// ---------------------------------------------------------------------------
// GGUF writer
// ---------------------------------------------------------------------------

struct GgufWriter {
    meta_kvs: Vec<(String, MetaValue)>,
    tensor_infos: Vec<TensorInfo>,
    tensor_data: Vec<u8>,
    alignment: usize,
}

#[allow(dead_code)]
enum MetaValue {
    Uint32(u32),
    Float32(f32),
    #[allow(dead_code)]
    Bool(bool),
    Str(String),
    #[allow(dead_code)]
    Uint64(u64),
    #[allow(dead_code)]
    ArrayStr(Vec<String>),
    #[allow(dead_code)]
    ArrayUint32(Vec<u32>),
}

struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    offset: u64,
}

impl GgufWriter {
    fn new() -> Self {
        Self {
            meta_kvs: Vec::new(),
            tensor_infos: Vec::new(),
            tensor_data: Vec::new(),
            alignment: DEFAULT_ALIGNMENT,
        }
    }

    fn add_meta(&mut self, key: &str, value: MetaValue) {
        self.meta_kvs.push((key.to_string(), value));
    }

    fn add_tensor(&mut self, name: &str, dims: &[usize], data: &[u8]) {
        // Align tensor data
        let pad = (self.alignment - (self.tensor_data.len() % self.alignment)) % self.alignment;
        self.tensor_data.extend(std::iter::repeat_n(0u8, pad));
        let offset = self.tensor_data.len() as u64;
        self.tensor_data.extend_from_slice(data);
        // GGUF stores dims in reverse (Fortran) order
        let mut gguf_dims: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
        gguf_dims.reverse();
        self.tensor_infos.push(TensorInfo {
            name: name.to_string(),
            dims: gguf_dims,
            offset,
        });
    }

    fn write_to_file(&self, path: &Path) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        // Header
        w.write_all(GGUF_MAGIC)?;
        w.write_all(&GGUF_VERSION.to_le_bytes())?;
        w.write_all(&(self.tensor_infos.len() as u64).to_le_bytes())?;
        w.write_all(&(self.meta_kvs.len() as u64).to_le_bytes())?;

        // Metadata KVs
        for (key, value) in &self.meta_kvs {
            write_gguf_string(&mut w, key)?;
            match value {
                MetaValue::Uint32(v) => {
                    w.write_all(&GGUF_META_UINT32.to_le_bytes())?;
                    w.write_all(&v.to_le_bytes())?;
                }
                MetaValue::Float32(v) => {
                    w.write_all(&GGUF_META_FLOAT32.to_le_bytes())?;
                    w.write_all(&v.to_le_bytes())?;
                }
                MetaValue::Bool(v) => {
                    w.write_all(&GGUF_META_BOOL.to_le_bytes())?;
                    w.write_all(&[*v as u8])?;
                }
                MetaValue::Str(v) => {
                    w.write_all(&GGUF_META_STRING.to_le_bytes())?;
                    write_gguf_string(&mut w, v)?;
                }
                MetaValue::Uint64(v) => {
                    w.write_all(&GGUF_META_UINT64.to_le_bytes())?;
                    w.write_all(&v.to_le_bytes())?;
                }
                MetaValue::ArrayStr(v) => {
                    w.write_all(&GGUF_META_ARRAY.to_le_bytes())?;
                    w.write_all(&GGUF_META_STRING.to_le_bytes())?;
                    w.write_all(&(v.len() as u64).to_le_bytes())?;
                    for s in v {
                        write_gguf_string(&mut w, s)?;
                    }
                }
                MetaValue::ArrayUint32(v) => {
                    w.write_all(&GGUF_META_ARRAY.to_le_bytes())?;
                    w.write_all(&GGUF_META_UINT32.to_le_bytes())?;
                    w.write_all(&(v.len() as u64).to_le_bytes())?;
                    for x in v {
                        w.write_all(&x.to_le_bytes())?;
                    }
                }
            }
        }

        // Tensor infos
        for info in &self.tensor_infos {
            write_gguf_string(&mut w, &info.name)?;
            w.write_all(&(info.dims.len() as u32).to_le_bytes())?;
            for &d in &info.dims {
                w.write_all(&d.to_le_bytes())?;
            }
            w.write_all(&GGML_TYPE_F32.to_le_bytes())?;
            w.write_all(&info.offset.to_le_bytes())?;
        }

        // Pad header to alignment
        let header_size = w.stream_position()? as usize;
        let pad = (self.alignment - (header_size % self.alignment)) % self.alignment;
        w.write_all(&vec![0u8; pad])?;

        // Tensor data
        w.write_all(&self.tensor_data)?;
        w.flush()?;
        Ok(())
    }
}

fn write_gguf_string(w: &mut impl Write, s: &str) -> std::io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor name mapping: safetensors → GGUF (qwentts.cpp convention)
// ---------------------------------------------------------------------------

fn safetensors_key_to_gguf(sf_key: &str) -> Option<String> {
    // Talker decoder layers
    if let Some(rest) = sf_key.strip_prefix("talker.model.layers.") {
        let (idx_str, rest) = rest.split_once('.')?;
        let idx: usize = idx_str.parse().ok()?;
        let gguf_rest = match rest {
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            _ => return None,
        };
        return Some(format!("talker.blk.{idx}.{gguf_rest}"));
    }

    // Code predictor decoder layers
    if let Some(rest) = sf_key.strip_prefix("talker.code_predictor.model.layers.") {
        let (idx_str, rest) = rest.split_once('.')?;
        let idx: usize = idx_str.parse().ok()?;
        let gguf_rest = match rest {
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            _ => return None,
        };
        return Some(format!("code_pred.blk.{idx}.{gguf_rest}"));
    }

    // Code predictor codec embeddings and lm_heads
    if let Some(rest) = sf_key.strip_prefix("talker.code_predictor.model.codec_embedding.") {
        let (idx_str, rest) = rest.split_once('.')?;
        if rest == "weight" {
            return Some(format!("code_pred.codec_embd.{idx_str}.weight"));
        }
    }
    if let Some(rest) = sf_key.strip_prefix("talker.code_predictor.lm_head.") {
        let (idx_str, rest) = rest.split_once('.')?;
        if rest == "weight" {
            return Some(format!("code_pred.lm_head.{idx_str}.weight"));
        }
    }

    // Top-level talker tensors
    let gguf = match sf_key {
        "talker.model.codec_embedding.weight" => "talker.codec_embd.weight",
        "talker.model.text_embedding.weight" => "talker.text_embd.weight",
        "talker.model.norm.weight" => "talker.output_norm.weight",
        "talker.codec_head.weight" => "talker.codec_head.weight",
        "talker.text_projection.linear_fc1.weight" => "talker.text_proj.fc1.weight",
        "talker.text_projection.linear_fc1.bias" => "talker.text_proj.fc1.bias",
        "talker.text_projection.linear_fc2.weight" => "talker.text_proj.fc2.weight",
        "talker.text_projection.linear_fc2.bias" => "talker.text_proj.fc2.bias",
        // Code predictor top-level
        "talker.code_predictor.model.norm.weight" => "code_pred.output_norm.weight",
        "talker.code_predictor.small_to_mtp_projection.weight" => "code_pred.mtp_proj.weight",
        _ => return None,
    };
    Some(gguf.to_string())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 && matches!(args[1].as_str(), "--help" | "-h") {
        println!("用法: convert-gguf talker <model-dir> <output.gguf>");
        println!("將 Qwen3-TTS Talker + Code Predictor safetensors 權重轉換為 GGUF F32 格式。");
        println!("<model-dir> 必須包含 config.json 與 model.safetensors。");
        return;
    }
    if args.len() < 4 {
        eprintln!("用法: convert-gguf talker <model-dir> <output.gguf>");
        eprintln!();
        eprintln!("將 Qwen3-TTS safetensors 權重轉換為 GGUF F32 格式。");
        eprintln!();
        eprintln!("參數:");
        eprintln!("  talker       轉換 Talker + Code Predictor 權重");
        eprintln!("  <model-dir>  HuggingFace 模型 snapshot 目錄（含 config.json + model.safetensors）");
        eprintln!("  <output.gguf> 輸出 GGUF 檔案路徑");
        std::process::exit(1);
    }

    let mode = &args[1];
    let model_dir = PathBuf::from(&args[2]);
    let output_path = PathBuf::from(&args[3]);

    if mode != "talker" {
        eprintln!("錯誤: 目前僅支持 'talker' 模式");
        std::process::exit(1);
    }

    if let Err(e) = convert_talker(&model_dir, &output_path) {
        eprintln!("錯誤: {e}");
        std::process::exit(1);
    }
}

fn convert_talker(model_dir: &Path, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Read config.json
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("無法讀取 config.json: {e}"))?;
    let config: serde_json::Value = serde_json::from_str(&config_str)?;

    let talker_config = &config["talker_config"];
    let hidden_size = talker_config["hidden_size"].as_u64().unwrap_or(1024) as u32;
    let num_layers = talker_config["num_hidden_layers"].as_u64().unwrap_or(28) as u32;
    let num_heads = talker_config["num_attention_heads"].as_u64().unwrap_or(16) as u32;
    let num_kv_heads = talker_config["num_key_value_heads"].as_u64().unwrap_or(8) as u32;
    let head_dim = talker_config["head_dim"].as_u64().unwrap_or(128) as u32;
    let intermediate_size = talker_config["intermediate_size"].as_u64().unwrap_or(3072) as u32;
    let vocab_size = talker_config["vocab_size"].as_u64().unwrap_or(3072) as u32;
    let text_vocab_size = talker_config["text_vocab_size"].as_u64().unwrap_or(151936) as u32;

    let cp_config = &talker_config["code_predictor_config"];
    let cp_hidden = cp_config["hidden_size"].as_u64().unwrap_or(1024) as u32;
    let cp_layers = cp_config["num_hidden_layers"].as_u64().unwrap_or(5) as u32;
    let cp_heads = cp_config["num_attention_heads"].as_u64().unwrap_or(16) as u32;
    let cp_kv_heads = cp_config["num_key_value_heads"].as_u64().unwrap_or(8) as u32;

    // Read generation_config.json
    let gen_config_path = model_dir.join("generation_config.json");
    let gen_config: Option<serde_json::Value> = std::fs::read_to_string(&gen_config_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());

    // Read model.safetensors
    let sf_path = model_dir.join("model.safetensors");
    let sf_data = std::fs::read(&sf_path)
        .map_err(|e| format!("無法讀取 model.safetensors: {e}"))?;
    let st = SafeTensors::deserialize(&sf_data)
        .map_err(|e| format!("無法解析 safetensors: {e}"))?;

    println!("[Convert] 讀取 {} 個 tensor", st.names().len());

    // Build GGUF
    let mut gguf = GgufWriter::new();

    // Architecture metadata
    gguf.add_meta("general.architecture", MetaValue::Str("qwen3-tts".into()));
    gguf.add_meta("general.name", MetaValue::Str("Qwen3-TTS Talker".into()));
    gguf.add_meta("qwen3-tts.talker.hidden_size", MetaValue::Uint32(hidden_size));
    gguf.add_meta("qwen3-tts.talker.num_layers", MetaValue::Uint32(num_layers));
    gguf.add_meta("qwen3-tts.talker.num_heads", MetaValue::Uint32(num_heads));
    gguf.add_meta("qwen3-tts.talker.num_kv_heads", MetaValue::Uint32(num_kv_heads));
    gguf.add_meta("qwen3-tts.talker.head_dim", MetaValue::Uint32(head_dim));
    gguf.add_meta("qwen3-tts.talker.intermediate_size", MetaValue::Uint32(intermediate_size));
    gguf.add_meta("qwen3-tts.talker.vocab_size", MetaValue::Uint32(vocab_size));
    gguf.add_meta("qwen3-tts.talker.text_vocab_size", MetaValue::Uint32(text_vocab_size));
    gguf.add_meta("qwen3-tts.code_predictor.hidden_size", MetaValue::Uint32(cp_hidden));
    gguf.add_meta("qwen3-tts.code_predictor.num_layers", MetaValue::Uint32(cp_layers));
    gguf.add_meta("qwen3-tts.code_predictor.num_heads", MetaValue::Uint32(cp_heads));
    gguf.add_meta("qwen3-tts.code_predictor.num_kv_heads", MetaValue::Uint32(cp_kv_heads));

    // Generation metadata
    if let Some(gc) = &gen_config {
        if let Some(v) = gc["top_k"].as_u64() {
            gguf.add_meta("generation.top_k", MetaValue::Uint32(v as u32));
        }
        if let Some(v) = gc["top_p"].as_f64() {
            gguf.add_meta("generation.top_p", MetaValue::Float32(v as f32));
        }
        if let Some(v) = gc["temperature"].as_f64() {
            gguf.add_meta("generation.temperature", MetaValue::Float32(v as f32));
        }
        if let Some(v) = gc["repetition_penalty"].as_f64() {
            gguf.add_meta("generation.repetition_penalty", MetaValue::Float32(v as f32));
        }
        if let Some(v) = gc["max_new_tokens"].as_u64() {
            gguf.add_meta("generation.max_new_tokens", MetaValue::Uint32(v as u32));
        }
    }

    // Convert tensors
    let mut converted = 0;
    let mut skipped = 0;
    for name in st.names() {
        let view = st.tensor(name)?;
        let gguf_name = match safetensors_key_to_gguf(name) {
            Some(n) => n,
            None => {
                skipped += 1;
                continue;
            }
        };

        // Get raw bytes and convert BF16 → F32
        let raw_data = view.data();
        let f32_data = bf16_bytes_to_f32_bytes(raw_data);
        let dims: Vec<usize> = view.shape().to_vec();

        gguf.add_tensor(&gguf_name, &dims, &f32_data);
        converted += 1;
    }

    println!("[Convert] 轉換 {converted} 個 tensor，跳過 {skipped} 個");

    // Write GGUF
    gguf.write_to_file(output_path)?;
    let size = std::fs::metadata(output_path)?.len();
    println!(
        "[Convert] 寫入 {} ({:.1} MB)",
        output_path.display(),
        size as f64 / 1_048_576.0
    );

    Ok(())
}
