# GGUF ↔ Safetensors Tensor 命名對照表

> 更新日期：2026-07-26
> 驗證方式：使用 `Serveurperso/Qwen3-TTS-GGUF qwen-talker-0.6b-base-Q4_K_M.gguf` 實測

## 背景

`qwentts.cpp` 專案發布的 GGUF talker 權重（HuggingFace `Serveurperso/Qwen3-TTS-GGUF`）
使用 GGUF/llama.cpp 生態的 tensor 命名慣例，與現有 `TalkerWeightLoader`
（讀取 HuggingFace `model.safetensors`）的鍵名不同。

此文件記錄兩者之間的對照關係，為 `gguf_key_to_safetensors()` 的實作依據。

---

## Shape 慣例

| 來源 | 維度順序 | 說明 |
|------|---------|------|
| GGUF 檔案內部 | `(out_features, in_features)` | llama.cpp 慣例 |
| Candle 讀取後 (`QTensor.shape`) | `(in_features, out_features)` | candle-core `gguf_file::Content::read()` 會自動 reverse 維度 |
| Safetensors (`model.safetensors`) | `(out_features, in_features)` | PyTorch 慣例 |

**結論：** 使用 `content.tensor()` 讀取 GGUF 後，`QTensor` 的 shape 已經 reverse。
dequantize 成 F32 `Tensor` 後的 shape 應與從 safetensors 讀到的相同，
**不需要額外轉置**。

---

## 完整命名對照表

### Talker 主模型每層（28 層，`i` = 0..27）

| # | GGUF 鍵名 | Safetensors 鍵名 | 說明 |
|---|-----------|-----------------|------|
| 1 | `talker.blk.{i}.attn_q.weight` | `talker.model.layers.{i}.self_attn.q_proj.weight` | Q 投影 |
| 2 | `talker.blk.{i}.attn_k.weight` | `talker.model.layers.{i}.self_attn.k_proj.weight` | K 投影 |
| 3 | `talker.blk.{i}.attn_v.weight` | `talker.model.layers.{i}.self_attn.v_proj.weight` | V 投影 |
| 4 | `talker.blk.{i}.attn_output.weight` | `talker.model.layers.{i}.self_attn.o_proj.weight` | O 投影 |
| 5 | `talker.blk.{i}.attn_q_norm.weight` | `talker.model.layers.{i}.self_attn.q_norm.weight` | QK-Norm（Q） |
| 6 | `talker.blk.{i}.attn_k_norm.weight` | `talker.model.layers.{i}.self_attn.k_norm.weight` | QK-Norm（K） |
| 7 | `talker.blk.{i}.attn_norm.weight` | `talker.model.layers.{i}.input_layernorm.weight` | 輸入層 RMSNorm |
| 8 | `talker.blk.{i}.ffn_norm.weight` | `talker.model.layers.{i}.post_attention_layernorm.weight` | FFN 前 RMSNorm |
| 9 | `talker.blk.{i}.ffn_gate.weight` | `talker.model.layers.{i}.mlp.gate_proj.weight` | SwiGLU Gate |
| 10 | `talker.blk.{i}.ffn_up.weight` | `talker.model.layers.{i}.mlp.up_proj.weight` | SwiGLU Up |
| 11 | `talker.blk.{i}.ffn_down.weight` | `talker.model.layers.{i}.mlp.down_proj.weight` | SwiGLU Down |

### Code Predictor 每層（5 層，`i` = 0..4）

| # | GGUF 鍵名 | Safetensors 鍵名 | 說明 |
|---|-----------|-----------------|------|
| 12 | `code_pred.blk.{i}.attn_q.weight` | `talker.code_predictor.model.layers.{i}.self_attn.q_proj.weight` | CP Q 投影 |
| 13 | `code_pred.blk.{i}.attn_k.weight` | `talker.code_predictor.model.layers.{i}.self_attn.k_proj.weight` | CP K 投影 |
| 14 | `code_pred.blk.{i}.attn_v.weight` | `talker.code_predictor.model.layers.{i}.self_attn.v_proj.weight` | CP V 投影 |
| 15 | `code_pred.blk.{i}.attn_output.weight` | `talker.code_predictor.model.layers.{i}.self_attn.o_proj.weight` | CP O 投影 |
| 16 | `code_pred.blk.{i}.attn_q_norm.weight` | `talker.code_predictor.model.layers.{i}.self_attn.q_norm.weight` | CP QK-Norm（Q） |
| 17 | `code_pred.blk.{i}.attn_k_norm.weight` | `talker.code_predictor.model.layers.{i}.self_attn.k_norm.weight` | CP QK-Norm（K） |
| 18 | `code_pred.blk.{i}.attn_norm.weight` | `talker.code_predictor.model.layers.{i}.input_layernorm.weight` | CP 輸入層 RMSNorm |
| 19 | `code_pred.blk.{i}.ffn_norm.weight` | `talker.code_predictor.model.layers.{i}.post_attention_layernorm.weight` | CP FFN 前 RMSNorm |
| 20 | `code_pred.blk.{i}.ffn_gate.weight` | `talker.code_predictor.model.layers.{i}.mlp.gate_proj.weight` | CP SwiGLU Gate |
| 21 | `code_pred.blk.{i}.ffn_up.weight` | `talker.code_predictor.model.layers.{i}.mlp.up_proj.weight` | CP SwiGLU Up |
| 22 | `code_pred.blk.{i}.ffn_down.weight` | `talker.code_predictor.model.layers.{i}.mlp.down_proj.weight` | CP SwiGLU Down |

### 頂層 Tensor

| # | GGUF 鍵名 | Safetensors 鍵名 | Shape（0.6B） | 說明 |
|---|-----------|-----------------|--------------|------|
| 23 | `talker.text_embd.weight` | `talker.model.text_embedding.weight` | `[151936, 2048]` | 文字嵌入表 |
| 24 | `talker.codec_embd.weight` | `talker.model.codec_embedding.weight` | `[3072, 1024]` | 碼本 0 嵌入表（合併詞彙） |
| 25 | `talker.text_proj.fc1.weight` | `talker.text_projection.linear_fc1.weight` | `[2048, 2048]` | 文字投影 fc1 |
| 26 | `talker.text_proj.fc1.bias` | `talker.text_projection.linear_fc1.bias` | `[2048]` | |
| 27 | `talker.text_proj.fc2.weight` | `talker.text_projection.linear_fc2.weight` | `[1024, 2048]` | 文字投影 fc2 |
| 28 | `talker.text_proj.fc2.bias` | `talker.text_projection.linear_fc2.bias` | `[1024]` | |
| 29 | `talker.codec_head.weight` | `talker.codec_head.weight` | `[3072, 1024]` | 碼本輸出頭（名稱相同） |
| 30 | `talker.output_norm.weight` | `talker.model.norm.weight` | `[1024]` | 最終層 RMSNorm |
| 31 | `code_pred.output_norm.weight` | `talker.code_predictor.model.norm.weight` | `[1024]` | CP 最終 RMSNorm |

### Code Predictor 子碼本嵌入（15 組，`i` = 0..14）

| # | GGUF 鍵名 | Safetensors 鍵名 | Shape（0.6B） | 說明 |
|---|-----------|-----------------|--------------|------|
| 32 | `code_pred.codec_embd.{i}.weight` | `talker.code_predictor.model.codec_embedding.{i}.weight` | `[2048, 1024]` | 子碼本嵌入表（注意：`_embd` → `_embedding` + 含 `.model.`） |
| 33 | `code_pred.lm_head.{i}.weight` | `talker.code_predictor.lm_head.{i}.weight` | `[2048, 1024]` | 子碼本輸出頭（注意：**不含** `.model.`） |

### 1.7B 限定

| # | GGUF 鍵名 | Safetensors 鍵名 | 說明 |
|---|-----------|-----------------|------|
| 34 | `code_pred.mtp_proj.weight` | `talker.code_predictor.small_to_mtp_projection.weight` | 僅 1.7B |
| 35 | `code_pred.mtp_proj.bias` | `talker.code_predictor.small_to_mtp_projection.bias` | 僅 1.7B |

### 不作處理的 Tensor

以下 GGUF 內的 tensor 在 `gguf_key_to_safetensors` 中直接 pass-through（不影響 talker 載入）：

- `spk_enc.*` — Speaker Encoder 權重（僅 Base 模型有）

---

## 命名差異摘要

GGUF 使用簡化、扁平化的命名規則，與 safetensors 的命名對照如下：

| 規則 | GGUF 範例 | Safetensors 對應 |
|------|----------|-----------------|
| 每層 11 tensor 固定模式 | `talker.blk.{i}.attn_q.weight` | `talker.model.layers.{i}.self_attn.q_proj.weight` |
| O 投影用全名 | `attn_output.weight` | `self_attn.o_proj.weight` |
| QK-Norm 用雙名 | `attn_q_norm.weight` | `self_attn.q_norm.weight` |
| Norm 必有 `.weight` | `attn_norm.weight` | `input_layernorm.weight` |
| FFN 用三字母 | `ffn_gate.weight` | `mlp.gate_proj.weight` |
| 嵌入表簡寫 | `codec_embd` | `codec_embedding` |
| 路徑扁平化 | `code_pred.codec_embd.{i}.weight` | `talker.code_predictor.model.codec_embedding.{i}.weight` |
| 路徑扁平化 | `code_pred.lm_head.{i}.weight` | `talker.code_predictor.lm_head.{i}.weight` |

---

## 實作與驗證狀態

### 實作位置

- `gguf_key_to_safetensors()` 函式：`src/talker/weight_loader.rs`
- `TalkerWeightLoader::from_gguf()`：同上
- `CandleLLM::from_gguf()`：`src/text_frontend/candle_backend.rs`
- CLI `--talker-backend gguf`：`examples/synthesize.rs`

### 測試

| 測試 | 檔案 | 狀態 |
|------|------|------|
| 9 個單元測試（gguf_key_to_safetensors 覆蓋） | `src/talker/weight_loader.rs` | ✅ 通過 |
| 29 tensor 端到端載入 + shape 驗證 | `tests/gguf_load_real_test.rs` | ✅ 通過 |
| `infer_config` 0.6B 維度正確 | `tests/gguf_load_real_test.rs` | ✅ 通過 |
| `build_talker` 完整建構 | `tests/gguf_load_real_test.rs` | ✅ 通過 |
| 數值對齊（codebook-0 logits cosine ≥ 0.995） | `tests/gguf_safetensors_alignment_test.rs` | ⚠️ Q4_K_M: 0.9903（需 Q8_0 達 0.995） |

### 已知限制

1. **數值對齊測試已執行**（`tests/gguf_safetensors_alignment_test.rs`）：
   - Q4_K_M GGUF vs safetensors codebook-0 logits cosine = **0.9903**
   - **未達 0.995 門檻**（Q4_K_M 為 4-bit 極致壓縮，0.99 屬合理範圍）。
   - 下載 **Q8_0 GGUF** 後應可達 0.995+。
   - Safetensors 路徑輸出已確認與 PyTorch fixture 完全一致（token exact match）。
2. **1.7B 尚未測試** — `mtp_proj` 的映射已在函式中處理，但未實際驗證。
3. **Speaker Encoder 權重** — GGUF 包含 `spk_enc.*` tensor，但當前的
   `TalkerWeightLoader` 不處理這些。實際用到 speaker encoder 時
   （voice clone 功能）可以從 GGUF 直接讀取 speaker encoder 權重，
   無需獨立下載 safetensors 版本。

## 探測指令

```bash
# 列出 GGUF 中所有 tensor 名稱、shape、dtype
cargo run --example gguf_talker_probe -- <path/to/qwen-talker-*.gguf>

# 端到端載入驗證
cargo test --release --test gguf_load_real_test
```
