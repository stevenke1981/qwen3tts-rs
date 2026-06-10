# Qwen3-TTS Rust Rewrite — 優化建議報告

> 基於 v0.1.16 原始碼分析，2026-06-10

---

## 目錄

1. [🔴 嚴重問題](#1-嚴重問題)
2. [🟡 性能瓶頸](#2-性能瓶頸)
3. [🔵 架構建議](#3-架構建議)
4. [🟢 程式碼品質](#4-程式碼品質)
5. [📦 倉庫整理](#5-倉庫整理)
6. [🧪 測試與基準](#6-測試與基準)
7. [📋 優先級矩陣](#7-優先級矩陣)

---

## 1. 🔴 嚴重問題

### 1.1 MTP 解碼器使用假權重 — 永遠輸出零

**檔案**: `src/codec/mtp.rs:86-87`  
**問題**: `MtpDecoder::new()` 使用 `VarBuilder::dummy()`，該方法回傳全零張量
（`quantization/mod.rs:615`）。因此 `MtpDecoder.forward()` 輸出全零 logits，
`decode_step()` 的 argmax 永遠選 index 0。

```rust
// mtp.rs:86
pub fn new(config: MtpConfig, _device: &Device) -> Self {
    let vb = VarBuilder::dummy();  // ← 所有權重都是 zeros!
```

**影響**: 任何使用 MTP 的 12Hz 解碼路徑實際上無法產生有效 Token。

**建議**:
- **短期**: 添加 `MtpDecoder::from_loader()` 方法，從真實 safetensors 載入權重
- **中期**: 從 PyTorch 參考實作導出 MTP 權重，補齊數值對齊測試

---

### 1.2 聲碼器完全未實作 — 端到端管線中斷

**檔案**: `src/vocoder/mod.rs:75-86`  
**問題**: `HifiGanVocoder::load()` 和 `Vocoder::decode()` 都回傳 `Err`。
雖然 spec.md 將聲碼器移植排在 Phase 3，但 v0.1.16 已無端到端可聽輸出。

**影響**: 解碼器輸出後無法轉換為可聽 PCM，無實際 TTS 產出。

**建議**:
- 優先移植輕量聲碼器（若原始碼使用 HiFi-GAN V1，Candle 可完整支援）
- 或先支援 `hound` 直接輸出 raw features 供外部聲碼器調試

---

### 1.3 25Hz 解碼器為空殼

**檔案**: `src/decoder_25hz.rs:75-80`  
**問題**: `decode_chunk()` 直接回傳 `Err("not yet implemented")`。

**影響**: 25Hz 高品質模式完全不可用。

**建議**:
- 標記 `DecoderMode::HighQuality` 在建構時回傳明確錯誤，而非執行時
- 考慮先行移除 25Hz 分支直到實作完成，減少維護負擔

---

### 1.4 Ring Buffer 與 Quantization 使用 unwrap

**檔案**: 
- `src/codec/causal_conv.rs:48` — `assert!(capacity > 0)` 在庫代碼中使用 assert
- `src/quantization/mod.rs:615-616` — `VarBuilder::get()` 使用 `.unwrap()`

**違反規範**: AGENTS.md §3.4：「禁止 `unwrap()` / `expect()` 出現在庫代碼中」

**建議**:
- `RingBuffer::new()` 改回傳 `Result<Self>`
- `VarBuilder::get()` 回傳 `Result<Tensor>` 而非直接 unwrap

---

## 2. 🟡 性能瓶頸

### 2.1 CausalConv1d::step() 每步重建完整歷史

**檔案**: `src/codec/causal_conv.rs:324-357`  
**問題**: `step()` 每次調用都將**所有歷史幀**複製到 `conv_input`，而非僅 kernel_size 幀。

```rust
// 341-344: 每次都拷貝全部 total_frames 幀
for ch in 0..in_channels {
    let history = self.state.get_history(ch, total_frames);  // ← O(total_frames) per channel
    conv_input.extend_from_slice(&history);
}
```

隨著幀數增長，每次 `step()` 的計算量呈 O(n) 成長（n = 幀數），違反了「constant-time per step」的流式推理要求。

**影響**: 流式解碼延遲隨音頻長度線性增加。

**建議**:
- 改為只取最近 `kernel_size` 幀作為 conv1d 輸入（因果卷積不需要更早的歷史）
- 移除 CausalConvState 的多餘歷史累積，讓 conv1d 的 padding 參數處理因果性

---

### 2.2 RingBuffer::last_n() 分配暫存 + get_history() 再分配

**檔案**: `src/codec/causal_conv.rs:224-230, 339-344`  
**問題**: 雙重分配：
1. `get_history()` 內部 `vec![0.0_f32; n]`（line 227）
2. `step()` 再用 `extend_from_slice` 拷貝到 `conv_input`

**影響**: 熱路徑中每幀分配 2 次 Vec。

**建議**:
- `get_history()` 改接受 `&mut [f32]` slice，調用者提供預分配緩衝區
- 或讓 `step()` 直接從 RingBuffer 內部資料構建 Tensor slice，完全避免拷貝

---

### 2.3 speed 插值手動迴圈 vs 張量運算

**檔案**: `src/decoder_12hz.rs:130-158`  
**問題**: 語速調整使用逐元素手動迴圈進行線性插值，而非 Candle 張量 ops：

```rust
for idx in 0..left_embed.len() {
    mixed.push(
        left_embed[idx] * (1.0 - weight as f32)
            + right_embed[idx] * (weight as f32),
    );
}
```

**影響**: CPU 路徑上損失向量化機會，GPU 上無法平行。

**建議**:
- 使用 `Tensor::lerp()` 或直接張量運算來批次內插
- 可考慮將 speed 調整移至 embedding 層之前（操作較小張量）

---

### 2.4 sliding_window mask 每次都重新建立

**檔案**: `src/codec/transformer.rs:217-235`  
**問題**: `apply_sliding_window_mask()` 每次 forward 都新建完整的 `[seq_len, seq_len]` mask：

```rust
let mut mask = Vec::with_capacity(seq_len * seq_len);
for q in 0..seq_len {
    for k in 0..seq_len {
        // ...
    }
}
```

**影響**: 對於 2048 seq_len，每次 forward 分配 4MB+ mask，O(n²) CPU 開銷。

**建議**:
- 對常見 seq_len 快取 mask
- 或使用 Candle 的 sparse mask / attention bias 機制

---

### 2.5 decode_chunk_inner Tensor ↔ Vec 來回轉換

**檔案**: `src/decoder_12hz.rs:193-224`  
**問題**: `decode_chunk_inner()` 將 embedding tensor 轉 `to_vec1()` 再從 slice 重建 tensor：

```rust
let frame_vec: Vec<f32> = frame_embed.to_vec1()?;  // Tensor → Vec
let pre_conv_out = self.pre_conv.step(&frame_vec)?;  // Vec → step
let x = Tensor::from_slice(&pre_conv_out, ...)?;  // Vec → Tensor again
```

**影響**: 不必要的 CPU-GPU 邊界往返。

**建議**:
- 為 `CausalConv1d::step` 添加張量直接輸入重載
- 或讓 `ParallelCodebook::decode()` 直接回傳 `[num_layers, embedding_dim]` 張量供後續處理

---

## 3. 🔵 架構建議

### 3.1 CausalConv1d 與 CausalConvNet 重複

**檔案**: 
- `src/codec/causal_conv.rs:238` — `CausalConv1d`（含 RingBuffer、streaming step）
- `src/codec/decoder_blocks.rs:15` — `CausalConvNet`（批處理 conv1d + padding 計算）

**問題**: 兩個結構體做同一件事（因果卷積），但使用不同的 padding 計算方式：
- `CausalConv1d` 用 Candle 原生 `conv1d(pad=kernel_size-1)`
- `CausalConvNet` 手動計算 `effective_kernel` 並用 `pad_with_zeros`

**建議**: 
- 統一為一個 `CausalConv1d`，將批處理模式（`forward`）和流式模式（`step`）放在同一個 impl 中
- 統一的 padding 邏輯可避免數值差異

---

### 3.2 ParallelCodebook 封裝過薄

**檔案**: `src/codec/codebook.rs:213-239`  
**問題**: `ParallelCodebook` 只做了一件事：呼叫 `batch_lookup()` 後 squeeze 維度 1。

```rust
pub fn decode(&self, tokens: &[u16]) -> crate::Result<Tensor> {
    let stacked = self.inner.batch_lookup(tokens)?;
    stacked.squeeze(1).map_err(Into::into)
}
```

**建議**: 
- 合併到 `CodebookLookup` 中，提供 `batch_lookup_squeezed()` 方法
- 或直接移除 `ParallelCodebook`，減少一層抽象

---

### 3.3 snake_beta 實作重複

**檔案**: 
- `src/codec/decoder_blocks.rs:6` — `snake_beta()`（含 alpha + beta）
- `src/codec/activation.rs:11` — `snake_beta_v2()`（僅 beta，dead_code）

**問題**: 兩個相似函數，一個有 alpha 參數一個沒有，後者標記 `#[allow(dead_code)]` 但從未被使用。

**建議**: 
- 刪除 `activation.rs` 中未使用的 `snake_beta_v2` 和 `SnakeBeta`
- 確認 `snake_beta` 的 alpha/beta 語義與 PyTorch 原版一致

---

### 3.4 Tokenizer 模組為空殼但有真實 tokenizer.json

**檔案**: `src/tokenizer/mod.rs` vs `models/tokenizer.json`  
**問題**: 倉庫中包含 `tokenizers-rs` crate 和真實 `models/tokenizer.json`，但 Rust Tokenizer 模組只是一個 stub，所有方法都回傳 `Err`。

**建議**:
- 實作 `tokenizer.json` 載入器（使用 `tokenizers` crate 或自訂解析）
- 或說明外部 tokenizer 的使用方式

---

### 3.5 WeightLoader 雙職責

**檔案**: `src/weights.rs`  
**問題**: `WeightLoader` 同時負責：
1. 低階 safetensors 解析與 dtype 轉換（`tensor_from_view`, `bf16_values_to_f32` 等）
2. 高階解碼器元件建構（`build_codebook`, `build_causal_conv`, `build_decoder_convs`）

**建議**: 將高階建構方法（build_*）拆分到各元件模組中，`WeightLoader` 只負責權重載入與基本存取。

---

## 4. 🟢 程式碼品質

### 4.1 `decode_frames` vs `decode_chunk_inner` 重複邏輯

**檔案**: `src/decoder_12hz.rs:115-191` vs `193-224`  
**問題**: `decode_frames()`（批處理路徑）和 `decode_chunk_inner()`（流式路徑）有大量重複的 forward 管線：
- pre_conv → pre_transformer → upsample → decoder_blocks → snake_beta → final_conv

**建議**: 提取為私有方法 `forward_pipeline(&self, x: Tensor) -> Result<Tensor>`。

---

### 4.2 `CausalTransConvNet` 未使用

**檔案**: `src/codec/decoder_blocks.rs:65-97`  
**問題**: `CausalTransConvNet` 定義了但僅 `DecoderBlock` 使用其自身對 `ConvTranspose1d` 的包裝，
`CausalTransConvNet` 作為獨立結構體未被其他任何程式碼使用。

**建議**:
- 將其內聯到 `DecoderBlock` 中，或移除獨立定義

---

### 4.3 DiT 模組大量 dead_code

**檔案**: `src/codec/flow_matching.rs`  
**問題**: 979 行中有 12 個結構體/函數標記 `#[allow(dead_code)]`，包括：
- `DiTTimestepEmbedding`, `AdaLayerNormZero`, `AdaLayerNormZeroFinal`, `DiTMLP`
- `DiTAttention`, `DiTDecoderLayer`, `DiTCodecEmbedding`, `DiTInputEmbedding`
- `DiTBackbone`, `FlowMatchingDecoder`

25Hz 路徑雖有完整骨架但皆不可用。

**建議**:
- 整併到 `#[cfg(feature = "dit")]` 閘門後，用 feature flag 控制編譯
- 或標記為 `pub(crate)` 並移除 `#[allow(dead_code)]`

---

### 4.4 decode_frames speed interpolation 邊界 bug

**檔案**: `src/decoder_12hz.rs:138-139`  
**問題**: 當 `new_num_frames == 2` 時：
```rust
let pos = (j as f64) * ((num_frames - 1) as f64) / ((new_num_frames - 1) as f64);
// pos = 0, then pos = (num_frames - 1)，正確
```
但當 `new_num_frames > num_frames`（speed < 1.0，放慢）時，`right = pos.ceil()` 可能等於 `num_frames` 導致 out-of-bounds。

**建議**: 限制 `right = pos.ceil().min((num_frames - 1) as f64) as usize`。

---

## 5. 📦 倉庫整理

### 5.1 根目錄大量暫存檔案

**問題**: 根目錄有 ~90 個 `.wav` 檔案、多個 `.tokens`、`.bin`、`.npy`、`agent_temp*.py`，
與 `.gitignore` 規則衝突（`*.wav`、`*.bin`、`*.npy` 已被忽略但許多已提交）。

**建議**:
- 執行 `git rm --cached` 清理已提交的大檔案
- 建立 `output/` 或 `tmp/` 目錄存放測試產出
- 考慮在 `.gitattributes` 中設定 WAV 檔案的 Git LFS

### 5.2 weights/ 目錄 gitignore 可能遺漏參考資料

**檔案**: `.gitignore` 包含 `/weights`，但 `weights/refs/` 和 `weights/refs2/` 中的 `.bin` 參考輸出
對 CI 數值對齊測試至關重要。

**建議**: 
- 將參考資料移至 `tests/fixtures/` 並取消 gitignore
- 或使用 `!weights/refs/*` 例外規則

### 5.3 Cargo.toml 中 candle-llm feature 無作用

**檔案**: `Cargo.toml:14`  
**問題**:
```toml
candle-llm = []
```
定義了 feature 但沒有啟用任何依賴。

**建議**: 移除未使用的 feature flag。

---

## 6. 🧪 測試與基準

### 6.1 benches/ 目錄完全空白

**檔案**: `benches/` （空目錄）  
**問題**: spec.md 要求 criterion 基準測試（首包延遲 p50/p99 ≤ 97ms），但完全未實作。

**建議**: 優先添加關鍵基準測試：
1. `ParallelCodebook::decode` — embedding lookup 延遲
2. `CausalConv1d::step` — 單幀處理延遲
3. `Decoder12Hz::decode_chunk` — 完整 12Hz 首包延遲

### 6.2 壓力測試缺失

**問題**: spec.md §5 要求「1000 次連續調用無記憶體洩漏」，但無對應測試。

**建議**: 添加循環調用測試，使用 `dhat` crate 記憶體分析。

### 6.3 數值對齊測試依賴外部檔案

**檔案**: `tests/debug_per_layer_compare.rs`  
**問題**: 多個對齊測試使用 `weights/` 下的 `.bin`/`.npy` 參考檔案，
但這些檔案的路徑在不同環境中可能不同。

**建議**:
- 使用 `env!("CARGO_MANIFEST_DIR")` 建立絕對路徑
- 對可選的對齊測試使用 `#[cfg(feature = "alignment_tests")]`

---

## 7. 📋 優先級矩陣

| 優先級 | 類別 | 項目 | 預估工時 | 影響 |
|--------|------|------|----------|------|
| P0 | 🔴 Bug | MTP 使用假權重 (1.1) | 4h | 12Hz 解碼全中斷 |
| P0 | 🔴 缺漏 | 聲碼器未實作 (1.2) | 40h | 無端到端音訊 |
| P0 | 🟡 性能 | step() 重建歷史 (2.1) | 8h | 流式延遲 O(n) 成長 |
| P1 | 🔵 架構 | CausalConv1d / CausalConvNet 重複 (3.1) | 4h | 維護成本 + 潛在數值差異 |
| P1 | 🟡 性能 | sliding_window mask 重複建立 (2.4) | 2h | 大 seq_len 性能瓶頸 |
| P1 | 🟡 性能 | Tensor↔Vec 來回轉換 (2.5) | 3h | 不必要的記憶體頻寬開銷 |
| P1 | 🧪 測試 | benches/ 空白 (6.1) | 8h | 無法量化延遲目標 |
| P2 | 🟢 品質 | snake_beta 重複實作 (4.3) | 1h | 程式碼膨脹 |
| P2 | 🟢 品質 | unwrap 在庫代碼中 (1.4) | 1h | 違反規範 |
| P2 | 📦 倉庫 | 根目錄清理 (5.1) | 1h | 開發體驗 |
| P3 | 🔵 架構 | Tokenizer stub v.s. 真實 tokenizer.json (3.4) | 8h | 功能缺漏 |
| P3 | 🔵 架構 | WeightLoader 雙職責 (3.5) | 4h | 模組化 |

---

> **總結**: 專案架構清晰、文件完整，12Hz 解碼器核心邏輯已大體完成。
> 當前最大風險是 **MTP 假權重**（永遠輸出零 token）和**聲碼器未實作**（端到端管線中斷）。
> 性能方面，`CausalConv1d::step()` 的 O(n) 歷史重建是流式延遲的首要瓶頸，
> 修復後應可接近 spec.md 要求的 97ms 首包延遲目標。
