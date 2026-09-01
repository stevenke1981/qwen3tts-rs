# Qwen3-TTS Rust Rewrite — 全面專案審查與優化改善建議報告

> 基準版本：v0.2.0 (對齊階段 P03-T04 完成 / P03-T05 驗收中)  
> 分析日期：2026-09-01  
> 審查規範依據：`AGENTS.md` (Alignment Controlled Workflow / 系統與效能規範)、`GAP_MATRIX.md`、`PERFORMANCE_BUDGET.md`

---

## 目錄

1. [執行摘要](#1-執行摘要)
2. [🔴 嚴重缺陷與規範違規 (Critical & Spec Violations)](#2-嚴重缺陷與規範違規)
3. [🟡 性能瓶頸與延遲優化 (Performance & Latency)](#3-性能瓶頸與延遲優化)
4. [🔵 架構精簡與技術債清理 (Architecture & Technical Debt)](#4-架構精簡與技術債清理)
5. [🧪 測試、基準與對齊驗收 (Testing, Benchmarks & Alignment)](#5-測試基準與對齊驗收)
6. [📋 優先級矩陣與修復路線圖 (Priority Matrix & Roadmap)](#6-優先級矩陣與修復路線圖)

---

## 1. 執行摘要

本專案自 2026 年 6 月（v0.1.16）以來已取得顯著架構進展：
- 完成了原先缺失的 `HifiGanVocoder` 權重加載與前向傳播。
- 建立了完整的 `talker` 模組（包含 `TalkerModel`、`CodePredictor`、Philox RNG 抽樣、Repetition Penalty 等）。
- 實作了純 Rust 的 `text_frontend`（包含 `CandleLLM`、Voice Clone、Speaker Presets 與 Model Catalog）。
- 通過了 Phase 00、01、02 及 P03-T01 ~ T04 的嚴格對齊驗證，現存 135 個單元/模組測試全數通過。

然而，根據 `AGENTS.md` 的嚴格標準（**首包延遲 ≤97ms、數值精度 ≥0.999、熱路徑零動態記憶體分配**），當前代碼庫在**串流解碼熱路徑**、**跨設備資料搬移**、**記憶體映射載入**以及**端到端流式管道**方面，仍存在關鍵瓶頸與違規現象。

---

## 2. 🔴 嚴重缺陷與規範違規

### 2.1 Clippy `deny` 錯誤導致建置閘門失敗

- **檔案**: `tests/mrope_reference_test.rs:268`
- **問題**: 測試常數陣列中包含 `6.28`，觸發了 Rust Clippy 預設 `deny` 的 `clippy::approx_constant`（判定近似 `std::f32::consts::TAU`），導致 `cargo clippy --all-targets` 直接以 exit code 1 失敗。
- **違反規範**: `AGENTS.md` §7.3：「Clippy 沒有新增警告」為任務驗收硬性條件。
- **解決方案**:
  ```rust
  // 在 tests/mrope_reference_test.rs 或該測試函式加上：
  #[allow(clippy::approx_constant)]
  // 或將測試隨機數 6.28 微調為非 TAU 數值（如 6.29）
  ```

---

### 2.2 TokenParser 硬編碼 EOS Token ID 與模型中繼資料不一致

- **檔案**: `src/text_frontend/token_parser.rs:16`
- **問題**: 代碼中硬編碼：
  ```rust
  const CODEC_EOS_TOKEN_ID: u16 = 0x7FFF; // 32767
  ```
  但 Qwen3-TTS 官方模型以及 `src/talker/config.rs:186` 定義的真實 `codec_eos_token_id` 為 `2150`！
- **影響**: 當 Talker 生成 2150（EOS 標記）時，`TokenParser::parse()` 永遠無法識別該幀為結尾，導致：
  1. 結尾填充幀或無效幀無法被正確截斷。
  2. 生成過度冗長的尾部空白或雜音。
- **建議**:
  - 將 `CODEC_EOS_TOKEN_ID` 從硬編碼改為由 `TalkerConfig` 或 `SynthesisOptions` 傳入，預設與 `config.codec_eos_token_id`（2150）保持一致。

---

### 2.3 `Decoder12Hz::decode_chunk` 累積歷史重新計算（O(N²) 複雜度）

- **檔案**: `src/decoder_12hz.rs:315-350`
- **問題**: 在串流模式下，每一幀 `decode_chunk_inner()` 都將當前幀輸出推入全域緩衝區 `self.step_buffer`，隨後將**整個歷史緩衝區**轉為 Tensor，並重新跑完整個下游卷積與聲碼器管線：
  ```rust
  self.step_buffer.extend(&h_vec);
  let total_frames = self.step_buffer.len() / self.config.latent_dim;
  // 每一幀都把整個 step_buffer (1..total_frames) 全部重算一遍！
  let mut h = h_tensor;
  for ub in &self.upsample_blocks { h = ub.forward(&h)?; }
  let h = self.decoder_start.forward(&h)?;
  for db in &self.decoder_blocks { h = db.forward(&h)?; }
  let h = snake_beta(&h, &self.final_snake_a, &self.final_snake_b)?;
  let h = self.final_conv.forward(&h)?;
  ```
- **影響**: 生成長音訊時，第 $T$ 幀需要計算 $T$ 幀卷積。生成 $N$ 幀音訊的總計算量為 $O(N^2)$。隨著音訊長度增加，幀延遲線性激增，徹底破壞流式首包與後續幀的延遲預算（違反 ≤97ms）。
- **建議**: 實作專案待辦之 `P05 — True stateful codec streaming`：
  - 針對 `ConvTranspose1d` 與空洞卷積（Dilated Convolutions）維護固定大小的重疊上下文（Overlap State）與因果緩衝區。
  - 每幀僅計算新幀所需特徵並流式吐出 PCM，使單幀複雜度維持嚴格 $O(1)$。

---

### 2.4 解碼熱路徑嚴重違反「零動態記憶體分配」與 CPU<->GPU 反覆同步

- **檔案**: `src/decoder_12hz.rs:314-362`, `src/codec/causal_conv.rs:451-483`
- **問題**:
  1. `h.squeeze(0)?.squeeze(2)?.to_vec1()?`：每幀將張量從 Device 搬移至 Host 並配置 `Vec<f32>`。
  2. `Tensor::from_slice(&self.step_buffer, ...)`：每幀重新把 Host Vec 複製至 Device。
  3. `all_output[prev_offset..].to_vec()`：每幀呼叫 `.to_vec()` 觸發堆記憶體分配。
  4. `CausalConv1d::step_tensor` 雖然宣稱「直接接受/回傳張量，避免 CPU-GPU 往返」，但其實作內部竟呼叫 `frame.to_vec1()?` 轉入 CPU `RingBuffer`，再透過 `Tensor::from_slice` 傳回 GPU！
- **違反規範**: `AGENTS.md` §3.1：「🔴 禁止：在 `decode_chunk` 熱路徑中使用 `Vec::new()`, `format!`, `Box::new`, `clone()`」。
- **影響**: 在 CUDA / Metal 模式下，每幀產生多次隱式 Stream Synchronize，GPU 管線完全被打斷，耗時由微秒級退化至數毫秒。
- **建議**:
  - `RingBuffer` 必須支援純張量狀態（Device-resident Tensor Ring Buffer），歷史切片直接以 `.narrow()`、`.slice_assign()` 或預先分配好的連續記憶體維護，杜絕任何 `to_vec1()` 與 `from_slice()`。

---

### 2.5 PreTransformer `KvRing` 內部採用 CPU `Vec<f32>`

- **檔案**: `src/codec/transformer.rs:350-417`
- **問題**: `KvRing` 的 `k_data` 與 `v_data` 為 CPU 向量（`Vec<f32>`）。在 `PreTransformer::step` 中（共 8 層 Transformer）：
  - `write()` 呼叫 `k.flatten_all()?.to_vec1()?` 與 `v.flatten_all()?.to_vec1()?`（8 層 × 2 = 16 次 GPU→CPU 傳輸 + 配置）。
  - `gather()` 在 CPU 上以巢狀迴圈拼接後，呼叫 `Tensor::from_slice`（8 層 × 2 = 16 次 CPU→GPU 傳輸 + 配置）。
  - 單單處理 1 幀音訊，就強制觸發 **32 次跨設備資料複製與同步屏障**！
- **建議**:
  - 將 `KvRing` 重構為純設備張量快取（如同 `talker/model.rs` 的 `kv_caches` 機制），形狀為 `(1, num_kv_heads, max_seq_len, head_dim)`，以指標更新與張量視圖操作，實現零 Host Transfer。

---

### 2.6 權重載入缺少 Memory Mapping (Mmap) 與手動慢速轉碼

- **檔案**: `src/weights.rs:40, 79` 與 `src/talker/weight_loader.rs:29-57`
- **問題**:
  1. 權重加載全部使用 `std::fs::read(path)` 將整個 1~3GB 檔案一口氣讀入 RAM，使程式啟動時記憶體峰值翻倍（檔案緩衝區 + 反序列化張量）。
  2. `TalkerWeightLoader` 載入 BF16 權重時，竟然使用單執行緒 CPU 逐位元組迴圈 `for chunk in raw_data.chunks_exact(2)` 手動轉成 F32！對於 1.7B 參數模型，這在啟動時浪費數十秒，並使記憶體使用量膨脹 2 倍。
- **建議**:
  - 改用 `memmap2::MmapOptions` 進行零拷貝檔案對齊映射。
  - 保留權重原本的 BF16 DType（Candle 原生支援 BF16），或若需轉為 F32，應使用 SIMD / `bytemuck` 平行向量化轉碼。

---

## 3. 🟡 性能瓶頸與延遲優化

### 3.1 `greedy_select_on_device` 每幀動態配置 10+ 個臨時張量

- **檔案**: `src/talker/sampling.rs:72-117`
- **問題**:
  在 Talker 自迴歸生成迴圈中，每一步預測 codebook 0 都會執行：
  ```rust
  let indices = Tensor::arange(0u32, vocab_u32, device)?;
  let finite = logits.abs()?.le(f32::MAX)?;
  let invalid = Tensor::full(f32::NEG_INFINITY, (1, vocab_size), device)?;
  let masked = valid.where_cond(logits, &invalid)?;
  let sentinel = Tensor::full(f32::MIN, (1, 1), device)?;
  Tensor::cat(&[&masked, &sentinel], 1)?.argmax(1)?
  ```
  每幀在 GPU 上臨時 allocate 多個張量，造成 Candle 內部記憶體配置器（CUDA allocator）頻繁申請/釋放與核心啟動開銷。
- **優化方案**:
  - `indices`、`invalid`、`sentinel` 等遮罩與常數張量在初始化時預先建立好並重複使用。
  - 對於貪婪選擇，直接以 `.narrow(1, 0, suppress_from)` 或原地遮罩取代動態分配。

---

### 3.2 `CodebookLookup::batch_lookup` Rayon 開銷 vs 單次張量索引

- **檔案**: `src/codec/codebook.rs:133-168`
- **問題**:
  對 16 層碼本使用 `rayon::par_iter()` 啟動 16 個微任務（Micro-tasks），每個任務僅執行一次 `.narrow(0, idx, 1)`，最後再於主線程呼叫 `Tensor::stack(&valid, 0)`。
  多執行緒排程與同步的 overhead 遠高於 16 次記憶體索引。
- **優化方案**:
  - 由於碼本權重已扁平化為 `(16 * 2048, embedding_dim)`，16 個 token 的全域索引為：
    $$\text{idx}_i = i \times 2048 + \text{token}_i$$
  - 直接構建一個 `[16]` 的索引張量，呼叫一次 `flat_weights.index_select(&indices, 0)` 即可在 GPU/CPU 向量化單核心內完成所有 16 層碼本查找，完全消除 Rayon 排程與 stack 開銷！

---

### 3.3 端到端 TTS 缺乏串流管線（TTS Streaming Gap）

- **檔案**: `src/text_frontend/candle_backend.rs:451-525`, `src/gui.rs:400-448`
- **現況**:
  目前執行路徑為：
  1. `talker.generate`：整段文字自迴歸生成完畢（耗時數秒）。
  2. `parser.parse`：將所有 Token 收集為 `TokenStream`。
  3. `decoder.decode_frames`：整批解碼為 PCM 波形。
- **影響**:
  使用者必須等待**整段話所有 Token 生成完畢**後，才能聽到第一個音訊樣本。首次發聲時間（Time to First Audio, TTFA）高達數千毫秒。
- **建議**:
  - 連通 `P06 — End-to-end generation streaming`：
  - 在 `talker.generate` 的迴圈中，每生成完一個 frame（16 個 tokens），立即回呼傳遞給串流解碼器 `decoder.decode_chunk()`，並透過回呼或頻道即時輸出 PCM 片段至音訊播放設備。

---

### 3.4 量化模型載入期全數解包為 F32（Quantization Gap）

- **檔案**: `src/quantization/mod.rs:90-100`, `src/weights.rs:242`
- **現況**:
  雖然專案定義了 `QuantizedTensor`（Q8_0, Q4_0），但在載入權重時，立即呼叫 `qt.dequantize_to_vec()` 解碼為 F32 向量再建立張量。
- **影響**:
  運行時的推論仍然是全精度 F32，記憶體頻寬與顯存佔用沒有享受到任何量化優勢（對應 `GAP_MATRIX.md` 中的 P0 項目）。
- **建議**:
  - 落實 `P08 — Native quantized runtime`：利用 Candle 的原生量化算子（如 QMatMul / Q8 Linear），在推論過程中直接運算量化權重。

---

## 4. 🔵 架構精簡與技術債清理

### 4.1 `HifiGanVocoder` 與 `Decoder12Hz` 結構與邏輯重複

- **檔案**: `src/vocoder/mod.rs` vs `src/decoder_12hz.rs:74-97`
- **問題**:
  `Decoder12Hz` 內部直接擁有 `decoder_start`、`decoder_blocks`、`final_snake_a/b`、`final_conv`，並自行加載與 forward；而 `src/vocoder/mod.rs` 的 `HifiGanVocoder` 又是完全相同的一組層。
  這導致相同權重鍵名（`0.conv` ~ `6.conv`）被兩套程式碼分別加載與維護。
- **建議**:
  - 將 `Decoder12Hz` 後半段的聲碼器邏輯統整調用 `HifiGanVocoder`，或明確規範 Vocoder 的介面職責，消除重複定義。

---

### 4.2 `src/tokenizer/mod.rs` 殘留空殼 Stub

- **檔案**: `src/tokenizer/mod.rs`
- **問題**:
  `qwen3tts::tokenizer::Tokenizer` 的所有方法（`new`, `encode`, `decode`）皆回傳 `Err("not yet implemented")`。然而 `text_frontend` 模組卻直接使用外部 `tokenizers::Tokenizer`。
- **建議**:
  - 移除或重新包裝該 stub，使 `qwen3tts::tokenizer` 成為對 `tokenizers::Tokenizer` 的正規封裝，避免使用者或外部整合者誤用該空殼結構。

---

### 4.3 `src/codec/activation.rs` 死代碼與數學定義歧異

- **檔案**: `src/codec/activation.rs:9-35`
- **問題**:
  `snake_beta_v2` 與 `SnakeBeta` 標記了 `#[allow(dead_code)]`，且未對 `beta` 取 `.exp()`（`decoder_blocks.rs:8-15` 的標準實作有取 `.exp()`）。
- **建議**:
  - 移除未使用的 `snake_beta_v2` 與 `SnakeBeta`，統一引用 `decoder_blocks.rs` 中的標準 `snake_beta`。

---

### 4.4 `convert_gguf.rs` 未使用的列舉變體

- **檔案**: `src/bin/convert_gguf.rs:68-72`
- **問題**: `MetaValue` 列舉中的 `Bool`, `Uint64`, `ArrayStr`, `ArrayUint32` 從未被建構，編譯時產生 dead code 警告。
- **建議**: 為未使用的變體加上 `#[allow(dead_code)]`，或補齊對應的 GGUF 中繼資料寫入邏輯。

---

### 4.5 GUI 每次點擊合成重複加載數百 MB 權重

- **檔案**: `src/gui.rs:381, 434`
- **問題**:
  每次使用者在 GUI 介面按下「開始合成」，背景執行緒都重新從硬碟讀取模型 safetensors 權重並重建 `CandleLLM` 與 `Decoder12Hz` 實例，產生數秒不必要的磁碟 I/O 阻塞。
- **建議**:
  - 在 GUI App 狀態中快取已加載的 `CandleLLM` 與 `Decoder12Hz`，僅在使用者切換模型或後端時重新加載。

---

## 5. 🧪 測試、基準與對齊驗收

### 5.1 基準測試 (Benches) 缺乏 Mock/Synthetic 權重支援

- **檔案**: `benches/bench_decoder_12hz.rs:23-30`
- **問題**:
  基準測試必須依賴本地硬碟存在的真實權重目錄（`weights/refs2`），若不存在則直接 skip。導致 CI 環境與開發者無法隨時量化評估首包延遲是否達成 ≤97ms 目標。
- **建議**:
  - 提供 `--synthetic` 或 mock 權重生成模式，使 Criterion 基準測試能在任何環境一鍵執行並監控延遲衰退。

---

### 5.2 P03-T05 Gate 阻塞修復

- **現狀**: 根據 `STATUS.md` 與 `artifacts/alignment/P03/P03-T05/review.md`：
  - 官方 Python 參考匯出包含 678 stages，而 Candle 產生 723 stages，導致比較器 fail-closed。
  - qwentts.cpp anchor 比對僅對應 10/16 anchors，形狀與閾值未完全通過。
- **建議**:
  - 優先修復 Python 匯出腳本使其完整涵蓋 723 stages，並更新 anchor 映射以打通 Phase 03 Phase Gate。

---

### 5.3 跨平台編譯限制（Windows Metal Gating）

- **問題**:
  在 Windows 上若使用 `cargo check --all-features`，會啟動 `metal` feature，拉入僅限 macOS 的 `objc2` crate 導致編譯失敗。
- **建議**:
  - 在 `Cargo.toml` 中將 `metal` feature 設定為 target-specific：
    ```toml
    [target.'cfg(target_os = "macos")'.dependencies]
    # metal-specific bindings
    ```

---

### 5.4 89+ 處 Clippy 警告集中清潔

- 包含 `unnecessary_cast`, `map_clone`, `needless_range_loop`, `manual_is_multiple_of`, `unnecessary_map_or`, `default_constructed_unit_structs`, `unusual_byte_groupings` 等。
- 透過 `cargo clippy --fix --lib -p qwen3tts-rs` 可自動修復 54 處，其餘手動修復以維持最高標準代碼品質。

---

## 6. 📋 優先級矩陣與修復路線圖

| 優先級 | 類別 | 項目 | 預估工時 | 影響程度與預期效益 |
|---|---|---|---|---|
| **P0** | 🔴 規範 | 修復 `tests/mrope_reference_test.rs` Clippy deny 錯誤 (§2.1) | 0.5h | 解除 `cargo clippy --all-targets` 失敗，恢復 Gate 通過 |
| **P0** | 🔴 正確性 | 修正 `TokenParser` 硬編碼 EOS ID (32767 → 2150) (§2.2) | 1h | 確保生成音訊結尾能被正確截斷，避免尾部噪音 |
| **P0** | 🔴 效能 | `Decoder12Hz` 串流解碼歷史全量重算改為 O(1) 狀態更新 (§2.3) | 12h | 消除 O(N²) 延遲累積，長音訊生成延遲下降 80%+ |
| **P0** | 🔴 規範 | 消除 `decode_chunk` 與 `KvRing` 中的跨設備拷貝與動態分配 (§2.4, §2.5) | 8h | 達成熱路徑零動態分配，GPU 模式單幀延遲降低 5~10x |
| **P1** | 🟡 效能 | `CodebookLookup` 改用單次張量索引取代 Rayon micro-tasks (§3.2) | 2h | 消除執行緒調度與微切片 stack 開銷，提升碼本查詢效能 |
| **P1** | 🟡 效能 | `greedy_select_on_device` 遮罩與常數張量預分配快取 (§3.1) | 3h | 減少每步 10+ 個臨時張量配置，降低顯存碎片與 Kernel 啟動時間 |
| **P1** | 🟡 功能 | 實作 TTS 端到端串流管線 (Frame Token → PCM Callback) (§3.3) | 16h | TTFA (首包音訊延遲) 從數秒大幅縮減至 <300ms |
| **P1** | 🔵 穩定性 | 權重載入採用 `memmap2` 零拷貝與 BF16 向量化 (§2.6) | 4h | 啟動記憶體峰值減半，模型載入時間縮短 60% |
| **P2** | 🔵 架構 | GUI 實作 Model/Session 快取機制 (§4.5) | 3h | 消除重複點擊合成時的重複加載磁碟 I/O 延遲 |
| **P2** | 🔵 架構 | 整併 `HifiGanVocoder` 與 `Decoder12Hz` 重複元件 (§4.1) | 4h | 減少維護負擔與權重重複加載 |
| **P2** | 🔵 清理 | 移除 `activation.rs` 死代碼與清理 `tokenizer/mod.rs` stub (§4.2, §4.3) | 2h | 保持代碼庫整潔一致 |
| **P2** | 🧪 測試 | 基準測試新增 Synthetic/Mock 權重模式 (§5.1) | 4h | 支援 CI 自動化基準測試與延遲防退化監控 |
| **P3** | 🟢 品質 | 全庫 89+ 處 Clippy 警告清理 (§5.4) | 2h | 達到最高代碼衛生標準 |
