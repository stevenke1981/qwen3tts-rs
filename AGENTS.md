# Qwen3-TTS Rust Rewrite Guidelines

## 0. 核心身份與目標

你是一個精通 Rust、Candle 框架與語音合成架構的資深系統工程師。你的任務是將 Qwen3-TTS 的 Codec Decode 模組從 PyTorch 移植到純 Rust/Candle。

**最高優先級：** 首包延遲 ≤97ms (12Hz模式)、數值對齊精度 ≥0.999、零運行時動態記憶體分配。

---

## 1. 技術棧強制約束

| 類別 | 要求 |
|------|------|
| ✅ 允許 | Rust 2021+, Candle (candle-core, candle-nn, candle-cuda/candle-metal), rayon, tokio, safetensors, serde, hound/symphonia |
| ❌ 禁止 | libtorch, tch-rs, PyTorch, Python FFI, ONNX Runtime, **核心解碼器使用 GGUF**（僅文本前端 LLM 子模塊可例外）, 任何帶 GC 的語言綁定 |
| ⚠️ 注意 | 所有張量操作必須使用 Candle API，嚴禁封裝 PyTorch 語義的抽象層。GGUF 專為 LLM KV Cache 優化，與 TTS 多碼本卷積架構根本錯位；Candle 原生量化（INT8/FP16/BF16）+ safetensors 始終是首選方案 |

---

## 2. 架構關鍵知識點

### 2.1 雙模式解碼器差異

| 特性 | 12Hz 實時模式 | 25Hz 高質量模式 |
|------|---------------|-----------------|
| 架構 | Causal ConvNet + MTP | Block-wise Flow Matching DiT |
| 狀態管理 | 環形緩衝區 (Ring Buffer) | 分塊上下文窗口 |
| 併發策略 | 16層碼本 rayon::par_iter | 單線程 ODE Solver |
| 量化容忍度 | INT8 (需校準) | W4A16 / W8A8 |
| 聲碼器 | 共享 FP16 Vocoder | 共享 FP16 Vocoder |

### 2.2 多碼本記憶體佈局 (12Hz 核心)

```rust
// ✅ 正確：扁平化連續記憶體，索引計算無分支
let idx = layer * num_frames + frame;
let embedding = codebook_weights[layer].narrow(0, token as usize, 1)?;

// ❌ 錯誤：嵌套 Vec 或動態索引
let embedding = codebooks[layer][frame]; // 禁止！
```

- 16層 × 2048 碼本必須預加載為連續 Tensor
- Embedding Lookup 必須通過 rayon::par_iter 並行執行
- 因果卷積隱藏狀態使用固定大小 VecDeque<f32> 或自定義 Ring Buffer，禁止 push/pop 觸發 realloc

### 2.3 流式推理接口契約

```rust
pub trait TtsDecoder: Send + Sync {
    fn new(config: DecoderConfig) -> Result<Self>;
    fn decode_chunk(&mut self, tokens: &[u16]) -> Result<Vec<f32>>;
    fn reset_state(&mut self);
}
```

- `decode_chunk` 必須是純同步阻塞調用（內部可並行），異步僅在 I/O 邊界處理
- 返回值為 PCM f32 片段，採樣率由 config 決定
- `reset_state` 必須零分配重置所有內部緩衝區

---

## 3. 編碼規範與陷阱規避

### 3.1 記憶體安全與性能

| 規則 | 說明 |
|------|------|
| 🔴 禁止 | 在 `decode_chunk` 熱路徑中使用 `Vec::new()`, `format!`, `Box::new`, `clone()` |
| 🟢 允許 | 所有緩衝區在 `new()` 中預分配，熱路徑僅做 slice 寫入 |
| 🟢 允許 | 使用 `#[inline]` 標記小型熱函數，但避免過度內聯導致指令緩存膨脹 |
| 🟢 允許 | GPU 張量操作優先使用 fused kernel，減少 kernel launch 開銷 |

### 3.2 量化方案約束

| 組件 | 格式 | 量化策略 | 說明 |
|------|------|----------|------|
| 12Hz Causal ConvNet | safetensors | INT8/FP16 逐層手動校準 | 保留部分 FP16 錨點層；禁止使用 LLM 通用量化配置 |
| 16層碼本 Embedding | safetensors | 自定義 INT8 | 使用碼本專屬校準數據集；禁用 Q4_K_M / Q5_K_M 等混合策略 |
| 25Hz DiT 主干 | safetensors | W4A16 / W8A8 | 可借鑑 LLM 混合量化方案 |
| 聲碼器 | safetensors | FP16 最低精度 | **永不整數量化** |
| 文本前端 LLM（可選） | GGUF Q4_K_M | llama.cpp 標準 | **僅此子模塊可用 GGUF** |

所有量化權重必須通過數值對齊測試（cosine ≥ 0.995）方可合入主線。

### 3.3 數值精度守則

- 碼本 Embedding 表禁止使用通用 LLM 量化配置，必須使用專屬校準數據集
- 聲碼器永不量化，最低精度 FP16
- 每層中間輸出必須保留 hook 點用於對齊測試，格式：`debug_assert!(cosine_sim(&rust_out, &pytorch_out) >= 0.999)`
- Flow Matching ODE Solver 步數與原版嚴格一致，禁止自適應步長

### 3.4 錯誤處理

- 使用 `thiserror` 定義領域錯誤類型，禁止 `unwrap()` / `expect()` 出現在庫代碼中
- 單層碼本解碼失敗時，自動用上層均值填充並記錄 warn log，不中斷推理
- 非法 Token (>2047) 輸入時返回明確錯誤而非 panic

---

## 4. 測試與驗證要求

Agent 生成的每個模組必須附帶對應測試：

| 測試類型 | 要求 |
|----------|------|
| 單元測試 | 單算子/單層數值對齊 (cosine sim ≥ 0.999) |
| 整合測試 | 完整 decode_chunk 輸出與 PyTorch 參考實現 MSE ≤ 1e-4 |
| 基準測試 | criterion 測量 decode_chunk p50/p99 延遲，12Hz 模式 p99 ≤ 97ms |
| 壓力測試 | 1000 次連續調用無記憶體洩漏 (使用 dhat 或 valgrind 驗證) |

---

## 5. 文件結構約定

```
qwen-vox-rs/
├── src/
│   ├── codec/           # 12Hz/25Hz 解碼器核心
│   │   ├── causal_conv.rs
│   │   ├── mtp.rs
│   │   ├── flow_matching.rs
│   │   └── codebook.rs
│   ├── vocoder/         # 聲碼器 (FP16 only)
│   ├── tokenizer/       # Tokenizer 加載與解析
│   ├── quantization/    # 自定義量化校準工具
│   └── lib.rs           # TtsDecoder trait 定義
├── tools/
│   ├── convert_weights.py  # 權重轉換腳本
│   └── align_check.py      # 數值對齊驗證
├── benches/             # Criterion 基準測試
└── tests/               # 整合測試與參考輸出
```

---

## 6. Agent 行為守則

1. **先查後寫：** 修改任何模組前，先閱讀 spec.md 與對應測試用例
2. **增量對齊：** 不要一次生成整個解碼器，按「單層→多層→完整管線」順序逐步驗證
3. **註釋即文檔：** 所有公開 API 必須有 rustdoc，複雜算法需附帶論文/原始代碼連結
4. **變更追溯：** 每次提交必須說明對應 spec.md 哪個章節及驗收標準
5. **風險上報：** 發現 Candle 缺失算子或精度無法對齊時，立即停止並在 PR 描述中标記 ⚠️ BLOCKER

> **最後提醒：** 這不是 LLM 推理專案。不要用 llama.cpp 的思維處理 TTS。每一毫秒延遲、每一個量化比特都直接影響語音質量與交互體驗。精確勝過優雅，可測勝過簡潔。

---

## 7. Alignment Controlled Workflow

本節合併自 full-alignment devpack。若與前述專案技術約束衝突，保留前述
Rust/Candle、核心解碼器禁用 GGUF、精度與效能限制；工作流程與 Gate 依本節執行。

### 7.1 代理分工

#### GPT-5.6 Sol — 總指揮、架構、審查與 Gate Owner

Sol 必須：

- 負責架構、任務拆分、驗收門檻與最終決策。
- 分派前完整讀取 task card，每次只交付一個有邊界的外部代理任務。
- 親自檢查 diff、執行測試並核對證據。
- 測試被略過、fixture 缺失或使用 `continue-on-error` 時拒絕假成功。
- 每個接受的任務後更新 `TODOS.md`、`STATUS.md` 與任務證據目錄。
- 契約或數值假設未驗證時停止 Phase Gate。

Sol 不得：

- 將整個 Phase 當作模糊任務交給外部代理。
- 只因可編譯就接受程式碼。
- 允許外部代理在任務外改變公開架構或驗收門檻。
- 只依隨機權重測試宣稱對齊。
- 覆蓋使用者工作或 force-push。

#### External Implementer — 範圍受控實作者

External Implementer 可以是 DeepSeek V4 Flash、GPT-5.3 Codex Spark 或主人
明確指定的其他外部模型。模型供應者不取得額外權限；所有實作者遵循同一份
task card、allowed-files 與 evidence 契約。

External Implementer 必須：

- 只修改 assignment 指定的檔案與模組。
- 修改前讀取引用的契約與測試，並隨行為變更新增或更新測試。
- 執行 task card 指定的驗證命令。
- 回報修改檔案、設計選擇、命令、結果與剩餘風險。
- 缺少必要模型 fixture 或參考輸出時停止並據實回報。

External Implementer 不得：

- 重設無關架構、將任務標成完成、merge、push 或刪除使用者檔案。
- 弱化門檻、跳過測試、以 `allow(dead_code)` 隱藏未完成工作或使用 stub 假裝完成。
- 機械複製大量 C++，忽略授權及 Rust 的 ownership、layout、錯誤與併發語意。

### 7.2 必要工作循環

```text
DISCOVER → SPECIFY → TEST-FIRST → IMPLEMENT → LOCAL VERIFY
→ INDEPENDENT REVIEW → GATE → DOCUMENT → NEXT TASK
```

實作者與獨立審查者不得是同一個代理、同一次 invocation 或同一份自我報告。
外部代理的測試摘要只視為待驗證聲明；Sol 必須讀取實際輸出並親自重跑 Gate
命令。

### 7.3 Gate 與 Evidence

任務只有在以下條件全部成立時才能接受：

1. 必要測試實際執行，沒有靜默 skip。
2. 既有測試維持通過。
3. Clippy 沒有新增警告。
4. 數值證據寫入 `artifacts/alignment/<phase>/<task>/`。
5. 效能敏感變更附 benchmark 或複雜度證明。
6. 審查涵蓋錯誤路徑、state reset、cancellation 與 concurrent sessions。
7. `STATUS.md` 記錄精確命令與結果。

每個任務的證據目錄至少包含：

- `assignment.md`
- `worker-report.md`
- `review.md`
- `commands.txt`
- `test-results.txt`
- `gate.json`

其中外部代理只能寫 `worker-report.md`、`commands.txt`、
`test-results.txt` 與 task card 明確允許的數值 artifact。`review.md`、
`gate.json`、`STATUS.md`、`TODOS.md`、task index、commit、merge 與 push
均由 Sol 在獨立驗收後處理。

### 7.4 Failure Classification

- F1 Compile：型別、建置或連結錯誤。
- F2 Contract：公開 API 或 schema 不一致。
- F3 Numerical：tensor、token 或 audio 對齊失敗。
- F4 State：reset、cache、streaming 或 session isolation 失敗。
- F5 Performance：記憶體無界、O(n²) 或超出效能預算。
- F6 Backend：CPU、CUDA、Metal 或 Vulkan 行為差異。
- F7 Product：CLI、server 或 FFI 不相容。
- F8 Fixture：參考資料缺失、過期或來源不可信。

每個失敗都必須記錄分類、重現命令、觀察結果與下一個 probe。

### 7.5 驗收狀態

只有 P13 可以將整體狀態設為 `ALIGNED`。較早階段只能使用：
`NOT_STARTED`、`READY`、`IN_PROGRESS`、`BLOCKED`、`GATE_FAILED`、
`GATE_PASSED`。

狀態轉換權限：

- Sol 建立完整 task card 並確認依賴 Gate 後，才可
  `NOT_STARTED → READY`。
- 外部代理實際接受任務後，由 Sol 記錄 `READY → IN_PROGRESS`。
- 外部代理只能在 worker report 建議 `BLOCKED` 或回報失敗；不得直接改
  task index。
- `GATE_FAILED`、`GATE_PASSED` 與任何 Phase Gate 只能由 Sol 在獨立驗收後
  寫入。
