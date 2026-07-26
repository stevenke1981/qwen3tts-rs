# qwen3tts-rs × qwentts.cpp 對齊 — Agent 派工指示

> 執行模型：`deepseek-v4-flash`（284B total / 13B active MoE，1M context，agentic coding 導向；推理深度弱於 Pro/Sonnet 等旗艦模型）
> 派工原則：每張工單（WO）獨立成一個 session，範圍窄、規格寫死、驗收標準可機械化檢查，**不要求 flash 模型自行做架構判斷**——判斷已經在規劃階段做完，工單只負責「照規格翻成 Rust + 通過驗收」。
> 完整背景與逐項證據見同批交付的 `qwentts_cpp_alignment_plan_2026-07-26.md`；本文件不重複論證過程，只給執行指令。
> 完成一張工單後：依專案慣例寫入 `lessons.md`（RSI loop）、Conventional Commits 提交、程式碼註解 zh-TW + 技術詞彙保留英文、禁止 `unwrap()`/`expect()` 出現在庫代碼中（`AGENTS.md:96`）。

---

## 派工單總覽

| WO | 標題 | 優先度 | 規模 | 依賴 |
|---|---|---|---|---|
| WO-1 | 停用 codec/vocoder 量化路徑，恢復強制 F32 | 🔴 P0 | XS (~2h) | 無 |
| WO-2 | Sampling 加入 repetition_penalty | 🔴 P0 | S (~3h) | 無 |
| WO-3a | `Attention`/`TransformerBlock` 加入固定大小 KV ring + `step()` | 🔴 P0 | M (~1d) | 無 |
| WO-3b | `Decoder12Hz` 串流路徑改用 `step()`，移除無限累積 buffer | 🔴 P0 | M (~1d) | WO-3a |
| WO-4 | GGUF talker 權重讀取 PoC（tensor 命名對照） | 🟡 P1 | M (~1-2d) | 無，可與 1/2/3 平行 |
| WO-5 | GGUF talker backend 正式整合，取代自製量化校準 | 🟡 P1 | L (~2-3d) | WO-4 |
| WO-6 | Philox4x32-10 PRNG 移植 | 🟡 P1 | S (~0.5-1d) | 無 |
| WO-7 | Prompt 模式驗證規則 7 條 checklist 核對 | 🔵 P2 | XS (~1-2h) | 無 |
| WO-8 | RoPE interleaved 交叉驗證腳本（僅驗證，不修改主線） | 🔵 P2 | S (~2-3h) | 無 |
| WO-9 | 25Hz decoder 改為建構期報錯，移出當期 sprint | ⚪️ 清理 | XS (~0.5h) | 無 |

建議排程：WO-1、WO-2、WO-7、WO-9 可先一次性掃過（都很小、獨立、無風險）；WO-3a→3b 是這批最重要也最花時間的項目，建議排實測驗證資源；WO-4→5 可以跟 WO-3 平行跑，兩條 agent 分頭進行不衝突（不同檔案）。

---

## WO-1：停用 codec/vocoder 量化路徑，恢復強制 F32

**優先度**：🔴 P0　**規模**：XS

**背景**：`AGENTS.md:82` 與 `src/vocoder/mod.rs:11-13` 都明文規定 vocoder 永不整數量化，但 `src/paths.rs:412-416` 的 `push_tokenizer_weight_candidates()` 目前把 `weights/tokenizer-q8` 排在 `weights/tokenizer`（F32）**之前**，等於預設偷跑量化過的 codec/vocoder 權重。

**涉及檔案**：
- `src/paths.rs`（`push_tokenizer_weight_candidates` 約第 412-416 行，以及 `q8_dir_for_f32_dir`／`default_tokenizer_q8_cache_dir` 等相關函式）
- `examples/quantize_tokenizer.rs`（量化工具入口）

**現況程式碼**（`src/paths.rs`）：
```rust
fn push_tokenizer_weight_candidates(candidates: &mut Vec<PathBuf>, root: &Path) {
    let weights = root.join("weights");
    push_unique(candidates, weights.join("tokenizer-q8"));
    push_unique(candidates, weights.join("tokenizer"));
}
```

**修改規格**：
1. `push_tokenizer_weight_candidates()` 移除 `tokenizer-q8` 候選路徑，只保留 `tokenizer`（F32）。
2. 找出所有「Q8/Q4 tokenizer 自動快取」相關邏輯（含載入後自動觸發 `quantize_tokenizer.exe` 產生 `tokenizer-q8` 快取的路徑，約在 `paths.rs` 內 v0.1.13 引入的自動轉檔區塊），一併移除或用 `#[deprecated]` + 明確 log 訊息標記為停用，**不要刪除函式本體**（避免破壞既有測試/呼叫點編譯），改成早期 return 並印出：
   ```
   log::warn!("codec/vocoder 量化已停用（違反 AGENTS.md §3.2，聲碼器永不整數量化）；強制使用 F32 權重");
   ```
3. `examples/quantize_tokenizer.rs`：在 `main()` 開頭加入明確拒絕，印出理由並以非零狀態碼結束，不執行任何量化動作。**不要刪除這個檔案**（保留給未來可能的 debug 用途，只是預設路徑擋掉）。
4. talker backbone 的量化路徑（如果 grep 到有獨立於 tokenizer 之外的 talker 量化函式）**不受影響**，維持原樣。

**驗收標準**：
- `cargo build` 通過。
- `cargo test` 全數通過（含既有 vocoder 測試）。
- 手動執行一次 `synthesize` 範例，log 中不應再出現「已使用 Q8 量化 tokenizer decoder 權重」字樣。
- `examples/quantize_tokenizer.exe` 執行後應印出拒絕訊息並以非零狀態碼結束。

**邊界（不要做）**：不要動 talker LM 的量化邏輯（那部分留給 WO-5 統一處理）；不要重構 `paths.rs` 其他無關函式。

**給 deepseek-v4-flash 的貼入式 prompt**：
````
你正在修改 Rust 專案 qwen3tts-rs。任務：停用 codec/vocoder（tokenizer decoder）的整數量化路徑，
因為這違反了 AGENTS.md §3.2「聲碼器永不整數量化，最低精度 FP16」的規範。

請完成以下修改：
1. 打開 src/paths.rs，找到 push_tokenizer_weight_candidates() 函式，移除其中把
   weights/tokenizer-q8 加入候選路徑的那一行，只保留 weights/tokenizer（F32）。
2. 搜尋 src/paths.rs 中所有跟 "tokenizer-q8" / "tokenizer-q4" / 自動量化快取產生
   （例如載入時自動呼叫 quantizer 產生 Q8 快取）相關的函式，將其邏輯改為：
   不執行任何量化動作，改印出 log::warn!("codec/vocoder 量化已停用（違反 AGENTS.md §3.2，
   聲碼器永不整數量化）；強制使用 F32 權重")，函式簽名與呼叫點維持不變（避免動到呼叫端程式碼），
   直接 return 原本代表「找不到/跳過」的值。
3. 打開 examples/quantize_tokenizer.rs，在 main() 函式最開頭加入：印出理由說明
  （codec/vocoder 依規範不可量化），然後 std::process::exit(1)，不要執行後面任何量化邏輯，
  但不要刪除檔案其餘程式碼。
4. 不要修改任何跟 talker backbone 量化相關的程式碼（如果你在搜尋時看到跟 "talker" 量化相關
   的函式，跳過不動）。
5. 完成後執行 `cargo build` 與 `cargo test`，確保全數通過，並回報你異動了哪些檔案、哪些行號。

程式碼註解請用繁體中文（zh-TW），技術詞彙（函式名、型別名等）保留英文。禁止在庫代碼中使用
unwrap() / expect()（若你新增的程式碼需要處理 Result，一律用 ? 或明確的 match/log 處理）。
````

---

## WO-2：Sampling 加入 repetition_penalty

**優先度**：🔴 P0　**規模**：S

**背景**：`src/talker/sampling.rs` 目前的取樣鏈只有 suppress → top-k → top-p → temperature-softmax → multinomial，沒有 repetition penalty，對長文本重複/跳針沒有任何抑制機制。

**涉及檔案**：`src/talker/sampling.rs`（`Sampler`、`SamplingOptions`、`sample_logits_with_scratch`）

**演算法規格**（HF 對齊規則，套用時機在 temperature 縮放**之前**）：

對於輸入 logits 向量 $\mathbf{z}$ 與歷史 token 集合 $H$（去重，每個 token 只處理一次）：

$$
z'_t = \begin{cases} z_t / p & z_t \ge 0,\ t \in H \\ z_t \times p & z_t < 0,\ t \in H \\ z_t & t \notin H \end{cases}
$$

其中 $p$ 為 `repetition_penalty`（`f32`，預設 `1.0` 表示關閉，`1.0` 時整段邏輯應該是 no-op）。

**修改規格**：
1. `SamplingOptions`（`sampling.rs:9-14`）新增欄位 `repetition_penalty: f32`，預設值 `1.0`（在 `SamplingOptions::greedy()` 與所有既有建構點補上 `1.0`，維持現有測試不動）。
2. `Sampler`（`sampling.rs:26-31`）新增一個 `history: Vec<u32>` 欄位（或由呼叫端傳入 `&[u32]`，二擇一，**優先選擇由呼叫端傳入 `history: &[u32]` 參數**，避免 `Sampler` 自己管理狀態導致跟現有 KV cache 重置邏輯打架）。
3. `Sampler::sample()`（`sampling.rs:42-64`）簽名改為多一個參數 `history: &[u32]`，傳給 `sample_logits_with_scratch`。
4. `sample_logits_with_scratch()`（`sampling.rs:98-106`）簽名加 `history: &[u32]`、`repetition_penalty: f32`（或直接從 `options: SamplingOptions` 讀取，若已在步驟 1 加進 `SamplingOptions` 則不用額外參數）。在函式最前面、`candidates.clear()` 之後、`top_k` 篩選之前，套用上述公式，就地修改 `logits` 副本（**不可修改呼叫端傳入的原始 slice**，需要先 clone 成 `Vec<f32>` 或在既有 `candidates` scratch buffer 上操作）。實作時用一個 `seen` 集合（`Vec<bool>` 長度等於 vocab size，或 `HashSet<usize>`，效能考量優先用前者搭配 scratch buffer 重用模式，仿照現有程式碼風格）避免同一 token 被套用兩次。
5. 呼叫端（搜尋 `Sampler::sample(` 或 `.sample(` 的所有呼叫點，應該在 `src/talker/talker.rs`、`src/talker/code_predictor.rs` 等檔案）補上 `history` 參數，內容為目前已生成的 codebook-0 token 序列（呼叫端應該已經有這個序列在手上，不需要新建狀態）。

**驗收標準**：
- 既有 `sampling.rs` 內 `#[cfg(test)]` 測試維持全數通過（因為預設 `repetition_penalty = 1.0` 應為 no-op）。
- 新增至少 2 個測試：(a) 給定重複出現在 history 的高分 token，套用 penalty 後其被選中機率應下降（可用固定 `rand01` closure 驗證輸出 token 改變）；(b) `repetition_penalty = 1.0` 時輸出與套用前完全一致（回歸測試）。

**給 deepseek-v4-flash 的貼入式 prompt**：
````
你正在修改 Rust 專案 qwen3tts-rs 的 src/talker/sampling.rs，加入 repetition_penalty 機制，
規則對齊 HuggingFace generate() 的做法：

對於 history（已生成過的 token，去重後每個只處理一次）中的每個 token t：
  若 logits[t] >= 0：logits[t] = logits[t] / penalty
  若 logits[t] <  0：logits[t] = logits[t] * penalty
不在 history 中的 token 不受影響。這一步在 top_k / top_p / temperature-softmax 之前套用。

請完成：
1. 在 SamplingOptions struct 新增欄位 repetition_penalty: f32，SamplingOptions::greedy()
   與其他既有建構點補上 repetition_penalty: 1.0（代表關閉，維持既有行為不變）。
2. Sampler::sample() 方法簽名新增參數 history: &[u32]，往下傳給 sample_logits_with_scratch。
3. sample_logits_with_scratch() 新增參數 history: &[u32]，在函式最開頭（candidates.clear() 之後、
   top_k 篩選之前）套用上述 repetition penalty 公式。實作時用一個長度等於 logits.len() 的
   Vec<bool>（scratch buffer，可加在 Sampler struct 裡重用，避免每次呼叫都配置新記憶體）
   標記哪些 token 已經在 history 中出現過並處理過，確保同一 token 不會被重複套用 penalty。
   注意：這一步要操作 logits 的可變副本，不可修改呼叫端傳入的原始資料。
4. 找出所有呼叫 Sampler::sample(...) 的地方（搜尋整個 src/talker/ 目錄），補上 history
   參數——history 應該是呼叫端已經持有的、目前已生成的 codebook-0 token 序列，不要新增
   額外的狀態追蹤機制，直接複用呼叫端既有的資料。
5. 在 sampling.rs 的 #[cfg(test)] mod tests 區塊新增兩個測試：
   a) 給一組 logits，某個高分 token 出現在 history 裡，套用 repetition_penalty=1.5 後，
      該 token 被選中的機率應該下降（用固定的 rand01 closure 驗證輸出 token 因此改變）。
   b) repetition_penalty=1.0 時，輸出必須跟套用前完全一致（用來當回歸測試）。
6. 確保既有測試全數維持通過，執行 cargo build 與 cargo test 驗證。

程式碼註解用繁體中文（zh-TW），技術詞彙保留英文；不可使用 unwrap()/expect()。
````

---

## WO-3a：`Attention` / `TransformerBlock` 加入固定大小 KV ring + `step()`

**優先度**：🔴 P0　**規模**：M

**背景**：`src/codec/transformer.rs` 目前只有 `forward()`，每次呼叫都是無狀態的整段重算，沒有任何 KV cache。`src/decoder_12hz.rs:106-109` 的 log 訊息自承「no trim – O(n²) for streaming」。這張工單只處理 transformer 本身的 KV cache 基礎設施，**不動 `Decoder12Hz` 的呼叫端**（那是 WO-3b）。

**涉及檔案**：`src/codec/transformer.rs`

**現況** `Attention::forward()` 簽名（`transformer.rs:143`）：
```rust
fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let (b, seq_len, _) = x.dims3()?;
    let q = self.q_proj.forward(x)?;
    let k = self.k_proj.forward(x)?;
    let v = self.v_proj.forward(x)?;
    let cos = self.cos.narrow(0, 0, seq_len)?;
    let sin = self.sin.narrow(0, 0, seq_len)?;
    // ... 全 seq_len 的 QK attention + sliding mask
}
```
`sliding_window` 目前設定值為 `72`（`transformer.rs:27` default，實際跑 12Hz decoder 時由 `config.sliding_window` 傳入，見 `decoder_12hz.rs:63`）。

**修改規格**：
1. 新增一個 `KvRing` struct（放在 `transformer.rs` 內或獨立小檔案 `src/codec/kv_ring.rs`，二擇一，優先放同檔案減少跨檔案改動）：
   - 固定容量 = `sliding_window`（例如 72）個 slot，每個 slot 存一個 timestep 的 K、V 向量（`num_kv_heads * head_dim` 維）。
   - 用扁平 `Vec<f32>` + 寫入游標實作（比照 `src/codec/causal_conv.rs` 現有 `RingBuffer` 已經驗證過的扁平化陣列模式，**風格要一致**，不要重新發明一套設計）。
   - 需要記錄「目前已寫入的絕對 position」（用於 RoPE 角度計算——即使只保留最近 72 個 slot，RoPE 的旋轉角度仍要用絕對 position，不是 slot 內的相對位置）。
2. `Attention` struct 新增 `Option<KvRing>` 欄位（每層一個），初始為 `None`（batch/離線 `forward()` 路徑不使用 ring，維持原樣不動）。
3. 新增 `Attention::step(&mut self, x: &Tensor, position: usize) -> Result<Tensor>` 方法：
   - `x` 為單一 timestep 輸入，shape `(b, 1, hidden_dim)`。
   - 計算當前 timestep 的 Q/K/V，K/V 寫入 ring（覆蓋最舊 slot）。
   - RoPE 用**絕對** `position` 算旋轉角度（沿用現有 `precompute_rope` 產生的 `cos`/`sin` table，用 `position` 做 index 而非 `0..seq_len`）。
   - Attention 只在 ring 目前實際持有的 slot 範圍內計算（causal + sliding window 天然滿足，因為 ring 本身只保留最近 `sliding_window` 筆），**不需要額外的 mask**（這點比 `forward()` 路徑簡單，因為 ring 容量本身就是視窗大小）。
   - 回傳 shape `(b, 1, hidden_dim)`。
4. `TransformerBlock` 新增對應 `step(&mut self, x: &Tensor, position: usize) -> Result<Tensor>`，內部呼叫 `Attention::step`，其餘（norm、FFN、layer scale）邏輯跟 `forward()` 一致，直接重用（可抽出共用私有函式避免重複程式碼，例如 `fn apply_ffn_block(&self, x: &Tensor) -> Result<Tensor>`）。
5. `PreTransformer` 新增：
   - `fn reset_state(&mut self)`：清空所有層的 KV ring（覆寫 position 游標歸零，資料不必真的清零，比照 `causal_conv.rs` 既有 ring buffer 的 reset 慣例）。
   - `fn step(&mut self, x: &Tensor, position: usize) -> Result<Tensor>`：對單一 timestep 輸入依序跑過所有層的 `step()`，其餘（`input_proj`/`norm`/`output_proj`）邏輯與 `forward()` 一致。
6. **不要修改** `forward()` 本身，也不要修改 `PreTransformer::from_loader()` 的既有簽名（新增的 ring 狀態應該在 `from_loader()` 建構時一併初始化好，但呼叫端介面不變）。

**驗收標準**：
- 新增測試：對同一組隨機輸入序列，逐 timestep 呼叫 `step()` 累積結果，與整段一次呼叫 `forward()` 的輸出比對，**每個 timestep 的輸出 cosine ≥ 0.999**（允許浮點誤差，不要求 bit-exact）。
- `reset_state()` 後再次從頭 `step()`，結果應與從未呼叫過的全新實例一致。
- `cargo bench`（若有對應 benchmark）顯示 `step()` 延遲不隨呼叫次數增長（常數時間）。

**給 deepseek-v4-flash 的貼入式 prompt**：
````
你正在修改 Rust 專案 qwen3tts-rs 的 src/codec/transformer.rs，目標是幫 Attention /
TransformerBlock / PreTransformer 加上串流推理用的固定大小 KV cache（ring buffer），
取代目前完全無狀態、每次都要整段重算的 forward()。這是效能修正，不是正確性修正——
forward() 保持不動，新增一組平行的 step() API 給串流路徑用。

背景知識：
- sliding_window 大小（例如 72）就是 KV ring 的固定容量：每個位置只需要看見最近
  sliding_window 個 timestep，ring 滿了之後新資料覆蓋最舊資料。
- RoPE 的旋轉角度要用「絕對 position」計算，不是 ring 內的相對位置——即使 ring 只留最近
  72 筆，第 500 個 timestep 進來時，它的 RoPE 角度仍然要用 position=500 去查表，不是
  position=500%72。
- 專案裡 src/codec/causal_conv.rs 已經有一個驗證過的扁平陣列 ring buffer 實作風格，
  新的 KV ring 請比照同樣的設計哲學（固定容量、扁平 Vec、寫入游標覆蓋最舊資料、不觸發
  realloc），保持風格一致。

請完成：
1. 設計一個 KvRing struct：固定容量 = sliding_window 個 slot，每個 slot 存一個 timestep
   的 K 向量與 V 向量（各為 num_kv_heads * head_dim 維），用扁平 Vec<f32> + 寫入游標實作，
   另外記錄目前已寫入的絕對 position 計數器。放在 transformer.rs 同檔案內即可。
2. Attention struct 新增 Option<KvRing> 欄位，初始為 None（不影響現有 forward() 路徑，
   forward() 完全不使用這個欄位）。
3. 新增 Attention::step(&mut self, x: &Tensor, position: usize) -> Result<Tensor>：
   輸入 x 是單一 timestep，shape (b, 1, hidden_dim)。計算該 timestep 的 Q/K/V，K/V 寫入
   ring（若 ring 是 None 要在第一次呼叫時初始化，容量用現有 sliding_window 欄位）。RoPE
   用 position 參數（不是 0）去查現有的 cos/sin table。Attention 計算只在 ring 目前實際
   持有的 slot 上做（causal + sliding window 天然滿足，不需要額外 mask，因為 ring 容量
   本身就是視窗大小）。回傳 shape (b, 1, hidden_dim)。
4. TransformerBlock 新增對應的 step(&mut self, x: &Tensor, position: usize) -> Result<Tensor>，
   內部呼叫 Attention::step，其餘 norm/FFN/layer-scale 邏輯與現有 forward() 完全一致
   （可以抽出共用私有函式避免程式碼重複）。
5. PreTransformer 新增：
   a) reset_state(&mut self)：清空所有層的 KV ring 狀態（position 歸零即可，資料不必清零）。
   b) step(&mut self, x: &Tensor, position: usize) -> Result<Tensor>：對單一 timestep 輸入
      依序跑過所有層的 step()，其餘 input_proj/norm/output_proj 邏輯與 forward() 一致。
6. 不要修改 forward() 本體，不要修改 PreTransformer::from_loader() 的對外呼叫簽名。
7. 新增測試：構造一段隨機輸入序列，(a) 用 forward() 整段算一次，(b) 逐 timestep呼叫
   reset_state() 後依序 step()，比較兩者每個 timestep 位置的輸出，要求 cosine similarity
   >= 0.999。另外測試 reset_state() 後重新從頭 step() 應與全新實例行為一致。
8. cargo build 與 cargo test 全數通過後回報異動摘要。

程式碼註解用繁體中文（zh-TW），技術詞彙保留英文；禁止 unwrap()/expect()；熱路徑（step()
內部）不可有 Vec::new()/format!/Box::new/clone()（比照 AGENTS.md §3.1 的既有規範）。
````

---

## WO-3b：`Decoder12Hz` 串流路徑改用 `step()`，移除無限累積 buffer

**優先度**：🔴 P0　**規模**：M　**依賴**：WO-3a 必須先完成並通過驗收

**背景**：`src/decoder_12hz.rs:30` 的 `pre_conv_buffer: Vec<f32>` 目前只增不減，每次 streaming 呼叫都把全部歷史丟給 `pre_transformer.forward()`。WO-3a 完成後 `PreTransformer` 已有 `step()`，這張工單負責把 `Decoder12Hz` 的串流入口接上去。

**涉及檔案**：`src/decoder_12hz.rs`

**修改規格**：
1. 找到 `Decoder12Hz` 中負責串流（非 `decode_frames` 批次路徑）的方法（`decode_chunk_inner` 或類似命名，`decoder_12hz.rs:193-224` 附近，實際名稱以目前原始碼為準）。
2. 把該方法內對 `pre_transformer.forward(...)` 的呼叫改成 `pre_transformer.step(&frame_tensor, position)`，`position` 用一個新增的 `usize` 計數器欄位（每次 streaming step 呼叫遞增 1，`reset_state()` 時歸零）。
3. `pre_conv_buffer` 不再需要累積全部歷史——確認 `CausalConv1d`（已於先前 commit 修好）本身已經是固定大小 ring，若 `pre_conv_buffer` 只是單純把每個 frame 的 conv 輸出暫存以便丟給 transformer，且 transformer 現在改用 `step()` 逐 frame 處理，這個 buffer 應該可以整個移除或縮小成只存「當前 frame」大小（不是整段歷史）。**移除前務必先確認沒有其他地方（例如 `decode_frames` 批次路徑）依賴這個欄位**——批次路徑應該維持原本呼叫 `forward()` 的邏輯不變。
4. `Decoder12Hz::reset_state()`（若尚未存在則新增，需實作 `TtsDecoder` trait 要求的 `reset_state`）呼叫 `self.pre_transformer.reset_state()` 並歸零 position 計數器。
5. 更新 `decoder_12hz.rs:106-109` 那則自承 O(n²) 的 log 訊息，改為描述新的常數時間行為。

**驗收標準**：
- 針對同一份測試音檔／token 序列，串流逐 frame 解碼的完整輸出，與既有批次 `decode_frames` 一次性解碼的輸出，逐樣本比對（或至少逐 frame 算 cosine），維持原本已有的一致性水準（batch 與 streaming 理論上應該產生相同或極接近的音訊，這點是 cpp 架構設計的核心保證，見對齊計畫 §3.3 表格）。
- 新增/更新 benchmark：streaming 模式下第 N 個 frame 的處理延遲不隨 N 增長。
- 現有 `tests/`（`talker_alignment_test.rs`、`integration_test.rs` 等）全數通過。

**給 deepseek-v4-flash 的貼入式 prompt**：
````
前提：WO-3a 已完成，src/codec/transformer.rs 的 PreTransformer 現在有了 step(&mut self,
x: &Tensor, position: usize) 與 reset_state(&mut self) 方法，KV cache 是固定大小的
sliding-window ring，不會隨呼叫次數增長。

你現在要修改 src/decoder_12hz.rs，讓 Decoder12Hz 的串流（streaming，非批次 decode_frames）
路徑改用新的 step() API，取代目前每次都把全部歷史丟給 pre_transformer.forward() 的做法。

請完成：
1. 找到 Decoder12Hz 中負責串流解碼的方法（搜尋呼叫 pre_transformer.forward 的地方，且該
   方法不是 decode_frames 批次路徑）。把該處呼叫改成 self.pre_transformer.step(&frame_tensor,
   position)。
2. 在 Decoder12Hz struct 新增一個 usize 欄位當作 position 計數器，串流路徑每處理一個 frame
   就遞增 1，reset_state() 時歸零。
3. 檢查 pre_conv_buffer 欄位（目前是累積全部歷史的 Vec<f32>）：如果它的唯一用途是把每個
   frame 的 conv 輸出暫存起來以便丟給 transformer，且現在 transformer 已經改用逐 frame 的
   step()，這個欄位應該可以移除或縮小成只暫存「當前這一個 frame」。動手前務必先確認
   decode_frames（批次路徑）是否也依賴這個欄位——如果有依賴，批次路徑要維持原樣不動，
   只影響串流路徑用到的部分。
4. 確保 Decoder12Hz 有實作 reset_state()（TtsDecoder trait 要求），內部呼叫
   self.pre_transformer.reset_state() 並把 position 計數器歸零。
5. 找到目前印出 "no trim – O(n²) for streaming" 的那行 log（在檔案開頭 from_safetensors
   附近），更新成描述現在已經是常數時間串流的訊息。
6. 寫一個測試或使用既有測試框架驗證：用同一份測試資料，串流逐 frame 呼叫的完整輸出，
   跟批次 decode_frames 一次性解碼的輸出要接近一致（逐 frame 算 cosine similarity 或直接
   比對樣本數值，容忍浮點誤差）。
7. 執行 cargo build、cargo test，全數通過，並在 benches/ 目錄下確認或新增一個 benchmark
   驗證：串流模式下第 N 個 frame 的處理延遲不會隨 N 增長（例如量測第 10 個跟第 500 個
   frame 的延遲應該接近）。

程式碼註解用繁體中文（zh-TW），技術詞彙保留英文；禁止 unwrap()/expect()；熱路徑不可有
Vec::new()/format!/Box::new/clone()。完成後回報異動檔案與行號摘要。
````

---

## WO-4：GGUF talker 權重讀取 PoC（tensor 命名對照）

**優先度**：🟡 P1　**規模**：M

**背景**：`AGENTS.md:16,83` 明文允許「文本前端 LLM 子模塊」（即 talker backbone）使用 GGUF，`qwentts.cpp` 專案已發布並驗證過 `qwen-talker-{0.6b,1.7b}-{base,customvoice,voicedesign}-{F32,BF16,Q8_0,Q4_K_M}.gguf`（HuggingFace `Serveurperso/Qwen3-TTS-GGUF`）。這張工單只做**讀取驗證**，不接線到正式推理路徑（那是 WO-5）。

**涉及檔案**：新增 `examples/gguf_talker_probe.rs`（或類似命名）；不動現有 `src/talker/*`。

**背景知識（GGUF metadata，來自 qwentts.cpp/docs/ARCHITECTURE.md）**：
- Tensor 命名：`talker.text_embd.weight`、`talker.codec_embd.weight`、`talker.text_proj.fc{1,2}.{weight,bias}`、`talker.codec_head.weight`、`talker.output_norm.weight`、`talker.blk.{0..27}.attn_{q,k,v,o}`、`talker.blk.{0..27}.attn_q.q_norm` / `attn_k.k_norm`、`talker.blk.{0..27}.attn_norm` / `ffn_norm`、`talker.blk.{0..27}.ffn.{gate,up,down}_proj`、`code_pred.blk.{0..4}.*`、`code_pred.output_norm.weight`、`code_pred.mtp_proj.{weight,bias}`（僅 1.7B）、`spk_enc.*`（僅 base）。
- Layout 慣例：ggml 對 `(out, in)` 的 Linear 存成 `ne[0]=in, ne[1]=out`；`ggml_mul_mat(A, B)` 等價於 PyTorch 的 `A @ B^T`。轉成 Candle Tensor 時需要對齊到 Candle/safetensors 慣用的 `(out, in)` 慣例，實務上等於讀出來的 shape 要 transpose 一次（實際是否需要 transpose 要以讀出來的 shape 實測為準，不要憑空假設）。

**修改規格**：
1. 新增一支範例程式，用 `candle_core::quantized::gguf_file`（`candle-core` 內建，`Cargo.toml` 已有 `candle-core = "0.10"` 依賴，不需加新 crate）讀取一個本機的 `qwen-talker-*.gguf` 檔案路徑（由 CLI 參數指定）。
2. 列印出該 GGUF 檔案內所有 tensor 的名稱、shape、dtype，並與上面列出的預期命名清單做逐一比對，印出「找到/缺少」報表。
3. 挑選其中 1-2 個 tensor（例如 `talker.text_embd.weight`）實際 dequantize 成 F32 `Tensor`，印出 shape 與前幾個數值，確認讀取管線可用。
4. 產出一份 `docs/gguf_tensor_mapping.md`，記錄實測到的 tensor 命名、shape、與 Rust 現有 `TalkerModel`/`WeightLoader`（`src/talker/weight_loader.rs`）期待的鍵名/shape 的對照表，包含是否需要 transpose 的實測結論。

**驗收標準**：
- 範例程式能成功列出 GGUF 檔案內所有 tensor 且無 panic。
- 至少 1 個 tensor 成功 dequantize 並印出合理數值（非全零、非 NaN）。
- `docs/gguf_tensor_mapping.md` 產出且內容基於實測（不是憑空推測）。

**給 deepseek-v4-flash 的貼入式 prompt**：
````
你正在替 Rust 專案 qwen3tts-rs 新增一支探測用範例程式，驗證能否用 candle-core 內建的
GGUF 讀取功能（candle_core::quantized::gguf_file 模組）讀取第三方專案 qwentts.cpp
（https://github.com/ServeurpersoCom/qwentts.cpp）發布的 GGUF talker 權重檔案。這是
PoC 階段，只需要讀取與驗證，不用接到正式推理管線。

背景：candle-core 版本是 "0.10"（Cargo.toml 已有此依賴，不需要新增），candle-core 內建
支援讀取 GGUF 檔案與其量化格式（Q8_0、Q4_K 等）。

請完成：
1. 新增 examples/gguf_talker_probe.rs，接受一個 CLI 參數（GGUF 檔案路徑）。
2. 用 candle_core::quantized::gguf_file 的 API 開啟並讀取該 GGUF 檔案，列出檔案內所有
   tensor 的名稱、shape、dtype，逐行印到 stdout。
3. 印出報表：比對讀到的 tensor 名稱清單跟下面這份預期清單，標出「找到」與「缺少」：
   talker.text_embd.weight, talker.codec_embd.weight, talker.text_proj.fc1.weight,
   talker.text_proj.fc1.bias, talker.text_proj.fc2.weight, talker.text_proj.fc2.bias,
   talker.codec_head.weight, talker.output_norm.weight,
   talker.blk.0.attn_q.weight, talker.blk.0.attn_k.weight, talker.blk.0.attn_v.weight,
   talker.blk.0.attn_o.weight, talker.blk.0.attn_q.q_norm, talker.blk.0.attn_k.k_norm,
   talker.blk.0.attn_norm, talker.blk.0.ffn_norm,
   talker.blk.0.ffn.gate_proj.weight, talker.blk.0.ffn.up_proj.weight, talker.blk.0.ffn.down_proj.weight,
   code_pred.blk.0.attn_q.weight, code_pred.output_norm.weight, code_pred.mtp_proj.weight
   （blk.0 只是示意，實際檔案裡應該有 blk.0 到 blk.27，talker 用 blk.{0..27}，
   code_pred 用 blk.{0..4}）。
4. 挑選 talker.text_embd.weight 這個 tensor，實際呼叫 dequantize 相關 API 把它轉成
   F32 的 candle_core::Tensor，印出它的 shape 與前 10 個數值，確認讀取管線真的可用
   （數值不應該全部是 0 或 NaN）。
5. 建立 docs/gguf_tensor_mapping.md，用表格記錄：實測到的 tensor 名稱、shape、dtype，
   對照 src/talker/weight_loader.rs 裡目前 WeightLoader 期待讀取的 safetensors 鍵名
   跟 shape（你需要先讀一下 src/talker/weight_loader.rs 跟 src/talker/config.rs 了解
   現有命名慣例），並記錄你實測出來是否需要 transpose 才能對上（GGML 的 Linear 權重
   慣例是 (in, out)，PyTorch/safetensors 通常是 (out, in)，實際要不要轉置以你程式跑出來
   的 shape 為準，不要憑空假設）。
6. 這支程式只是探測用途，不需要處理成生產品質，但仍要能編譯、執行、不 panic。

如果你手邊沒有實際的 GGUF 檔案可以測試，先把程式寫好、能編譯通過，並在
docs/gguf_tensor_mapping.md 裡註明「尚未用真實檔案驗證，待有檔案後執行 cargo run
--example gguf_talker_probe -- <path> 補完」。

程式碼註解用繁體中文（zh-TW），技術詞彙保留英文。
````

---

## WO-5：GGUF talker backend 正式整合，取代自製量化校準

**優先度**：🟡 P1　**規模**：L　**依賴**：WO-4 完成且產出的 tensor mapping 已驗證可用

**背景**：`docs/talker_native_todo.md` 記錄 talker 量化仍停留在「需要 activation calibration、真正 int8/int4 compute kernel，目前走 dequant 回 F32」的 experimental 階段。WO-4 驗證讀取管線可行後，這張工單接上正式推理路徑。

**涉及檔案**：`src/talker/weight_loader.rs`、`src/talker/model.rs`、`src/talker/mod.rs`（新增 backend 分支）、CLI 入口（`examples/synthesize.rs`、`examples/synthesize_batch.rs`）

**修改規格**：
1. 在 `src/talker/weight_loader.rs` 新增一個從 GGUF 載入的路徑（例如 `WeightLoader::from_gguf(path, device)` 或獨立的 `GgufWeightLoader`），沿用 WO-4 驗證出來的 tensor 命名與 shape 轉換規則。
2. 新增一個 backend 選項（例如 CLI `--talker-backend gguf|safetensors`，或沿用現有 `--backend` 概念擴充），讓使用者可以指定用 GGUF 或既有 safetensors 載入 talker。**兩條路徑並存**，不要移除既有 safetensors 路徑。
3. `WeightLoader::from_dir()`（既有 safetensors 路徑）與新的 GGUF 路徑最終都要能建構出同一個 `TalkerModel`/`CodePredictor` 結構，確保下游 `forward`/`step` 邏輯完全不用改。
4. 新增數值比對測試：用同一組固定輸入，分別跑 safetensors 路徑與 GGUF 路徑（需要一份對應的 GGUF 測試檔案，可用 WO-4 探測過的檔案），比對兩者輸出的 codebook-0 logits，**cosine ≥ 0.995**（沿用 `AGENTS.md:85` 既有量化驗收門檻）。
5. 若驗收通過，可在 `docs/talker_native_todo.md` 補一則進度記錄（比照現有的日期戳記格式），並視情況安排淘汰 `quantize_tokenizer.exe` 相關的 talker 量化舊路徑（若存在的話）——**這一步需要 Steven 本人確認後才執行刪除，agent 只需要在報告中列出建議刪除清單，不要自行刪除**。

**驗收標準**：
- 兩條 backend 路徑（safetensors / GGUF）都能成功合成語音並產生非靜音、非爆音的 WAV。
- 數值比對測試 cosine ≥ 0.995。
- 現有 CLI 與既有測試不受影響（safetensors 路徑維持原樣可用）。

**給 deepseek-v4-flash 的貼入式 prompt**：
````
前提：WO-4 已完成，docs/gguf_tensor_mapping.md 已經有實測過的 GGUF tensor 命名與 shape
對照表（讀取這份文件了解命名規則跟是否需要 transpose）。

你現在要幫 qwen3tts-rs 的 talker 模組新增一條「從 GGUF 載入權重」的正式路徑，跟現有的
safetensors 路徑並存，讓使用者可以二選一。目的是之後可以直接用第三方專案 qwentts.cpp
已經發布、已經驗證過的量化 GGUF 檔案（Q8_0 / Q4_K_M）取代目前還在 experimental 階段、
自製的 talker 量化校準流程，藉此縮短工作量、提高量化品質可信度。

請完成：
1. 在 src/talker/weight_loader.rs（先讀一遍現有的 WeightLoader / from_dir 實作，理解
   現有介面長什麼樣）新增一個從 GGUF 讀取的路徑，可以是 WeightLoader 上新增
   from_gguf(path, device) 建構子，或是新增一個獨立的 GgufWeightLoader 但要能產出跟
   現有 WeightLoader 相容的介面，讓下游 TalkerModel/CodePredictor 建構程式碼不用改。
   套用 docs/gguf_tensor_mapping.md 裡記錄的命名對照與 transpose 規則。
2. 在 CLI 範例（examples/synthesize.rs、examples/synthesize_batch.rs）新增一個參數
   （例如 --talker-backend gguf 或 safetensors，預設維持 safetensors 不變既有行為），
   讓使用者可以選擇要用哪條路徑載入 talker 權重。兩條路徑都要能正常運作，不要移除或
   破壞既有的 safetensors 路徑。
3. 新增一個數值比對測試：準備同一組固定輸入（可以用專案裡既有的 fixture 或新建一個
   小的），分別用 safetensors 路徑跟 GGUF 路徑建構 talker、跑同一次 forward，比較兩者
   輸出的 codebook-0 logits 的 cosine similarity，要求 >= 0.995。如果沒有實際的 GGUF
   測試檔案可用，把測試寫成 #[ignore] 並在測試註解說明需要什麼檔案才能執行，同時確保
   程式碼本身能編譯通過。
4. 完成後在 docs/talker_native_todo.md 檔案最後面，比照現有的日期戳記格式（例如
   "- 2026-XX-XX GGUF talker backend update:"）補一段進度記錄，簡述做了什麼、驗證結果
   如何。
5. 如果你在過程中發現有現成的、自製的 talker 量化校準相關程式碼因為這個新路徑而變得
   多餘，不要自行刪除，改成在你的回報中列出「建議淘汰清單」讓人工確認。

執行 cargo build 與 cargo test（含新測試，若因缺少 GGUF 檔案而 #[ignore] 也沒關係）
確保全數通過或明確標記為 ignored。程式碼註解用繁體中文（zh-TW），技術詞彙保留英文，
禁止 unwrap()/expect()。
````

---

## WO-6：Philox4x32-10 PRNG 移植

**優先度**：🟡 P1（若不需要跨實作 bit-exact 可重現性可降為 P2）　**規模**：S

**背景**：`src/talker/sampling.rs:67-75` 目前用 SplitMix64，統計性質沒問題，但跟 PyTorch/HF/cpp 用的 Philox4x32-10 不同演算法，同一個 `seed` 在不同實作間不會產生相同亂數序列，無法做到跨實作可重現性驗證。

**涉及檔案**：`src/talker/sampling.rs`（新增獨立的 PRNG 模組，例如 `src/talker/philox.rs`）

**修改規格**：
1. Philox4x32-10 是公開發表的演算法（Random123 論文，非 `qwentts.cpp` 專屬程式碼），可自行用 Rust 實作 counter-based PRNG：4 個 32-bit word 的 key/counter，10 輪 round function（每輪用 Philox 的固定 multiplier 常數對 counter 做 high/low 乘法混合 + key 遞增）。**不要抄 `qwentts.cpp/src/philox.h` 的程式碼**，只參考其角色定位（counter-based、可跳位）自行依公開演算法規格重新實作並附上單元測試（例如與已知的 Philox4x32-10 test vector 比對，若無法取得 test vector，至少確保同 `(seed, counter)` 輸入永遠得到相同輸出、不同 `counter` 得到不同輸出）。
2. 新增 `PhiloxSampler`（或在既有 `Sampler` 上加一個 backend enum：`SplitMix64` / `Philox4x32_10`），介面與現有 `next_f64_state` 對齊（回傳 `[0,1)` 均勻分布浮點數），讓 `Sampler` 可以擇一使用，預設維持 `SplitMix64`（避免影響既有測試/行為），新 backend 需要顯式指定才啟用。

**驗收標準**：
- 新增單元測試證明同 `(seed, counter)` 輸入具確定性、不同輸入分布近似均勻（可用簡單卡方檢定或至少檢查跨大量樣本的 mean/variance 落在合理範圍）。
- 既有預設行為（`SplitMix64`）不受影響，既有測試全數通過。

**給 deepseek-v4-flash 的貼入式 prompt**：
````
你正在替 Rust 專案 qwen3tts-rs 新增一個 Philox4x32-10 counter-based PRNG 實作，作為
現有 SplitMix64 取樣 RNG 之外的另一個選項，目的是未來可以做跨實作（PyTorch/HuggingFace）
的可重現性比對。

重要：Philox4x32-10 是公開發表的演算法（出自 Random123 論文），不是任何特定專案的專屬
程式碼，請直接依照公開演算法規格自行實作，不要參考或抄任何第三方專案的原始碼。

Philox4x32-10 演算法規格：
- 狀態是 4 個 32-bit word 當作 counter：(c0, c1, c2, c3)，加上 2 個 32-bit word 當作 key：(k0, k1)。
- 每一輪 round function：
  - 用兩個固定的 64-bit multiplier 常數（Philox4x32 標準常數：M0 = 0xD2511F53,
    M1 = 0xCD9CC53B）分別對 (c0, c2) 做寬乘法，得到高低 32 位元。
  - 新的 counter 用高低位元與 key 做 XOR 重新排列（標準 Philox4x32 的 round 混合公式）。
  - key 每輪遞增固定的 Weyl 常數（W0 = 0x9E3779B9, W1 = 0xBB67AE85）。
- 總共執行 10 輪。
- 最終輸出 4 個 32-bit word，可以轉換成 [0,1) 區間的浮點數（例如取其中一個 32-bit word
  除以 2^32）。

如果你對這個演算法的精確位元運算細節不夠有把握，請先搜尋 "Philox4x32-10 algorithm
specification" 或 "Random123 Philox paper" 確認正確的常數與運算順序後再實作，寧可多花
一點時間確認正確性，也不要憑印象拼湊出錯誤的變體。

請完成：
1. 新增 src/talker/philox.rs，實作一個 struct（例如 Philox4x32_10），提供：
   - new(seed: u64) -> Self 建構子
   - next_u32(&mut self) -> u32 或 next_f64(&mut self) -> f64（回傳 [0,1) 均勻分布），
     介面設計上要能相容現有 sampling.rs 裡 next_f64_state 的用法（回傳型別、呼叫方式
     盡量一致，方便之後替換）。
2. 在 src/talker/sampling.rs 的 Sampler 加一個 backend 選項（例如一個 enum
   RngBackend { SplitMix64, Philox4x32_10 }，或是把 next_f64_state 抽成一個 trait，
   兩種 RNG 都實作它），預設值維持現有的 SplitMix64（不要改變既有預設行為），新的
   Philox backend 需要顯式指定才會啟用。
3. 新增單元測試：
   a) 同樣的 (seed, 呼叫序列) 應該每次執行都得到完全相同的輸出序列（確定性）。
   b) 產生幾千個樣本，檢查平均值接近 0.5、分布沒有明顯偏態（簡單統計檢查即可，不用
      嚴謹的卡方檢定）。
   c) 不同 seed 應該產生不同的序列。
4. 確保現有 sampling.rs 的既有測試（使用 SplitMix64 預設行為）完全不受影響，維持全數通過。

程式碼註解用繁體中文（zh-TW），技術詞彙保留英文。完成後執行 cargo build 與 cargo test
確認通過。
````

---

## WO-7：Prompt 模式驗證規則 7 條 checklist 核對

**優先度**：🔵 P2　**規模**：XS

**背景**：`qwentts.cpp` 對合成參數組合有 7 條明確驗證規則（見對齊計畫 §5.2）。Rust CLI 已有部分覆蓋（`docs/talker_native_todo.md` v0.1.11 記錄），這張工單只是逐條核對補缺，不是從零開始寫。

**涉及檔案**：`examples/synthesize.rs`、`examples/synthesize_batch.rs`（或參數驗證邏輯所在的共用模組）

**待核對的 7 條規則**：
1. `--speaker` 給了但 model_type != custom_voice → 應報錯
2. `--instruct` 給了但 model_type == base → 應報錯
3. model_type == custom_voice 但沒給 `--speaker` → 應報錯
4. model_type == voice_design 但 `--instruct`/`--instruct-file` 缺失或空 → 應報錯
5. `--ref-wav`（或 Rust 對應的 `--reference-audio`）給了但 model_type != base → 應報錯
6. `--speaker` 與 `--ref-wav`/`--reference-audio` 同時給 → 應報錯（互斥）
7. `--reference-text` 給了但沒給 `--ref-wav`/`--reference-audio` → 應報錯

**給 deepseek-v4-flash 的貼入式 prompt**：
````
你正在核對 Rust 專案 qwen3tts-rs 的 CLI 參數驗證邏輯（examples/synthesize.rs、
examples/synthesize_batch.rs）是否涵蓋以下 7 條驗證規則。這些規則移植自第三方專案
qwentts.cpp 的生產驗證邏輯。

請先閱讀這兩個檔案裡現有的參數驗證程式碼（搜尋跟 --mode、--speaker、--instruct、
--reference-audio、--reference-text 相關的驗證邏輯），逐條核對下面 7 條規則是否已經
被涵蓋、涵蓋的方式是否會回傳清楚的錯誤訊息：

1. --speaker 給了但目前載入的模型 model_type 不是 custom-voice → 應報錯
2. --instruct 給了但 model_type 是 base → 應報錯
3. model_type 是 custom-voice 但沒給 --speaker → 應報錯
4. model_type 是 voice-design 但 --instruct 跟 --instruct-file 都缺失或空 → 應報錯
5. --reference-audio 給了但 model_type 不是 base → 應報錯
6. --speaker 與 --reference-audio 同時給了（這兩個應該互斥）→ 應報錯
7. --reference-text 給了但沒給 --reference-audio → 應報錯

對於每一條規則，請在報告中列出：
- 目前是否已經有對應的檢查（是/否/部分涵蓋）
- 如果有，程式碼位置（檔案:行號）
- 如果沒有或只是部分涵蓋，補上缺少的檢查，錯誤訊息要清楚說明是哪個參數組合不合法

補齊程式碼後，如果專案有既有的 CLI 參數驗證測試，比照既有測試風格新增涵蓋這 7 條規則
的測試案例（每條規則至少一個測試）。執行 cargo build 與 cargo test 確認通過。

程式碼註解用繁體中文（zh-TW），技術詞彙保留英文；禁止 unwrap()/expect()。
````

---

## WO-8：RoPE interleaved 交叉驗證腳本（僅驗證，不修改主線）

**優先度**：🔵 P2　**規模**：S

**背景**：`src/talker/config.rs` 的 `rope_interleaved: true` 與第三方專案 `qwentts.cpp` 的 GGUF metadata `mrope_interleaved: false` 不一致，但 Rust 現有實作已通過官方 PyTorch 參考的 exact-token-match 驗證，證據力較強。**這張工單只做獨立驗證，不修改 `src/talker/primitives.rs` 或 `config.rs` 的既有邏輯**，除非驗證結果明確顯示現有實作有誤才回報建議、交由人工決定是否採納。

**涉及檔案**：新增獨立測試/範例（例如 `examples/rope_pairing_probe.rs`），不動主線程式碼。

**修改規格**：
1. 寫一支獨立小程式，用固定的假輸入向量（例如 `head_dim=128`，數值為 `0..128` 遞增），分別跑：
   - 現有 `src/talker/primitives.rs` 裡 `rope_interleaved=true` 分支的計算邏輯
   - 現有同一段程式碼但假設 `rope_interleaved=false` 分支的計算邏輯（若程式碼本身有 if/else 兩個分支，直接呼叫兩種）
2. 印出兩種模式下，輸出向量中「哪些維度索引被兩兩配對做旋轉」的具體規則（例如 `(0,64)` pairing 還是 `(0,1)` pairing），文字描述清楚兩種模式的具體差異。
3. 產出一份 `docs/rope_pairing_verification.md`，記錄兩種模式的具體配對規則、與現有 `docs/talker_native_todo.md` 中「exact token match」驗證的關係說明（也就是誠實記錄：現有驗證是否真的會被這個欄位的取值影響，或者在目前測試涵蓋的 fixture 下這個欄位剛好不影響輸出）。**不要下結論說哪個是對的，只陳述觀察到的事實**，讓人工判斷。

**給 deepseek-v4-flash 的貼入式 prompt**：
````
你正在替 Rust 專案 qwen3tts-rs 寫一支「觀察用」的驗證程式，目的是搞清楚
src/talker/config.rs 裡 rope_interleaved 欄位（在 src/talker/primitives.rs 大約第
238 行左右有 if self.rope_interleaved { ... } else { ... } 分支）的 true/false 兩種
模式，具體上是怎麼對向量的維度做配對旋轉的。這是純粹的觀察/記錄任務，不要修改
primitives.rs 或 config.rs 裡任何現有邏輯或預設值。

請完成：
1. 先閱讀 src/talker/primitives.rs 裡跟 RoPE 相關的程式碼，找到 rope_interleaved 的
   if/else 兩個分支，理解兩種模式各自的維度配對規則。
2. 新增 examples/rope_pairing_probe.rs，構造一個固定的假輸入向量（例如 head_dim=128，
   數值就用 0.0 到 127.0 遞增，方便肉眼追蹤每個維度的來源），分別呼叫兩個分支的邏輯
   （如果程式碼結構上不方便直接呼叫兩個分支，就複製一份邏輯出來個別測試，並在程式
   註解說明這只是探測用途）。
3. 印出兩種模式下，輸出向量的每個維度是由輸入的哪兩個維度組合而來（例如維度 0 和
   維度 64 配對，或維度 0 和維度 1 配對），用清楚的文字或表格列出配對規則。
4. 建立 docs/rope_pairing_verification.md，記錄：
   - 兩種模式各自實際觀察到的配對規則
   - 目前 docs/talker_native_todo.md 裡記錄的 exact token match 驗證（talker 單幀與
     兩幀自回歸測試）用的是哪個模式（讀 config.rs 裡的預設值確認）
   - 誠實記錄你的觀察，不要下結論說哪個模式「正確」或「錯誤」，只陳述事實，讓人工
     之後自行判斷是否需要進一步跟官方 PyTorch 參考實作或其他來源比對。
5. 這支程式跟文件都是輔助分析用途，不需要納入正式測試套件，但要能編譯通過。

程式碼註解用繁體中文（zh-TW），技術詞彙保留英文。
````

---

## WO-9：25Hz decoder 改為建構期報錯，移出當期 sprint

**優先度**：⚪️ 清理　**規模**：XS

**背景**：`qwentts.cpp` 只做 12Hz，沒有 25Hz Flow-Matching DiT 可參照，這塊工作沒有「對齊」對象。目前 `src/decoder_25hz.rs:75-80` 是在**呼叫時**才回傳 `Err`，建議改成**建構時**就報錯，減少維護負擔（沿用專案自己在 `docs/optimization_suggestions_2026-06-10.md:63-65` 提過的建議）。

**涉及檔案**：`src/decoder_25hz.rs`

**修改規格**：
1. `Decoder25Hz::new()`（`decoder_25hz.rs:34-56`）在建構完成前，若確認整條 Flow Matching 管線本來就無法產生有效輸出（依現有 `#[allow(dead_code)]` 標記的範圍判斷），改為直接回傳明確的 `Err`（例如 `Error::Config("25Hz Flow-Matching decoder 尚未實作，且無可參照的第三方實作，暫緩此模式；如需啟用請參考 spec.md Phase 2".into())`），而不是等到 `decode_chunk()` 才報錯。
2. **不要刪除** `src/codec/flow_matching.rs` 裡任何現有程式碼（那些 `#[allow(dead_code)]` 的結構體保留，未來可能還會用到），只調整 `Decoder25Hz` 對外的失敗時機。
3. 更新任何呼叫 `Decoder25Hz::new()` 的地方（CLI、測試），確保新的建構期錯誤被正確處理（不要 panic）。
4. 既有 `#[test] fn test_decoder_creation()`（`decoder_25hz.rs` 檔案內測試）需要相應調整：如果這個測試原本預期建構會成功，現在應該改成預期建構回傳 `Err` 並驗證錯誤訊息內容，或標記為 `#[ignore]` 並說明原因。

**給 deepseek-v4-flash 的貼入式 prompt**：
````
你正在修改 Rust 專案 qwen3tts-rs 的 src/decoder_25hz.rs。目前 Decoder25Hz::new() 可以
成功建構，但呼叫 decode_chunk() 時才回傳 "25Hz decoder not yet implemented (Phase 2)"
的錯誤。因為這個模式短期內沒有明確的實作依據可以參照，希望把失敗時機提前到建構階段，
減少後續維護負擔（讓呼叫端一開始選錯模式就立刻知道，而不是走到解碼那一步才發現）。

請完成：
1. 修改 Decoder25Hz::new()，在建構完成前直接回傳一個明確的 Err，錯誤訊息使用
   Error::Config("25Hz Flow-Matching decoder 尚未實作，且無可參照的第三方實作，
   暫緩此模式；如需啟用請參考 spec.md Phase 2".into())（或風格相近、意思相同的訊息，
   請用繁體中文撰寫錯誤訊息）。
2. 不要刪除 src/codec/flow_matching.rs 裡任何現有程式碼（那些標記 #[allow(dead_code)]
   的結構體要保留，未來可能還會用到），只調整 Decoder25Hz 這一層對外的失敗時機。
3. 搜尋整個專案裡呼叫 Decoder25Hz::new() 的地方（CLI 範例、測試），確認新的建構期
   錯誤都被妥善處理（用 ? 或明確的 match 處理，不要 panic 或 unwrap）。
4. 修改 decoder_25hz.rs 檔案內既有的 #[test] fn test_decoder_creation()：這個測試目前
   應該是預期建構成功，現在請改成預期建構回傳 Err，並驗證回傳的錯誤訊息內容確實提到
   「尚未實作」或類似字樣。
5. decode_chunk() 方法本身可以保留原本的錯誤處理邏輯不用特別動（既然建構階段就會擋
   下來，這裡實務上不會再被呼叫到，但保留防禦性程式碼沒有壞處）。
6. 執行 cargo build 與 cargo test，確保全數通過。

程式碼註解用繁體中文（zh-TW），技術詞彙保留英文；禁止 unwrap()/expect()。
````

---

## 附註：給協調 agent（orchestrator）的分派建議

- WO-1、WO-2、WO-7、WO-9 彼此獨立、風險低，可以同時分給多個 deepseek-v4-flash session 平行處理，不會互相衝突（各自碰的檔案不重疊，除了都可能觸碰 `examples/synthesize.rs`——WO-7 與 WO-9 若剛好都要改這個檔案，建議序列執行避免 merge 衝突）。
- WO-3a → WO-3b 有嚴格順序依賴，且是本次最核心的效能修正，建議指派後先跑完整的 `cargo test` + 新增的 streaming/batch 一致性測試再進下一張工單，不要跳過驗收就繼續。
- WO-4 → WO-5 可以跟 WO-3 系列平行進行（不同檔案），但 WO-5 的驗收依賴 WO-4 產出的 `docs/gguf_tensor_mapping.md` 是否可靠，若 WO-4 的探測結果顯示 tensor mapping 不確定/有疑慮，WO-5 應該暫緩，回報人工判斷。
- 每張工單完成後，依專案 RSI（Recursive Self-Improvement）流程把過程中發現的意外狀況（例如某個函式簽名跟工單描述的不完全一樣、需要臨場調整）寫進 `lessons.md`，方便後續工單或未來類似任務參考。
- 因為 deepseek-v4-flash 是 flash 級模型，建議每張工單完成後，由另一個 session（可以是同一個模型、或換更強的模型做 review）快速過一次 diff，確認範圍沒有超出工單邊界（尤其是「邊界（不要做）」段落列出的項目）。
