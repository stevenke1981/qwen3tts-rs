# RoPE Interleaved 配對規則驗證報告

> 更新日期：2026-07-26
> 驗證工具：`examples/rope_pairing_probe.rs`
> 標的 config：`head_dim=128`、`mrope_section=[24,20,20]`、`rope_interleaved=true`（預設）

## 背景

`src/talker/config.rs` 的 `rope_interleaved` 欄位控制 3D Multimodal RoPE 如何將 `head_dim`
維度分配到三個模態軸（text=0、audio=1、vision=2）。`qwentts.cpp` 的 GGUF metadata 記錄
`mrope_interleaved=false`，但 Rust 預設 `rope_interleaved=true`，且已通過多項 exact token
match 驗證。

**此文件只記錄觀察事實，不下結論。** 詳見 §4。

---

## 1. 兩種模式的具體配對規則

### 1.1 Interleaved（`rope_interleaved=true`）

維度以 **每 3 維度一組交錯** 分配：

| 維度區間 | 分配模式 |
|---------|---------|
| 0..60   | `T A V T A V ...` （3 維度循環，每模態各 20 維度） |
| 60..64  | `T T T T` （殘餘 4 維度全歸 text） |
| 64..124 | 同 0..60（因為 `dim % 64` 鏡像） |
| 124..128| `T T T T` |

總計：
- **Text (T)**: 48 維度（20 + 4 + 20 + 4）
- **Audio (A)**: 40 維度（20 + 20）
- **Vision (V)**: 40 維度（20 + 20）

配對（`rotate_half_and_apply` 中 `d` 與 `d+64` 配對）：
- 全部 64 對都是 **同軸配對** → 保證正確旋轉
- Text pairs: [0, 3, 6, ..., 61, 62, 63]（24 pairs）
- Audio pairs: [1, 4, 7, ..., 55, 58]（20 pairs）
- Vision pairs: [2, 5, 8, ..., 56, 59]（20 pairs）

### 1.2 Blocked（`rope_interleaved=false`）

維度以 **連續區塊** 分配：

| 維度區間 | 軸 | 說明 |
|---------|----|------|
| 0..23   | Text (0) | text 第一半（24 維度） |
| 24..43  | Audio (1) | audio 第一半（20 維度） |
| 44..63  | Vision (2) | vision 第一半（20 維度） |
| 64..87  | Text (0) | text 第二半（24 維度） |
| 88..107 | Audio (1) | audio 第二半（20 維度） |
| 108..127| Vision (2) | vision 第二半（20 維度） |

總計：
- **Text (T)**: 48 維度（24 + 24）
- **Audio (A)**: 40 維度（20 + 20）
- **Vision (V)**: 40 維度（20 + 20）

配對：全部 64 對都是同軸配對。

### 1.3 量化差異

- **86/128** 個維度的 axis 分配在兩種模式下不同
- 兩種模式在 42/128 個維度上相同（這些是維度 0, 3, 6, ..., 57, 60-63 中屬於 text 的維度）
- 兩種模式在每個軸的總維度數完全相同（T=48, A=40, V=40）

---

## 2. 目前驗證覆蓋狀態

參考 `docs/talker_native_todo.md` 確認：

| 驗證項目 | 使用的 config | rope_interleaved 值 | 狀態 |
|---------|-------------|-------------------|------|
| Text projection fixture | Default | `true` | cosine 1.0 |
| Codec embedding/head fixture | Default | `true` | cosine 1.0 |
| Talker attention layer 0 | Default | `true` | cosine 1.0 |
| Talker decoder layer 0 | Default | `true` | cosine 1.0 |
| Talker model prefill | Default | `true` | cosine 1.0 |
| Code predictor first step | Default | `true` | cosine 1.0, next=1965 |
| Code predictor greedy | Default | `true` | exact token match |
| Talker single frame | Default | `true` | exact token match |
| Talker two frame AR | Default | `true` | exact token match |
| Talker prompt generation | Default | `true` | exact token match |

**所有既有驗證都使用 `rope_interleaved=true`（Default config）。**

---

## 3. 與 qwentts.cpp GGUF metadata 的關係

- `qwentts.cpp` GGUF metadata 記錄 `mrope_interleaved=false`
- Rust `TalkerConfig::default()` 設定 `rope_interleaved=true`
- 兩者不一致，但：

**重要觀察：** 現有 PyTorch fixture 驗證全部通過（cosine 1.0 或 exact token match），
這些驗證從 PyTorch 參考實作產出 fixture → Rust 讀取 fixture 比對。
PyTorch 參考實作本身也有 `rope_interleaved` 參數，產生 fixture 時使用的值
與 Rust `Config::default()` 一致（即 `true`）。

因此，**在目前 fixture 涵蓋的輸入範圍內**，`rope_interleaved=true` 是與 PyTorch
參考一致的設定。`qwentts.cpp` 的 GGUF metadata 是否使用了不同的 `rope_interleaved` 值
來產生 GGUF 權重，或者 GGUF 檔案的 `mrope_interleaved=false` 是否正確，需要由人工
進一步比對。

---

## 4. 結論（僅陳述事實）

1. 兩種模式的軸維度分配**截然不同**（86/128 維度不同）。
2. 兩種模式的配對一致性**相同**（全部同軸配對）。
3. 既有 PyTorch fixture 驗證全部使用 `rope_interleaved=true`，且全部通過。
4. GGUF metadata 標示 `mrope_interleaved=false`，但 Rust 預設 `true`。
5. **無法僅由此文件判定哪個值是「正確」的**——需要以下任一才能判定：
   - 用 `rope_interleaved=false` 跑一次同樣的 PyTorch 參考 fixture，比對輸出
   - 確認 PyTorch 參考實作在 fixture export 時使用的實際 `rope_interleaved` 值
   - 用 GGUF 路徑跑一次完整的生成輸出，與 safetensors 路徑比對（前提是兩者使用相同的 RoPE 設定）

---

## 5. 建議下一步

1. 在 `rope_pairing_probe.rs` 中新增實際的角速度計算：對同一組固定輸入，
   用兩種模式分別計算 `cos/sin`，輸出完整張量，比對數值差異。
2. 如果決定追蹤 GGUF metadata，修改 `TalkerConfig::default()` 的
   `rope_interleaved` 為 `false` 後，重新執行所有 alignment test 確認是否
   會破壞既有 fixture 驗證。

---

## 6. 執行方式

```bash
# 重新產生此報告
cargo run --example rope_pairing_probe
```
