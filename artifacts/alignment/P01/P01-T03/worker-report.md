# P01-T03 Worker Report

## 修改摘要
- `src/talker/primitives.rs`
  - 調整 `MultimodalRotaryEmbedding::mrope_axis_for_dim` 的 interleaved/non-interleaved 映射邏輯，使其對應到:
    - interleaved：`axis_dim = dim % half_dim`，對應 `axis_idx` 在 `axis_idx..mrope_section[axis_idx]*3`，以步進 3 選軸。
    - non-interleaved：完整 head 遍歷 `[s0,s1,s2,s0,s1,s2]` 模式。
  - 修正 `cached_positions_from_delta` 呼叫基底邏輯（保留 `query_len` 逐位累加）。
- `src/talker/talker.rs`
  - 將 `compute_position_ids` 的 pad 行為修正為：只將 pad 位置替換為 `1`，保留 valid token 的 `cumsum-1` 結果。
  - generation 快取位置改為 `cache_position_start = seq_len + gen_step`，避免錯誤對齊。
- `tests/mrope_reference_test.rs`
  - 重新對齊預期公式，覆蓋：
    - prefill 的 position_ids 與 delta
    - cached positions 與 query/base/delta 關係
    - interleaved/non-interleaved axis 邏輯邊界
    - q/k 旋轉與手算參考向量比對
    - 非法輸入 fail-fast
  - 修正混合 padding 測試案例預期值（`[0,1,1,0,1]` 對應為 `[1,0,1,1,2]`）。
- `docs/alignment/mrope-parity.md`
  - 補充測試與命令執行紀錄狀態：本輪嘗試執行所有 assignment 指定 cargo 指令，但環境缺少 `cargo`。

## 目的
- 對齊 Qwen3 官方 M-RoPE 官方語意：
  - `cumsum(mask)-1` + pad=1 的位置計算
  - `[batch,1]` delta
  - 快取 decode 的 `cache_position_start + rope_delta + arange(query_len)`
  - 通道 axis 映射邏輯
