# M-RoPE Parity Notes (P01-T03)

## 變更摘要
- 將 `MultimodalRotaryEmbedding::forward_single_position` 改為接收
  `[batch, query_len]` 的位置張量，與快取解碼公式一致。
- 補齊 `forward` 形狀驗證，限定 `position_ids` 必為 `[3, batch, seq_len]`。
- 修正 `compute_position_ids`：
  - `position_ids = cumsum(mask) - 1`
  - padding token 位置填成 `1`
  - `rope_delta = max_position + 1 - valid_length`
  - 回傳 `pos` 展成 `[3, batch, seq_len]` 與 `delta` 為 `[batch, 1]`
- 生成步驟改為
  `positions = arange(query_len) + cache_position_start + rope_delta`
  並改用 `cached_positions_from_delta` 取得每步位置。
- 新增 M-RoPE 對齊整合測試 `tests/mrope_reference_test.rs`。

## 驗證測試清單
- `compute_position_ids_prefill_and_delta_match_expected_formula`
  - 驗證 `cumsum - 1`、padding 設為 1、`delta = max + 1 - valid`
- `cached_positions_follow_cache_position_plus_delta_formula`
  - 驗證快取位址 `cache_position_start + delta + query_idx`
  - 驗證 `[batch,1]` 形狀檢核
- `forward_single_position_matches_general_forward_with_equal_axes`
  - 驗證 fast-path 與一般 forward 在各軸同值情境一致
- `apply_multimodal_rotary_pos_emb_matches_manual_reference_values`
  - 驗證旋轉公式（q/k）結果
- `invalid_mrope_inputs_fail_fast`
  - 驗證 `theta`、mrope_section 長度/總和與 shape 錯誤的早期失敗

## 目前狀態
- 指派檔中要求的程式碼與測試檔皆已新增/修改。
- 主代理已使用 `C:\Users\steven\.cargo\bin\cargo.exe` 完成格式、CPU
  編譯、相關 library 測試、7 項 reference integration tests 與兩個
  Candle 範例編譯；全部通過。
- 獨立 GPT-5.3 Codex Spark 審查結果：`ACCEPT`。
