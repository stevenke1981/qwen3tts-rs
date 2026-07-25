# P02-T02 Repetition Penalty Parity

## 目標
- 對齊 qwentts.cpp（`82cd05b9f3a175612dc89fd6943e610fab096ef5`）的 repetition penalty 在取樣鏈路中的行為。
- 保持操作順序固定：
  `suppression -> repetition penalty -> f32 temperature divide -> top-k -> top-p -> softmax/multinomial`。
- `temperature <= 0`（Greedy）時不套用 repetition penalty、也不消耗 Philox。

## 參考資料
- Oracle：`tools/generate_repetition_penalty_vectors.py`
- Fixture：`fixtures/alignment/p02_repetition_penalty_vectors.json`
- 參照版本：`82cd05b9f3a175612dc89fd6943e610fab096ef5`

## 驗證要點
- 重複 history token 只轉換一次（去重後一次處理）。
- 超出字彙表大小的 history token 忽略。
- `penalty == 1.0` 或 empty history 不改變 logits。
- Greedy（`temperature <= 0`）直接回傳最高候選，不進入 Philox 多項抽樣，也不轉換 repetition。
- 取樣鏈路保留 `top-k` 之後才做 `top-p`。
- talker c0 歷史只接受已完成 frame 的非 EOS token；`code_predictor` 使用空 history。

## 已新增檔案
- `tools/generate_repetition_penalty_vectors.py`
- `fixtures/alignment/p02_repetition_penalty_vectors.json`
- `tests/repetition_penalty_test.rs`
- `config/fixtures.json`（新增 `p02-repetition-penalty` 項目）
