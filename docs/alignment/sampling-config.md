# P02-T03 Sampling Config Matrix

## 目標
- 建立 `generation_config.json` 與 CLI 取樣參數整合邏輯的可審計契約。
- 固定 `talker`/`subtalker` 在 `do_sample`、`temperature`、`top_k`、`top_p`、`repetition_penalty` 上的優先順序與預設回退。
- 限制生成路徑分支：`talker.do_sample=true` 時走 `generate_sampled`，否則走 `generate`。

## 參照來源
- 產生腳本：`tools/generate_sampling_config_matrix.py`
- 參照版本：`82cd05b9f3a175612dc89fd6943e610fab096ef5`
- Fixture：`fixtures/alignment/p02_sampling_config_matrix.json`

## 驗證重點
- 無檔案 `generation_config.json` 時回退到預設：
  - `talker` 預設：`do_sample=true, temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.05`
  - `subtalker` 預設：`do_sample=true, temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.0`
- `subtalker_dosample=false` 時，`subtalker` 仍保留設定溫度值，實際採樣模式由 `do_sample=false` 的布林值直接控制，不再以 `temperature` 是否大於 0 作為模式開關。
- `talker.do_sample=false` 時整體走 `generate`；`talker` 是否 greedy 只受 `do_sample` 決定，不再用 `temperature <= 0` 分支。
- CLI `--temperature / --top_k / --top-p` 只覆寫 talker 的三個主參數，不覆寫 `subtalker` 參數與 `repetition_penalty`。

## 交付檔案
- `tools/generate_sampling_config_matrix.py`
- `fixtures/alignment/p02_sampling_config_matrix.json`
- `tests/sampling_config_test.rs`
- `config/fixtures.json`（新增 `p02-sampling-config-matrix`）
- `docs/alignment/sampling-config.md`

## 指令
- 生成：`python tools/generate_sampling_config_matrix.py`
- 檢查：`python tools/generate_sampling_config_matrix.py --check fixtures/alignment/p02_sampling_config_matrix.json`

## 測試
- `tests/sampling_config_test.rs` 會驗證：
  - 每個案例預期路由 (`generate` / `generate_sampled`)
  - 兩條分支有效參數結果
  - `config/fixtures.json` 的 hash 錄入與一致性
