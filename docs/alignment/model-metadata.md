# Qwen3-TTS model metadata contract (`config.json`)

本文件定義 text frontend/rust examples 的 metadata 驗證規則，取代以 model-id 字串判斷 runtime 能力的做法。

## 主要欄位（`ModelMetadata`）

讀取來源：`model_dir/config.json`（`ModelMetadata::from_model_dir`）。

- `model_type`：必須為 `qwen3_tts`
- `tokenizer_type`：必須為 `qwen3_tts_tokenizer_12hz`
- `tts_model_size`：必須為 `0b6` 或 `1b7`
- `tts_model_type`：`base` / `custom_voice` / `voice_design`
- `talker_config`
  - `model_type`: `qwen3_tts_talker`
  - `num_hidden_layers`: `> 0`
  - `hidden_size`: `> 0`
  - `head_dim`: `> 0`
  - `num_attention_heads`: `> 0`
  - `code_predictor_config`
    - `model_type`: `qwen3_tts_talker_code_predictor`
    - `num_hidden_layers`: `> 0`
    - `hidden_size`: `> 0`
    - `num_attention_heads`: `> 0`
  - `rope_scaling`
    - `mrope_section`: 長度必須為 **3**
    - `mrope_section` 每一項必須 `> 0`
    - `sum(mrope_section) * 2 == head_dim`
    - `interleaved`: 任意布林值（目前僅保留展示）

## Runtime mode mapping

- `tts_model_type = base` -> `voice-clone`
- `tts_model_type = custom_voice` -> `custom-voice`
- `tts_model_type = voice_design` -> `voice-design`

`GenerationMode::Auto` 不再繞過驗證；其行為會先以
`ModelMetadata::runtime_generation_mode()` 解析為實際模式（base -> voice-clone、custom_voice -> custom-voice、voice_design -> voice-design），再套用同一組規則驗證。無法取得有效 metadata 仍視為錯誤。

## 主要 runtime 驗證規則

`validate_generation_request(metadata, requested_mode, ...)` 的行為：

- `auto`
  - 先映射為 metadata 對應的 runtime 模式，再進行同一組規則驗證
- `custom-voice`
  - 只允許 `tts_model_type = custom_voice`
  - 必須提供 `--speaker`
  - 若提供 `--instruct`，僅允許 1.7B 變種
- `voice-design`
  - 只允許 `tts_model_type = voice_design`
  - 必須提供 `--instruct`（或 `--instruct-file`）
- `voice-clone`
  - 只允許 `tts_model_type = base`
  - 必須提供 `--reference-audio`

## Example 變更

- `examples/synthesize.rs`
  - 任何後端（Python/Candle）在 runtime 解析參數前先呼叫 `ModelMetadata::from_model_dir` 取得 metadata
  - 不再使用 `model_capability()` 做 runtime 檢查
  - `validate_generation_request(...)` 改為無條件執行（含 `Auto`，先映射再驗證）
  - `model_capability()` 僅保留 UI 顯示用途
- `examples/synthesize_batch.rs`
  - 執行前先解析 metadata 並進行 `validate_generation_request(...)`

## 測試與實測檔

- 單元：`src/text_frontend/model_catalog.rs` 新增更多 metadata 與 `mrope_section` 長度/加總測試
- 整合（手動）：`tests/model_metadata_real_test.rs`
  - `#[ignore]`
  - 透過 `QWEN3_TTS_REAL_MODEL_DIR` 指向真實 0.6B Base snapshot 目錄
