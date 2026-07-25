# Runtime token metadata flow (`runtime-token-metadata.md`)

本文件記錄 `CandleLLM` 與 `InputBuilder` 在 runtime 導入 `config.json` metadata 的實作與驗證順序。

## 1) 載入順序

1. `CandleLLM::from_pretrained_dir` 呼叫 `ModelMetadata::from_model_dir(model_dir)`。
2. 從 `config.json` 解析 `ModelMetadata`，包括以下 token 與 mapping：
   - `assistant_token_id`
   - `im_start_token_id`
   - `im_end_token_id`
   - `tts_bos_token_id`
   - `tts_eos_token_id`
   - `tts_pad_token_id`
   - `codec_bos_id`
   - `codec_eos_token_id`
   - `codec_think_id`
   - `codec_nothink_id`
   - `codec_think_bos_id`
   - `codec_think_eos_id`
   - `codec_pad_id`
   - `codec_language_id`
   - `spk_id`
   - `spk_is_dialect`
3. 呼叫 `apply_metadata_to_talker_config` 將欄位套回 `TalkerConfig`。

`from_files` 路徑則透過 `load_metadata_from_safetensors` 在 safetensors 同資料夾尋找 `config.json`，行為一致。

## 2) 生成階段語言與 speaker 解析

`InputBuilder` 在 `build_inner` 中改為 metadata-aware 行為：

- `language` 先行 `resolve_codec_language_id`。
  - `auto` 時，若 `speaker` 有對應 dialect，會用該方言語系。
  - 明確語言時，若語言為中文且 speaker 有 dialect，亦套用 dialect。
  - 未知語言直接回傳 `unsupported language` 錯誤（fail-closed）。
- `speaker` 改為 `resolve_speaker_id`。
  - 依 `config.spk_id` 做大小寫無關比對。
  - 未知 speaker 直接回傳 `unsupported speaker` 錯誤（fail-closed）。

## 3) 與測試的關聯

- 單元測試
  - `src/talker/input_builder.rs`
    - `resolve_codec_language_for_chinese_substitutes_dialect`
    - `resolve_speaker_is_case_insensitive`
  - `src/text_frontend/model_catalog.rs`
    - `synthetic_base_and_custom_voice_metadata_cover_required_tables`
    - `malformed_and_inconsistent_dialect_maps_fail_closed`
- 整合（可選，需本地模型）
  - `tests/model_runtime_metadata_real_test.rs`
    - `runtime_metadata_contract_is_loaded_from_real_model_dir`
  - `tests/model_metadata_real_test.rs`
    - `parse_real_model_06b_base_metadata`

執行方式：

```powershell
$env:QWEN3_TTS_REAL_MODEL_DIR = "C:\path\to\Qwen3-TTS-12Hz-0.6B-Base"
cargo test --test model_runtime_metadata_real_test -- --ignored
```
