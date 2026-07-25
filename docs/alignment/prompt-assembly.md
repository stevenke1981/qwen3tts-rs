# Prompt Assembly Alignment (P01-T04)

## Scope

- 文件對應修改：`src/text_frontend/candle_backend.rs`, `tests/prompt_assembly_test.rs`
- 主題：`prompt_ids`、`InputBuilder::build`、`InputBuilder::build_voice_clone` 的拼接規則與模式門檻一致性

## 官方格式邏輯

- 主提示固定為
  - `"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"`
- 指令提示固定為
  - `"<|im_start|>user\n{INSTRUCT}<|im_end|>\n"`
- 參考文字（ICL 路徑）固定為
  - `"<|im_start|>assistant\n{REF_TEXT}<|im_end|>\n"`
- `build_reference_text_ids` 會取 `ids[3..len-2]`，對應官方 `role+content+tail` 切法（去除 chat template 的首 3 與尾 2 token）

## 模式/參數矩陣

以 `validate_generation_request` 視角整理：

- Base（runtime `0.6/1.7`）允許 `--reference-audio`，不接受 `--speaker`
- Base 不接受 `--instruct`
- CustomVoice（`0.6`）：需要 `--speaker`；不接受 `--reference-audio`
- CustomVoice（`1.7`）：允許 `--speaker`，若附 `--instruct` 不阻檔
- VoiceDesign（`1.7`）：需要 `--instruct`，不接受 `--speaker`、不接受 `--reference-audio`

## 幾何與決策規則

### Base／通用 `build`
1. 先驗證 `text_token_ids.len() >= 8`
2. 構建 `codec_prefill`
   - 有語言時使用 `codec_think` 路徑
   - 無語言時使用 `codec_nothink` 路徑
3. 插入 `codec_pad + codec_bos`
4. CustomVoice named speaker 由 metadata ID 插入；Base voice clone 則插入
   reference audio 產生的 x-vector，兩者都不會產生額外 instruction
5. 文字尾段 append `tts_eos`

### Voice-clone `build_voice_clone`
1. 共用 `build` 主幹
2. `voice_clone` 決策：
   - `reference_codes` 與 `reference_text_token_ids` 均空：僅 x-vector 路徑
   - 兩者皆非空：ICL 路徑
   - 任一為空：失敗（回傳錯誤）
3. ICL 路徑將 `reference_text + current text content` 與 `reference_codes(+codec_bos)` 串接，若長度不足以對齊則以 `tts_pad` 補位

## 測試對應

- `wrapper_strings_match_official_format`
- `reference_text_slice_is_official_range_3_to_len_minus_2`
- `validate_generation_request_mode_matrix_for_five_variants`
- `input_builder_standard_and_custom_voice_base_layouts`
- `input_builder_xvector_path_uses_speaker_embedding_without_invented_instruction`
- `input_builder_icl_geometry_handles_short_and_long_body`
- `input_builder_rejects_malformed_prompt_geometry`
- `native_voice_clone_plan_without_reference_text_uses_speaker_embedding_only_mode`
- `native_voice_clone_plan_rejects_empty_reference_audio`

## 驗證

- `cargo fmt --all -- --check`: PASS
- CPU check: PASS
- text frontend: 32 passed
- input builder: 7 passed
- prompt assembly integration: 8 passed
- native voice clone: 7 passed
- both Candle synthesis examples: PASS
- independent GPT-5.3 Codex Spark review: no blocking findings
