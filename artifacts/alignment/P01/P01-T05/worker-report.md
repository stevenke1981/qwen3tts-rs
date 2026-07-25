# P01-T05 Worker Report

- 建立官方 Python oracle 來源檔：`fixtures/alignment/p01_prompt_id_matrix.json`，改由
  `C:\Users\steven\Qwen3-TTS\.venv\Scripts\python.exe tools/generate_prompt_id_matrix.py` 產生與驗證。
- 將 prompt oracle 比對資料對齊 `config/model-matrix.json`，並在 `tests/prompt_id_matrix_real_test.rs` 中只比對
  官方輸出的五個模型條目（5 entries）、主要/參考/指令 prompt ID 與 reference body slice。
- 先以 `expected_model_revision`/`expected_config_sha` 固化預期 hash，並透過真實 0.6B Base snapshot 反查 `model_revision`、`config_sha` 做故障封閉檢查。
- 在 `src/text_frontend/model_catalog.rs` 與 `src/text_frontend/prompt_templates.rs` 強化 metadata 與 prompt case 描述，並提供
  literal helper（主文、參考文、reference body slice）與版本分支一致性防呆。
- 更新 `config/fixtures.json` 加入 `p01-prompt-id-matrix` provenance，含 SHA-256 與產生指令。
- 更新 `docs/alignment/prompt-id-matrix.md` 補齊模型清單、案例矩陣、oracle 來源與 gate 命令。
- 更新 `tools/generate_prompt_id_matrix.py`，保留官方模型/分詞器來源判斷與共享 tokenizer 驗證流程（不下載權重）。
