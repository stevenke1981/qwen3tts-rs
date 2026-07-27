# Qwen3-TTS 模型與模式完整對照

> 來源：HuggingFace 官方模型卡（2026-07 查證）

## 模型矩陣

| 模型 | 參數 | `tts_model_type` | API 方法 | 硬性要求 | 可選 |
|---|---|---|---|---|---|
| `Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | `base` | `generate_voice_clone()` | `ref_audio` + `ref_text` | `x_vector_only_mode` |
| `Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | `base` | `generate_voice_clone()` | `ref_audio` + `ref_text` | `x_vector_only_mode`, `voice_clone_prompt` |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B | `custom_voice` | `generate_custom_voice()` | `speaker` | — |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | `custom_voice` | `generate_custom_voice()` | `speaker` | `instruct` |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1.7B | `voice_design` | `generate_voice_design()` | `instruct` | — |

## 各模式詳細需求

### 1. Voice Clone（Base 模型）

**用途**：3 秒參考音頻快速語音複製

**硬性要求**：
- `--reference-audio`：≥3 秒的參考音頻（WAV/MP3/URL/base64/numpy tuple）
- `--reference-text`：參考音頻的文字轉錄（`x_vector_only_mode` 可省略，但品質較差）

**不允許**：
- `--speaker`（Base 模型無預設音色）
- `--instruct`（Base 模型不支援指令控制）

**Rust 用法**：
```bash
cargo run --example synthesize --features candle-llm -- \
    --text "你好，世界。" \
    --backend candle \
    --model-dir <path-to-0.6B-Base> \
    --reference-audio ref.wav \
    --reference-text "這是參考音頻的文字內容。" \
    --output clone.wav
```

**config.json 特徵**：
- `tts_model_type: "base"`
- `spk_id: {}`（空）
- `spk_is_dialect: {}`（空）

---

### 2. Custom Voice（CustomVoice 模型）

**用途**：9 個預設高級音色 + 可選風格指令

**硬性要求**：
- `--speaker`：必須指定一個預設音色名稱

**可選**：
- `--instruct`：自然語言風格控制（僅 1.7B 支援）

**不允許**：
- `--reference-audio`（CustomVoice 不支援語音複製）

**9 個預設音色**：

| Speaker | 描述 | 母語 |
|---|---|---|
| `Vivian` | 明亮略帶稜角的年輕女聲 | 中文 |
| `Serena` | 溫暖柔和的年輕女聲 | 中文 |
| `Uncle_Fu` | 沉穩男聲，醇厚音色 | 中文 |
| `Dylan` | 青春北京男聲，清晰自然 | 中文（北京話） |
| `Eric` | 活潑成都男聲，略帶沙啞明亮 | 中文（四川話） |
| `Ryan` | 動感男聲，節奏感強 | 英文 |
| `Aiden` | 陽光美式男聲，中頻清晰 | 英文 |
| `Ono_Anna` | 俏皮日本女聲，輕快靈巧 | 日文 |
| `Sohee` | 溫暖韓國女聲，情感豐富 | 韓文 |

**Rust 用法**：
```bash
# 0.6B（無 instruct）
cargo run --example synthesize --features candle-llm -- \
    --text "你好，世界。" \
    --backend candle \
    --model-dir <path-to-0.6B-CustomVoice> \
    --speaker Vivian \
    --output custom.wav

# 1.7B（含 instruct 風格控制）
cargo run --example synthesize --features candle-llm -- \
    --text "你好，世界。" \
    --backend candle \
    --model-dir <path-to-1.7B-CustomVoice> \
    --speaker Vivian \
    --instruct "用特別憤怒的語氣說" \
    --output custom_angry.wav
```

**config.json 特徵**：
- `tts_model_type: "custom_voice"`
- `spk_id`：9 個 speaker → token ID 映射
- `spk_is_dialect`：Eric → `sichuan_dialect`、Dylan → `beijing_dialect`
- `codec_language_id` 含方言 ID（`beijing_dialect: 2074`、`sichuan_dialect: 2062`）

---

### 3. Voice Design（VoiceDesign 模型）

**用途**：用自然語言描述創造全新聲音

**硬性要求**：
- `--instruct`：自然語言聲音描述（例如「體現撒嬌稚嫩的蘿莉女聲，音調偏高且起伏明顯」）

**不允許**：
- `--speaker`（VoiceDesign 無預設音色）
- `--reference-audio`（VoiceDesign 不做語音複製）

**僅 1.7B 版本**（hidden_size=2048, intermediate_size=6144）

**Rust 用法**：
```bash
cargo run --example synthesize --features candle-llm -- \
    --text "哥哥，你回來啦，人家等了你好久好久了，要抱抱！" \
    --backend candle \
    --model-dir <path-to-1.7B-VoiceDesign> \
    --instruct "體現撒嬌稚嫩的蘿莉女聲，音調偏高且起伏明顯，帶有黏人、做作又刻意賣萌的聽覺效果。" \
    --output design.wav
```

**config.json 特徵**：
- `tts_model_type: "voice_design"`
- `spk_id: {}`（空）
- `spk_is_dialect: {}`（空）
- `hidden_size: 2048`、`intermediate_size: 6144`（1.7B 架構）

---

### 4. 進階：Voice Design → Voice Clone 管線

先用 VoiceDesign 產生參考音頻，再用 Base 模型複製：

```bash
# Step 1: VoiceDesign 產生參考音頻
cargo run --example synthesize --features candle-llm -- \
    --text "H-hey! You dropped your... uh... calculus notebook?" \
    --backend candle \
    --model-dir <path-to-1.7B-VoiceDesign> \
    --instruct "Male, 17 years old, tenor range, gaining confidence" \
    --output ref_designed.wav

# Step 2: Base 模型複製該聲音
cargo run --example synthesize --features candle-llm -- \
    --text "No problem! I actually... kinda finished those already?" \
    --backend candle \
    --model-dir <path-to-1.7B-Base> \
    --reference-audio ref_designed.wav \
    --reference-text "H-hey! You dropped your... uh... calculus notebook?" \
    --output clone_designed.wav
```

---

## 架構差異

| 欄位 | 0.6B | 1.7B |
|---|---|---|
| `hidden_size` | 1024 | 2048 |
| `intermediate_size` | 3072 | 6144 |
| `num_hidden_layers` | 28 | 28 |
| `num_attention_heads` | 16 | 16 |
| `num_key_value_heads` | 8 | 8 |
| `head_dim` | 128 | 128 |
| `text_hidden_size` | 2048 | 2048 |
| Code Predictor `hidden_size` | 1024 | 1024 |
| Code Predictor `num_hidden_layers` | 5 | 5 |

## 權重下載

```bash
# 0.6B-Base（已有）
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base

# 0.6B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

# 1.7B-Base
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base

# 1.7B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

# 1.7B-VoiceDesign
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

# Tokenizer 解碼器（共用）
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz
```

## Rust 實作狀態

| 功能 | 狀態 | 備註 |
|---|---|---|
| config.json 解析（Base/CustomVoice/VoiceDesign） | ✅ | `model_catalog.rs` 已支持 `tts_model_type` 三種 |
| Speaker ID 解析 | ✅ | `input_builder.rs` `resolve_speaker_id()` |
| 方言解析 | ✅ | `spk_is_dialect` → `codec_language_id` 映射 |
| Instruct 提示詞構建 | ✅ | `build_instruction_prompt()` |
| Voice Clone 輸入構建 | ✅ | `InputBuilder.build_voice_clone()` |
| CustomVoice 輸入構建 | ✅ | `InputBuilder.build()` + speaker 參數 |
| VoiceDesign 輸入構建 | ✅ | `InputBuilder.build()` + instruct 參數 |
| 模式驗證（硬性要求） | ✅ | `validate_generation_request()` |
| 0.6B 權重載入 | ✅ | safetensors + GGUF |
| 1.7B 權重載入 | ⚠️ 未驗證 | 架構相同但 hidden_size=2048，需實際測試 |
| Speaker Encoder（voice clone） | ✅ | `NativeSpeakerEncoder` |
| 端到端 TTS | ✅ | 0.6B-Base 已驗證 |
