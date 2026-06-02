# Qwen3-TTS Codec Decode Voice Fix - 2026-06-02

## 背景

症狀：目前生成的 WAV 有音訊波形，但聽起來不像人聲，且早期煙測輸出偏低振幅或假波形。

本次先用 `codebase-memory-mcp` 為 `D:\qwen3tts-rs` 建立知識圖：

- Project: `D-qwen3tts-rs`
- Artifact: `.codebase-memory/graph.db.zst`

主要追蹤路徑：

```text
examples/synthesize.rs
  -> Decoder12Hz::decode_chunk / decode_frames
  -> codebook
  -> pre_conv
  -> pre_transformer
  -> upsample
  -> decoder blocks
  -> final PCM
```

## 根因

逐層對齊測試顯示，`codebook` 與 `pre_conv` 起初已經完全對齊 PyTorch reference，但後續模組存在多個偏差：

1. `PreTransformer` attention mask 錯誤
   - 原本 `seq_len <= window` 時直接不套 mask。
   - 這會讓 batch decode 偷看未來 token，與 PyTorch causal decoder 不一致。

2. `SnakeBeta` 公式錯誤
   - PyTorch reference:
     ```text
     alpha = exp(alpha)
     beta = exp(beta)
     y = x + sin(x * alpha)^2 / (beta + 1e-9)
     ```
   - Rust 版原本把 `beta` 放進 `sin()`，並額外乘 `alpha`，導致 decoder residual/final activation 全面偏離。

3. Decoder causal conv padding 不符合 PyTorch
   - PyTorch `Qwen3TTSTokenizerV2CausalConvNet` 是左側 causal padding：
     ```text
     padding = (kernel_size - 1) * dilation + 1 - stride
     ```
   - Rust 後段 decoder 原本使用一般 symmetric padding。

4. Decoder transposed conv 不符合 PyTorch
   - PyTorch `Qwen3TTSTokenizerV2CausalTransConvNet` 是 `padding=0` 後裁掉右側 `kernel_size - stride`。
   - Rust 原本用 `padding`/`output_padding` 嘗試湊長度，時間位置錯位。

5. Residual unit dilation 遺漏
   - PyTorch residual conv1 dilation 是 `1, 3, 9`。
   - Rust 原本固定 dilation `1`。

6. 離線範例逐幀 decode，丟失 transformer 上下文
   - `examples/synthesize.rs` 原本逐幀呼叫 `decode_chunk()` 再拼接。
   - 原版 PyTorch decoder 是整段 causal batch decode，pre-transformer 需要整段上下文。

7. 現有 token fixture 是 raw u16 frame bytes
   - CLI 的 `--tokens` parser 只接受 `[num_frames: u32] + frames` 格式。
   - `weights/test_tokens.bin` 是 raw u16 frame array，導致煙測無法直接跑。

8. `auto` 語言固定映射成英文
   - `tools/generate_tokens.py` 原本把 `--language auto` 直接轉成 `english`。
   - 中文輸入在英文 codec language 條件下生成 token，容易聽起來像多語言混雜或咬字漂移。
   - Rust CLI 已有 `--speaker`，但 Python token generator 沒有接收到 speaker，導致 speaker 條件被忽略。

## 修復內容

### `src/codec/transformer.rs`

- 修正 `apply_sliding_window_mask`。
- 現在任何 `seq_len` 都會套 causal mask。
- 超過 sliding window 時才裁掉舊 key。

### `src/codec/decoder_blocks.rs`

- 修正 `snake_beta()` 為 PyTorch reference 公式。
- 新增 `CausalConvNet`：
  - 顯式 causal padding。
  - 支援 stride、dilation、groups。
- 新增 `CausalTransConvNet`：
  - `ConvTranspose1d` 使用 `padding=0`。
  - 輸出後裁掉右側 `kernel_size - stride`。
- `ConvNeXtBlock.dwconv` 改用 `CausalConvNet`。
- `ResidualUnit` 改用 `CausalConvNet`。
- `DecoderBlock` 的 residual dilation 改為 `1, 3, 9`。

### `src/codec/mod.rs`

- 匯出 `CausalConvNet`，供 integration tests 和 12Hz decoder 使用。

### `src/decoder_12hz.rs`

- `decoder_start` 與 `final_conv` 改用 `CausalConvNet`。
- 新增 `Decoder12Hz::decode_frames(&[[u16; 16]]) -> Result<Vec<f32>>`。
- `decode_frames()` 使用整段 batch path：
  - codebook lookup
  - `pre_conv.forward()`
  - `pre_transformer.forward()`
  - upsample
  - decoder blocks
  - final snake + conv

### `examples/synthesize.rs`

- 離線合成改用 `decoder.decode_frames(&stream.frames)`。
- 不再逐幀呼叫 `decode_chunk()`。
- 新增 `--save-tokens <path>`：
  - 先用已驗證可正常生成的文字前端產生 token。
  - 保存 token 後，之後可用 `--tokens <path>` 跳過 LLM/Python，直接走 Rust/Candle codec decode。

### `src/text_frontend/mod.rs`

- `TokenStream` 新增 `to_binary()` 與 `write_binary()`。
- 產生的語義 token 現在可保存成標準二進位格式：
  - header: `num_frames: u32`
  - payload: `num_frames * 16` 個 little-endian `u16`
- 這讓已驗證正常的文字前端輸出可以被後續 Rust 原生 decode 重跑使用，不必每次重新呼叫 Python。

### `src/text_frontend/token_parser.rs`

- `parse_bytes()` 新增 raw u16 frame fallback。
- 若 header 宣稱長度不合理，但檔案大小符合 `N * 16 * u16`，則直接解析為 raw frames。
- 新增 `test_parse_raw_u16_frames`。

### `tools/generate_tokens.py`

- `auto` language 改為依文字 Unicode 範圍偵測：
  - 中文：`chinese`
  - 日文：`japanese`
  - 韓文：`korean`
  - 俄文：`russian`
  - 拉丁字母：`english`
- 新增 `--speaker` 參數，並傳入 `mm.generate(..., speakers=[speaker_actual])`。

### `src/text_frontend/python_bridge.rs`

- Rust `SynthesisOptions.speaker` 現在會傳給 Python bridge 的 `--speaker`。

### `tests/integration_test.rs`

新增 PyTorch reference 對齊測試：

- `test_decode_frames_matches_pytorch_reference`
- `test_decoder_start_conv_matches_pytorch_reference`
- `test_decoder_blocks_match_pytorch_reference`
- `test_decoder_final_stage_matches_pytorch_reference`

關鍵驗收數據：

```text
decoder_start conv cosine_vs_pt=1.00000000
decoder block 1 cosine_vs_pt=1.00000000
decoder block 2 cosine_vs_pt=1.00000000
decoder block 3 cosine_vs_pt=1.00000000
decoder block 4 cosine_vs_pt=1.00000000
decoder final snake cosine_vs_pt=1.00000000
decoder final conv cosine_vs_pt=1.00000000
decode_frames cosine_vs_pt=0.99999962
```

## 驗證命令

```powershell
cargo check --examples
cargo test
cargo run --example synthesize -- --tokens weights\test_tokens.bin --output output-fixed.wav
cargo run --example synthesize -- --text "你好，這是一段中文語音測試。" --language auto --output output-zh-auto-fixed.wav
cargo run --example synthesize -- --text "你好，這是一段中文語音測試。" --language auto --save-tokens tokens-zh-auto.bin --output output-zh-auto-fixed.wav
cargo run --example synthesize -- --tokens tokens-zh-auto.bin --output output-zh-native-rerun.wav
```

## 驗證結果

`cargo test` 通過：

```text
39 lib tests passed
debug_compare_decoder_outputs passed
debug_per_layer_compare passed
depthwise_conv tests passed
integration tests passed
doctest passed
```

煙測輸出：

```text
output-fixed.wav
sample_rate=24000
channels=1
duration=4.00s
frames=96000
peak=0.922330
rms=0.151029
```

真實中文文字前端煙測輸出：

```text
output-zh-auto-fixed.wav
sample_rate=24000
channels=1
duration=2.72s
frames=65280
peak=0.467513
rms=0.047704
```

Rust 原生 token decode 重跑：

```text
tokens-zh-auto.bin
frames=37

output-zh-auto-save.wav
sample_rate=24000
channels=1
duration=2.96s
frames=71040
peak=0.200165
rms=0.041599
sha256=B06E80BD87F6065E5756D39711F51CB1C2DBB380D01D96F1FEB304BB47221C21

output-zh-native-rerun.wav
sample_rate=24000
channels=1
duration=2.96s
frames=71040
peak=0.200165
rms=0.041599
sha256=B06E80BD87F6065E5756D39711F51CB1C2DBB380D01D96F1FEB304BB47221C21
```

## 產物

- `.codebase-memory/graph.db.zst`
- `output-fixed.wav`
- `output-zh-auto-fixed.wav`
- `tokens-zh-auto.bin`
- `output-zh-auto-save.wav`
- `output-zh-native-rerun.wav`

## 結論

本次修復後，12Hz codec decode 的 full batch path 已和 PyTorch reference 高精度對齊。`decode_frames` cosine 達 `0.99999962`，煙測 WAV 不再是低振幅或假波形，已恢復可生成正常人聲所需的 codec decoder 行為。

後續針對「像俄文跟其他語言同時出現」的聽感，再修正文字前端 `auto` 語言選擇。中文文字現在會自動送 `chinese` codec language，而不是錯送 `english`。若仍聽到語言混雜，下一步應比對 Python 原生 `qwen_tts` 直接輸出的 WAV 與 Rust decoder 輸出，以區分是 talker token 生成問題還是 Rust decode 問題。
