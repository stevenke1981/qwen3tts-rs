# Qwen3-TTS Rust Release 使用說明

適用版本：`qwen3tts-rs v0.1.10 Windows x64 / Windows x64 CUDA`

這個 release 包提供純 Rust/Candle 可執行檔：

- `synthesize.exe`：單句文字轉語音。
- `synthesize_batch.exe`：批次文字轉語音，模型只載入一次，適合連續測試多句。
- `quantize_tokenizer.exe`：將已轉換的 tokenizer decoder safetensors 量化成 Q8/Q4-hybrid。

## 重要限制

release 包不包含大型模型權重。你需要另外準備：

1. Qwen3-TTS talker 模型 snapshot，例如：

```text
C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\<sha>
```

該目錄內需有：

```text
model.safetensors
tokenizer.json
```

若使用 `1.7B-VoiceDesign` snapshot 且該目錄缺少 `tokenizer.json`，app 會自動嘗試使用本機 HuggingFace cache 裡的 Base 模型 tokenizer。你也可以手動把 Base 模型的 `tokenizer.json` 放到 `models\tokenizer.json`。

2. 12Hz tokenizer decoder 權重。

`v0.1.3` 開始，若 app 找不到已轉換的 Rust 權重，會優先自動執行 release 包內的 Rust converter：

```text
convert_tokenizer.exe
```

這個轉換器不需要 Python。它會從本機 HuggingFace cache 尋找：

```text
Qwen/Qwen3-TTS-Tokenizer-12Hz
```

也可以手動指定 tokenizer decoder snapshot：

```powershell
.\convert_tokenizer.exe --input C:\path\to\Qwen3-TTS-Tokenizer-12Hz\snapshot --output weights\tokenizer
```

若找不到 Rust converter，app 才會退回舊的 Python converter：

```text
tools\convert_weights.py
```

自動轉換輸出會優先快取到使用者目錄，避免每個新 release 目錄重複轉換超過 1GB 權重：

```text
%LOCALAPPDATA%\qwen3tts-rs\tokenizer-12hz\
```

也可以用環境變數覆蓋：

```powershell
$env:QWEN3TTS_TOKENIZER_CACHE_DIR = "D:\qwen3tts-cache"
$env:QWEN3TTS_TOKENIZER_WEIGHT_DIR = "D:\qwen3tts-cache\tokenizer-12hz"
```

手動轉換輸出仍可放到 release 目錄下：

```text
weights\tokenizer\
```

只有使用 Python fallback 時，才需要 Python 以及 `torch safetensors huggingface_hub numpy`。

## CPU 與 CUDA 版本

- `qwen3tts-rs-v0.1.10-windows-x64.zip`：CPU/Candle build。
- `qwen3tts-rs-v0.1.10-windows-x64-cuda.zip`：CUDA/Candle build，啟動時會優先嘗試 `CUDA:0`，若 CUDA 初始化失敗才回退 CPU。

建置 CUDA 版 release：

```powershell
.\tools\package_release.ps1 -Cuda
```

預設 `-CudaComputeCap 86` 對應 RTX 3070 Ti；其他 GPU 可自行指定 Candle 接受的整數格式，例如 `75`、`89`。

若你已經有轉好的權重，也可以直接放在下列其中一個位置：

```text
<目前 PowerShell 所在目錄>\weights\tokenizer\
<synthesize.exe 所在目錄>\weights\tokenizer\
%LOCALAPPDATA%\qwen3tts-rs\tokenizer-12hz\
```

release app 會依照上面的順序檢查。若你從其他目錄呼叫 exe，建議直接把 `weights\tokenizer` 放在 exe 同一層，例如：

```text
weights\tokenizer\
```

至少需包含：

```text
codebook.safetensors
lightweight.safetensors
pre_transformer.safetensors
upsample.safetensors
decoder_blocks.safetensors
```

若找不到或轉換失敗，app 會列出它實際檢查過的完整路徑。

## 單句合成

在 release 目錄執行：

```powershell
$model = "C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\<sha>"
.\synthesize.exe `
  --text "今天天氣真好" `
  --backend candle `
  --language chinese `
  --model-dir $model `
  --output output.wav `
  --max-new-tokens 16
```

輸出：

```text
output.wav
```

低硬體需求可改用 `0.6B-Base` snapshot；這也是 CLI 預設模型：

```powershell
$model = "C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\<sha>"
.\synthesize.exe `
  --text "今天天氣真好" `
  --backend candle `
  --language chinese `
  --model-dir $model `
  --output output_0p6b.wav `
  --max-new-tokens 32
```

## VoiceDesign 音色指令

`1.7B-VoiceDesign` 可用 `--instruct` 以自然語言描述音色與語氣：

```powershell
$model = "C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign\snapshots\<sha>"
.\synthesize.exe `
  --text "歡迎使用 Qwen3-TTS Rust 原生版本" `
  --backend candle `
  --language chinese `
  --model-dir $model `
  --instruct "年輕女性，台灣口語，溫柔親切，語速自然" `
  --seed 20260603 `
  --output voicedesign.wav `
  --max-new-tokens 64
```

常用音色設定可存成 UTF-8 文字檔：

```powershell
"年輕女性，台灣口語，溫柔親切，語速自然" | Set-Content .\instruct.txt -Encoding UTF8

.\synthesize.exe `
  --text "歡迎使用 Qwen3-TTS Rust 原生版本" `
  --backend candle `
  --language chinese `
  --model-dir $model `
  --instruct-file .\instruct.txt `
  --seed 20260603 `
  --output voicedesign_file.wav `
  --max-new-tokens 64
```

Base 模型不支援穩定音色控制；若需要指定音色，請使用 VoiceDesign 搭配 `--instruct` 或下面的內建 speaker preset。`--speaker` 在 CustomVoice 權重提供 speaker id map 時會使用真實 speaker；若模型沒有 speaker map，app 會把已知 speaker 名稱轉成 VoiceDesign `--instruct` fallback。

## 內建 speaker preset

`v0.1.9` 起支援 Qwen CustomVoice 官方 9 個 speaker 名稱。可先列出清單：

```powershell
.\synthesize.exe --list-speakers
.\synthesize_batch.exe --list-speakers
```

| Speaker | 說明 | 建議語言 |
| --- | --- | --- |
| `Vivian` | 明亮的年輕女性聲線 | Chinese |
| `Serena` | 溫暖、柔和的年輕女性聲線 | Chinese |
| `Uncle_Fu` | 成熟男性，音色醇厚 | Chinese |
| `Dylan` | 年輕北京男性聲線 | Chinese (Beijing) |
| `Eric` | 活潑成都男性聲線 | Chinese (Sichuan) |
| `Ryan` | 節奏感較強的男性聲線 | English |
| `Aiden` | 陽光美式男性聲線 | English |
| `Ono_Anna` | 活潑日文女性聲線 | Japanese |
| `Sohee` | 溫暖韓文女性聲線 | Korean |

CustomVoice 模型會把這些名稱當作真實 speaker id：

```powershell
$model = "C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice\snapshots\<sha>"
.\synthesize.exe `
  --text "你好，今天想和你聊聊天" `
  --backend candle `
  --language chinese `
  --model-dir $model `
  --speaker Vivian `
  --output vivian.wav `
  --max-new-tokens 64
```

VoiceDesign 或 Base 模型沒有 speaker map 時，相同指令會自動加入對應的 instruct preset；也可以與 `--instruct` / `--instruct-file` 疊加，讓 preset 控制大方向、文字指令控制語氣：

```powershell
$model = "C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign\snapshots\<sha>"
.\synthesize.exe `
  --text "歡迎使用 Qwen3-TTS Rust 原生版本" `
  --backend candle `
  --language chinese `
  --model-dir $model `
  --speaker Uncle_Fu `
  --instruct "語氣親切，像在錄製教學旁白" `
  --seed 20260604 `
  --output uncle_fu_voicedesign.wav `
  --max-new-tokens 80
```

長中文句子若 `--max-new-tokens` 太低會被截斷。保守估算可用：

```text
max-new-tokens ≈ 中文字元數 × 4
```

## 批次合成

批次模式會先載入一次模型，再連續生成多句，避免每句都重新讀取 1.7B 權重。

```powershell
$model = "C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\<sha>"
.\synthesize_batch.exe `
  --model-dir $model `
  --output-dir batch-output `
  --prefix qwen1p7b `
  --language chinese `
  --max-new-tokens 16 `
  --instruct-file .\instruct.txt `
  --seed 20260603 `
  --text "今天天氣真好" `
  --text "你好，測試第二句"
```

若已下載預設 0.6B Base 到 HuggingFace cache，batch 可以省略 `--model-dir`：

```powershell
.\synthesize_batch.exe `
  --text "今天天氣真好" `
  --output-dir batch-output `
  --language chinese
```

輸出檔名只使用數字索引，例如：

```text
batch-output\qwen1p7b_0001.wav
batch-output\qwen1p7b_0002.wav
```

也可以使用文字檔：

```powershell
@"
今天天氣真好
你好，測試第二句
"@ | Set-Content .\texts.txt -Encoding UTF8

.\synthesize_batch.exe `
  --model-dir $model `
  --texts .\texts.txt `
  --output-dir batch-output `
  --language chinese `
  --no-save-tokens
```

逐句切換音色與 seed 時，重複傳入 `--instruct-file` 與 `--seed`，數量需等於句數：

```powershell
.\synthesize_batch.exe `
  --model-dir $model `
  --text "第一句" `
  --text "第二句" `
  --instruct-file .\voice_a.txt `
  --instruct-file .\voice_b.txt `
  --seed 101 `
  --seed 202 `
  --output-dir batch-output
```

## Tokenizer decoder Q8/Q4 量化

`v0.1.10` 新增 `quantize_tokenizer.exe`。它輸入已轉好的 Rust tokenizer decoder 權重，輸出另一個可直接被 `synthesize.exe` / `synthesize_batch.exe` 讀取的量化目錄：

```powershell
.\quantize_tokenizer.exe `
  --input weights\tokenizer `
  --output weights\tokenizer-q8 `
  --format q8_0 `
  --group-size 64 `
  --min-cosine 0.995
```

使用量化權重：

```powershell
$env:QWEN3TTS_TOKENIZER_WEIGHT_DIR = "D:\qwen3tts-rs\weights\tokenizer-q8"
.\synthesize.exe --tokens .\sample.tokens --output q8.wav
```

量化策略：

| 模式 | 狀態 | 說明 |
| --- | --- | --- |
| `q8_0` | 建議測試 | 本機 tokenizer decoder 量化結果：99 個 tensor 量化、137 個保留，大小約為原始 26.6%；decoder smoke cosine vs base `0.99974333` |
| `q4_0` | experimental | 預設會自動把低於 `--min-cosine` 的 tensor 保留為 F32 anchor；本機 0.995 門檻下只有 12 個 tensor 量化、224 個保留，大小約為原始 94.7%；decoder smoke cosine vs base `0.99235672` |

`quantize_tokenizer.exe` 會寫出：

```text
weights\tokenizer-q8\quantization_report.json
```

報告內含每個 tensor 是否量化、cosine、最大絕對誤差、保留原因。若你要強制檢查「低 cosine 就失敗，不自動保留」，加上：

```powershell
.\quantize_tokenizer.exe --format q4_0 --fail-low-cosine
```

目前量化 loader 採取相容優先：讀取 Q8/Q4 safetensors 後會反量化回 F32 Tensor 再進現有 decoder。這已能降低磁碟大小，但還不是 int8/int4 kernel 熱路徑加速。vocoder / HiFi-GAN 類權重仍預設保留，避免音質劣化。

可把校準文字清單記入報告，方便不同 agent 比對同一批評估資料：

```powershell
.\quantize_tokenizer.exe `
  --input weights\tokenizer `
  --output weights\tokenizer-q8 `
  --format q8_0 `
  --calibration-texts .\calibration_texts.txt
```

## 常用參數

| 參數 | 說明 |
| --- | --- |
| `--model-dir` | Qwen3-TTS 模型 snapshot 目錄 |
| `--model` | batch 模式可指定 HuggingFace model id；省略 `--model-dir` 時會自動找本機 HF cache |
| `--language` | 語言，中文建議使用 `chinese` |
| `--speaker` | 說話者條件；支援內建 `Vivian`、`Uncle_Fu`、`Dylan` 等 preset，CustomVoice 走真實 speaker id，其他模型轉成 instruct fallback |
| `--list-speakers` | 顯示內建 speaker preset 清單 |
| `--instruct` | VoiceDesign/CustomVoice 音色或語氣指令 |
| `--instruct-file` | 從 UTF-8 文字檔讀取音色或語氣指令；batch 可重複傳入做到逐句切換 |
| `--seed` | 固定取樣 seed；batch 可重複傳入做到逐句 seed |
| `--max-new-tokens` | 最大生成 frame 數，短句可先用 `16` 測試；長中文可用中文字數 × 4 估算 |
| `--temperature` | 取樣溫度，預設 `0.9` |
| `--top-k` | top-k 取樣，預設 `50` |
| `--top-p` | top-p 取樣，預設 `1.0` |
| `--speed` | 語速倍率，預設 `1.0` |
| `--save-tokens-dir` | batch 模式同步輸出每句 `.tokens` 檔 |
| `--no-save-tokens` | batch 模式關閉 token 檔輸出 |
| `quantize_tokenizer.exe --format` | tokenizer decoder 權重量化格式：`q8_0` 或 `q4_0` |
| `quantize_tokenizer.exe --min-cosine` | 量化 tensor 的 cosine 門檻，預設 `0.995`；低於門檻會自動保留為 F32 anchor |

## 驗證輸出不是靜音

```powershell
@'
import wave, struct, math
name = "output.wav"
with wave.open(name, "rb") as w:
    n = w.getnframes()
    sr = w.getframerate()
    data = w.readframes(n)
    samples = struct.unpack("<" + "h" * (len(data) // 2), data)
    rms = math.sqrt(sum(x*x for x in samples) / len(samples)) if samples else 0
    peak = max((abs(x) for x in samples), default=0)
    print(f"{name}: duration={n/sr:.3f}s sr={sr} rms={rms:.1f} peak={peak}")
'@ | python -
```

若 `rms=0` 且 `peak=0`，代表 WAV 是靜音。

## 已知狀態

- 1.7B native Candle 短句已可產生非靜音語音。
- `v0.1.4` 提供 CPU 與 CUDA 兩種 release；CUDA binary 會優先使用 `CUDA:0`。
- `v0.1.5` 新增 VoiceDesign/CustomVoice `--instruct`，並在 VoiceDesign snapshot 缺少 `tokenizer.json` 時自動 fallback 使用 Base tokenizer。
- `v0.1.6` 新增 `--instruct-file`、`--seed`、batch `--no-save-tokens`，並讓 batch 對長中文提示 `--max-new-tokens` 建議。
- `v0.1.7` 新增 `--speed` 與 `--version`，並修正中文截斷建議。
- `v0.1.8` 新增 batch 逐句 `--instruct-file`/`--seed`、batch 0.6B 模型自動尋找，以及 tokenizer decoder 全域快取。
- `v0.1.9` 新增 9 個內建 CustomVoice speaker presets、`--list-speakers`，並讓 Base/VoiceDesign 在無 speaker map 時自動轉成 instruct fallback。
- `v0.1.10` 新增 `quantize_tokenizer.exe`、Q8/Q4-hybrid safetensors 格式、WeightLoader 自動反量化，以及 tokenizer decoder 量化報告。
- decoder 容量會依實際 frame 數或 batch `--max-new-tokens` 擴展，修復超過 64 幀時的 `narrow` crash。
- 批次模式可避免每句都重新載入模型。
- 批次模式輸出檔名固定為 `prefix_0001.wav`，不再把完整文字放入檔名。
- 批次模式可用 `--save-tokens-dir <dir>` 同步輸出每句 `.tokens` 檔。
- `--instruct` 已走 VoiceDesign/CustomVoice 的獨立 user prompt embedding，不會混入待朗讀文字。
- 大型模型權重與 tokenizer decoder 權重未包含在 zip 內。
- `v0.1.1` 修正 release app 只用目前工作目錄找 `weights/tokenizer` 的問題；現在也會檢查 exe 所在目錄。
- `v0.1.2` 開始，若找不到 Rust tokenizer decoder 權重，app 會自動嘗試執行 bundled `tools/convert_weights.py tokenizer`。
- `v0.1.3` 開始，app 優先使用 bundled `convert_tokenizer.exe` 做 Rust 原生轉換，不需要 Python；Python converter 只作為 fallback。
