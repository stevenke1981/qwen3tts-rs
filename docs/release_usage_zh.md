# Qwen3-TTS Rust Release 使用說明

適用版本：`qwen3tts-rs v0.1.14 Windows x64 / Windows x64 CUDA`

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

- `qwen3tts-rs-v0.1.14-windows-x64.zip`：CPU/Candle build。
- `qwen3tts-rs-v0.1.14-windows-x64-cuda.zip`：CUDA/Candle build，啟動時會優先嘗試 `CUDA:0`，若 CUDA 初始化失敗才回退 CPU。

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

## 模型能力與模式

`v0.1.11` 起，CLI 內建 Qwen3-TTS 模型能力表：

```powershell
.\synthesize.exe --list-models
.\synthesize_batch.exe --list-models
```

| 模型名稱 | 參數量 | 主要功能 | 語言支援 | 流式 | 指令控制 | 推薦場景 |
| --- | --- | --- | --- | --- | --- | --- |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1.7B | 文字描述聲音設計 | 10 種 | yes | yes | 自訂聲音創作 |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | 9 種預設音色 + 指令風格控制 | 10 種 | yes | yes | 高品質敘事、多角色 |
| `Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | 3 秒聲音克隆 + 微調基礎 | 10 種 | yes | - | 克隆、Fine-tuning |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B | 9 種預設音色（無指令） | 10 種 | yes | - | 輕量部署 |
| `Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | 3 秒聲音克隆 + 微調基礎 | 10 種 | yes | - | 資源受限環境 |

支援語言清單：

```text
Chinese, English, French, German, Italian, Spanish, Portuguese, Japanese, Korean, Russian
```

CLI 也新增模式檢查：

| 參數 | 說明 |
| --- | --- |
| `--mode auto` | 預設，不阻擋既有流程 |
| `--mode custom-voice` | 要求 CustomVoice 模型與 `--speaker`；0.6B CustomVoice 不允許 `--instruct` |
| `--mode voice-design` | 要求 `1.7B-VoiceDesign` 與 `--instruct` / `--instruct-file` |
| `--mode voice-clone` | 要求 Base 模型與 `--reference-audio`；單句 Python backend 可用，Candle 原生 conditioning 尚未實作 |

目前 release 的「流式」能力指模型支援 12Hz streaming token/audio 架構；CLI 仍是整句寫 WAV，尚未提供即時 audio chunk callback。

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

等價於 Python 高階 API `generate_voice_design(...)` 的 Rust CLI 用法是：

```powershell
.\synthesize.exe `
  --mode voice-design `
  --model "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" `
  --model-dir $model `
  --text "哥哥，你回來啦..." `
  --language Chinese `
  --instruct "一位溫柔可愛、帶點撒嬌語氣的年輕女孩聲音，語速稍慢" `
  --output voice_design.wav
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

等價於 Python 高階 API `generate_custom_voice(...)` 的 Rust CLI 用法是：

```powershell
.\synthesize.exe `
  --mode custom-voice `
  --model "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" `
  --model-dir $model `
  --text "其實我真的有發現，我是一個特別善於觀察別人情緒的人。" `
  --language Chinese `
  --speaker Vivian `
  --instruct "用特別憤怒的語氣說" `
  --output custom_voice.wav
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

## Voice Clone 狀態

Base 模型對應 Python 高階 API `generate_voice_clone(...)`：

```python
wavs, sr = model.generate_voice_clone(
    text="測試文字...",
    language="Chinese",
    reference_audio="reference.wav",
)
```

Rust CLI 已新增參數與能力檢查：

```powershell
.\synthesize.exe `
  --mode voice-clone `
  --model "Qwen/Qwen3-TTS-12Hz-1.7B-Base" `
  --model-dir $model `
  --text "測試文字..." `
  --language Chinese `
  --reference-audio reference.wav
```

`v0.1.14` 起，單句模式可用 Python 官方 `qwen_tts` 路徑實際生成 Voice Clone WAV：

```powershell
.\synthesize.exe `
  --backend python `
  --mode voice-clone `
  --model "Qwen/Qwen3-TTS-12Hz-0.6B-Base" `
  --text "測試文字..." `
  --language Chinese `
  --reference-audio reference.wav `
  --reference-text "參考音訊的逐字稿" `
  --output clone.wav
```

`--reference-text` 可省略；省略時會使用 speaker-embedding-only 模式。Candle/Rust 原生 reference-audio conditioning 仍未完成，因為還需要移植 speech tokenizer encoder、speaker encoder 與 ICL prompt，現在會明確報錯而不是靜默忽略參考音訊。batch voice-clone 也尚未接上，請先用單句 `synthesize.exe --backend python`。

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

`v0.1.12` 起，Q8 量化權重會自動優先於 F32 權重。只要放在下列任一路徑，不需要設定環境變數：

```text
weights\tokenizer-q8
%LOCALAPPDATA%\qwen3tts-rs\tokenizer-12hz-q8
```

搜尋順序是：`QWEN3TTS_TOKENIZER_WEIGHT_DIR` 明確指定、目前目錄 `weights\tokenizer-q8`、目前目錄 `weights\tokenizer`、exe 旁 `weights\tokenizer-q8`、exe 旁 `weights\tokenizer`、全域 cache Q8、全域 cache F32。

`v0.1.13` 起，第一次執行若完全找不到 tokenizer decoder 權重，app 會先自動執行 `convert_tokenizer.exe` 建立 F32 cache，接著自動嘗試用 `quantize_tokenizer.exe` 建立 Q8 cache。若 Q8 建立成功，後續會直接使用 Q8；若 quantizer 不存在或失敗，會保留 F32 fallback，不中斷合成。

當實際使用 Q8 權重且目錄內有 `quantization_report.json` 時，terminal 會顯示摘要，例如：

```text
已使用 Q8 量化 tokenizer decoder 權重: ...\tokenizer-12hz-q8 (約 125 MB，-73%，99/236 tensors quantized)
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
| `--list-models` | 顯示模型名稱、參數量、主要功能、語言支援、流式、指令控制、推薦場景 |
| `--mode` | `auto`、`custom-voice`、`voice-design`、`voice-clone` |
| `--reference-audio` | Voice Clone 參考音訊；單句 `synthesize.exe --backend python` 可實際生成，Candle 原生仍待 speech tokenizer encoder/speaker encoder |
| `--reference-text` | Voice Clone 參考音訊逐字稿；未提供時使用 speaker-embedding-only 模式 |
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
- `v0.1.11` 新增模型能力 catalog、`--list-models`、`--mode`、`--reference-audio`，並對 CustomVoice / VoiceDesign / VoiceClone 做明確能力檢查。
- `v0.1.12` 自動優先使用 `weights\tokenizer-q8` 與 `%LOCALAPPDATA%\qwen3tts-rs\tokenizer-12hz-q8`，讓已驗證的 Q8 tokenizer decoder 不必每次手動指定環境變數。
- `v0.1.13` 第一次自動轉換 F32 tokenizer decoder 後會自動嘗試建立 Q8 cache，並在使用 Q8 時顯示大小、節省比例與量化 tensor 摘要。
- `v0.1.14` 單句 `synthesize.exe --backend python --mode voice-clone` 會呼叫官方 `generate_voice_clone()` 並直接輸出 WAV；`--reference-text` 可選。Candle 原生 Voice Clone 仍待 speech tokenizer encoder/speaker encoder/ICL prompt。
- decoder 容量會依實際 frame 數或 batch `--max-new-tokens` 擴展，修復超過 64 幀時的 `narrow` crash。
- 批次模式可避免每句都重新載入模型。
- 批次模式輸出檔名固定為 `prefix_0001.wav`，不再把完整文字放入檔名。
- 批次模式可用 `--save-tokens-dir <dir>` 同步輸出每句 `.tokens` 檔。
- `--instruct` 已走 VoiceDesign/CustomVoice 的獨立 user prompt embedding，不會混入待朗讀文字。
- 大型模型權重與 tokenizer decoder 權重未包含在 zip 內。
- `v0.1.1` 修正 release app 只用目前工作目錄找 `weights/tokenizer` 的問題；現在也會檢查 exe 所在目錄。
- `v0.1.2` 開始，若找不到 Rust tokenizer decoder 權重，app 會自動嘗試執行 bundled `tools/convert_weights.py tokenizer`。
- `v0.1.3` 開始，app 優先使用 bundled `convert_tokenizer.exe` 做 Rust 原生轉換，不需要 Python；Python converter 只作為 fallback。
