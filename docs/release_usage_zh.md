# Qwen3-TTS Rust Release 使用說明

適用版本：`qwen3tts-rs v0.1.6 Windows x64 / Windows x64 CUDA`

這個 release 包提供純 Rust/Candle 可執行檔：

- `synthesize.exe`：單句文字轉語音。
- `synthesize_batch.exe`：批次文字轉語音，模型只載入一次，適合連續測試多句。

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

轉換輸出會放到 release 目錄下：

```text
weights\tokenizer\
```

只有使用 Python fallback 時，才需要 Python 以及 `torch safetensors huggingface_hub numpy`。

## CPU 與 CUDA 版本

- `qwen3tts-rs-v0.1.6-windows-x64.zip`：CPU/Candle build。
- `qwen3tts-rs-v0.1.6-windows-x64-cuda.zip`：CUDA/Candle build，啟動時會優先嘗試 `CUDA:0`，若 CUDA 初始化失敗才回退 CPU。

建置 CUDA 版 release：

```powershell
.\tools\package_release.ps1 -Cuda
```

預設 `-CudaComputeCap 86` 對應 RTX 3070 Ti；其他 GPU 可自行指定 Candle 接受的整數格式，例如 `75`、`89`。

若你已經有轉好的權重，也可以直接放在下列其中一個位置：

```text
<目前 PowerShell 所在目錄>\weights\tokenizer\
<synthesize.exe 所在目錄>\weights\tokenizer\
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

Base 模型不支援穩定音色控制；若需要指定音色，請使用 VoiceDesign 搭配 `--instruct`。`--speaker` 只有在模型權重本身提供 speaker id 對應時才有實際效果，Base/VoiceDesign 常見 snapshot 的 speaker map 通常是空的。

長中文句子若 `--max-new-tokens` 太低會被截斷。簡單估算可用：

```text
max-new-tokens ≈ 中文字元數 × 3
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

## 常用參數

| 參數 | 說明 |
| --- | --- |
| `--model-dir` | Qwen3-TTS 模型 snapshot 目錄 |
| `--language` | 語言，中文建議使用 `chinese` |
| `--speaker` | 說話者條件；若模型沒有 speaker id map，通常無作用 |
| `--instruct` | VoiceDesign/CustomVoice 音色或語氣指令 |
| `--instruct-file` | 從 UTF-8 文字檔讀取音色或語氣指令 |
| `--seed` | 固定取樣 seed，讓相同文字/條件更容易重現 |
| `--max-new-tokens` | 最大生成 frame 數，短句可先用 `16` 測試；長中文可用中文字數 × 3 估算 |
| `--temperature` | 取樣溫度，預設 `0.9` |
| `--top-k` | top-k 取樣，預設 `50` |
| `--top-p` | top-p 取樣，預設 `1.0` |
| `--save-tokens-dir` | batch 模式同步輸出每句 `.tokens` 檔 |
| `--no-save-tokens` | batch 模式關閉 token 檔輸出 |

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
- decoder 容量會依實際 frame 數或 batch `--max-new-tokens` 擴展，修復超過 64 幀時的 `narrow` crash。
- 批次模式可避免每句都重新載入模型。
- 批次模式輸出檔名固定為 `prefix_0001.wav`，不再把完整文字放入檔名。
- 批次模式可用 `--save-tokens-dir <dir>` 同步輸出每句 `.tokens` 檔。
- `--instruct` 已走 VoiceDesign/CustomVoice 的獨立 user prompt embedding，不會混入待朗讀文字。
- 大型模型權重與 tokenizer decoder 權重未包含在 zip 內。
- `v0.1.1` 修正 release app 只用目前工作目錄找 `weights/tokenizer` 的問題；現在也會檢查 exe 所在目錄。
- `v0.1.2` 開始，若找不到 Rust tokenizer decoder 權重，app 會自動嘗試執行 bundled `tools/convert_weights.py tokenizer`。
- `v0.1.3` 開始，app 優先使用 bundled `convert_tokenizer.exe` 做 Rust 原生轉換，不需要 Python；Python converter 只作為 fallback。
