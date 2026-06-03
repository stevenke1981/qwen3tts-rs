# Qwen3-TTS Rust Release 使用說明

適用版本：`qwen3tts-rs v0.1.2 Windows x64`

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

2. 12Hz tokenizer decoder 權重。

`v0.1.2` 開始，若 app 找不到已轉換的 Rust 權重，會自動嘗試執行 release 包內的：

```text
tools\convert_weights.py
```

自動轉換會下載/讀取 HuggingFace `Qwen/Qwen3-TTS-Tokenizer-12Hz`，並輸出到 release 目錄下：

```text
weights\tokenizer\
```

自動轉換需要本機有 Python 以及這些 Python 套件：

```powershell
pip install torch safetensors huggingface_hub numpy
```

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
  --text "今天天氣真好" `
  --text "你好，測試第二句"
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
  --language chinese
```

## 常用參數

| 參數 | 說明 |
| --- | --- |
| `--model-dir` | Qwen3-TTS 模型 snapshot 目錄 |
| `--language` | 語言，中文建議使用 `chinese` |
| `--speaker` | 說話者條件，可省略 |
| `--max-new-tokens` | 最大生成 frame 數，短句可先用 `16` 測試 |
| `--temperature` | 取樣溫度，預設 `0.9` |
| `--top-k` | top-k 取樣，預設 `50` |
| `--top-p` | top-p 取樣，預設 `1.0` |

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
- release 是 CPU/Candle build。第一次載入 1.7B 權重會花較久時間。
- 批次模式可避免每句都重新載入模型。
- 大型模型權重與 tokenizer decoder 權重未包含在 zip 內。
- `v0.1.1` 修正 release app 只用目前工作目錄找 `weights/tokenizer` 的問題；現在也會檢查 exe 所在目錄。
- `v0.1.2` 開始，若找不到 Rust tokenizer decoder 權重，app 會自動嘗試執行 bundled `tools/convert_weights.py tokenizer`。
