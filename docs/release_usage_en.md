# Qwen3-TTS Rust Release Guide

Target version: `qwen3tts-rs v0.1.0 Windows x64`

This release package contains pure Rust/Candle executables:

- `synthesize.exe`: single-utterance text-to-speech.
- `synthesize_batch.exe`: batch text-to-speech. The model is loaded once and
  reused across multiple lines.

## Important Limitations

The release package does not include large model weights. You need to provide:

1. A Qwen3-TTS talker model snapshot, for example:

```text
C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\<sha>
```

The directory must contain:

```text
model.safetensors
tokenizer.json
```

2. The 12Hz tokenizer decoder weights under the current working directory:

```text
weights\tokenizer\
```

At minimum, this directory must include `codebook.safetensors` and the
additional safetensors files needed by the decoder.

## Single Utterance

Run from the release directory:

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

Output:

```text
output.wav
```

## Batch Synthesis

Batch mode loads the model once, then synthesizes multiple lines. This avoids
reloading the 1.7B weights for every sentence.

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

You can also use a text file:

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

## Common Options

| Option | Description |
| --- | --- |
| `--model-dir` | Qwen3-TTS model snapshot directory |
| `--language` | Language condition. Use `chinese` for Chinese prompts |
| `--speaker` | Optional speaker condition |
| `--max-new-tokens` | Maximum generated frame count. Use `16` for short smoke tests |
| `--temperature` | Sampling temperature, default `0.9` |
| `--top-k` | Top-k sampling, default `50` |
| `--top-p` | Top-p sampling, default `1.0` |

## Verify The Output Is Not Silent

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

If `rms=0` and `peak=0`, the WAV is silent.

## Known Status

- Native Candle 1.7B short Chinese prompts now produce non-silent audio.
- This is a CPU/Candle release build. Loading 1.7B weights can take noticeable
  time on startup.
- Batch mode avoids reloading the model for every sentence.
- Large model weights and tokenizer decoder weights are not bundled in the zip.
