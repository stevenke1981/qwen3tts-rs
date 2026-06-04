# Qwen3-TTS Rust Release Guide

Target version: `qwen3tts-rs v0.1.10 Windows x64 / Windows x64 CUDA`

This release package contains pure Rust/Candle executables:

- `synthesize.exe`: single-utterance text-to-speech.
- `synthesize_batch.exe`: batch text-to-speech. The model is loaded once and
  reused across multiple lines.
- `quantize_tokenizer.exe`: quantizes converted tokenizer decoder safetensors
  into Q8/Q4-hybrid directories.

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

If you use a `1.7B-VoiceDesign` snapshot that does not include
`tokenizer.json`, the app automatically tries to reuse the Base model tokenizer
from the local HuggingFace cache. You can also copy the Base model
`tokenizer.json` to `models\tokenizer.json`.

2. The 12Hz tokenizer decoder weights.

Starting from `v0.1.3`, if the app cannot find converted Rust weights, it first
attempts to run the bundled Rust converter:

```text
convert_tokenizer.exe
```

This converter does not require Python. It searches the local HuggingFace cache
for:

```text
Qwen/Qwen3-TTS-Tokenizer-12Hz
```

You can also manually pass a tokenizer decoder snapshot:

```powershell
.\convert_tokenizer.exe --input C:\path\to\Qwen3-TTS-Tokenizer-12Hz\snapshot --output weights\tokenizer
```

If the Rust converter is missing, the app falls back to the older Python
converter:

```text
tools\convert_weights.py
```

Automatic conversion now prefers a user-level cache, so a new release directory
does not need to reconvert more than 1GB of tokenizer decoder weights:

```text
%LOCALAPPDATA%\qwen3tts-rs\tokenizer-12hz\
```

You can override the cache or weight directory with environment variables:

```powershell
$env:QWEN3TTS_TOKENIZER_CACHE_DIR = "D:\qwen3tts-cache"
$env:QWEN3TTS_TOKENIZER_WEIGHT_DIR = "D:\qwen3tts-cache\tokenizer-12hz"
```

Manual conversion can still write into the release directory:

```text
weights\tokenizer\
```

Python plus `torch safetensors huggingface_hub numpy` is only required when the
Python fallback is used.

## CPU And CUDA Packages

- `qwen3tts-rs-v0.1.10-windows-x64.zip`: CPU/Candle build.
- `qwen3tts-rs-v0.1.10-windows-x64-cuda.zip`: CUDA/Candle build. On startup it
  first tries `CUDA:0` and falls back to CPU only if CUDA cannot initialize.

Build the CUDA release package:

```powershell
.\tools\package_release.ps1 -Cuda
```

The default `-CudaComputeCap 86` targets RTX 3070 Ti. For other GPUs, pass the
integer format Candle expects, such as `75` or `89`.

If you already have converted weights, place them in one of these locations:

```text
<current PowerShell directory>\weights\tokenizer\
<directory containing synthesize.exe>\weights\tokenizer\
%LOCALAPPDATA%\qwen3tts-rs\tokenizer-12hz\
```

The release app checks those paths in that order. If you run the exe from a
different directory, the safest layout is to place `weights\tokenizer` next to
the exe:

```text
weights\tokenizer\
```

At minimum, this directory must include:

```text
codebook.safetensors
lightweight.safetensors
pre_transformer.safetensors
upsample.safetensors
decoder_blocks.safetensors
```

If the weights are missing or conversion fails, the app prints the full paths it
checked.

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

For lower hardware requirements, point `--model-dir` to a `0.6B-Base`
snapshot. This is also the CLI default model:

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

## VoiceDesign Instructions

`1.7B-VoiceDesign` supports natural-language voice/style control through
`--instruct`:

```powershell
$model = "C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign\snapshots\<sha>"
.\synthesize.exe `
  --text "Welcome to the native Rust Qwen3-TTS build" `
  --backend candle `
  --language english `
  --model-dir $model `
  --instruct "young female voice, warm and friendly, natural speaking pace" `
  --seed 20260603 `
  --output voicedesign.wav `
  --max-new-tokens 64
```

You can keep reusable voice settings in a UTF-8 text file:

```powershell
"young female voice, warm and friendly, natural speaking pace" | Set-Content .\instruct.txt -Encoding UTF8

.\synthesize.exe `
  --text "Welcome to the native Rust Qwen3-TTS build" `
  --backend candle `
  --language english `
  --model-dir $model `
  --instruct-file .\instruct.txt `
  --seed 20260603 `
  --output voicedesign_file.wav `
  --max-new-tokens 64
```

Base models do not provide stable voice control. Use VoiceDesign plus
`--instruct`, or the built-in speaker presets below, when you need a specific
voice. `--speaker` uses a real speaker id when CustomVoice weights provide a
speaker map. If the loaded model has no speaker map, known speaker names are
translated into a VoiceDesign `--instruct` fallback.

## Built-In Speaker Presets

Starting in `v0.1.9`, the CLI understands the 9 official Qwen CustomVoice
speaker names. List them with:

```powershell
.\synthesize.exe --list-speakers
.\synthesize_batch.exe --list-speakers
```

| Speaker | Description | Recommended language |
| --- | --- | --- |
| `Vivian` | Bright young female voice | Chinese |
| `Serena` | Warm, gentle young female voice | Chinese |
| `Uncle_Fu` | Mature male voice with a mellow timbre | Chinese |
| `Dylan` | Youthful Beijing male voice | Chinese (Beijing) |
| `Eric` | Lively Chengdu male voice | Chinese (Sichuan) |
| `Ryan` | Dynamic male voice with rhythmic delivery | English |
| `Aiden` | Sunny American male voice | English |
| `Ono_Anna` | Playful Japanese female voice | Japanese |
| `Sohee` | Warm Korean female voice | Korean |

CustomVoice models use these names as real speaker ids:

```powershell
$model = "C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice\snapshots\<sha>"
.\synthesize.exe `
  --text "Hello from the native Rust Qwen3-TTS build" `
  --backend candle `
  --language english `
  --model-dir $model `
  --speaker Ryan `
  --output ryan.wav `
  --max-new-tokens 64
```

For VoiceDesign or Base models without a speaker map, the same flag injects the
matching instruction preset. You can combine `--speaker` with `--instruct` or
`--instruct-file`: the speaker preset gives the broad voice, and your instruction
adds style or delivery details.

```powershell
$model = "C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign\snapshots\<sha>"
.\synthesize.exe `
  --text "Welcome to the native Rust Qwen3-TTS build" `
  --backend candle `
  --language english `
  --model-dir $model `
  --speaker Vivian `
  --instruct "warm and friendly, natural speaking pace" `
  --seed 20260604 `
  --output vivian_voicedesign.wav `
  --max-new-tokens 80
```

If long Chinese text is truncated, increase `--max-new-tokens`. A conservative
starting point is:

```text
max-new-tokens ~= Chinese character count x 4
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
  --instruct-file .\instruct.txt `
  --seed 20260603 `
  --text "今天天氣真好" `
  --text "你好，測試第二句"
```

If the default 0.6B Base snapshot is already in the HuggingFace cache, batch
mode can omit `--model-dir`:

```powershell
.\synthesize_batch.exe `
  --text "今天天氣真好" `
  --output-dir batch-output `
  --language chinese
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
  --language chinese `
  --no-save-tokens
```

For per-line voice and seed control, repeat `--instruct-file` and `--seed`.
The repeated counts must match the number of text lines:

```powershell
.\synthesize_batch.exe `
  --model-dir $model `
  --text "First line" `
  --text "Second line" `
  --instruct-file .\voice_a.txt `
  --instruct-file .\voice_b.txt `
  --seed 101 `
  --seed 202 `
  --output-dir batch-output
```

Batch output file names use only the numeric index:

```text
batch-output\qwen1p7b_0001.wav
batch-output\qwen1p7b_0002.wav
```

## Tokenizer Decoder Q8/Q4 Quantization

`v0.1.10` adds `quantize_tokenizer.exe`. It reads converted Rust tokenizer
decoder weights and writes another directory that `synthesize.exe` and
`synthesize_batch.exe` can load directly:

```powershell
.\quantize_tokenizer.exe `
  --input weights\tokenizer `
  --output weights\tokenizer-q8 `
  --format q8_0 `
  --group-size 64 `
  --min-cosine 0.995
```

Use the quantized weights:

```powershell
$env:QWEN3TTS_TOKENIZER_WEIGHT_DIR = "D:\qwen3tts-rs\weights\tokenizer-q8"
.\synthesize.exe --tokens .\sample.tokens --output q8.wav
```

Quantization status:

| Mode | Status | Notes |
| --- | --- | --- |
| `q8_0` | Recommended for testing | Local tokenizer decoder result: 99 tensors quantized, 137 preserved, about 26.6% of the original size; decoder smoke cosine vs base `0.99974333` |
| `q4_0` | Experimental | Tensors below `--min-cosine` are automatically preserved as F32 anchors. With a 0.995 gate locally, only 12 tensors quantized, 224 preserved, about 94.7% of the original size; decoder smoke cosine vs base `0.99235672` |

The tool writes:

```text
weights\tokenizer-q8\quantization_report.json
```

The report lists every tensor, whether it was quantized, cosine similarity,
maximum absolute error, and preservation reason. To force a hard failure instead
of auto-preserving low-cosine tensors:

```powershell
.\quantize_tokenizer.exe --format q4_0 --fail-low-cosine
```

The current loader prioritizes compatibility: it reads Q8/Q4 safetensors and
dequantizes back to F32 tensors before calling the existing decoder. This
reduces disk size, but it is not an int8/int4 kernel speedup yet. Vocoder /
HiFi-GAN-style weights remain preserved by default to avoid audio quality loss.

You can record the calibration/evaluation text manifest in the report so
multiple agents compare the same sample set:

```powershell
.\quantize_tokenizer.exe `
  --input weights\tokenizer `
  --output weights\tokenizer-q8 `
  --format q8_0 `
  --calibration-texts .\calibration_texts.txt
```

## Common Options

| Option | Description |
| --- | --- |
| `--model-dir` | Qwen3-TTS model snapshot directory |
| `--model` | Batch mode HuggingFace model id; used for local HF cache auto-search when `--model-dir` is omitted |
| `--language` | Language condition. Use `chinese` for Chinese prompts |
| `--speaker` | Speaker condition; supports built-in presets such as `Vivian`, `Uncle_Fu`, and `Dylan`. CustomVoice uses real speaker ids; other models use instruct fallback |
| `--list-speakers` | Show built-in speaker presets |
| `--instruct` | VoiceDesign/CustomVoice voice or style instruction |
| `--instruct-file` | Read voice or style instruction from a UTF-8 text file; in batch mode repeat it once per line to switch voices |
| `--seed` | Fixed sampling seed; in batch mode repeat it once per line to vary seeds |
| `--max-new-tokens` | Maximum generated frame count. Use `16` for short smoke tests; for long Chinese text, start with character count x 4 |
| `--temperature` | Sampling temperature, default `0.9` |
| `--top-k` | Top-k sampling, default `50` |
| `--top-p` | Top-p sampling, default `1.0` |
| `--speed` | Speech speed factor, default `1.0` |
| `--save-tokens-dir` | Batch mode token-file output directory |
| `--no-save-tokens` | Disable batch token-file output |
| `quantize_tokenizer.exe --format` | Tokenizer decoder quantization format: `q8_0` or `q4_0` |
| `quantize_tokenizer.exe --min-cosine` | Tensor cosine gate, default `0.995`; low-cosine tensors are preserved as F32 anchors |

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
- `v0.1.4` provides both CPU and CUDA release packages. The CUDA binary prefers
  `CUDA:0`.
- `v0.1.5` adds VoiceDesign/CustomVoice `--instruct` support and automatically
  falls back to a Base tokenizer when a VoiceDesign snapshot does not include
  `tokenizer.json`.
- `v0.1.6` adds `--instruct-file`, `--seed`, batch `--no-save-tokens`, and
  batch warnings for long Chinese text with low `--max-new-tokens`.
- `v0.1.7` adds `--speed` and `--version`, and fixes Chinese truncation
  guidance.
- `v0.1.8` adds per-line batch `--instruct-file`/`--seed`, batch 0.6B model
  auto-search, and a global tokenizer decoder cache.
- `v0.1.9` adds the 9 built-in CustomVoice speaker presets, `--list-speakers`,
  and automatic instruct fallback for Base/VoiceDesign models without a speaker
  map.
- `v0.1.10` adds `quantize_tokenizer.exe`, Q8/Q4-hybrid safetensors,
  WeightLoader auto-dequantization, and tokenizer decoder quantization reports.
- Decoder capacity now expands from the actual frame count, or from batch
  `--max-new-tokens`, fixing the `narrow` crash above 64 frames.
- Batch mode avoids reloading the model for every sentence.
- Batch mode writes fixed index-only names like `prefix_0001.wav`; it no longer
  embeds the full text in the filename.
- Batch mode can write per-line `.tokens` files with `--save-tokens-dir <dir>`.
- `--instruct` follows the VoiceDesign/CustomVoice separate user prompt
  embedding path; it is not mixed into the text to be spoken.
- Large model weights and tokenizer decoder weights are not bundled in the zip.
- `v0.1.1` fixes the release app only checking `weights/tokenizer` relative to
  the current working directory. It now also checks the executable directory.
- Starting from `v0.1.2`, if Rust tokenizer decoder weights are missing, the app
  automatically attempts to run bundled `tools/convert_weights.py tokenizer`.
- Starting from `v0.1.3`, the app prefers bundled `convert_tokenizer.exe` for
  Rust-native conversion without Python. The Python converter is only a fallback.
