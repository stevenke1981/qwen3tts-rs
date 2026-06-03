# Qwen3-TTS Rust Agent Voice Failure Playbook

This note is for agents porting Qwen3-TTS to Rust/Candle. It records the fixes
for the earlier "high-frequency noise / waveform but no real human voice" issue,
and separates it from the later native talker silence and mixed-language issues.

## Golden rule

Do not judge this project only by "a WAV file was written". A broken pipeline can
produce non-zero PCM that sounds like high-frequency noise, or valid-length PCM
that is complete silence.

Always verify all three layers:

1. Token source: text frontend / talker generated valid 16-codebook frames.
2. Codec decoder: Rust `decode_frames` matches PyTorch reference numerically.
3. Audio artifact: WAV has non-zero RMS/peak and sounds like speech.

## Failure mode A: high-frequency noise, fake waveform, no human voice

### Symptoms

- WAV has visible waveform or non-zero samples.
- Audio does not sound like speech.
- It can sound like high-frequency noise, buzzy artifacts, or unstable codec
  output.
- Frame count and sample rate may look correct, so the issue is easy to miss.

### Root causes found

The decoder path was structurally close but not PyTorch-equivalent:

- `PreTransformer` skipped causal masking when `seq_len <= window`.
- `SnakeBeta` used the wrong formula.
- Decoder conv blocks used symmetric padding instead of PyTorch-style causal
  left padding.
- Transposed conv used padding/output-padding to force length instead of
  PyTorch's `padding=0` plus right-side crop.
- Residual unit dilation was fixed at `1` instead of `1, 3, 9`.
- Offline synthesis decoded frame-by-frame with `decode_chunk`, losing the
  full causal transformer context used by PyTorch batch decoding.
- Raw u16 token fixtures were not accepted by the CLI token parser.

### Fix pattern

Use full-path PyTorch alignment, not subjective listening first.

Required decoder fixes:

- Use causal mask for every sequence length in `apply_sliding_window_mask`.
- Match `SnakeBeta` exactly:

```text
alpha = exp(alpha)
beta = exp(beta)
y = x + sin(x * alpha)^2 / (beta + 1e-9)
```

- Use explicit causal conv padding:

```text
padding = (kernel_size - 1) * dilation + 1 - stride
```

- For causal transposed conv, use `padding=0`, then crop the right side by
  `kernel_size - stride`.
- Use residual dilations `1, 3, 9`.
- For offline synthesis, call `Decoder12Hz::decode_frames(&stream.frames)`.
  Do not concatenate many isolated `decode_chunk()` calls for a full utterance.
- Accept both standard token binary format and raw u16 frame arrays.

### Validation target

The reference alignment target for this class of bug:

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

Recommended commands:

```powershell
cargo test --features candle-llm
cargo run --example synthesize -- --tokens weights\test_tokens.bin --output output-fixed.wav
```

Audio smoke check:

```powershell
@'
import wave, struct, math
name = "output-fixed.wav"
with wave.open(name, "rb") as w:
    n = w.getnframes()
    sr = w.getframerate()
    data = w.readframes(n)
    s = struct.unpack("<" + "h" * (len(data) // 2), data)
    rms = math.sqrt(sum(x*x for x in s) / len(s)) if s else 0
    peak = max((abs(x) for x in s), default=0)
    print(f"{name}: duration={n/sr:.3f}s sr={sr} rms={rms:.1f} peak={peak}")
'@ | python -
```

## Failure mode B: mixed-language or unstable human-like speech

### Symptoms

- Audio has speech-like components.
- The utterance sounds like multiple languages mixed together.
- Chinese input may sound like Russian or other languages blended in.

### Root causes found

- `auto` language routing previously mapped to English too aggressively.
- Python token generation did not receive the selected speaker.
- Chinese text sent through English codec conditions can produce confusing
  multilingual prosody or token distributions.

### Fix pattern

- Detect language from Unicode ranges when `--language auto`.
- Route CJK text to `chinese`.
- Pass `--speaker` through `PythonBridge` into the Python token generator.
- When checking native Rust output, save known-good tokens first, then replay
  them through Rust decoder:

```powershell
cargo run --example synthesize -- --text "你好，這是一段中文語音測試。" --language auto --save-tokens tokens-zh-auto.bin --output output-zh-auto.wav
cargo run --example synthesize -- --tokens tokens-zh-auto.bin --output output-zh-native-rerun.wav
```

The two WAVs from token replay should match if the decoder is correct.

## Failure mode C: 1.7B native Candle short text is silent

### Symptoms

- `cn_candle_1.7b_hello.wav` has audible "hello".
- `cn_candle_1.7b_short.wav` has correct duration but RMS/peak are zero.
- Greedy token generation can produce valid-looking token IDs while the codec
  output is silence.

### Root causes found

- 1.7B talker and code predictor have different hidden widths:
  - talker/codebook side: 2048
  - code predictor transformer side: 1024
- The 1.7B checkpoint includes:

```text
talker.code_predictor.small_to_mtp_projection.weight [1024, 2048]
talker.code_predictor.small_to_mtp_projection.bias   [1024]
```

- The Rust loader originally assumed 0.6B-style 1024-wide identity projection.
- Greedy codec generation was too brittle for practical 1.7B generation.

### Fix pattern

- Infer `TalkerConfig` from safetensors shapes instead of using static defaults.
- Load optional `small_to_mtp_projection` when present.
- Project code predictor inputs before transformer layers.
- Keep greedy generation for PyTorch alignment tests.
- Use deterministic sampling for practical synthesis when `temperature > 0`.
- Suppress control tokens from the main codec head (`>= 2048`) while allowing
  EOS.

The current practical default path uses:

```text
temperature = 0.9
top_k = 50
top_p = 1.0
```

Validation data from the fixed 1.7B path:

```text
cn_candle_1.7b_short.wav:               duration=1.280s rms=0.0    peak=0
cn_candle_1.7b_short_sampled_final.wav: duration=1.280s rms=1837.7 peak=7067
```

## Agent checklist

Before changing model logic:

- Use `codebase-memory-mcp` first:
  - `search_graph`
  - `trace_path`
  - `get_code_snippet`
- Identify whether the bug is in token generation, decoder math, or audio I/O.
- Save token frames with `--save-tokens` whenever possible.
- Replay saved tokens through Rust decoder before blaming talker generation.
- Compare tensors against PyTorch fixtures for math changes.
- Measure WAV RMS and peak; do not rely only on file size.
- Keep Python/PyTorch out of the Rust runtime path. Python is only a reference
  or fixture-export tool.

## Known good verification commands

```powershell
cargo test --features candle-llm
cargo test --test talker_alignment_test talker_prompt_generation_matches_pytorch_fixture -- --ignored --nocapture
cargo build --release --example synthesize --features candle-llm
```

1.7B native smoke:

```powershell
$model = "C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\fd4b254389122332181a7c3db7f27e918eec64e3"
.\target\release\examples\synthesize.exe `
  --text "今天天氣真好" `
  --backend candle `
  --language chinese `
  --model-dir $model `
  --output cn_candle_1.7b_short_sampled_final.wav `
  --max-new-tokens 16
```

## Do not repeat these mistakes

- Do not use GGUF for the codec decoder, codebooks, vocoder, or causal convnet.
- Do not evaluate decoder correctness from one hand-listened WAV.
- Do not decode offline utterances as independent one-frame chunks.
- Do not assume 0.6B and 1.7B share all hidden dimensions.
- Do not use `--language auto` without verifying the resolved language.
- Do not treat a valid-length WAV as proof of valid speech.
