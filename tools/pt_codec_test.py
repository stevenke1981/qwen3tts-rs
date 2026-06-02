"""Properly run the PyTorch codec decoder and save reference output."""

import sys, struct, os, json
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
model = Qwen3TTSModel.from_pretrained(
    model_id,
    device_map="cpu",
    dtype=torch.float32,
    trust_remote_code=True,
)
mm = model.model
mm.eval()
st = mm.speech_tokenizer

# st.model is Qwen3TTSTokenizerV2Model
codec = st.model
dec = codec.decoder  # Qwen3TTSTokenizerV2Decoder

print(f"=== decoder structure ===", file=sys.stderr)
for name, mod in dec.named_children():
    n = sum(p.numel() for p in mod.parameters()) / 1e6
    print(f"  {name}: {type(mod).__name__}, {n:.0f}M params", file=sys.stderr)

# Try to understand what the decode method actually does
# Let's look at the source
import inspect

try:
    src = inspect.getsource(st.decode)
    print(f"\n=== st.decode source ===", file=sys.stderr)
    print(src[:2000], file=sys.stderr)
except:
    print("Cannot get source", file=sys.stderr)

# Try calling codec.decode() directly if it exists
if hasattr(codec, "decode"):
    print(f"\n=== codec.decode source ===", file=sys.stderr)
    try:
        src = inspect.getsource(codec.decode)
        print(src[:2000], file=sys.stderr)
    except:
        print("Cannot get source", file=sys.stderr)

# The model has an encoder and decoder.
# Let's try: encode speech, then decode the codes
# Load a test audio file or create a synthetic one
print("\n=== Trying encode/decode cycle ===", file=sys.stderr)

# Create synthetic audio (1 second of 24kHz sine sweep)
sr = st.get_input_sample_rate()
print(f"Input sample rate: {sr}Hz", file=sys.stderr)

# Use st.load_audio or just create a simple audio
# Let's see if there's a default audio loading method
try:
    # Create synthetic test audio
    t = torch.linspace(0, 1.0, int(sr))
    audio = 0.5 * torch.sin(2 * np.pi * 220 * t)  # 220Hz sine
    audio = audio.unsqueeze(0)  # [1, samples]

    # Encode
    encoded = st.encode(audio)
    print(f"encode output type: {type(encoded).__name__}", file=sys.stderr)
    if isinstance(encoded, dict):
        for k, v in encoded.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}, {v.dtype}", file=sys.stderr)
            else:
                print(f"  {k}: {type(v).__name__}", file=sys.stderr)

    # Decode
    decoded = st.decode(encoded)
    print(f"decode output type: {type(decoded).__name__}", file=sys.stderr)
    if isinstance(decoded, torch.Tensor):
        print(f"  shape: {decoded.shape}", file=sys.stderr)
        print(
            f"  range: {decoded.min().item():.4f} - {decoded.max().item():.4f}",
            file=sys.stderr,
        )
        print(f"  first 20: {decoded[0, :20].tolist()}", file=sys.stderr)
    elif isinstance(decoded, list):
        for i, d in enumerate(decoded):
            if isinstance(d, torch.Tensor):
                print(f"  [{i}] shape: {d.shape}", file=sys.stderr)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback

    traceback.print_exc(file=sys.stderr)
