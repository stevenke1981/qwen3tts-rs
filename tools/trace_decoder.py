"""Trace decoder.forward() to find quantizer usage."""

import sys, numpy as np
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
codec = mm.speech_tokenizer.model
dec = codec.decoder

import inspect

# Get the decoder's forward method
print("=== decoder forward source ===", file=sys.stderr)
src = inspect.getsource(dec.forward)
print(src[:3000], file=sys.stderr)

# Also check the quantizer's decode method
quantizer = dec.quantizer
print("\n=== quantizer methods ===", file=sys.stderr)
for attr in dir(quantizer):
    if not attr.startswith("__"):
        print(f"  {attr}", file=sys.stderr)

# Check decode method
if hasattr(quantizer, "decode"):
    print("\n=== quantizer.decode source ===", file=sys.stderr)
    src = inspect.getsource(quantizer.decode)
    print(src[:2000], file=sys.stderr)

# Also decompile the forward
# Let's look at the quantizer type
print(f"\nquantizer type: {type(quantizer).__name__}", file=sys.stderr)
print(f"quantizer file: {type(quantizer).__module__}", file=sys.stderr)
