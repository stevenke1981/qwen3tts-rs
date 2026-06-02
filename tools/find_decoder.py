"""Find codec decoder in the Qwen3TTS model."""

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

# Check speech_tokenizer
st = mm.speech_tokenizer
print(f"speech_tokenizer type: {type(st).__name__}", file=sys.stderr)
print(f"speech_tokenizer module: {type(st).__module__}", file=sys.stderr)

# List speech_tokenizer attributes
for attr in dir(st):
    if not attr.startswith("_"):
        print(f"  st.{attr}", file=sys.stderr)

# Check if it has a decoder
if hasattr(st, "decoder"):
    dec = st.decoder
    print(f"\nst.decoder type: {type(dec).__name__}", file=sys.stderr)
    for attr in dir(dec):
        if not attr.startswith("_"):
            print(f"  dec.{attr}", file=sys.stderr)

# Check parameters in speech_tokenizer
total = sum(p.numel() for p in st.parameters()) / 1e6
print(f"\nst params: {total:.0f}M", file=sys.stderr)

# Try loading the separate tokenizer model
# The weights at weights/tokenizer/ are from Qwen3-TTS-Tokenizer-12Hz
# Let's load that separately
print("\n--- Trying separate Tokenizer model ---", file=sys.stderr)
try:
    from transformers import AutoModel

    tok = AutoModel.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    tok.eval()
    print(f"Tokenizer type: {type(tok).__name__}", file=sys.stderr)
    for attr in dir(tok):
        if not attr.startswith("_"):
            print(f"  tok.{attr}", file=sys.stderr)

    # Try decoding
    test_tokens = torch.tensor(
        [
            [
                [
                    1221,
                    1052,
                    1114,
                    1364,
                    1468,
                    1760,
                    974,
                    1318,
                    746,
                    391,
                    161,
                    1013,
                    663,
                    837,
                    216,
                    1929,
                ],
            ]
        ],
        dtype=torch.long,
    )
    print(f"Test tokens shape: {test_tokens.shape}", file=sys.stderr)

    with torch.no_grad():
        out = tok.decode(test_tokens)
    print(f"Tokenizer decode output shape: {out.shape}", file=sys.stderr)
    print(
        f"Output stats: min={out.min().item():.6f}, max={out.max().item():.6f}, mean={out.mean().item():.6f}",
        file=sys.stderr,
    )
    print(f"First 20 samples: {out[0, 0, :20].tolist()}", file=sys.stderr)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback

    traceback.print_exc(file=sys.stderr)
