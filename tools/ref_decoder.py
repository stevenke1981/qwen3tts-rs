"""Compare PyTorch codec decoder output vs our Rust decoder.

This script loads the Qwen3-TTS model, gets its codec decoder,
runs inference with known tokens, and saves the reference output.
"""

import sys, struct, os
import numpy as np
import torch

# Load model
from qwen_tts import Qwen3TTSModel

model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
print(f"[ref] Loading {model_id}...", file=sys.stderr)
model = Qwen3TTSModel.from_pretrained(
    model_id,
    device_map="cpu",
    dtype=torch.float32,
    trust_remote_code=True,
)
mm = model.model
mm.eval()
device = next(mm.parameters()).device
print(f"[ref] Model loaded on {device}", file=sys.stderr)

# Create test tokens (same format as what LLM produces: [1, num_frames, 16])
# Use tokens from our earlier test
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
            [
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
                1100,
                1200,
                1300,
                1400,
                1500,
                1600,
            ],
            [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
        ]
    ],
    dtype=torch.long,
)

print(f"[ref] Test tokens shape: {test_tokens.shape}", file=sys.stderr)

# Find the codec decoder in the model
# The model has mm.codec or mm.decoder or similar
# Let's explore
print("[ref] Model components:", file=sys.stderr)
for name, mod in mm.named_children():
    n = sum(p.numel() for p in mod.parameters()) / 1e6
    print(f"  {name}: {n:.0f}M params", file=sys.stderr)

# Try to find codec decoder
# In Qwen3-TTS, the codec decoder might be mm.codec_decoder or similar
# Let's look for decode-related methods
for attr in dir(mm):
    if (
        "codec" in attr.lower()
        or "decode" in attr.lower()
        or "tokenizer" in attr.lower()
    ):
        print(f"  method/attr: {attr}", file=sys.stderr)

# Check if there's a separate codec model
for name, child in mm.named_children():
    if "codec" in name.lower() or "decode" in name.lower():
        print(f"  codec child: {name}, type={type(child).__name__}", file=sys.stderr)

# Actually, let's try using mm.generate() directly as we did before
# and also try to get the intermediate codec decoder outputs

# The model.generate() returns (codes_list, hiddens)
# Let's trace: talker_codes are used by codec decoder to produce audio
# But in this model, decode_chunk is probably handled by mm.codec_decoder

# Check for decode method
if hasattr(mm, "codec_decoder"):
    print("[ref] Found mm.codec_decoder!", file=sys.stderr)
    cd = mm.codec_decoder
    # Try to run it
    with torch.no_grad():
        out = cd(test_tokens)
    print(f"[ref] codec_decoder output shape: {out.shape}", file=sys.stderr)
elif hasattr(mm, "decode"):
    print("[ref] Found mm.decode()!", file=sys.stderr)
    with torch.no_grad():
        out = mm.decode(test_tokens)
    print(f"[ref] decode output shape: {out.shape}", file=sys.stderr)
else:
    print("[ref] No codec_decoder or decode found", file=sys.stderr)
    # Try to look at the model architecture more carefully
    print(f"[ref] mm type: {type(mm).__name__}", file=sys.stderr)
    print(f"[ref] mm module file: {type(mm).__module__}", file=sys.stderr)
