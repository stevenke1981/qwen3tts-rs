"""Test the codec decoder via speech_tokenizer."""

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
st = mm.speech_tokenizer

# Examine st.model (the neural network inside the tokenizer)
codec_model = st.model
print(f"st.model type: {type(codec_model).__name__}", file=sys.stderr)
print(f"st.model module: {type(codec_model).__module__}", file=sys.stderr)

# List children
for name, child in codec_model.named_children():
    n = sum(p.numel() for p in child.parameters()) / 1e6
    print(f"  {name}: {type(child).__name__}, {n:.0f}M params", file=sys.stderr)

# List all layers
print("\nAll layers:", file=sys.stderr)
for name, param in codec_model.named_parameters():
    print(f"  {name}: {param.shape}", file=sys.stderr)

# Test decode
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

with torch.no_grad():
    # Try the decode method
    out = st.decode(test_tokens)
print(f"\nst.decode output type: {type(out).__name__}", file=sys.stderr)
if isinstance(out, torch.Tensor):
    print(f"st.decode output shape: {out.shape}", file=sys.stderr)
elif isinstance(out, (list, tuple)):
    for i, o in enumerate(out):
        if isinstance(o, torch.Tensor):
            print(f"  [{i}] shape: {o.shape}")
        else:
            print(f"  [{i}] type: {type(o).__name__}")
elif isinstance(out, np.ndarray):
    print(f"st.decode output shape: {out.shape}", file=sys.stderr)

# Check what decode returns
print(f"\nOutput value range:", file=sys.stderr)
if isinstance(out, torch.Tensor):
    print(
        f"  min={out.min().item():.6f}, max={out.max().item():.6f}, mean={out.mean().item():.6f}",
        file=sys.stderr,
    )
    print(f"  first 20: {out.flatten()[:20].tolist()}", file=sys.stderr)
    print(f"  output len: {out.shape[-1]}", file=sys.stderr)
