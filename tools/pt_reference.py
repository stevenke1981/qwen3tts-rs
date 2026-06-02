"""Compare PyTorch codec decoder output with same tokens."""

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

# Get the codec decoder directly
codec = mm.speech_tokenizer.model  # Qwen3TTSTokenizerV2Model
dec = codec.decoder  # Qwen3TTSTokenizerV2Decoder

print(f"dec type: {type(dec).__name__}", file=sys.stderr)

# Check if chunked_decode exists
if hasattr(dec, "chunked_decode"):
    print(f"Found dec.chunked_decode!", file=sys.stderr)
    import inspect

    src = inspect.getsource(dec.chunked_decode)
    print(src[:3000], file=sys.stderr)

# Also look at the decoder module structure
print(f"\ndec._modules keys: {list(dec._modules.keys())[:5]}", file=sys.stderr)

# Now test with actual tokens
# Use tokens from LLM generate (shape: [1, num_frames, 16])
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
)  # [1, 3, 16]

print(f"\nTest tokens shape: {test_tokens.shape}", file=sys.stderr)

# Method 1: Use codec.decode() directly (what st.decode calls internally)
with torch.no_grad():
    result = codec.decode(test_tokens)
    print(f"\ncodec.decode result type: {type(result).__name__}", file=sys.stderr)
    if isinstance(result, tuple):
        print(f"  len={len(result)}", file=sys.stderr)
        for i, r in enumerate(result):
            if isinstance(r, torch.Tensor):
                print(
                    f"  [{i}] shape={r.shape}, min={r.min().item():.6f}, max={r.max().item():.6f}",
                    file=sys.stderr,
                )
            else:
                print(f"  [{i}] type={type(r).__name__}", file=sys.stderr)

# Method 2: Use chunked_decode directly
with torch.no_grad():
    # The decode method does: audio_codes.transpose(1, 2) then chunked_decode
    codes_t = test_tokens.transpose(1, 2)  # [1, 16, 3]
    result2 = dec.chunked_decode(codes_t)
    print(f"\nchunked_decode result type: {type(result2).__name__}", file=sys.stderr)
    if isinstance(result2, torch.Tensor):
        print(f"  shape={result2.shape}", file=sys.stderr)
        print(
            f"  min={result2.min().item():.6f}, max={result2.max().item():.6f}",
            file=sys.stderr,
        )
        print(f"  mean={result2.mean().item():.6f}", file=sys.stderr)
        print(f"  first 20: {result2[0, 0, :20].tolist()}", file=sys.stderr)
    elif isinstance(result2, tuple):
        for i, r in enumerate(result2):
            if isinstance(r, torch.Tensor):
                print(
                    f"  [{i}] shape={r.shape}, range=[{r.min().item():.6f}, {r.max().item():.6f}]",
                    file=sys.stderr,
                )

# Try directly calling the decoder modules step by step
# to see each intermediate output
print("\n=== Step-by-step decode ===", file=sys.stderr)

# 1. Codebook lookup via quantizer
quantizer = dec.quantizer
print(f"  quantizer: {type(quantizer).__name__}", file=sys.stderr)

# codes_t shape: [1, 16, 3] -> [batch, num_quantizers, time]
# Actually chunked_decode probably does:
#   quantizer codes → embeddings → pre_conv → pre_transformer → upsample → decoder

# Let me just call chunked_decode and get the full output
# and also compute a reference audio we can compare
with torch.no_grad():
    # Clamp tokens like codec.decode does
    clamped = torch.clamp(test_tokens, min=0)
    audio_codes = clamped.transpose(1, 2)  # [1, 16, 3]
    audio_values = dec.chunked_decode(audio_codes).squeeze(1)
    print(f"\n  Final audio shape: {audio_values.shape}", file=sys.stderr)
    print(
        f"  Range: [{audio_values.min().item():.6f}, {audio_values.max().item():.6f}]",
        file=sys.stderr,
    )
    print(f"  Mean: {audio_values.mean().item():.6f}", file=sys.stderr)
    print(f"  First 40 samples: {audio_values[0, :40].tolist()}", file=sys.stderr)
    print(f"  Last 10 samples: {audio_values[0, -10:].tolist()}", file=sys.stderr)

    # Save reference for comparison
    ref_path = "weights/pt_reference.npy"
    np.save(ref_path, audio_values[0].numpy())
    print(f"\n  Saved reference to {ref_path}", file=sys.stderr)
