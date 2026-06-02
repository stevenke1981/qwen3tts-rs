"""
Detailed comparison: PyTorch codec decoder vs our Rust decoder.
Saves per-layer intermediate outputs as .npy files for comparison.
"""

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
dec = codec.decoder  # Qwen3TTSTokenizerV2Decoder

# Test tokens: 3 frames, same as before
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

codes_t = test_tokens.transpose(1, 2)  # [1, 16, 3]

# ---- Step through the decoder manually ----
with torch.no_grad():
    # 1. Quantizer (codebook lookup + residual)
    print("=== Quantizer ===", file=sys.stderr)
    quantizer = dec.quantizer

    # Check quantizer structure
    for name, mod in quantizer.named_children():
        n = sum(p.numel() for p in mod.parameters()) / 1e3
        print(f"  {name}: {type(mod).__name__}, {n:.0f}K params", file=sys.stderr)

    # Forward through quantizer
    codes = codes_t.clone()
    # The quantizer likely does: embed each codebook index → sum → project
    quant_out = quantizer(codes)
    if isinstance(quant_out, tuple):
        quant_out = quant_out[0]
    print(f"  quantizer output shape: {quant_out.shape}", file=sys.stderr)
    print(
        f"  quantizer output range: [{quant_out.min().item():.6f}, {quant_out.max().item():.6f}]",
        file=sys.stderr,
    )
    np.save("weights/pt_quant_out.npy", quant_out[0].cpu().numpy())

    # 2. pre_conv
    print("\n=== pre_conv ===", file=sys.stderr)
    pre_conv = dec.pre_conv
    print(f"  pre_conv type: {type(pre_conv).__name__}", file=sys.stderr)
    pre_conv_out = pre_conv(quant_out)
    if isinstance(pre_conv_out, tuple):
        pre_conv_out = pre_conv_out[0]
    print(f"  pre_conv output shape: {pre_conv_out.shape}", file=sys.stderr)
    print(
        f"  pre_conv output range: [{pre_conv_out.min().item():.6f}, {pre_conv_out.max().item():.6f}]",
        file=sys.stderr,
    )
    np.save("weights/pt_pre_conv_out.npy", pre_conv_out[0].cpu().numpy())

    # 3. pre_transformer
    print("\n=== pre_transformer ===", file=sys.stderr)
    pre_trans = dec.pre_transformer
    pt_out = pre_trans(pre_conv_out)
    if isinstance(pt_out, tuple):
        pt_out = pt_out[0]
    print(f"  pre_transformer output shape: {pt_out.shape}", file=sys.stderr)
    print(
        f"  pre_transformer output range: [{pt_out.min().item():.6f}, {pt_out.max().item():.6f}]",
        file=sys.stderr,
    )
    np.save("weights/pt_pre_trans_out.npy", pt_out[0].cpu().numpy())

    # 4. upsample
    print("\n=== upsample ===", file=sys.stderr)
    for i, up_mod in enumerate(dec.upsample):
        pt_out = up_mod(pt_out)
        if isinstance(pt_out, tuple):
            pt_out = pt_out[0]
        print(f"  upsample.{i} output shape: {pt_out.shape}", file=sys.stderr)
        print(
            f"  upsample.{i} output range: [{pt_out.min().item():.6f}, {pt_out.max().item():.6f}]",
            file=sys.stderr,
        )
    np.save("weights/pt_upsample_out.npy", pt_out[0].cpu().numpy())

    # 5. decoder blocks
    print("\n=== decoder ===", file=sys.stderr)
    for i, dec_mod in enumerate(dec.decoder):
        pt_out = dec_mod(pt_out)
        if isinstance(pt_out, tuple):
            pt_out = pt_out[0]
        print(f"  decoder.{i} output shape: {pt_out.shape}", file=sys.stderr)
        print(
            f"  decoder.{i} output range: [{pt_out.min().item():.6f}, {pt_out.max().item():.6f}]",
            file=sys.stderr,
        )
    np.save("weights/pt_decoder_out.npy", pt_out[0].cpu().numpy())

print("\n=== Done! Saved intermediates ===", file=sys.stderr)
