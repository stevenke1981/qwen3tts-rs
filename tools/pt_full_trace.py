"""Direct comparison: PyTorch decoder.forward() per-layer outputs."""

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
codes_t = test_tokens.transpose(1, 2)  # [1, 16, 3]

with torch.no_grad():
    # Step 1: quantizer
    hidden = dec.quantizer.decode(codes_t)
    print(
        f"quantizer output: {hidden.shape}, range=[{hidden.min():.4f}, {hidden.max():.4f}, mean={hidden.mean():.6f}]",
        file=sys.stderr,
    )
    np.save("weights/pt_quant_out.npy", hidden[0].cpu().numpy())

    # Step 2: pre_conv
    hidden = dec.pre_conv(hidden)
    print(
        f"pre_conv output: {hidden.shape}, range=[{hidden.min():.4f}, {hidden.max():.4f}]",
        file=sys.stderr,
    )
    np.save("weights/pt_pre_conv_out.npy", hidden[0].cpu().numpy())

    # Step 2.5: transpose for transformer
    hidden = hidden.transpose(1, 2)
    print(f"transposed: {hidden.shape}", file=sys.stderr)

    # Step 3: pre_transformer - check what it outputs
    pt_result = dec.pre_transformer(inputs_embeds=hidden)
    print(f"pre_transformer result type: {type(pt_result).__name__}", file=sys.stderr)
    if hasattr(pt_result, "last_hidden_state"):
        hidden = pt_result.last_hidden_state
    elif isinstance(pt_result, tuple):
        hidden = pt_result[0]
    else:
        hidden = pt_result
    print(
        f"pre_transformer output: {hidden.shape}, range=[{hidden.min():.4f}, {hidden.max():.4f}]",
        file=sys.stderr,
    )
    np.save("weights/pt_pre_trans_out.npy", hidden[0].cpu().numpy())

    # Step 4: permute to channel-first
    hidden = hidden.permute(0, 2, 1)
    print(f"permuted: {hidden.shape}", file=sys.stderr)

    # Step 5: upsample
    for i, blocks in enumerate(dec.upsample):
        for j, block in enumerate(blocks):
            hidden = block(hidden)
            print(
                f"  upsample.{i}.{j} output: {hidden.shape}, range=[{hidden.min():.4f}, {hidden.max():.4f}]",
                file=sys.stderr,
            )
    np.save("weights/pt_upsample_out.npy", hidden[0].cpu().numpy())

    # Step 6: decoder blocks
    wav = hidden
    for i, block in enumerate(dec.decoder):
        wav = block(wav)
        print(
            f"  decoder.{i} output: {wav.shape}, range=[{wav.min():.4f}, {wav.max():.4f}]",
            file=sys.stderr,
        )
    np.save("weights/pt_decoder_out.npy", wav[0].cpu().numpy())

    # Final clamp
    wav = wav.clamp(min=-1, max=1)
    print(
        f"\nFINAL: {wav.shape}, range=[{wav.min():.6f}, {wav.max():.6f}]",
        file=sys.stderr,
    )
    print(f"First 20: {wav[0, 0, :20].tolist()}", file=sys.stderr)

print("\nDone! Saved all intermediates to weights/pt_*.npy", file=sys.stderr)
