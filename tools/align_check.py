#! /usr/bin/env python3
"""
Qwen3-TTS Rust 數值對齊驗證腳本

比較 PyTorch 原版模型與 Rust Candle 實作的中間層輸出，
確保餘弦相似度 >= 0.999。

用法:
    # 對齊檢查 (需要原版 + Rust 輸出)
    python tools/align_check.py --pytorch-output ./ref_output.pt --rust-output ./rust_output.npz

    # 導出參考輸出 (用於 Rust 測試)
    python tools/align_check.py export --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --tokens "0,1,2,...,15" --output ./refs

    # 生成測試 Token 序列
    python tools/align_check.py gen_tokens --length 100 --output ./refs/test_tokens.bin
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 餘弦相似度
# ---------------------------------------------------------------------------


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """計算兩個張量的餘弦相似度。"""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    dot = (a_flat * b_flat).sum()
    norm_a = a_flat.norm()
    norm_b = b_flat.norm()
    if norm_a.item() < 1e-10 or norm_b.item() < 1e-10:
        return 1.0 if torch.equal(a_flat, b_flat) else 0.0
    return (dot / (norm_a * norm_b)).item()


def mean_squared_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """計算均方誤差。"""
    return ((a.float() - b.float()) ** 2).mean().item()


# ---------------------------------------------------------------------------
# 導出參考輸出
# ---------------------------------------------------------------------------


def export_reference_outputs(model_id: str, tokens: list[int], output_dir: str):
    """從 PyTorch 原版模型導出中間層輸出作為參考。

    用 hook 註冊所有卷積層、Transformer 層的中間輸出。
    """
    try:
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Model,
            Qwen3TTSTokenizerV2Config,
        )
    except ImportError:
        print("ERROR: qwen-tts not installed. Run: pip install qwen-tts")
        sys.exit(1)

    device = torch.device("cpu")
    print(f"Loading model {model_id}...")
    config = Qwen3TTSTokenizerV2Config.from_pretrained(model_id)
    model = Qwen3TTSTokenizerV2Model(config)
    model = model.to(device)
    model.eval()

    # Load weights
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    weights_path = hf_hub_download(model_id, "model.safetensors")
    state_dict = load_file(weights_path)

    # Filter only decoder keys
    decoder_keys = {
        k.removeprefix("decoder."): v
        for k, v in state_dict.items()
        if k.startswith("decoder.")
    }
    missing, unexpected = model.decoder.load_state_dict(decoder_keys, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 收集中間輸出
    intermediates = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            intermediates[name] = output.detach().cpu()

        return hook

    # 註冊 hook (decoder 部分)
    decoder = model.decoder
    hooks = []

    # pre_conv
    hooks.append(decoder.pre_conv.register_forward_hook(make_hook("pre_conv")))

    # pre_transformer layers
    for i, layer in enumerate(decoder.pre_transformer.layers):
        hooks.append(
            layer.register_forward_hook(make_hook(f"pre_transformer.layer_{i}"))
        )

    # pre_transformer output
    hooks.append(
        decoder.pre_transformer.norm.register_forward_hook(
            make_hook("pre_transformer.norm")
        )
    )

    # upsample blocks
    for i, block_list in enumerate(decoder.upsample):
        for j, block in enumerate(block_list):
            hooks.append(block.register_forward_hook(make_hook(f"upsample.{i}.{j}")))

    # decoder blocks
    for i, block in enumerate(decoder.decoder):
        hooks.append(block.register_forward_hook(make_hook(f"decoder_block.{i}")))

    # 構造 fake codes: (1, num_quantizers, seq_len)
    seq_len = len(tokens) // 16
    codes = torch.tensor(tokens[: 16 * seq_len], dtype=torch.long).reshape(
        1, seq_len, 16
    )
    codes = codes.transpose(1, 2)  # (1, 16, seq_len)
    codes = codes.clamp(min=0)

    print(f"Running forward pass with codes shape {codes.shape}...")
    with torch.no_grad():
        output = decoder(codes)

    # 保存中間輸出
    for name, tensor in intermediates.items():
        np_path = out / f"{name}.npy"
        np.save(str(np_path), tensor.numpy())
        print(f"  Saved {name}: {list(tensor.shape)}")

    # 保存最終輸出
    np_path = out / "output.npy"
    np.save(str(np_path), output.numpy())
    print(f"  Saved output: {list(output.shape)}")

    # 清理 hook
    for h in hooks:
        h.remove()

    print(f"\nReference outputs saved to {out}")
    return str(out)


# ---------------------------------------------------------------------------
# 對齊檢查
# ---------------------------------------------------------------------------


def check_alignment(pytorch_path: str, rust_path: str, threshold: float = 0.999):
    """對齊檢查入口。"""
    print(f"Loading PyTorch ref: {pytorch_path}")
    ref = np.load(pytorch_path)

    print(f"Loading Rust output: {rust_path}")
    rust = (
        np.load(rust_path) if rust_path.endswith(".npy") else dict(np.load(rust_path))
    )

    results = {}
    all_pass = True

    for key in ref.files if hasattr(ref, "files") else [""]:
        ref_t = torch.from_numpy(ref[key] if key else ref)
        rust_t = torch.from_numpy(rust[key] if key else rust)

        cos = cosine_similarity(ref_t, rust_t)
        mse = mean_squared_error(ref_t, rust_t)
        results[key] = {"cosine": cos, "mse": mse, "pass": cos >= threshold}

        status = "PASS" if cos >= threshold else "FAIL"
        print(f"  {key or 'output'}: cos={cos:.6f} mse={mse:.8e} [{status}]")
        if cos < threshold:
            all_pass = False

    if all_pass:
        print(f"\n*** ALL ALIGNMENT CHECKS PASSED (threshold={threshold})")
    else:
        print(f"\n*** SOME CHECKS FAILED (threshold={threshold})")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 生成測試 Token 序列
# ---------------------------------------------------------------------------


def gen_test_tokens(
    length: int = 100, seed: int = 42, output: str = "./refs/test_tokens.bin"
):
    """生成隨機但可重現的測試 Token 序列。"""
    rng = np.random.RandomState(seed)
    tokens = rng.randint(0, 2048, size=(length, 16), dtype=np.uint16)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(str(out_path))

    print(f"Generated {length}x16 tokens -> {out_path}")
    print(f"  Shape: {tokens.shape}, dtype: uint16")
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS numerical alignment checker",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check
    cp = subparsers.add_parser(
        "check", help="Check alignment between PyTorch and Rust outputs"
    )
    cp.add_argument("--pytorch-output", required=True)
    cp.add_argument("--rust-output", required=True)
    cp.add_argument("--threshold", type=float, default=0.999)

    # export
    ep = subparsers.add_parser(
        "export", help="Export reference outputs from PyTorch model"
    )
    ep.add_argument("--model", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    ep.add_argument("--tokens", type=lambda s: [int(x) for x in s.split(",")])
    ep.add_argument("--length", type=int, default=100)
    ep.add_argument("--output", "-o", default="./refs")

    # gen_tokens
    gp = subparsers.add_parser("gen_tokens", help="Generate test token sequences")
    gp.add_argument("--length", type=int, default=100)
    gp.add_argument("--seed", type=int, default=42)
    gp.add_argument("--output", "-o", default="./refs/test_tokens.bin")

    args = parser.parse_args()

    if args.command == "check":
        check_alignment(args.pytorch_output, args.rust_output, args.threshold)
    elif args.command == "export":
        if args.tokens:
            tokens = args.tokens
        else:
            rng = np.random.RandomState(42)
            tokens = rng.randint(0, 2048, size=args.length * 16).tolist()
        export_reference_outputs(args.model, tokens, args.output)
    elif args.command == "gen_tokens":
        gen_test_tokens(args.length, args.seed, args.output)


if __name__ == "__main__":
    main()
