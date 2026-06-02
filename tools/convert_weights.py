#! /usr/bin/env python3
"""
Qwen3-TTS → Rust/Candle 權重轉換腳本

將 HuggingFace Qwen3-TTS 模型權重轉換為 Rust/Candle
可直接載入的 safetensors 格式。

用法:
    # 轉換 Tokenizer Decoder 權重 (12Hz codec decoder)
    python tools/convert_weights.py tokenizer --output ./weights/tokenizer

    # 轉換完整 TTS 模型 (Talker + CodePredictor + Tokenizer)
    python tools/convert_weights.py full --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --output ./weights

    # 僅轉換 Codebook 與 Conv 權重 (為 decoder_12hz.rs 準備)
    python tools/convert_weights.py lightweight --output ./weights/lightweight
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import safetensors.torch as st
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# 路徑輔助
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 模型載入
# ---------------------------------------------------------------------------


def load_safetensors(
    repo_id: str, filename: str = "model.safetensors", subfolder: Optional[str] = None
) -> dict[str, torch.Tensor]:
    """從 HuggingFace Hub 下載並載入 safetensors 權重。"""
    path = hf_hub_download(repo_id, filename, subfolder=subfolder)
    tensors = st.load_file(path)
    print(
        f"  Loaded {len(tensors)} tensors from {repo_id}/{filename or subfolder or ''}"
    )
    return tensors


def load_config(
    repo_id: str, filename: str = "config.json", subfolder: Optional[str] = None
) -> dict:
    """從 HuggingFace Hub 下載 config.json。"""
    path = hf_hub_download(repo_id, filename, subfolder=subfolder)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Codebook 權重處理
# ---------------------------------------------------------------------------


def extract_codebook_weights(decoder_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    """從 SplitResidualVectorQuantizer 提取並預計算 16 層碼本權重。

    PyTorch 中:
      - rvq_first: 1 層 (semantic), dim=256, codebook_size=2048
      - rvq_rest: 15 層 (acoustic), dim=256, codebook_size=2048
      - 每層有 input_proj (512→256) 與 output_proj (256→512)
      - 實際碼本 = embedding_sum / cluster_usage[:, None]

    回傳形狀: (16, 2048, 512) — 直接可用的嵌入表
    """
    num_layers = 16
    codebook_size = 2048
    embed_dim = 512  # codebook_dim

    # rvq_first: 1 層
    first_cu = decoder_tensors[
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"
    ]
    first_es = decoder_tensors[
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"
    ]
    first_out_proj = decoder_tensors[
        "decoder.quantizer.rvq_first.output_proj.weight"
    ]  # (512, 256, 1)

    # 計算實際碼本嵌入
    first_embed = first_es / first_cu.clamp(min=1e-5)[:, None]  # (2048, 256)

    # 投影到 512-dim: output_proj 是 Conv1d(256→512, kernel=1)
    # weight: (512, 256, 1) → Linear(256→512)
    first_embed_proj = first_embed @ first_out_proj.squeeze(-1).T  # (2048, 512)
    first_embed_proj = first_embed_proj.unsqueeze(0)  # (1, 2048, 512)

    # rvq_rest: 15 層
    rest_embeds = []
    rest_out_proj = decoder_tensors[
        "decoder.quantizer.rvq_rest.output_proj.weight"
    ]  # (512, 256, 1)

    for i in range(15):
        cu = decoder_tensors[
            f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"
        ]
        es = decoder_tensors[
            f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"
        ]
        embed = es / cu.clamp(min=1e-5)[:, None]  # (2048, 256)
        embed_proj = embed @ rest_out_proj.squeeze(-1).T  # (2048, 512)
        rest_embeds.append(embed_proj.unsqueeze(0))

    # 合併: (16, 2048, 512)
    all_embeds = torch.cat([first_embed_proj] + rest_embeds, dim=0)
    return all_embeds.contiguous()


# ---------------------------------------------------------------------------
# 卷積權重處理
# ---------------------------------------------------------------------------


def extract_conv_weights(decoder_tensors: dict[str, torch.Tensor]) -> list[dict]:
    """提取所有 CausalConv1d 權重 (pre_conv + decoder blocks 中的 conv)。

    回傳 list of dict: [{"name": ..., "weight": Tensor, "bias": Tensor|None}, ...]
    """
    convs = []

    # 1. pre_conv (codebook_dim→latent_dim, kernel=3)
    convs.append(
        {
            "name": "pre_conv",
            "weight": decoder_tensors["decoder.pre_conv.conv.weight"],
            "bias": decoder_tensors["decoder.pre_conv.conv.bias"],
        }
    )

    # 2. decoder 起始 conv
    convs.append(
        {
            "name": "decoder_start",
            "weight": decoder_tensors["decoder.decoder.0.conv.weight"],
            "bias": decoder_tensors["decoder.decoder.0.conv.bias"],
        }
    )

    # 3. decoder blocks 中的 conv (transposed + residual)
    for i in range(1, 6):  # 5 decoder blocks
        prefix = f"decoder.decoder.{i}"
        # transposed conv (block.1)
        convs.append(
            {
                "name": f"decoder_block_{i}_transposed",
                "weight": decoder_tensors[f"{prefix}.block.1.conv.weight"],
                "bias": decoder_tensors[f"{prefix}.block.1.conv.bias"],
            }
        )
        # residual unit convs (block.2,3,4)
        for j in range(2, 5):
            for k in [1, 2]:  # conv1, conv2
                convs.append(
                    {
                        "name": f"decoder_block_{i}_ru{j - 1}_conv{k}",
                        "weight": decoder_tensors[
                            f"{prefix}.block.{j}.conv{k}.conv.weight"
                        ],
                        "bias": decoder_tensors[
                            f"{prefix}.block.{j}.conv{k}.conv.bias"
                        ],
                    }
                )

    # 4. final conv (decoder.6)
    convs.append(
        {
            "name": "decoder_final",
            "weight": decoder_tensors["decoder.decoder.6.conv.weight"],
            "bias": decoder_tensors["decoder.decoder.6.conv.bias"],
        }
    )

    return convs


def extract_lightweight_weights(
    decoder_tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """提取輕量級權重 (為 decoder_12hz.rs 準備)。

    僅提取:
    - codebook_weights: (16, 2048, 512) — 所有 16 層碼本嵌入
    - pre_conv.weight, pre_conv.bias — 首層 causal conv
    """
    result = {}

    # Codebook 權重
    result["codebook_weights"] = extract_codebook_weights(decoder_tensors)

    # pre_conv
    result["pre_conv.weight"] = decoder_tensors["decoder.pre_conv.conv.weight"]
    result["pre_conv.bias"] = decoder_tensors["decoder.pre_conv.conv.bias"]

    return result


# ---------------------------------------------------------------------------
# Transformer 權重處理 (pre_transformer)
# ---------------------------------------------------------------------------


def extract_transformer_weights(
    decoder_tensors: dict[str, torch.Tensor], prefix: str = "decoder.pre_transformer"
) -> dict[str, torch.Tensor]:
    """提取 pre_transformer 權重。

    結構:
    - input_proj: Linear(latent_dim, hidden_size)
    - output_proj: Linear(hidden_size, latent_dim)
    - norm: RMSNorm(hidden_size)
    - layers.0.* ~ layers.7.* (8 層 Transformer)
    """
    result = {}
    for key in decoder_tensors:
        if key.startswith(prefix):
            # 保留原命名以便對應
            rust_key = key.replace(f"{prefix}.", "pre_transformer.")
            result[rust_key] = decoder_tensors[key]
    return result


def extract_upsample_weights(
    decoder_tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """提取 upsample blocks 權重。"""
    result = {}
    for key in decoder_tensors:
        if key.startswith("decoder.upsample"):
            rust_key = key.replace("decoder.", "")
            result[rust_key] = decoder_tensors[key]
    return result


def extract_decoder_blocks(
    decoder_tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """提取 decoder blocks (波形重建)。"""
    result = {}
    for key in decoder_tensors:
        if key.startswith("decoder.decoder"):
            rust_key = key.replace("decoder.", "")
            result[rust_key] = decoder_tensors[key]
    return result


# ---------------------------------------------------------------------------
# Tokenizer 權重提取總入口
# ---------------------------------------------------------------------------


def convert_tokenizer(output_dir: str, repo_id: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz"):
    """轉換 Tokenizer 模型全部權重。"""
    print(f"\n{'=' * 60}")
    print(f"Converting tokenizer: {repo_id}")
    print(f"{'=' * 60}")

    tensors = load_safetensors(repo_id)
    config = load_config(repo_id)

    out = ensure_dir(output_dir)

    # 1. Codebook 權重
    print("\n[1/5] Extracting codebook weights...")
    codebook = extract_codebook_weights(tensors)
    st.save_file(
        {"codebook_weights": codebook}, os.path.join(out, "codebook.safetensors")
    )
    print(f"  Saved: codebook.safetensors ({list(codebook.shape)})")

    # 2. Lightweight conv weights
    print("\n[2/5] Extracting lightweight conv weights...")
    light = extract_lightweight_weights(tensors)
    st.save_file(
        {k: v for k, v in light.items() if k != "codebook_weights"},
        os.path.join(out, "lightweight.safetensors"),
    )
    for k, v in light.items():
        if k != "codebook_weights":
            print(f"  {k}: {list(v.shape)}")

    # 3. Pre-transformer
    print("\n[3/5] Extracting pre-transformer...")
    transformer = extract_transformer_weights(tensors)
    st.save_file(transformer, os.path.join(out, "pre_transformer.safetensors"))
    print(f"  Saved: pre_transformer.safetensors ({len(transformer)} tensors)")

    # 4. Upsample blocks
    print("\n[4/5] Extracting upsample blocks...")
    upsample = extract_upsample_weights(tensors)
    st.save_file(upsample, os.path.join(out, "upsample.safetensors"))
    print(f"  Saved: upsample.safetensors ({len(upsample)} tensors)")

    # 5. Decoder blocks (full waveform decoder)
    print("\n[5/5] Extracting decoder blocks...")
    decoder_blocks = extract_decoder_blocks(tensors)
    st.save_file(decoder_blocks, os.path.join(out, "decoder_blocks.safetensors"))
    print(f"  Saved: decoder_blocks.safetensors ({len(decoder_blocks)} tensors)")

    # 6. 保存配置
    config_out = {
        "model_type": "qwen3_tts_tokenizer_12hz",
        "decoder_config": config.get("decoder_config", {}),
        "encoder_config": config.get("encoder_config", {}),
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"  Saved: config.json")

    print(f"\n*** Tokenizer conversion complete -> {out}")
    return out


# ---------------------------------------------------------------------------
# Lightweight 模式 (僅 codebook + conv)
# ---------------------------------------------------------------------------


def convert_lightweight(output_dir: str):
    """僅提取輕量級權重給現有 decoder_12hz.rs 使用。"""
    print(f"\n{'=' * 60}")
    print("Converting lightweight weights (codebook + pre_conv)")
    print(f"{'=' * 60}")

    tensors = load_safetensors("Qwen/Qwen3-TTS-Tokenizer-12Hz")
    out = ensure_dir(output_dir)

    light = extract_lightweight_weights(tensors)
    st.save_file(light, os.path.join(out, "lightweight.safetensors"))
    print(f"  codebook_weights: {list(light['codebook_weights'].shape)}")
    print(f"  pre_conv.weight: {list(light['pre_conv.weight'].shape)}")
    print(f"  pre_conv.bias: {list(light['pre_conv.bias'].shape)}")
    print(f"\n*** Lightweight conversion complete -> {out}")
    return out


# ---------------------------------------------------------------------------
# Full model (Talker + CodePredictor + Tokenizer)
# ---------------------------------------------------------------------------


def convert_talker(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """提取 Talker LM 權重 (28 層 Transformer)。"""
    result = {}
    for key in tensors:
        if key.startswith("model.talker."):
            rust_key = key.replace("model.talker.", "talker.")
            result[rust_key] = tensors[key]
    return result


def convert_code_predictor(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """提取 CodePredictor 權重 (5 層 Transformer)。"""
    result = {}
    for key in tensors:
        if key.startswith("model.code_predictor."):
            rust_key = key.replace("model.code_predictor.", "code_predictor.")
            result[rust_key] = tensors[key]
    return result


def convert_full(model_id: str, output_dir: str):
    """轉換完整 TTS 模型。"""
    print(f"\n{'=' * 60}")
    print(f"Converting full model: {model_id}")
    print(f"{'=' * 60}")

    base_out = ensure_dir(output_dir)

    # 載入主模型權重
    print("\n[1/3] Loading main model...")
    talker_tensors = load_safetensors(model_id, "model.safetensors")

    # Talker
    print("  Extracting Talker weights...")
    talker = convert_talker(talker_tensors)
    st.save_file(talker, os.path.join(base_out, "talker.safetensors"))
    print(f"  Saved: talker.safetensors ({len(talker)} tensors)")

    # CodePredictor
    print("  Extracting CodePredictor weights...")
    cp = convert_code_predictor(talker_tensors)
    if cp:
        st.save_file(cp, os.path.join(base_out, "code_predictor.safetensors"))
        print(f"  Saved: code_predictor.safetensors ({len(cp)} tensors)")
    else:
        print("  (no CodePredictor weights found)")

    # 載入 tokenizer 權重 (from subfolder)
    print("\n[2/3] Loading speech tokenizer...")
    tokenizer_out = os.path.join(base_out, "tokenizer")
    convert_tokenizer(tokenizer_out)

    # 保存主配置
    print("\n[3/3] Saving config...")
    config = load_config(model_id)
    with open(os.path.join(base_out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: config.json")

    print(f"\n*** Full model conversion complete -> {base_out}")
    return base_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS → Rust/Candle weight converter",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # tokenizer
    tp = subparsers.add_parser("tokenizer", help="Convert tokenizer decoder weights")
    tp.add_argument("--repo", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    tp.add_argument("--output", "-o", default="./weights/tokenizer")

    # lightweight
    lp = subparsers.add_parser("lightweight", help="Convert only codebook+conv weights")
    lp.add_argument("--output", "-o", default="./weights/lightweight")

    # full
    fp = subparsers.add_parser("full", help="Convert full TTS model")
    fp.add_argument("--model", "-m", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    fp.add_argument("--output", "-o", default="./weights/full")

    args = parser.parse_args()

    if args.command == "tokenizer":
        convert_tokenizer(args.output, args.repo)
    elif args.command == "lightweight":
        convert_lightweight(args.output)
    elif args.command == "full":
        convert_full(args.model, args.output)


if __name__ == "__main__":
    main()
