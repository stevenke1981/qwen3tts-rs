#!/usr/bin/env python3
"""
Qwen3-TTS Token Generator — Python bridge for Rust text_frontend.

Usage:
    python generate_tokens.py --text "Hello world" > tokens.bin
    python generate_tokens.py --text "你好" --model "Qwen/Qwen3-TTS-12Hz-0.6B-Base" > tokens.bin

Output (binary, little-endian):
    [num_frames: u32]
    [frame0_token0: u16] ... [frame0_token15: u16]
    [frame1_token0: u16] ... [frame1_token15: u16]
    ...

Each frame = 16 codebook tokens (one per quantizer layer, 12Hz mode).
"""

import argparse
import sys
import struct
import warnings
from typing import List, Optional

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*flash-attn is not installed.*")


def load_model(model_id: str, device: str = "auto"):
    """Load Qwen3-TTS model and tokenizer."""
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
    )
    from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[bridge] Loading model {model_id} on {device}...", file=sys.stderr)

    processor = Qwen3TTSProcessor.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        trust_remote_code=True,
    )
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    print(
        f"[bridge] Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)",
        file=sys.stderr,
    )
    return model, processor, device


def generate_codes(
    model,
    processor,
    text: str,
    device: str,
    language: str = "auto",
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    max_new_tokens: int = 4096,
) -> List[np.ndarray]:
    """
    Generate codec tokens from text using the talker.

    Returns list of [seq_len, 16] uint16 arrays (one per batch item).
    """
    # Process text input
    inputs = processor(
        text=[text],
        language=[language],
        return_tensors="pt",
        padding=True,
    )
    input_ids = [ids.to(device) for ids in inputs["input_ids"]]
    instruct_ids = [
        ids.to(device) if ids is not None else None for ids in inputs["instruct_ids"]
    ]
    languages = inputs["language"]

    # Generate
    with torch.no_grad():
        talker_codes_list, _ = model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=[None],
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.05,
        )

    # Convert to numpy uint16
    result = []
    for codes in talker_codes_list:
        codes_np = codes.cpu().numpy().astype(np.uint16)  # [seq_len, 16]
        result.append(codes_np)

    return result


def write_codes_binary(codes_list: List[np.ndarray], output_stream):
    """Write token frames as binary to output_stream."""
    for codes in codes_list:
        num_frames = codes.shape[0]
        # Header: number of frames
        output_stream.buffer.write(struct.pack("<I", num_frames))
        # Frames: each frame is 16 × u16 (32 bytes)
        for frame in codes:
            for token in frame:
                output_stream.buffer.write(struct.pack("<H", int(token)))


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS token generator")
    parser.add_argument(
        "--text", type=str, default=None, help="Input text to synthesize"
    )
    parser.add_argument(
        "--text-file", type=str, default=None, help="Read text from file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Model ID (default: Qwen/Qwen3-TTS-12Hz-0.6B-Base)",
    )
    parser.add_argument("--language", type=str, default="auto", help="Language")
    parser.add_argument(
        "--temperature", type=float, default=0.9, help="Sampling temperature"
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument(
        "--max-new-tokens", type=int, default=4096, help="Max new tokens"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cuda/cpu)"
    )

    args = parser.parse_args()

    # Get text input
    if args.text:
        text = args.text
    elif args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        # Read from stdin
        text = sys.stdin.read().strip()

    if not text:
        print("[bridge] Error: no input text", file=sys.stderr)
        sys.exit(1)

    # Load model
    model, processor, device = load_model(args.model, args.device)

    # Generate codes
    print(
        f"[bridge] Generating tokens for text ({len(text)} chars)...", file=sys.stderr
    )
    codes_list = generate_codes(
        model=model,
        processor=processor,
        text=text,
        device=device,
        language=args.language,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    num_frames = codes_list[0].shape[0]
    print(
        f"[bridge] Generated {num_frames} frames, saving to stdout...", file=sys.stderr
    )

    # Write binary output to stdout
    write_codes_binary(codes_list, sys.stdout)

    print(f"[bridge] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
