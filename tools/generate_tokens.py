#!/usr/bin/env python3
"""
Qwen3-TTS Token Generator — Python bridge for Rust text_frontend.

Usage:
    python tools/generate_tokens.py --text "Hello world" > tokens.bin

Output (binary, little-endian stdout):
    [num_frames: u32] [frame0_16xu16] [frame1_16xu16] ...
"""

import argparse
import contextlib
import io
import struct
import sys
import warnings
from typing import Optional

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*flash-attn is not installed.*")


def load_model(model_id: str):
    """Load Qwen3-TTS model via the high-level API."""
    from qwen_tts import Qwen3TTSModel

    print(f"[bridge] Loading {model_id}...", file=sys.stderr)
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cpu",
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model.model.eval()
    n = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"[bridge] Loaded ({n:.0f}M params)", file=sys.stderr)
    return model


def generate_codes(
    model,
    text: str,
    language: str = "auto",
    speaker: Optional[str] = None,
    instruct: Optional[str] = None,
    temperature=0.9,
    top_k=50,
    top_p=1.0,
    max_new_tokens=4096,
):
    """
    Generate codec tokens from plain text.

    Builds the chat-format input and calls model.generate().
    Returns list of [seq_len, 16] uint16 arrays.
    """
    mm = model.model
    tokenizer = model.processor.tokenizer
    device = next(mm.parameters()).device

    # ── Build chat-format input ──────────────────────────────────────
    # Format: <|im_start|>assistant\nTEXT<|im_end|>\n<|im_start|>assistant\n
    im_start = tokenizer("<|im_start|>", return_tensors="pt")["input_ids"][0].to(device)
    im_end = tokenizer("<|im_end|>", return_tensors="pt")["input_ids"][0].to(device)
    asst = tokenizer("assistant\n", return_tensors="pt")["input_ids"][0].to(device)
    user = tokenizer("user\n", return_tensors="pt")["input_ids"][0].to(device)
    nl = tokenizer("\n", return_tensors="pt")["input_ids"][0].to(device)

    text_ids = tokenizer(text, return_tensors="pt")["input_ids"][0].to(device)
    full_ids = torch.cat([im_start, asst, text_ids, im_end, nl, im_start, asst])
    input_ids = [full_ids.unsqueeze(0)]

    instruct_ids = None
    if instruct:
        instruct_text_ids = tokenizer(instruct, return_tensors="pt")["input_ids"][0].to(device)
        instruct_full_ids = torch.cat([im_start, user, instruct_text_ids, im_end, nl])
        instruct_ids = [instruct_full_ids.unsqueeze(0)]

    # Map language string → ID
    lang_map = getattr(mm.config.talker_config, "codec_language_id", {})
    lang_lower = language.lower()
    if lang_lower == "auto":
        lang_actual = detect_language(text, lang_map)
    elif lang_lower in lang_map:
        lang_actual = lang_lower
    else:
        # Try to find case-insensitive match
        matches = [k for k in lang_map if k.lower() == lang_lower]
        lang_actual = matches[0] if matches else "english"

    speaker_actual = speaker if speaker else None

    print(f"[bridge] language={lang_actual}", file=sys.stderr)
    if speaker_actual:
        print(f"[bridge] speaker={speaker_actual}", file=sys.stderr)
    if instruct:
        print(f"[bridge] instruct={instruct}", file=sys.stderr)

    # ── Generate ─────────────────────────────────────────────────────
    with torch.no_grad():
        codes_list, _ = mm.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=[lang_actual],
            speakers=[speaker_actual],
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.05,
        )

    result = []
    for codes in codes_list:
        result.append(codes.cpu().numpy().astype(np.uint16))
    return result


def detect_language(text: str, lang_map: dict) -> str:
    """Infer Qwen3-TTS language key from Unicode ranges."""
    ranges = [
        ("chinese", "\u4e00", "\u9fff"),
        ("japanese", "\u3040", "\u30ff"),
        ("korean", "\uac00", "\ud7af"),
        ("russian", "\u0400", "\u04ff"),
    ]
    for language, start, end in ranges:
        if language in lang_map and any(start <= ch <= end for ch in text):
            return language

    latin = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text)
    if latin > 0 and "english" in lang_map:
        return "english"

    return "english" if "english" in lang_map else next(iter(lang_map), "english")


def write_codes_binary(codes_list, stream):
    """Write token frames as binary to an output stream."""
    for codes in codes_list:
        num_frames = codes.shape[0]
        stream.buffer.write(struct.pack("<I", num_frames))
        for frame in codes:
            for token in frame:
                stream.buffer.write(struct.pack("<H", int(token)))


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS token generator")
    parser.add_argument("--text", type=str, default=None, help="Input text")
    parser.add_argument(
        "--text-file", type=str, default=None, help="Read text from file"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", help="Model ID"
    )
    parser.add_argument("--language", type=str, default="auto")
    parser.add_argument("--speaker", type=str, default=None)
    parser.add_argument("--instruct", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)

    args = parser.parse_args()

    # Get text
    if args.text:
        text = args.text
    elif args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        text = sys.stdin.read().strip()

    if not text:
        print("[bridge] Error: no input text", file=sys.stderr)
        sys.exit(1)

    # Load model (first call downloads ~1GB)
    # Redirect stdout to stderr to prevent library warnings/prints corrupting binary output
    true_stdout = sys.stdout
    sys.stdout = sys.stderr

    loaded = load_model(args.model)

    # Generate codes
    print(f"[bridge] Generating tokens ({len(text)} chars)...", file=sys.stderr)
    codes_list = generate_codes(
        loaded,
        text,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    n = codes_list[0].shape[0]
    print(f"[bridge] Generated {n} frames ~ {n / 12.5:.1f}s audio", file=sys.stderr)

    # Restore true stdout and write binary
    sys.stdout = true_stdout
    write_codes_binary(codes_list, sys.stdout)
    print(f"[bridge] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
