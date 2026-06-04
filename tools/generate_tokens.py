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
import random
import struct
import sys
import wave
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
    seed: Optional[int] = None,
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
    if seed is not None:
        set_seed(seed)
        print(f"[bridge] seed={seed}", file=sys.stderr)

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


def set_seed(seed: int):
    """Seed Python, NumPy, and Torch sampling for reproducible token generation."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def api_language(language: str) -> str:
    """Map CLI language names to qwen_tts generate_* API names."""
    if not language or language.lower() == "auto":
        return "Auto"

    aliases = {
        "zh": "Chinese",
        "cn": "Chinese",
        "chinese": "Chinese",
        "en": "English",
        "english": "English",
        "fr": "French",
        "french": "French",
        "de": "German",
        "german": "German",
        "it": "Italian",
        "italian": "Italian",
        "es": "Spanish",
        "spanish": "Spanish",
        "pt": "Portuguese",
        "portuguese": "Portuguese",
        "ja": "Japanese",
        "japanese": "Japanese",
        "ko": "Korean",
        "korean": "Korean",
        "ru": "Russian",
        "russian": "Russian",
    }
    return aliases.get(language.lower(), language)


def as_numpy_wav(wav):
    """Convert qwen_tts output into a mono float32 NumPy array."""
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.squeeze(wav)
    if wav.ndim > 1:
        wav = wav[0]
    return np.clip(wav, -1.0, 1.0)


def write_wav(path: str, wav, sample_rate: int):
    """Write mono PCM16 WAV without depending on soundfile."""
    samples = as_numpy_wav(wav)
    pcm = (samples * 32767.0).astype("<i2", copy=False)
    with wave.open(path, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(int(sample_rate))
        writer.writeframes(pcm.tobytes())


def generate_voice_clone_wav(
    model,
    text: str,
    language: str,
    reference_audio: str,
    output_wav: str,
    reference_text: Optional[str] = None,
    temperature=0.9,
    top_k=50,
    top_p=1.0,
    max_new_tokens=4096,
    seed: Optional[int] = None,
):
    """Generate Voice Clone audio with the official qwen_tts high-level API."""
    if seed is not None:
        set_seed(seed)
        print(f"[bridge] seed={seed}", file=sys.stderr)

    reference_text = reference_text.strip() if reference_text else None
    x_vector_only = reference_text is None
    language_name = api_language(language)
    print(f"[bridge] voice-clone language={language_name}", file=sys.stderr)
    print(f"[bridge] reference_audio={reference_audio}", file=sys.stderr)
    if reference_text:
        print(f"[bridge] reference_text={reference_text}", file=sys.stderr)
    else:
        print("[bridge] reference_text not provided; using speaker-embedding-only mode", file=sys.stderr)

    with torch.no_grad():
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language_name,
            ref_audio=reference_audio,
            ref_text=reference_text,
            x_vector_only_mode=x_vector_only,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )

    write_wav(output_wav, wavs[0], sr)
    duration = as_numpy_wav(wavs[0]).shape[0] / float(sr)
    print(f"[bridge] Wrote {output_wav} ({duration:.2f}s @ {sr} Hz)", file=sys.stderr)


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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--voice-clone",
        action="store_true",
        help="Run official generate_voice_clone() and write --output-wav",
    )
    parser.add_argument("--reference-audio", type=str, default=None)
    parser.add_argument("--reference-text", type=str, default=None)
    parser.add_argument("--output-wav", type=str, default=None)

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

    if args.voice_clone:
        if not args.reference_audio:
            print("[bridge] Error: --voice-clone requires --reference-audio", file=sys.stderr)
            sys.exit(1)
        if not args.output_wav:
            print("[bridge] Error: --voice-clone requires --output-wav", file=sys.stderr)
            sys.exit(1)

        print(f"[bridge] Generating voice clone ({len(text)} chars)...", file=sys.stderr)
        generate_voice_clone_wav(
            loaded,
            text,
            language=args.language,
            reference_audio=args.reference_audio,
            output_wav=args.output_wav,
            reference_text=args.reference_text,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        print("[bridge] Done.", file=sys.stderr)
        return

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
        seed=args.seed,
    )

    n = codes_list[0].shape[0]
    print(f"[bridge] Generated {n} frames ~ {n / 12.5:.1f}s audio", file=sys.stderr)

    # Restore true stdout and write binary
    sys.stdout = true_stdout
    write_codes_binary(codes_list, sys.stdout)
    print(f"[bridge] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
