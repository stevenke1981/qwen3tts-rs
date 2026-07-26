#!/usr/bin/env python3
"""Compare mono PCM WAV outputs using only the Python standard library."""

from __future__ import annotations
import argparse
import json
import math
import struct
import wave
from pathlib import Path


def read_wav(path: Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono WAV")
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM WAV")
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    values = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    return rate, [v / 32768.0 for v in values]


def metrics(a: list[float], b: list[float]) -> dict:
    n = min(len(a), len(b))
    if n == 0:
        raise ValueError("empty audio")
    aa, bb = a[:n], b[:n]
    dot = sum(x*y for x,y in zip(aa,bb))
    na = math.sqrt(sum(x*x for x in aa))
    nb = math.sqrt(sum(y*y for y in bb))
    cosine = dot / max(na*nb, 1e-30)
    diffs = [abs(x-y) for x,y in zip(aa,bb)]
    mse = sum((x-y)**2 for x,y in zip(aa,bb)) / n
    return {
        "samples_a": len(a), "samples_b": len(b), "compared": n,
        "cosine": cosine, "max_abs": max(diffs), "mse": mse,
        "length_equal": len(a) == len(b),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--max-abs", type=float, default=1e-4)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    ra, a = read_wav(args.reference)
    rb, b = read_wav(args.candidate)
    result = metrics(a,b)
    result.update({"sample_rate_a":ra, "sample_rate_b":rb})
    passed = (ra == rb and result["length_equal"] and
              result["cosine"] >= args.min_cosine and
              result["max_abs"] <= args.max_abs)
    result["passed"] = passed
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
