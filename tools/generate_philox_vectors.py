#!/usr/bin/env python3
"""Generate and validate Philox vector fixture for P02-T01 sampling parity.

The generator is independent from the Rust implementation under test and uses a
pure-Python Philox4x32-10 implementation with Random123 constants.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = ROOT / "fixtures" / "alignment" / "p02_philox_vectors.json"
PYTHON = "python tools/generate_philox_vectors.py"

PHILOX_M0 = 0xD251_1F53
PHILOX_M1 = 0xCD9E_8D57
PHILOX_W0 = 0x9E37_79B9
PHILOX_W1 = 0xBB67_AE85
CURAND_2POW32_INV = 2.3283064365386963e-10
PINNED_COMMIT = "82cd05b9f3a175612dc89fd6943e610fab096ef5"
SOURCE_REPO = "https://github.com/ServeurpersoCom/qwentts.cpp"


def f32(value: float) -> float:
    """Apply f32 rounding exactly like a single-precision CUDA/C++ conversion."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


# Non-standard numeric widths intentionally use Python integers because they are
# arbitrary precision and then wrapped with masks.
def mulhilo32(a: int, b: int) -> tuple[int, int]:
    prod = (a & 0xFFFF_FFFF) * (b & 0xFFFF_FFFF)
    return (prod >> 32) & 0xFFFF_FFFF, prod & 0xFFFF_FFFF


def philox_round(state: tuple[int, int, int, int], k0: int, k1: int) -> tuple[int, int, int, int]:
    x, y, z, w = state
    hi0, lo0 = mulhilo32(PHILOX_M0, x)
    hi1, lo1 = mulhilo32(PHILOX_M1, z)
    return (
        (hi1 ^ y ^ k0) & 0xFFFF_FFFF,
        lo1,
        (hi0 ^ w ^ k1) & 0xFFFF_FFFF,
        lo0,
    )


def philox4x32_10(seed: int, subseq: int, ctr_lo: int) -> tuple[int, int, int, int]:
    seed_lo = seed & 0xFFFF_FFFF
    seed_hi = (seed >> 32) & 0xFFFF_FFFF
    state = (
        ctr_lo & 0xFFFF_FFFF,
        0,
        subseq & 0xFFFF_FFFF,
        (subseq >> 32) & 0xFFFF_FFFF,
    )
    state = philox_round(state, seed_lo, seed_hi)
    k0, k1 = seed_lo, seed_hi
    for _ in range(9):
        k0 = (k0 + PHILOX_W0) & 0xFFFF_FFFF
        k1 = (k1 + PHILOX_W1) & 0xFFFF_FFFF
        state = philox_round(state, k0, k1)
    return state


def philox_uniform(seed: int, subseq: int, ctr_lo: int) -> tuple[int, int, int, int, str, float]:
    x, y, z, w = philox4x32_10(seed, subseq, ctr_lo)
    u = f32((f32(float(x)) + f32(0.5)) * f32(CURAND_2POW32_INV))
    return x, y, z, w, f"0x{f32_bits(u):08x}", float(u)


def expected_cases() -> list[dict[str, Any]]:
    definitions = [
        ("zero_seed_zero_counter", 0, 0, 0),
        ("seed1_subseq0_ctr0", 1, 0, 0),
        ("seed123456789abcdef0_subseq1_ctr7", 0x1234_5678_9ABC_DEF0, 1, 7),
        ("seed_deadbeef_subseq42_ctr0", 0xDEAD_BEEF_DEAD_BEEF, 42, 0),
        ("seed_beef_subseq_rollover", 0x1234_5678_90AB_CDEF, 0xFFFF_FFFF_FFFF_FFFF, 0),
    ]

    samples_per_case = [1, 1, 1, 1, 2]

    entries = []
    for (name, seed, subseq_start, ctr_lo), n in zip(definitions, samples_per_case):
        samples = []
        for i in range(n):
            subseq = (subseq_start + i) & 0xFFFF_FFFF_FFFF_FFFF
            x, y, z, w, uniform_bits, uniform = philox_uniform(seed, subseq, ctr_lo)
            samples.append(
                {
                    "subsequence": str(subseq),
                    "uniform_bits": uniform_bits,
                    "uniform": uniform,
                    "words": [x, y, z, w],
                }
            )
        entries.append(
            {
                "name": name,
                "seed": str(seed),
                "seed_hex": f"0x{seed:016x}",
                "ctr_lo": ctr_lo,
                "subsequence_start": str(subseq_start),
                "samples": samples,
            }
        )
    return entries


def build_fixture() -> dict[str, Any]:
    return {
        "version": 1,
        "fixture_id": "p02-philox-vectors",
        "source_repo": SOURCE_REPO,
        "source_revision": PINNED_COMMIT,
        "generated_by": "tools/generate_philox_vectors.py independent Philox implementation",
        "command": PYTHON,
        "cases": expected_cases(),
    }


def validate(path: Path) -> None:
    expected = build_fixture()
    if not path.exists():
        raise RuntimeError(f"FIXTURE_MISSING: {path}")
    current = json.loads(path.read_text(encoding="utf-8"))
    if current != expected:
        raise RuntimeError("FIXTURE_MISMATCH: p02 Philox vectors do not match generator output")


def write_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_fixture()
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate/validate Philox vectors for P02-T01")
    p.add_argument(
        "--check",
        type=Path,
        help="Validate an existing fixture file instead of generating",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=FIXTURE_PATH,
        help="Write location for generated vectors",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.check is not None:
        validate(args.check)
        print(f"PHILOX_VECTORS_CHECK_OK {args.check}")
        return 0

    write_fixture(args.output)
    print(f"PHILOX_VECTORS_GENERATED {args.output}")
    print(f"CASES={len(expected_cases())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
