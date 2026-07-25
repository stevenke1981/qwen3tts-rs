#!/usr/bin/env python3
"""Generate and validate repetition-penalty parity vectors for P02-T02.

The generator is independent from the Rust implementation under test and records:
- signed-zero and NaN handling in filtered logits
- duplicate / out-of-range history application (deduplicated once)
- operation order results (repetition penalty then temperature scaling)
- expected stochastic samples for fixed uniform draws
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = ROOT / "fixtures" / "alignment" / "p02_repetition_penalty_vectors.json"
PYTHON = "python tools/generate_repetition_penalty_vectors.py"
SOURCE_REPO = "https://github.com/ServeurpersoCom/qwentts.cpp"
PINNED_COMMIT = "82cd05b9f3a175612dc89fd6943e610fab096ef5"
PHILOX_M0 = 0xD251_1F53
PHILOX_M1 = 0xCD9E_8D57
PHILOX_W0 = 0x9E37_79B9
PHILOX_W1 = 0xBB67_AE85
CURAND_2POW32_INV = 2.328_306_436_538_696_3e-10


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def to_f32(bits: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits, 16) & 0xFFFF_FFFF))[0]


def filter_candidates(
    logits: List[float],
    suppress_from: int | None,
    allow_suppressed_token: int | None,
) -> List[Tuple[int, float]]:
    candidates: List[Tuple[int, float]] = []
    for idx, score in enumerate(logits):
        if not math.isfinite(score):
            continue
        if suppress_from is not None:
            if idx >= suppress_from and idx != allow_suppressed_token:
                continue
        candidates.append((idx, score))
    return candidates


def sort_by_desc_score(candidates: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    return sorted(candidates, key=lambda x: (x[1], -x[0]), reverse=True)


def apply_repetition_penalty(
    logits: List[float],
    history: Iterable[int],
    penalty: float,
) -> List[float]:
    if penalty == 1.0:
        return list(logits)

    seen = [False] * len(logits)
    result = list(logits)
    p = f32(penalty)
    for raw in history:
        idx = int(raw)
        if idx < 0 or idx >= len(result):
            continue
        if seen[idx]:
            continue
        seen[idx] = True

        score = result[idx]
        result[idx] = f32(score * p) if score < 0.0 else f32(score / p)
    return result


def to_probabilities(candidates: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    max_logit = f32(candidates[0][1])
    probs: List[Tuple[int, float]] = []
    total = 0.0
    for idx, logit in candidates:
        p = f32(math.exp(float(f32(logit - max_logit))))
        probs.append((idx, p))
        total += p
    if total <= 0.0 or not math.isfinite(total):
        return [(candidates[0][0], 1.0)]
    return [(idx, f32(p / total)) for idx, p in probs]


def apply_temperature_in_place(candidates: List[Tuple[int, float]], temperature: float) -> None:
    if not math.isfinite(temperature):
        return
    inv_temp = float(f32(1.0 / temperature)) if temperature > 0.0 else 0.0
    for i in range(len(candidates)):
        idx, logit = candidates[i]
        candidates[i] = (idx, float(f32(logit * inv_temp)))


def apply_top_p(candidates: List[Tuple[int, float]], top_p: float) -> List[Tuple[int, float]]:
    if not (0.0 <= top_p < 1.0):
        return candidates
    cumulative = 0.0
    kept: List[Tuple[int, float]] = []
    for idx, p in candidates:
        cumulative += p
        kept.append((idx, p))
        if cumulative >= top_p:
            break
    total = sum(p for _, p in kept)
    if total > 0.0:
        kept = [(idx, p / total) for idx, p in kept]
    return kept


def sample_with_draw(
    candidates: List[Tuple[int, float]],
    draw: float,
) -> int:
    cumulative = 0.0
    for idx, p in candidates:
        cumulative += p
        if cumulative >= f32(draw):
            return idx
    return candidates[-1][0]


def mulhilo32(a: int, b: int) -> Tuple[int, int]:
    product = (a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)
    return ((product >> 32) & 0xFFFFFFFF, product & 0xFFFFFFFF)


def philox_round(state: Tuple[int, int, int, int], k0: int, k1: int) -> Tuple[int, int, int, int]:
    hi0, lo0 = mulhilo32(PHILOX_M0, state[0])
    hi1, lo1 = mulhilo32(PHILOX_M1, state[2])
    return (
        (hi1 ^ state[1] ^ k0) & 0xFFFFFFFF,
        lo1,
        (hi0 ^ state[3] ^ k1) & 0xFFFFFFFF,
        lo0,
    )


def philox4x32_10(ctr: Tuple[int, int, int, int], seed_lo: int, seed_hi: int) -> Tuple[int, int, int, int]:
    k0 = seed_lo & 0xFFFFFFFF
    k1 = seed_hi & 0xFFFFFFFF
    state = philox_round(ctr, k0, k1)
    for _ in range(9):
        k0 = (k0 + PHILOX_W0) & 0xFFFFFFFF
        k1 = (k1 + PHILOX_W1) & 0xFFFFFFFF
        state = philox_round(state, k0, k1)
    return state


def next_uniform(seed: int, subsequence: int, ctr_lo: int) -> float:
    seed_lo = seed & 0xFFFF_FFFF
    seed_hi = (seed >> 32) & 0xFFFF_FFFF
    ctr = (
        ctr_lo & 0xFFFFFFFF,
        0,
        subsequence & 0xFFFFFFFF,
        (subsequence >> 32) & 0xFFFFFFFF,
    )
    words = philox4x32_10(ctr, seed_lo, seed_hi)
    uniform = f32((f32(words[0] & 0xFFFFFFFF) + f32(0.5)) * f32(CURAND_2POW32_INV))
    return float(uniform)


def run_reference_sampler(
    logits: List[float],
    suppress_from: int | None,
    allow_suppressed_token: int | None,
    history: Iterable[int],
    options: Dict[str, Any],
    draws: List[float],
) -> List[Dict[str, Any]]:
    temp = float(options["temperature"])
    repetition_penalty = float(options["repetition_penalty"])
    top_k = int(options["top_k"])
    top_p = float(options["top_p"])

    candidates = filter_candidates(logits, suppress_from, allow_suppressed_token)
    if not candidates:
        return []
    candidates = sort_by_desc_score(candidates)

    if temp <= 0.0:
        idx = candidates[0][0]
        return [{"uniform_bits": None, "expected": int(idx)}]

    after_penalty_logits = apply_repetition_penalty(
        [float(x) for x in logits], history, repetition_penalty
    )
    penalized = filter_candidates(
        after_penalty_logits, suppress_from, allow_suppressed_token
    )
    if not penalized:
        return []

    penalized = sort_by_desc_score(penalized)
    apply_temperature_in_place(penalized, temp)

    if top_k > 0 and len(penalized) > top_k:
        penalized = penalized[:top_k]

    probs = to_probabilities(penalized)
    probs = apply_top_p(probs, top_p)
    # qwentts.cpp uses sorted candidates only for filtering, then performs the
    # final multinomial accumulation in original vocabulary-id order.
    probs.sort(key=lambda item: item[0])
    return [
        {
            "uniform_bits": f"0x{f32_bits(draw):08x}",
            "expected": sample_with_draw(probs, draw),
            "draw": float(draw),
        }
        for draw in draws
    ]


def bits_from_vec(values: List[float]) -> List[str]:
    return [f"0x{f32_bits(value):08x}" for value in values]


def f64(value: float) -> float:
    return float(value)


def f64_bits_from_f32(value: float) -> str:
    return f"0x{f32_bits(value):08x}"


def build_case(
    name: str,
    logits: List[float],
    options: Dict[str, Any],
    history: List[int],
    suppress_from: int | None,
    allow_suppressed_token: int | None,
    draws: List[float],
    seed: int,
) -> Dict[str, Any]:
    filtered = filter_candidates(logits, suppress_from, allow_suppressed_token)
    filtered_sorted = sort_by_desc_score(filtered.copy())
    if float(options["temperature"]) <= 0.0:
        after_penalty = list(logits)
    else:
        after_penalty = apply_repetition_penalty(
            list(logits), history, float(options["repetition_penalty"])
        )
    after_penalty_candidates = filter_candidates(
        after_penalty, suppress_from, allow_suppressed_token
    )
    after_penalty_sorted = sort_by_desc_score(after_penalty_candidates)
    after_temperature = after_penalty_sorted.copy()
    if float(options["temperature"]) > 0.0:
        apply_temperature_in_place(after_temperature, float(options["temperature"]))
    sample_draws = [next_uniform(seed, i, 0) for i in range(len(draws))]
    samples = run_reference_sampler(
        logits,
        suppress_from,
        allow_suppressed_token,
        history,
        options,
        sample_draws,
    )

    return {
        "name": name,
        "version": 1,
        "logit_bits": bits_from_vec(logits),
        "history": [int(x) for x in history],
        "suppress_from": suppress_from,
        "allow_suppressed_token": allow_suppressed_token,
        "options": {
            "temperature": float(options["temperature"]),
            "top_k": int(options["top_k"]),
            "top_p": float(options["top_p"]),
            "repetition_penalty": float(options["repetition_penalty"]),
        },
        "candidates_before_repetition": [
            {"idx": int(i), "logit_bits": f64_bits_from_f32(v)}
            for i, v in filtered_sorted
        ],
        "after_repetition_bits": [
            f64_bits_from_f32(logit) for _, logit in after_penalty_sorted
        ],
        "candidates_after_repetition": [int(i) for i, _ in after_penalty_sorted],
        "after_temperature_bits": [f64_bits_from_f32(logit) for _, logit in after_temperature],
        "candidates_after_temperature": [int(i) for i, _ in after_temperature],
        "seed": str(seed),
        "samples": samples,
    }


def build_fixture() -> Dict[str, Any]:
    case_specs = [
        (
            "positive-negative-repetition-history-unique",
            [1.0, -2.0, 0.75, float("nan"), 2.0, 0.0, -0.0],
            {
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "repetition_penalty": 1.5,
            },
            [2, 2, 6, 2, 7, 4, 4_000, 1_000],
            1024,
            None,
            [0.15, 0.95, 0.42],
            1234,
        ),
        (
            "greedy-bypass-with-signed-zero",
            [3.0, -0.0, 0.0, -1.0, 2.5],
            {
                "temperature": 0.0,
                "top_k": 3,
                "top_p": 0.2,
                "repetition_penalty": 2.0,
            },
            [0, 2, 1, 4, 2],
            None,
            None,
            [0.12],
            777,
        ),
        (
            "topk-topp-order-after-repetition",
            [0.0, 4.0, 1.5, 2.5, 3.2, -0.2],
            {
                "temperature": 0.75,
                "top_k": 3,
                "top_p": 0.8,
                "repetition_penalty": 2.0,
            },
            [1, 1, 1, 3, 20],
            None,
            None,
            [0.99, 0.15],
            2025,
        ),
        (
            "suppression-blocked-control-token",
            [1.1, 2.2, 3.3, 4.4, 5.5],
            {
                "temperature": 1.0,
                "top_k": 2,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
            },
            [4],
            3,
            2,
            [0.2],
            77,
        ),
        (
            "empty-history-no-penalty-effect",
            [2.0, 2.0, -1.2, 0.4],
            {
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "repetition_penalty": 2.5,
            },
            [],
            None,
            None,
            [0.4],
            88,
        ),
    ]

    return {
        "version": 1,
        "fixture_id": "p02-repetition-penalty-vectors",
        "source_repo": SOURCE_REPO,
        "source_revision": PINNED_COMMIT,
        "generated_by": "tools/generate_repetition_penalty_vectors.py",
        "command": PYTHON,
        "cases": [build_case(*spec) for spec in case_specs],
    }


def validate(path: Path) -> None:
    expected = build_fixture()
    if not path.exists():
        raise RuntimeError(f"FIXTURE_MISSING: {path}")
    current = json.loads(path.read_text(encoding="utf-8"))
    if current != expected:
        raise RuntimeError("FIXTURE_MISMATCH: repetition penalty vectors changed")


def write_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_fixture()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate/check repetition-penalty parity vectors for P02-T02"
    )
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
        print(f"REPETITION_PENALTY_VECTORS_CHECK_OK {args.check}")
        return 0

    write_fixture(args.output)
    print(f"REPETITION_PENALTY_VECTORS_GENERATED {args.output}")
    print(f"CASES={len(build_fixture()['cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
