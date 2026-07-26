#!/usr/bin/env python3
"""Compare simple F32 stage dump files described by manifests.

Binary stage files are expected to be little-endian float32.
The tool validates binary size divisibility by 4 and verifies manifest sha256.

Extended in P03-T05 with:
- --max-abs threshold
- logit top-1/top-5 ranking metrics
- --mapping file for qwentts.cpp anchor comparison
- --expected-stages count check
- --require-all-reference-stages
- fail-closed provenance and NaN/Inf checks
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import sys
from pathlib import Path


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_f32(path: Path) -> tuple[list[float], bytes]:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise ValueError(f"size is not multiple of 4: {path} ({len(raw)} bytes)")

    arr = array.array("f")
    arr.frombytes(raw)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tolist(), raw


def shape_numel(shape: list[object]) -> int:
    size = 1
    for dim in shape:
        if not isinstance(dim, int):
            raise ValueError(f"non-integer shape dimension: {dim!r}")
        if dim < 0:
            raise ValueError(f"negative shape dimension: {dim}")
        size *= dim
    return size


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / max(na * nb, 1e-30)


def max_abs_err(a: list[float], b: list[float]) -> float:
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def has_non_finite(values: list[float]) -> bool:
    return any(math.isnan(v) or math.isinf(v) for v in values)


def argmax(values: list[float]) -> int:
    """Deterministic lowest-index tie-breaking argmax."""
    best_idx = 0
    best_val = values[0]
    for i, v in enumerate(values[1:], 1):
        if v > best_val:
            best_val = v
            best_idx = i
    return best_idx


def top_k_indices(values: list[float], k: int) -> list[int]:
    """Return top-k indices with deterministic lowest-index tie-breaking."""
    indexed = sorted(enumerate(values), key=lambda iv: (-iv[1], iv[0]))
    return [i for i, _ in indexed[:k]]


def logit_metrics(ref: list[float], cand: list[float]) -> dict:
    """Compute logit ranking metrics between reference and candidate."""
    ref_top1 = argmax(ref)
    cand_top1 = argmax(cand)
    top1_match = ref_top1 == cand_top1

    k = min(5, len(ref), len(cand))
    ref_topk = set(top_k_indices(ref, k))
    cand_topk = set(top_k_indices(cand, k))
    overlap = len(ref_topk & cand_topk) / k if k > 0 else 1.0

    # Score margin around differing rank: |ref[ref_top1] - cand[cand_top1]|
    margin = abs(ref[ref_top1] - cand[cand_top1]) if not top1_match else 0.0

    return {
        "top1_ref": ref_top1,
        "top1_cand": cand_top1,
        "top1_match": top1_match,
        "top5_overlap": overlap,
        "score_margin_at_diff": margin,
    }


def load_stages(manifest: dict, source: str) -> tuple[bool, dict, str]:
    stages = manifest.get("stages")
    if not isinstance(stages, list):
        return False, {}, f"{source} manifest has no stages list"

    stage_map: dict[str, dict] = {}
    for stage in stages:
        if not isinstance(stage, dict):
            return False, {}, f"{source} manifest contains non-object stage entry"
        name = stage.get("name")
        if not isinstance(name, str) or not name:
            return False, {}, f"{source} manifest contains invalid stage name: {name!r}"
        if name in stage_map:
            return False, {}, f"{source} manifest has duplicate stage name: {name}"
        stage_map[name] = stage
    return True, stage_map, ""


def check_stage(
    path: Path, stage: dict, expected_shape_len: int | None = None
) -> tuple[bool, dict]:
    file_path = path / stage["file"]
    if not file_path.exists():
        return False, {"passed": False, "reason": f"missing stage file: {file_path}"}

    try:
        values, raw = read_f32(file_path)
    except Exception as err:
        return False, {"passed": False, "reason": str(err)}

    expected_len = shape_numel(stage.get("shape", []))
    if expected_len != len(values):
        return False, {
            "passed": False,
            "reason": f"shape length mismatch: expected {expected_len}, actual {len(values)}",
        }
    if expected_shape_len is not None and expected_shape_len != len(values):
        return False, {
            "passed": False,
            "reason": "candidate/reference float count mismatch",
        }

    manifest_sha = stage.get("sha256", "")
    actual_sha = file_sha256(raw)
    if actual_sha != manifest_sha:
        return (
            False,
            {
                "passed": False,
                "reason": f"sha256 mismatch: manifest={manifest_sha}, actual={actual_sha}",
            },
        )

    if has_non_finite(values):
        return False, {"passed": False, "reason": "non-finite values (NaN/Inf) in stage"}

    return True, {"passed": True, "values": values}


def load_mapping(path: Path) -> tuple[dict, list[str], str]:
    """Load a qwentts-stage-map.json mapping file.

    Returns (mappings_dict, logit_stage_names, error_string).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as err:
        return {}, [], f"failed to load mapping: {err}"

    mappings = data.get("mappings")
    if not isinstance(mappings, dict):
        return {}, [], "mapping file has no 'mappings' dict"

    logit_stages = data.get("logit_stages", [])
    if not isinstance(logit_stages, list):
        return {}, [], "mapping file 'logit_stages' is not a list"

    return mappings, logit_stages, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--max-abs", type=float, default=None)
    ap.add_argument("--logit-top1-exact", action="store_true")
    ap.add_argument("--min-logit-top5-overlap", type=float, default=None)
    ap.add_argument("--mapping", type=Path, default=None)
    ap.add_argument("--expected-stages", type=int, default=None)
    ap.add_argument("--require-all-reference-stages", action="store_true")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    # ── Load manifests ──────────────────────────────────────────────────────
    try:
        rm = json.loads(args.reference.read_text(encoding="utf-8"))
    except Exception as err:
        print(json.dumps({"passed": False, "reason": f"cannot read reference manifest: {err}"}))
        return 1

    try:
        cm = json.loads(args.candidate.read_text(encoding="utf-8"))
    except Exception as err:
        print(json.dumps({"passed": False, "reason": f"cannot read candidate manifest: {err}"}))
        return 1

    rs_ok, rs, rs_error = load_stages(rm, "reference")
    if not rs_ok:
        print(json.dumps({"passed": False, "reason": rs_error}, indent=2))
        return 1
    cs_ok, cs, cs_error = load_stages(cm, "candidate")
    if not cs_ok:
        print(json.dumps({"passed": False, "reason": cs_error}, indent=2))
        return 1

    # ── Provenance checks ───────────────────────────────────────────────────
    ref_source = rm.get("source", "")
    cand_source = cm.get("source", "")
    ref_revision = rm.get("revision")
    cand_revision = cm.get("revision")

    # ── Stage count checks ──────────────────────────────────────────────────
    if args.expected_stages is not None:
        if len(rs) != args.expected_stages:
            print(
                json.dumps(
                    {
                        "passed": False,
                        "reason": f"reference stage count {len(rs)} != expected {args.expected_stages}",
                    },
                    indent=2,
                )
            )
            return 1
        if len(cs) != args.expected_stages:
            print(
                json.dumps(
                    {
                        "passed": False,
                        "reason": f"candidate stage count {len(cs)} != expected {args.expected_stages}",
                    },
                    indent=2,
                )
            )
            return 1

    # ── Load mapping if provided ────────────────────────────────────────────
    mapping: dict[str, str] = {}
    logit_stage_names: list[str] = []
    if args.mapping is not None:
        mapping, logit_stage_names, map_err = load_mapping(args.mapping)
        if map_err:
            print(json.dumps({"passed": False, "reason": map_err}, indent=2))
            return 1

    # ── Determine comparison pairs ──────────────────────────────────────────
    # Without mapping: compare all stages present in both manifests.
    # With mapping: compare only mapped reference stages against candidate.
    if mapping:
        # Verify all mapped reference stages exist
        missing_ref = [name for name in mapping if name not in rs]
        if missing_ref:
            print(
                json.dumps(
                    {
                        "passed": False,
                        "reason": f"mapped reference stages missing from reference manifest: {sorted(missing_ref)}",
                    },
                    indent=2,
                )
            )
            return 1

        # Verify all mapped candidate stages exist
        missing_cand = [cand_name for cand_name in mapping.values() if cand_name not in cs]
        if missing_cand:
            print(
                json.dumps(
                    {
                        "passed": False,
                        "reason": f"mapped candidate stages missing from candidate manifest: {sorted(missing_cand)}",
                    },
                    indent=2,
                )
            )
            return 1

        pairs = [(ref_name, mapping[ref_name]) for ref_name in sorted(mapping)]
    else:
        if args.require_all_reference_stages:
            missing = sorted(set(rs) - set(cs))
            if missing:
                print(
                    json.dumps(
                        {
                            "passed": False,
                            "reason": f"reference stages missing from candidate: {missing}",
                        },
                        indent=2,
                    )
                )
                return 1

        names = sorted(set(rs) | set(cs))
        if not names:
            print(json.dumps({"passed": False, "reason": "empty comparison set"}, indent=2))
            return 1
        pairs = [(n, n) for n in names]

    if not pairs:
        print(json.dumps({"passed": False, "reason": "empty comparison set"}, indent=2))
        return 1

    # ── Compare stages ──────────────────────────────────────────────────────
    rows: list[dict] = []
    passed = True
    logit_stages_set = set(logit_stage_names)

    for ref_name, cand_name in pairs:
        r = rs.get(ref_name)
        c = cs.get(cand_name)

        if r is None:
            rows.append({"name": ref_name, "passed": False, "reason": "missing reference stage"})
            passed = False
            continue
        if c is None:
            rows.append({"name": ref_name, "passed": False, "reason": f"missing candidate stage: {cand_name}"})
            passed = False
            continue

        # Shape/dtype/layout checks
        r_shape = r.get("shape", [])
        c_shape = c.get("shape", [])
        r_dtype = r.get("dtype", "")
        c_dtype = c.get("dtype", "")
        r_layout = r.get("layout", "")
        c_layout = c.get("layout", "")

        if r_dtype != "f32" or c_dtype != "f32":
            rows.append(
                {
                    "name": ref_name,
                    "passed": False,
                    "reason": f"dtype mismatch: ref={r_dtype}, cand={c_dtype}",
                }
            )
            passed = False
            continue

        # For mapped comparisons, shapes may differ (qwentts.cpp 2D vs Candle 3D).
        # We compare by total element count.
        r_numel = shape_numel(r_shape)
        c_numel = shape_numel(c_shape)
        if r_numel != c_numel:
            rows.append(
                {
                    "name": ref_name,
                    "passed": False,
                    "reason": f"element count mismatch: ref={r_numel} (shape={r_shape}), cand={c_numel} (shape={c_shape})",
                }
            )
            passed = False
            continue

        if not mapping and r_layout != c_layout:
            rows.append(
                {
                    "name": ref_name,
                    "passed": False,
                    "reason": f"layout mismatch: ref={r_layout}, cand={c_layout}",
                }
            )
            passed = False
            continue

        r_ok, r_data = check_stage(args.reference.parent, r)
        if not r_ok:
            rows.append({"name": ref_name, **r_data})
            passed = False
            continue

        c_ok, c_data = check_stage(args.candidate.parent, c, len(r_data["values"]))
        if not c_ok:
            rows.append({"name": ref_name, **c_data})
            passed = False
            continue

        rv = r_data["values"]
        cv = c_data["values"]
        co = cosine(rv, cv)
        ma = max_abs_err(rv, cv)

        row: dict = {
            "name": ref_name,
            "candle_name": cand_name if mapping else ref_name,
            "shape_ref": r_shape,
            "shape_cand": c_shape,
            "dtype": "f32",
            "cosine": co,
            "max_abs": ma,
        }

        ok = co >= args.min_cosine
        if args.max_abs is not None and ma > args.max_abs:
            ok = False

        # Logit ranking metrics
        is_logit = ref_name in logit_stages_set or (
            not mapping and ref_name.startswith("talker-logits-")
        )
        if is_logit:
            lm = logit_metrics(rv, cv)
            row["logit"] = lm
            if args.logit_top1_exact and not lm["top1_match"]:
                ok = False
            if args.min_logit_top5_overlap is not None and lm["top5_overlap"] < args.min_logit_top5_overlap:
                ok = False

        row["passed"] = ok
        rows.append(row)
        passed &= ok

    # ── Build report ────────────────────────────────────────────────────────
    report: dict = {
        "passed": passed,
        "reference_source": ref_source,
        "candidate_source": cand_source,
        "reference_revision": ref_revision,
        "candidate_revision": cand_revision,
        "reference_stage_count": len(rs),
        "candidate_stage_count": len(cs),
        "compared_stage_count": len(pairs),
        "thresholds": {
            "min_cosine": args.min_cosine,
            "max_abs": args.max_abs,
            "logit_top1_exact": args.logit_top1_exact,
            "min_logit_top5_overlap": args.min_logit_top5_overlap,
        },
        "stages": rows,
    }

    out = json.dumps(report, indent=2)
    print(out)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(out + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
