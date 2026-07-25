#!/usr/bin/env python3
"""Compare simple F32 stage dump files described by manifests.

Binary stage files are expected to be little-endian float32.
The tool validates binary size divisibility by 4 and verifies manifest sha256.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
from pathlib import Path


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_f32(path: Path) -> tuple[list[float], bytes]:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise ValueError(f"size is not multiple of 4: {path} ({len(raw)} bytes)")

    arr = array.array("f")
    arr.frombytes(raw)
    if __import__("sys").byteorder != "little":
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


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / max(na * nb, 1e-30)


def load_stages(manifest: dict, source: str) -> tuple[bool, dict, str]:
    stages = manifest.get("stages")
    if not isinstance(stages, list):
        return False, {}, f"{source} manifest has no stages list"

    stage_map = {}
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


def check_stage(path: Path, stage: dict, expected_shape_len: int | None = None) -> tuple[bool, dict]:
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
            {"passed": False, "reason": f"sha256 mismatch: manifest={manifest_sha}, actual={actual_sha}"},
        )

    return True, {"passed": True, "values": values}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    rm = json.loads(args.reference.read_text(encoding="utf-8"))
    cm = json.loads(args.candidate.read_text(encoding="utf-8"))

    rs_ok, rs, rs_error = load_stages(rm, "reference")
    if not rs_ok:
        print(json.dumps({"passed": False, "reason": rs_error}, indent=2))
        return 1
    cs_ok, cs, cs_error = load_stages(cm, "candidate")
    if not cs_ok:
        print(json.dumps({"passed": False, "reason": cs_error}, indent=2))
        return 1
    names = sorted(set(rs) | set(cs))

    rows = []
    passed = True

    for name in names:
        if name not in rs or name not in cs:
            rows.append({"name": name, "passed": False, "reason": "missing stage"})
            passed = False
            continue

        r = rs[name]
        c = cs[name]
        if r["shape"] != c["shape"] or r["dtype"] != "f32" or c["dtype"] != "f32" or r["layout"] != c["layout"]:
            rows.append({"name": name, "passed": False, "reason": "shape/dtype/layout mismatch"})
            passed = False
            continue

        r_ok, r_data = check_stage(args.reference.parent, r)
        if not r_ok:
            rows.append({"name": name, **r_data})
            passed = False
            continue

        c_ok, c_data = check_stage(args.candidate.parent, c, len(r_data["values"]))
        if not c_ok:
            rows.append({"name": name, **c_data})
            passed = False
            continue

        rv = r_data["values"]
        cv = c_data["values"]
        co = cosine(rv, cv)
        ma = max((abs(x - y) for x, y in zip(rv, cv)), default=0.0)
        ok = co >= args.min_cosine
        rows.append({"name": name, "cosine": co, "max_abs": ma, "passed": ok})
        passed &= ok

    report = {"passed": passed, "stages": rows}
    out = json.dumps(report, indent=2)
    print(out)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(out + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
