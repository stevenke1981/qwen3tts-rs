#!/usr/bin/env python3
"""Compare JSON token streams. Accepts a list of frames or {"frames": [...]}."""

from __future__ import annotations
import argparse, json
from pathlib import Path

def load(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj["frames"] if isinstance(obj, dict) else obj

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    a, b = load(args.reference), load(args.candidate)
    mismatches = []
    for i in range(max(len(a), len(b))):
        av = a[i] if i < len(a) else None
        bv = b[i] if i < len(b) else None
        if av != bv:
            mismatches.append({"frame":i,"reference":av,"candidate":bv})
            if len(mismatches) >= 100:
                break
    result = {
        "reference_frames":len(a),
        "candidate_frames":len(b),
        "exact":a == b,
        "mismatch_count_shown":len(mismatches),
        "mismatches":mismatches,
    }
    out = json.dumps(result, ensure_ascii=False, indent=2)
    print(out)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(out + "\n", encoding="utf-8")
    return 0 if a == b else 1

if __name__ == "__main__":
    raise SystemExit(main())
