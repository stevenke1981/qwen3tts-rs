#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("manifest",type=Path)
    ap.add_argument("--require-all",action="store_true")
    args=ap.parse_args()
    data=json.loads(args.manifest.read_text(encoding="utf-8"))
    errors=[]
    for item in data.get("fixtures",[]):
        p=Path(item["path"])
        if not p.exists():
            errors.append(f"missing: {p}"); continue
        expected=item.get("sha256")
        if expected and sha256(p)!=expected:
            errors.append(f"sha256 mismatch: {p}")
    if args.require_all and not data.get("fixtures"):
        errors.append("manifest contains no fixtures")
    for e in errors: print(e)
    return 1 if errors else 0
if __name__=="__main__":
    raise SystemExit(main())
