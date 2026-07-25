#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, sys

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/"MANIFEST.json").read_text(encoding="utf-8"))
bad=[]
for item in manifest["files"]:
    p=root/item["path"]
    if not p.exists():
        bad.append({"path":item["path"],"error":"missing"}); continue
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    if digest!=item["sha256"]:
        bad.append({"path":item["path"],"error":"sha256","actual":digest})
print(json.dumps({"passed":not bad,"errors":bad},indent=2))
sys.exit(0 if not bad else 1)
