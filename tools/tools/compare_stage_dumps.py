#!/usr/bin/env python3
"""Compare simple F32 stage dump files described by manifests.

Binary stage files are expected to be little-endian float32. This tool intentionally
fails on missing stages or shape differences.
"""
from __future__ import annotations
import argparse, array, json, math
from pathlib import Path

def read_f32(path: Path) -> list[float]:
    arr = array.array("f")
    with path.open("rb") as f:
        arr.fromfile(f, path.stat().st_size // 4)
    if __import__("sys").byteorder != "little":
        arr.byteswap()
    return arr.tolist()

def cosine(a,b):
    dot=sum(x*y for x,y in zip(a,b))
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return dot/max(na*nb,1e-30)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("reference",type=Path)
    ap.add_argument("candidate",type=Path)
    ap.add_argument("--min-cosine",type=float,default=0.999)
    ap.add_argument("--report",type=Path)
    args=ap.parse_args()
    rm=json.loads(args.reference.read_text(encoding="utf-8"))
    cm=json.loads(args.candidate.read_text(encoding="utf-8"))
    rs={x["name"]:x for x in rm["stages"]}; cs={x["name"]:x for x in cm["stages"]}
    names=sorted(set(rs)|set(cs))
    rows=[]; passed=True
    for name in names:
        if name not in rs or name not in cs:
            rows.append({"name":name,"passed":False,"reason":"missing stage"})
            passed=False; continue
        r,c=rs[name],cs[name]
        if r["shape"]!=c["shape"] or r["dtype"]!="f32" or c["dtype"]!="f32":
            rows.append({"name":name,"passed":False,"reason":"shape/dtype mismatch"})
            passed=False; continue
        rv=read_f32(args.reference.parent/r["file"])
        cv=read_f32(args.candidate.parent/c["file"])
        if len(rv)!=len(cv):
            rows.append({"name":name,"passed":False,"reason":"length mismatch"})
            passed=False; continue
        co=cosine(rv,cv)
        ma=max((abs(x-y) for x,y in zip(rv,cv)),default=0.0)
        ok=co>=args.min_cosine
        rows.append({"name":name,"cosine":co,"max_abs":ma,"passed":ok})
        passed &= ok
    report={"passed":passed,"stages":rows}
    out=json.dumps(report,indent=2)
    print(out)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(out+"\n",encoding="utf-8")
    return 0 if passed else 1
if __name__=="__main__":
    raise SystemExit(main())
