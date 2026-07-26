#!/usr/bin/env python3
"""Verify bounded streaming latency from a JSONL trace.

Each line: {"frame": 0, "latency_ms": 1.2, "resident_bytes": 123}
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("trace",type=Path)
    ap.add_argument("--max-latency-ratio",type=float,default=1.20)
    ap.add_argument("--max-memory-growth-ratio",type=float,default=1.05)
    ap.add_argument("--report",type=Path)
    args=ap.parse_args()
    rows=[json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if len(rows)<40:
        raise SystemExit("need at least 40 frames")
    q=max(10,len(rows)//4)
    first=rows[5:q]
    last=rows[-q:]
    fmed=statistics.median(x["latency_ms"] for x in first)
    lmed=statistics.median(x["latency_ms"] for x in last)
    latency_ratio=lmed/max(fmed,1e-12)
    fmem=max(x["resident_bytes"] for x in first)
    lmem=max(x["resident_bytes"] for x in last)
    memory_ratio=lmem/max(fmem,1)
    passed=latency_ratio<=args.max_latency_ratio and memory_ratio<=args.max_memory_growth_ratio
    result={"passed":passed,"frames":len(rows),"first_median_ms":fmed,
            "last_median_ms":lmed,"latency_ratio":latency_ratio,
            "first_peak_bytes":fmem,"last_peak_bytes":lmem,"memory_ratio":memory_ratio}
    out=json.dumps(result,indent=2)
    print(out)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(out+"\n",encoding="utf-8")
    return 0 if passed else 1
if __name__=="__main__":
    raise SystemExit(main())
