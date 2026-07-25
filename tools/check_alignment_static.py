#!/usr/bin/env python3
"""Static guard for known false-parity patterns.

This is not a proof. It blocks obvious regressions and forces manual review.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

PATTERNS = [
    ("accumulated_history", re.compile(r"pre_conv_buffer|accumulat(?:e|ed).*history", re.I)),
    ("full_dequantize", re.compile(r"dequantize_to_vec|load_quantized_f32", re.I)),
    ("permissive_ci", re.compile(r"continue-on-error\s*:\s*true", re.I)),
    ("silent_fixture_skip", re.compile(r"Skipping:\s*real weights|return;\s*//.*weights", re.I)),
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("repo",type=Path)
    ap.add_argument("--report",type=Path)
    args=ap.parse_args()
    hits=[]
    for p in args.repo.rglob("*"):
        if not p.is_file() or any(x in p.parts for x in [".git","target","artifacts"]):
            continue
        if p.suffix.lower() not in {".rs",".py",".yml",".yaml",".toml",".md"}:
            continue
        try: text=p.read_text(encoding="utf-8")
        except UnicodeDecodeError: continue
        for name,rx in PATTERNS:
            for m in rx.finditer(text):
                line=text.count("\n",0,m.start())+1
                hits.append({"rule":name,"file":str(p.relative_to(args.repo)),"line":line,
                             "snippet":text[m.start():m.start()+120].splitlines()[0]})
    result={"hits":hits,"note":"Hits require review; P05/P08/P13 production hits should be removed or justified."}
    out=json.dumps(result,indent=2)
    print(out)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(out+"\n",encoding="utf-8")
    return 0
if __name__=="__main__":
    raise SystemExit(main())
