#!/usr/bin/env python3
"""Generate offline suppression/EOS parity vectors."""
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures/alignment/p02_suppression_eos_vectors.json"
REVISION = "82cd05b9f3a175612dc89fd6943e610fab096ef5"
OFFICIAL_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"

def build():
    return {
        "version": 1,
        "fixture_id": "p02-suppression-eos-vectors",
        "path": "fixtures/alignment/p02_suppression_eos_vectors.json",
        "source_repo": "https://github.com/ServeurpersoCom/qwen3tts-rs",
        "source_revision": REVISION,
        "official_qwen_revision": OFFICIAL_REVISION,
        "qwentts_cpp_revision": REVISION,
        "generated_by": "tools/generate_suppression_eos_vectors.py",
        "command": "python tools/generate_suppression_eos_vectors.py --check fixtures/alignment/p02_suppression_eos_vectors.json",
        "cases": [
            {"vocab_size": 2048, "suppress_from": 1024, "eos": 1024, "step": 0, "allowed": list(range(1024))},
            {"vocab_size": 3072, "suppress_from": 2048, "eos": 2150, "step": 1, "allowed": list(range(2048))},
            {"vocab_size": 4096, "suppress_from": 3072, "eos": 3072, "step": 2, "allowed": list(range(3072)) + [3072]},
        ],
    }

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def validate(path):
    current = json.loads(path.read_text(encoding="utf-8"))
    expected = build()
    for key in ("version", "fixture_id", "path", "source_repo", "source_revision", "official_qwen_revision", "qwentts_cpp_revision", "generated_by", "command"):
        if current.get(key) != expected[key]:
            raise RuntimeError(f"FIXTURE_MISMATCH: {key}")
    if canonical(current.get("cases")) != canonical(expected["cases"]):
        raise RuntimeError("FIXTURE_MISMATCH: cases")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.check:
        validate(args.check)
        print(f"SUPPRESSION_EOS_CHECK_OK {args.check}")
    else:
        args.output.write_text(json.dumps(build(), indent=2), encoding="utf-8")
        print(f"SUPPRESSION_EOS_GENERATED {args.output}")

if __name__ == "__main__":
    main()
