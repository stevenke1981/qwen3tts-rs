#!/usr/bin/env bash
set -euo pipefail
phase="${1:?usage: run-gates.sh P00 [repo]}"
repo="${2:-.}"
cd "$repo"
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
python3 tools/check_alignment_static.py . --report "artifacts/alignment/${phase}/static-report.json"
echo "${phase} base gate passed; run phase-specific commands from its task card."
