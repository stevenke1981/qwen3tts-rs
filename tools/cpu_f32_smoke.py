#!/usr/bin/env python3
"""Build deterministic CPU F32 baseline reports from reference manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
REPORT_BACKEND = "CPU/F32"
RUN_KINDS = ("official-python", "qwentts.cpp", "qwentts-cpp")
RUN_KIND_CANONICAL = {"qwentts-cpp": "qwentts.cpp"}
INPUT_ARTIFACT_KINDS = ("prompt", "stdout", "tensor", "audio", "token", "stderr")
KIND_CANONICAL = {"stdout": "token"}
REQUIRED_REPORT_KINDS = ("prompt", "token", "tensor", "audio")


class BaselineError(SystemExit):
    def __init__(self, prefix: str, message: str, code: int = 1) -> None:
        super().__init__(code)
        self.prefix = prefix
        self.message = message


def _error(prefix: str, message: str) -> None:
    raise BaselineError(prefix, message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _safe_relative_path(value: str, *, prefix: str, empty_ok: bool = False) -> Path:
    if value != value.strip():
        _error(prefix, "path contains leading/trailing whitespace")
    if value == "" and not empty_ok:
        _error(prefix, "path cannot be empty")
    if value == "":
        return Path(".")

    if os.path.isabs(value):
        _error(prefix, f"absolute path is not allowed: {value}")
    parts = [segment for segment in re.split(r"[\\\\/]", value) if segment != ""]
    if not parts:
        _error(prefix, f"invalid path: {value}")
    if any(part in (".", "..") for part in parts):
        _error(prefix, f"path traversal is not allowed: {value}")
    if any(re.search(r"[<>:\"|?*]", part) for part in parts):
        _error(prefix, f"unsafe path characters in: {value}")

    return Path(*parts)


def _relative_to_repo_root(root: Path, value: str, *, prefix: str) -> Path:
    rel = _safe_relative_path(value, prefix=prefix)
    path = root / rel
    try:
        common = os.path.commonpath([str(path.resolve()), str(root.resolve())])
    except ValueError:
        _error(prefix, f"path leaves repository root: {value}")
    if common != str(root.resolve()):
        _error(prefix, f"path leaves repository root: {value}")
    return path


def _ensure_file_is_within(
    path: Path,
    *,
    base: Path,
    label: str,
    ensure_file: bool = True,
    symlink_guard: bool = True,
) -> None:
    try:
        common = os.path.commonpath([str(path.resolve()), str(base.resolve())])
    except ValueError:
        _error(label, f"path escapes repository root: {path}")
    if common != str(base.resolve()):
        _error(label, f"path escapes repository root: {path}")

    if ensure_file and not path.exists():
        _error(label, f"missing artifact file: {path}")
    if ensure_file and not path.is_file():
        _error(label, f"artifact path is not a file: {path}")
    if path.is_symlink():
        _error(label, f"symlink path is not allowed: {path}")

    if symlink_guard:
        current = path.parent
        root = base
        while current != root.parent:
            if current.is_symlink():
                _error(label, f"symlink escape detected: {current}")
            if current == root:
                break
            current = current.parent


def _read_json(path: Path, *, prefix: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        _error(prefix, f"failed to read json: {path}: {exc}")
    except json.JSONDecodeError as exc:
        _error(prefix, f"invalid json: {path}: {exc}")


def _normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n")


@dataclass(frozen=True)
class Case:
    case_id: str
    language: str
    text_sha256: str


def _load_corpus(repo_root: Path, path_arg: str, *, required_cases: list[str] | None) -> tuple[Path, str, dict[str, Case]]:
    corpus_path = _relative_to_repo_root(repo_root, path_arg, prefix="BASELINE_CONFIG")
    if corpus_path.exists() and corpus_path.is_symlink():
        _error("BASELINE_CONFIG", f"corpus file path is a symlink: {path_arg}")
    if not corpus_path.exists():
        _error("BASELINE_CONFIG", f"corpus file not found: {path_arg}")
    if not corpus_path.is_file():
        _error("BASELINE_CONFIG", f"corpus path is not a file: {path_arg}")

    corpus_bytes = corpus_path.read_bytes()
    corpus_sha = _sha256(corpus_bytes)
    raw = _read_json(corpus_path, prefix="BASELINE_INPUT")

    cases_payload = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(cases_payload, list):
        _error("BASELINE_INPUT", "corpus must contain a list at `cases`")

    cases: list[Case] = []
    for idx, item in enumerate(cases_payload):
        if not isinstance(item, dict):
            _error("BASELINE_INPUT", f"malformed corpus item at index {idx}")
        case_id = item.get("id")
        language = item.get("language")
        if not isinstance(case_id, str) or not case_id.strip():
            _error("BASELINE_INPUT", f"corpus case id must be a non-empty string at index {idx}")
        if not isinstance(language, str) or not language.strip():
            _error("BASELINE_INPUT", f"corpus case {case_id} missing language")

        has_text = "text" in item
        has_text_file = "text_file" in item
        if has_text == has_text_file:
            _error("BASELINE_INPUT", f"case {case_id} must define exactly one of `text` or `text_file`")

        if has_text:
            text = item.get("text")
            if not isinstance(text, str):
                _error("BASELINE_INPUT", f"case {case_id} text must be a UTF-8 string")
            normalized = _normalize_text(text)
            if normalized == "":
                _error("BASELINE_INPUT", f"case {case_id} text is empty")
        else:
            text_file = item.get("text_file")
            if not isinstance(text_file, str) or not text_file.strip():
                _error("BASELINE_INPUT", f"case {case_id} text_file must be a non-empty string")
            text_file_path = _relative_to_repo_root(
                repo_root, text_file, prefix="BASELINE_INPUT"
            )
            if text_file_path.exists() and text_file_path.is_symlink():
                _error("BASELINE_INPUT", f"case {case_id} text_file is a symlink: {text_file}")
            if not text_file_path.exists():
                _error("BASELINE_INPUT", f"case {case_id} text_file not found: {text_file}")
            if not text_file_path.is_file():
                _error("BASELINE_INPUT", f"case {case_id} text_file is not a file: {text_file}")
            _ensure_file_is_within(
                text_file_path,
                base=repo_root,
                label="BASELINE_INPUT",
                symlink_guard=True,
            )
            normalized = _normalize_text(text_file_path.read_text(encoding="utf-8"))
            if normalized == "":
                _error("BASELINE_INPUT", f"case {case_id} text_file is empty: {text_file}")

        cases.append(Case(case_id=case_id, language=language, text_sha256=_sha256(normalized.encode("utf-8"))))

    case_map: dict[str, Case] = {}
    for case in cases:
        if case.case_id in case_map:
            _error("BASELINE_INPUT", f"duplicate case id in corpus: {case.case_id}")
        case_map[case.case_id] = case

    selected_case_ids = required_cases if required_cases else [case.case_id for case in cases]
    for case_id in selected_case_ids:
        if case_id not in case_map:
            _error("BASELINE_INPUT", f"required case not found in corpus: {case_id}")

    return corpus_path, corpus_sha, case_map


def _validate_hash(value: Any) -> str:
    if not isinstance(value, str):
        _error("BASELINE_INPUT", "artifact.sha256 must be a hex string")
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        _error("BASELINE_INPUT", f"malformed artifact.sha256: {value}")
    return value


def _validate_revision(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        _error("BASELINE_INPUT", f"{field} must be a string")
    value = value.strip()
    if not value:
        _error("BASELINE_INPUT", f"{field} cannot be empty")
    return value


def _load_manifest(
    path: Path,
    repo_root: Path,
    case_map: dict[str, Case],
) -> dict[str, Any]:
    data = _read_json(path, prefix="BASELINE_INPUT")
    if not isinstance(data, dict):
        _error("BASELINE_INPUT", f"reference run is not an object: {path}")

    required = ("schema_version", "kind", "source", "revision", "model", "case_id", "seed", "argv", "exit_status", "artifacts", "stages")
    missing = [key for key in required if key not in data]
    if missing:
        _error("BASELINE_INPUT", f"reference run missing fields {missing}: {path}")

    if data["schema_version"] != SCHEMA_VERSION:
        _error("BASELINE_INPUT", f"unsupported schema_version: {path}")

    kind = data["kind"]
    source = data["source"]
    if kind not in RUN_KINDS:
        _error("BASELINE_INPUT", f"unsupported run kind: {kind}")
    if source not in RUN_KINDS:
        _error("BASELINE_INPUT", f"unsupported source kind: {source}")
    canonical_kind = RUN_KIND_CANONICAL.get(kind, kind)
    canonical_source = RUN_KIND_CANONICAL.get(source, source)
    if canonical_kind != canonical_source:
        _error("BASELINE_INPUT", f"kind/source mismatch in manifest: {path}")

    revision = _validate_revision(data["revision"], field="run revision")
    model = data["model"]
    if not isinstance(model, str) or not model.strip():
        _error("BASELINE_INPUT", f"run model is empty: {path}")
    case_id = data["case_id"]
    if not isinstance(case_id, str) or not case_id.strip():
        _error("BASELINE_INPUT", f"run case_id is empty: {path}")
    if case_id not in case_map:
        _error("BASELINE_INPUT", f"run case_id not found in corpus: {case_id}")
    if data["exit_status"] != 0:
        _error("BASELINE_INPUT", f"non-zero reference run status for {case_id}: {data['exit_status']}")

    if data["argv"] is not None and not isinstance(data["argv"], list):
        _error("BASELINE_INPUT", f"manifest argv must be array: {path}")

    raw_artifacts = data["artifacts"]
    if not isinstance(raw_artifacts, list):
        _error("BASELINE_INPUT", f"manifest artifacts must be list: {path}")
    if not raw_artifacts:
        _error("BASELINE_INPUT", f"manifest has no artifacts: {path}")

    artifact_base = path.parent
    seen_paths: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    kinds: set[str] = set()
    for item in raw_artifacts:
        if not isinstance(item, dict):
            _error("BASELINE_INPUT", f"artifact entry must be an object: {path}")
        if (
            not isinstance(item.get("kind"), str)
            or not isinstance(item.get("relative_path"), str)
            or not isinstance(item.get("byte_length"), int)
            or "sha256" not in item
        ):
            _error("BASELINE_INPUT", f"malformed artifact entry: {path}")
        if item["byte_length"] <= 0:
            _error(
                "BASELINE_INPUT",
                f"artifact byte_length must be positive: {item['relative_path']} in {path}",
            )
        raw_kind = item["kind"]
        if raw_kind not in INPUT_ARTIFACT_KINDS:
            _error("BASELINE_INPUT", f"unsupported artifact kind: {raw_kind}")
        kind = KIND_CANONICAL.get(raw_kind, raw_kind)
        rel = _safe_relative_path(
            item["relative_path"], prefix="BASELINE_INPUT"
        )
        artifact_path = artifact_base / rel
        _ensure_file_is_within(
            artifact_path,
            base=artifact_base,
            label="BASELINE_INPUT",
            ensure_file=True,
            symlink_guard=True,
        )
        artifact_sha = _validate_hash(item["sha256"])
        actual_size = artifact_path.stat().st_size
        if actual_size != item["byte_length"]:
            _error(
                "BASELINE_INPUT",
                f"artifact size mismatch for {item['relative_path']} in {path}",
            )
        actual_sha = _sha256_file(artifact_path)
        if actual_sha != artifact_sha:
            _error(
                "BASELINE_INPUT",
                f"artifact sha mismatch for {item['relative_path']} in {path}",
            )
        artifact_record = {
            "kind": kind,
            "relative_path": rel.as_posix(),
            "byte_length": item["byte_length"],
            "sha256": actual_sha,
        }
        if kind == "tensor":
            shape = item.get("shape")
            dtype = item.get("dtype")
            layout = item.get("layout")
            if shape is not None:
                if not isinstance(shape, list) or any(
                    not isinstance(dim, int) or dim < 0 for dim in shape
                ):
                    _error("BASELINE_INPUT", f"malformed tensor shape: {item['relative_path']}")
                artifact_record["shape"] = shape
            if dtype is not None:
                artifact_record["dtype"] = str(dtype)
            if layout is not None:
                artifact_record["layout"] = str(layout)

        rel_key = artifact_record["relative_path"]
        if rel_key in seen_paths:
            _error("BASELINE_INPUT", f"duplicate artifact path in manifest: {rel_key}")
        seen_paths.add(rel_key)
        kinds.add(kind)
        artifacts.append(artifact_record)

    return {
        "adapter": canonical_kind,
        "case": case_id,
        "model": model,
        "seed": data["seed"],
        "source_revision": revision,
        "source_manifest_path": str(path.relative_to(repo_root).as_posix()),
        "source_manifest_sha256": _sha256(path.read_bytes()),
        "artifacts": sorted(artifacts, key=lambda item: item["relative_path"]),
        "present_kinds": kinds,
    }


def _normalize_requirements(items: list[str] | None, *, prefix: str, allowed: tuple[str, ...]) -> list[str]:
    if not items:
        return []
    seen: set[str] = set()
    output: list[str] = []
    for value in items:
        if value not in allowed:
            _error(prefix, f"unsupported value: {value}")
        value = RUN_KIND_CANONICAL.get(value, value)
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _build_coverage(
    required_cases: list[str],
    required_adapters: list[str],
    required_kinds: list[str],
    runs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    run_index = {(run["adapter"], run["case"]): run["artifacts"] for run in runs}
    failures: list[str] = []
    coverage: dict[str, Any] = {}
    total_cases = len(required_cases)

    case_kinds: dict[str, set[str]] = {case_id: set() for case_id in required_cases}
    for case_id in required_cases:
        for adapter in required_adapters:
            artifacts = run_index.get((adapter, case_id))
            if not artifacts:
                failures.append(f"missing run for adapter={adapter}, case={case_id}")
                continue
            for artifact in artifacts:
                if artifact["byte_length"] > 0:
                    case_kinds[case_id].add(artifact["kind"])

    for kind in required_kinds:
        present = 0
        for case_id in required_cases:
            if kind in case_kinds[case_id]:
                present += 1
        missing = total_cases - present
        if missing:
            coverage[kind] = {
                "required": total_cases,
                "present": present,
                "missing": missing,
                "status": "FAIL",
            }
            failures.append(f"missing required kind={kind} for {missing} case(s)")
        else:
            coverage[kind] = {
                "required": total_cases,
                "present": present,
                "missing": 0,
                "status": "PASS",
            }

    return coverage, failures


def _build_report(
    *,
    repo_root: Path,
    corpus_path: Path,
    corpus_sha: str,
    corpus_cases: dict[str, Case],
    required_cases: list[str],
    required_adapters: list[str],
    required_kinds: list[str],
    runs: list[dict[str, Any]],
    target_revision: str,
    reference_revision: str,
    official_revision: str,
) -> tuple[dict[str, Any], list[str]]:
    normalized_runs = []
    for run in runs:
        normalized_runs.append(
            {
                "adapter": run["adapter"],
                "case": run["case"],
                "model": run["model"],
                "seed": run["seed"],
                "source_revision": run["source_revision"],
                "source_manifest_path": run["source_manifest_path"],
                "source_manifest_sha256": run["source_manifest_sha256"],
                "artifacts": run["artifacts"],
            }
        )

    normalized_runs.sort(key=lambda item: (item["adapter"], item["case"]))
    normalized_runs = [
        {
            "adapter": run["adapter"],
            "case": run["case"],
            "model": run["model"],
            "seed": run["seed"],
            "source_revision": run["source_revision"],
            "source_manifest_path": run["source_manifest_path"],
            "source_manifest_sha256": run["source_manifest_sha256"],
            "artifacts": run["artifacts"],
        }
        for run in normalized_runs
    ]

    selected_case_entries = [
        {
            "id": corpus_cases[case_id].case_id,
            "language": corpus_cases[case_id].language,
            "text_sha256": corpus_cases[case_id].text_sha256,
        }
        for case_id in required_cases
    ]

    coverage, failures = _build_coverage(required_cases, required_adapters, required_kinds, normalized_runs)
    status = "PASS" if not failures and all(
        entry["status"] == "PASS" for entry in coverage.values()
    ) else "FAIL"

    report = {
        "schema_version": SCHEMA_VERSION,
        "backend": REPORT_BACKEND,
        "target_revision": target_revision,
        "reference_revision": reference_revision,
        "official_revision": official_revision,
        "corpus": {
            "path": str(corpus_path.relative_to(repo_root).as_posix()),
            "sha256": corpus_sha,
            "cases": selected_case_entries,
        },
        "runs": [
            {
                "adapter": run["adapter"],
                "case": run["case"],
                "model": run["model"],
                "seed": run["seed"],
                "source_revision": run["source_revision"],
                "source_manifest_path": run["source_manifest_path"],
                "source_manifest_sha256": run["source_manifest_sha256"],
                "artifacts": run["artifacts"],
            }
            for run in normalized_runs
        ],
        "coverage": coverage,
        "required": {
            "adapters": sorted(required_adapters),
            "cases": required_cases,
            "kinds": required_kinds,
        },
        "status": status,
    }

    return report, failures


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cpu_f32_smoke.py")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--reference-revision", required=True)
    parser.add_argument("--official-revision", required=True)
    parser.add_argument("--require-adapter", action="append", default=[])
    parser.add_argument("--require-case", action="append", default=[])
    parser.add_argument("--require-kind", action="append", default=[])
    parser.add_argument("--run", action="append", required=True, dest="runs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    required_adapters = _normalize_requirements(args.require_adapter, prefix="BASELINE_CONFIG", allowed=RUN_KINDS)
    required_kinds = _normalize_requirements(
        args.require_kind,
        prefix="BASELINE_CONFIG",
        allowed=REQUIRED_REPORT_KINDS,
    )
    required_cases = args.require_case

    if args.repo_root != args.repo_root.strip():
        _error("BASELINE_CONFIG", "repo-root path contains leading/trailing whitespace")
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        _error("BASELINE_CONFIG", f"repo-root does not exist: {args.repo_root}")
    if repo_root.is_file():
        _error("BASELINE_CONFIG", f"repo-root is not a directory: {args.repo_root}")

    target_revision = _validate_revision(args.target_revision, field="target revision")
    reference_revision = _validate_revision(args.reference_revision, field="reference revision")
    official_revision = _validate_revision(args.official_revision, field="official revision")

    corpus_path, corpus_sha, corpus_cases = _load_corpus(
        repo_root,
        args.corpus,
        required_cases=required_cases,
    )

    run_paths: list[Path] = []
    if not args.runs:
        _error("BASELINE_CONFIG", "at least one --run is required")
    for run_arg in args.runs:
        run_path = _relative_to_repo_root(repo_root, run_arg, prefix="BASELINE_INPUT")
        if run_path.is_symlink():
            _error("BASELINE_INPUT", f"run manifest path is a symlink: {run_arg}")
        if not run_path.exists():
            _error("BASELINE_INPUT", f"run manifest not found: {run_arg}")
        if not run_path.is_file():
            _error("BASELINE_INPUT", f"run manifest path is not a file: {run_arg}")
        _ensure_file_is_within(run_path, base=repo_root, label="BASELINE_INPUT")
        run_paths.append(run_path)

    manifests = [_load_manifest(path, repo_root, case_map=corpus_cases) for path in run_paths]
    if not manifests:
        _error("BASELINE_INPUT", "no reference runs available")

    seen: set[tuple[str, str]] = set()
    for item in manifests:
        key = (item["adapter"], item["case"])
        if key in seen:
            _error("BASELINE_INPUT", f"duplicate adapter/case identity: {key}")
        seen.add(key)

    if not required_adapters:
        required_adapters = sorted({item["adapter"] for item in manifests})
    if not required_cases:
        required_cases = sorted(corpus_cases.keys())

    selected_runs = [item for item in manifests if item["case"] in required_cases and item["adapter"] in required_adapters]

    report, failures = _build_report(
        repo_root=repo_root,
        corpus_path=corpus_path,
        corpus_sha=corpus_sha,
        corpus_cases=corpus_cases,
        required_cases=sorted(required_cases),
        required_adapters=sorted(required_adapters),
        required_kinds=required_kinds,
        runs=selected_runs,
        target_revision=target_revision,
        reference_revision=reference_revision,
        official_revision=official_revision,
    )

    output_path = _relative_to_repo_root(repo_root, args.output, prefix="BASELINE_OUTPUT")
    if output_path.exists() and not args.dry_run:
        _error("BASELINE_OUTPUT", f"output already exists: {args.output}")
    if output_path.is_dir():
        _error("BASELINE_OUTPUT", f"output must be a file path: {args.output}")
    if output_path.exists() and args.dry_run:
        _error("BASELINE_OUTPUT", f"dry-run would overwrite existing output: {args.output}")

    payload = json.dumps(report, sort_keys=True, indent=2)
    if args.dry_run:
        print(payload)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8", newline="\n")

    if report["status"] != "PASS":
        if failures:
            if not args.dry_run and not output_path.exists():
                # Defensive branch: maintain fail-closed behavior even if filesystem race deletes output.
                _error("BASELINE_OUTPUT", f"could not create output: {args.output}")
            _error("BASELINE_COVERAGE", "required adapter/case/kind coverage missing")
    return 0


def main() -> None:
    try:
        raise SystemExit(_run())
    except BaselineError as exc:
        print(f"{exc.prefix} {exc.message}", file=sys.stderr)
        raise SystemExit(exc.code)


if __name__ == "__main__":
    main()
