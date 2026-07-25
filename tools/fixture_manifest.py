#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED_MANIFEST_VERSION = 1
REQUIRED_FIXTURE_FIELDS = {
    "id",
    "path",
    "source",
    "revision",
    "sha256",
    "license",
    "generated_by",
    "command",
    "required",
}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PROVENANCE_FIELDS = ("source", "revision", "license", "generated_by", "command")


@dataclass
class ManifestResult:
    required_count: int = 0
    verified_count: int = 0
    optional_missing_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _record_error(result: ManifestResult, prefix: str, message: str) -> None:
    result.errors.append(f"{prefix} {message}")


def _record_warning(result: ManifestResult, prefix: str, message: str) -> None:
    result.warnings.append(f"{prefix} {message}")


def validate_manifest(path: Path) -> tuple[dict, list]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"FIXTURE_INVALID invalid JSON: {exc}"]


def is_valid_hash(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return bool(SHA256_RE.fullmatch(value))


def is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def resolve_fixture_path(manifest_path: Path, raw_path: str) -> Path:
    return (manifest_path.parent / raw_path).resolve()


def describe_path(manifest_path: Path, resolved: Path) -> str:
    try:
        return str(resolved.relative_to(manifest_path.parent))
    except ValueError:
        return str(resolved)


def collect_issues(manifest_path: Path, data: dict, require_all: bool) -> ManifestResult:
    result = ManifestResult()
    if not isinstance(data, dict):
        _record_error(result, "FIXTURE_INVALID", "manifest root is not a JSON object")
        return result

    if data.get("version") != REQUIRED_MANIFEST_VERSION:
        _record_error(
            result,
            "FIXTURE_INVALID",
            f"unsupported manifest version: {data.get('version')}",
        )
        return result

    fixtures = data.get("fixtures")
    if not isinstance(fixtures, list):
        _record_error(result, "FIXTURE_INVALID", "fixtures must be an array")
        return result

    ids_seen: set[str] = set()
    for item in fixtures:
        if not isinstance(item, dict):
            _record_error(result, "FIXTURE_INVALID", "fixture entry is not an object")
            continue

        missing = REQUIRED_FIXTURE_FIELDS - item.keys()
        if missing:
            _record_error(
                result,
                "FIXTURE_INVALID",
                f"fixture missing field(s): {','.join(sorted(missing))}",
            )
            continue

        fixture_id = item["id"]
        required = item.get("required")
        path_field = item.get("path")
        file_hash = item.get("sha256")
        if not isinstance(fixture_id, str) or not fixture_id:
            _record_error(result, "FIXTURE_INVALID", "fixture id must be a non-empty string")
            continue
        if fixture_id in ids_seen:
            _record_error(result, "FIXTURE_INVALID", f"duplicate fixture id: {fixture_id}")
            continue
        ids_seen.add(fixture_id)

        if required:
            result.required_count += 1
        if not isinstance(required, bool):
            _record_error(result, "FIXTURE_INVALID", f"{fixture_id}: required must be true/false")
            continue
        invalid = False
        for field in PROVENANCE_FIELDS:
            if not is_non_empty_string(item.get(field)):
                _record_error(
                    result,
                    "FIXTURE_INVALID",
                    f"{fixture_id}: {field} must be a non-empty string",
                )
                invalid = True
                break
        if invalid:
            continue
        if not isinstance(path_field, str) or not path_field:
            _record_error(result, "FIXTURE_INVALID", f"{fixture_id}: path must be a non-empty string")
            continue
        if not is_valid_hash(file_hash):
            _record_error(result, "FIXTURE_INVALID", f"{fixture_id}: malformed sha256")
            continue

        resolved = resolve_fixture_path(manifest_path, path_field)
        if not resolved.exists():
            if required:
                _record_error(
                    result,
                    "FIXTURE_MISSING",
                    f"{fixture_id}: {describe_path(manifest_path, resolved)}",
                )
            else:
                _record_warning(
                    result,
                    "FIXTURE_OPTIONAL_MISSING",
                    f"{fixture_id}: {describe_path(manifest_path, resolved)}",
                )
                result.optional_missing_count += 1
            continue
        if not resolved.is_file():
            _record_error(
                result,
                "FIXTURE_INVALID",
                f"{fixture_id}: {describe_path(manifest_path, resolved)} is not a regular file",
            )
            continue
        actual = sha256(resolved)
        if actual != file_hash.lower():
            _record_error(result, "FIXTURE_HASH_MISMATCH", f"{fixture_id}: {resolved.name}")
            continue

        result.verified_count += 1

    if require_all and result.required_count == 0:
        _record_error(result, "FIXTURE_MISSING", "manifest has no required fixtures")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--require-all", action="store_true")
    args = ap.parse_args()

    data, json_errors = validate_manifest(args.manifest)
    if json_errors:
        for item in json_errors:
            print(item)
        return 1

    result = collect_issues(args.manifest, data, args.require_all)
    if result.errors:
        for item in result.errors:
            print(item)
        return 1

    for item in result.warnings:
        print(item)

    print(
        "FIXTURE_VERIFIED "
        f"required={result.required_count} "
        f"verified={result.verified_count} "
        f"optional_missing={result.optional_missing_count}"
    )
    return 0


if __name__=="__main__":
    raise SystemExit(main())
