#!/usr/bin/env python3
"""Generate and validate sampling-config fixture for P02-T03.

The generator now ties the matrix to public-model provenance from
`p01_prompt_id_matrix.json`, resolves `generation_config.json` from immutable
model revisions, and records SHA256 + resolved talker/subtalker values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
PYTHON = "python tools/generate_sampling_config_matrix.py"
SOURCE_REPO = "https://github.com/ServeurpersoCom/qwen3tts-rs"
SOURCE_REVISION = "82cd05b9f3a175612dc89fd6943e610fab096ef5"
FIXTURE_PATH = ROOT / "fixtures" / "alignment" / "p02_sampling_config_matrix.json"
P01_FIXTURE_PATH = ROOT / "fixtures" / "alignment" / "p01_prompt_id_matrix.json"


class PublicModelEntry(TypedDict):
    model_id: str
    model_revision: str
    config_sha256: str


class PromptIdMatrix(TypedDict):
    models: List[PublicModelEntry]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def hf_cache_root() -> Path:
    if (cache := os.environ.get("HUGGINGFACE_HUB_CACHE")) is not None:
        return Path(cache)
    if (cache := os.environ.get("HF_HOME")) is not None:
        return Path(cache) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_snapshot_path(repo_id: str, revision: str) -> Path:
    return (
        hf_cache_root()
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )


def resolve_snapshot(repo_id: str, revision: str, allow_patterns: List[str]) -> Path:
    path = model_snapshot_path(repo_id, revision)
    if path.exists() and all((path / pattern).exists() for pattern in allow_patterns):
        return path

    downloaded = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(hf_cache_root()),
        allow_patterns=allow_patterns,
    )
    candidate = Path(downloaded) / "snapshots" / revision
    if candidate.exists():
        return candidate
    return Path(downloaded)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_generation_config_snapshot(repo_id: str, revision: str) -> Path:
    path = resolve_snapshot(repo_id, revision, ["generation_config.json"])
    cfg_path = path / "generation_config.json"
    if not cfg_path.exists():
        raise RuntimeError(
            f"GENCFG_MISSING: no generation_config.json in {repo_id}@{revision}"
        )
    return cfg_path


def _validate_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"GENCFG_BAD_TYPE: {name} must be finite number, got {type(value).__name__}")
    if not float(value) == float(value) or value in (float("inf"), float("-inf")):
        raise ValueError(f"GENCFG_INVALID: {name} must be finite")
    return float(value)


def _validate_probability(value: Any, name: str) -> float:
    value = _validate_finite_number(value, name)
    if not (0.0 < value <= 1.0):
        raise ValueError(f"GENCFG_RANGE: {name} must be in (0.0, 1.0]")
    return value


def _validate_top_k(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"GENCFG_BAD_TYPE: {name} must be integer")
    as_int = int(value)
    if float(as_int) != float(value) or as_int < 0:
        raise ValueError(f"GENCFG_RANGE: {name} must be non-negative integer")
    return as_int


def _validate_do_sample(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"GENCFG_BAD_TYPE: {name} must be boolean")
    return value


def resolve_sampling_config(raw: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], bool, Dict[str, Any]]:
    talker_do_sample = _validate_do_sample(
        raw.get("do_sample", True), "generation_config.talker.do_sample"
    )
    subtalker_do_sample = _validate_do_sample(
        raw.get("subtalker_dosample", True),
        "generation_config.subtalker.dosample",
    )

    talker = {
        "temperature": _validate_finite_number(
            raw.get("temperature", 0.9), "generation_config.talker.temperature"
        ),
        "top_k": _validate_top_k(
            raw.get("top_k", 50), "generation_config.talker.top_k"
        ),
        "top_p": _validate_probability(
            raw.get("top_p", 1.0), "generation_config.talker.top_p"
        ),
        "repetition_penalty": _validate_finite_number(
            raw.get("repetition_penalty", 1.05),
            "generation_config.talker.repetition_penalty",
        ),
    }
    subtalker = {
        "temperature": _validate_finite_number(
            raw.get("subtalker_temperature", 0.9),
            "generation_config.subtalker.temperature",
        ),
        "top_k": _validate_top_k(
            raw.get("subtalker_top_k", 50),
            "generation_config.subtalker.top_k",
        ),
        "top_p": _validate_probability(
            raw.get("subtalker_top_p", 1.0),
            "generation_config.subtalker.top_p",
        ),
        "repetition_penalty": 1.0,
    }

    if talker["repetition_penalty"] <= 0.0:
        raise ValueError(
            "GENCFG_RANGE: generation_config.talker.repetition_penalty must be > 0.0"
        )
    if subtalker["repetition_penalty"] <= 0.0:
        raise ValueError(
            "GENCFG_RANGE: generation_config.subtalker.repetition_penalty must be > 0.0"
        )
    if not float(talker["temperature"]) > 0.0 and talker_do_sample:
        raise ValueError(
            "GENCFG_RANGE: generation_config.talker.temperature must be > 0.0 when do_sample=true"
        )
    if not float(subtalker["temperature"]) > 0.0 and subtalker_do_sample:
        raise ValueError(
            "GENCFG_RANGE: generation_config.subtalker.temperature must be > 0.0 when do_sample=true"
        )

    return (
        talker_do_sample,
        talker,
        subtalker_do_sample,
        subtalker,
    )


def build_case(
    name: str,
    generation_config: Dict[str, Any],
    synthesis_options: Dict[str, Any],
    expected: Dict[str, Any] | None = None,
    expect_error: bool = False,
) -> Dict[str, Any]:
    case: Dict[str, Any] = {
        "name": name,
        "generation_config": generation_config,
        "synthesis_options": synthesis_options,
    }
    if expect_error:
        case["expect_error"] = True
    else:
        if expected is None:
            raise ValueError("expected must be provided when expect_error is false")
        case["expected"] = expected
    return case


def expected_case(
    route: str,
    talker: Dict[str, Any],
    subtalker: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "route": route,
        "talker": talker,
        "subtalker": subtalker,
    }


def build_cases() -> List[Dict[str, Any]]:
    return [
        build_case(
            name="defaults_to_model_json_when_file_missing",
            generation_config={},
            synthesis_options={
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 1.0,
            },
        expected=expected_case(
                "generate_sampled",
                {
                    "do_sample": True,
                    "options": {
                        "temperature": 0.9,
                        "top_k": 50,
                        "top_p": 1.0,
                        "repetition_penalty": 1.05,
                    },
                },
                {
                    "do_sample": True,
                    "options": {
                        "temperature": 0.9,
                        "top_k": 50,
                        "top_p": 1.0,
                        "repetition_penalty": 1.0,
                    },
                },
            ),
        ),
        build_case(
            name="explicit_talker_subtalker_overrides",
            generation_config={
                "do_sample": True,
                "temperature": 1.2,
                "top_k": 64,
                "top_p": 0.92,
                "repetition_penalty": 1.15,
                "subtalker_dosample": False,
                "subtalker_temperature": 0.8,
                "subtalker_top_k": 40,
                "subtalker_top_p": 0.7,
            },
            synthesis_options={
                "temperature": 0.35,
                "top_k": 16,
                "top_p": 0.75,
            },
            expected=expected_case(
                "generate_sampled",
                {
                    "do_sample": True,
                    "options": {
                        "temperature": 0.35,
                        "top_k": 16,
                        "top_p": 0.75,
                        "repetition_penalty": 1.15,
                    },
                },
                {
                    "do_sample": False,
                    "options": {
                        "temperature": 0.8,
                        "top_k": 40,
                        "top_p": 0.7,
                        "repetition_penalty": 1.0,
                    },
                },
            ),
        ),
        build_case(
            name="cli_temperature_zero_stays_greedy_talker_with_sampling_mode",
            generation_config={
                "do_sample": True,
                "temperature": 1.0,
                "top_k": 20,
                "top_p": 0.8,
                "repetition_penalty": 1.0,
                "subtalker_dosample": True,
                "subtalker_temperature": 0.55,
                "subtalker_top_k": 32,
                "subtalker_top_p": 0.9,
            },
            synthesis_options={
                "temperature": 0.0,
                "top_k": 9,
                "top_p": 0.9,
            },
            expected=expected_case(
                "generate_sampled",
                {
                    "do_sample": True,
                    "options": {
                        "temperature": 0.0,
                        "top_k": 9,
                        "top_p": 0.9,
                        "repetition_penalty": 1.0,
                    },
                },
                {
                    "do_sample": True,
                    "options": {
                        "temperature": 0.55,
                        "top_k": 32,
                        "top_p": 0.9,
                        "repetition_penalty": 1.0,
                    },
                },
            ),
        ),
        build_case(
            name="talker_greedy_forced_by_generation_config",
            generation_config={
                "do_sample": False,
                "temperature": 1.0,
                "top_k": 64,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
                "subtalker_dosample": True,
                "subtalker_temperature": 0.55,
                "subtalker_top_k": 32,
                "subtalker_top_p": 0.85,
            },
            synthesis_options={
                "temperature": 0.9,
                "top_k": 20,
                "top_p": 0.8,
            },
            expected=expected_case(
                "generate_sampled",
                {
                    "do_sample": False,
                    "options": {
                        "temperature": 0.9,
                        "top_k": 20,
                        "top_p": 0.8,
                        "repetition_penalty": 1.05,
                    },
                },
                {
                    "do_sample": True,
                    "options": {
                        "temperature": 0.55,
                        "top_k": 32,
                        "top_p": 0.85,
                        "repetition_penalty": 1.0,
                    },
                },
            ),
        ),
        build_case(
            name="greedy_talker_and_greedy_subtalker",
            generation_config={
                "do_sample": False,
                "temperature": 1.0,
                "top_k": 64,
                "top_p": 0.95,
                "repetition_penalty": 1.0,
                "subtalker_dosample": False,
                "subtalker_temperature": 0.55,
                "subtalker_top_k": 32,
                "subtalker_top_p": 0.85,
            },
            synthesis_options={
                "temperature": 0.9,
                "top_k": 20,
                "top_p": 0.8,
            },
            expected=expected_case(
                "generate",
                {
                    "do_sample": False,
                    "options": {
                        "temperature": 0.9,
                        "top_k": 20,
                        "top_p": 0.8,
                        "repetition_penalty": 1.0,
                    },
                },
                {
                    "do_sample": False,
                    "options": {
                        "temperature": 0.55,
                        "top_k": 32,
                        "top_p": 0.85,
                        "repetition_penalty": 1.0,
                    },
                },
            ),
        ),
        build_case(
            name="invalid_generation_config_rejected",
            generation_config={
                "do_sample": True,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 2.5,
                "repetition_penalty": 1.05,
            },
            synthesis_options={
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 1.0,
            },
            expect_error=True,
        ),
    ]


def build_model_cases() -> List[Dict[str, Any]]:
    raw = json.loads(P01_FIXTURE_PATH.read_text(encoding="utf-8"))
    models = raw["models"]
    if not isinstance(models, list) or len(models) == 0:
        raise RuntimeError("p01 matrix is missing models list")

    payload: List[Dict[str, Any]] = []
    for entry in models:
        model_id = entry["model_id"]
        revision = entry["model_revision"]
        cfg_path = ensure_generation_config_snapshot(model_id, revision)
        raw_bytes = cfg_path.read_bytes()
        raw_text = raw_bytes.decode("utf-8")
        cfg = json.loads(raw_text)
        resolved = {}
        talker_do_sample, talker, subtalker_do_sample, subtalker = resolve_sampling_config(cfg)
        resolved = {
            "talker": {
                "do_sample": talker_do_sample,
                "options": talker,
            },
            "subtalker": {
                "do_sample": subtalker_do_sample,
                "options": subtalker,
            },
        }
        payload.append(
            {
                "model_id": model_id,
                "model_revision": revision,
                "generation_config": {
                    "path": "generation_config.json",
                    "sha256": sha256_file(cfg_path),
                    "raw_bytes": raw_text,
                    "raw": cfg,
                    "resolved": resolved,
                },
            }
        )
    return payload


def build_fixture(*, include_models: bool = True) -> Dict[str, Any]:
    return {
        "version": 1,
        "fixture_id": "p02-sampling-config-matrix",
        "path": "fixtures/alignment/p02_sampling_config_matrix.json",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "generated_by": "tools/generate_sampling_config_matrix.py",
        "command": PYTHON,
        "cases": build_cases(),
        "models": build_model_cases() if include_models else [],
    }


def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=None)


def validate(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"FIXTURE_MISSING: {path}")
    current = json.loads(path.read_text(encoding="utf-8"))
    metadata = {
        "version": 1,
        "fixture_id": "p02-sampling-config-matrix",
        "path": "fixtures/alignment/p02_sampling_config_matrix.json",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "generated_by": "tools/generate_sampling_config_matrix.py",
        "command": PYTHON,
    }
    for key, value in metadata.items():
        if current.get(key) != value:
            raise RuntimeError(f"FIXTURE_MISMATCH: top-level {key} differs")
    expected_cases = build_fixture(include_models=False)["cases"]
    if _canonical_json(current.get("cases")) != _canonical_json(expected_cases):
        raise RuntimeError("FIXTURE_MISMATCH: synthetic sampling cases changed")
    expected_models = {entry["model_id"]: entry for entry in json.loads(
        P01_FIXTURE_PATH.read_text(encoding="utf-8")
    )["models"]}
    models = current.get("models")
    if not isinstance(models, list) or len(models) != len(expected_models):
        raise RuntimeError("FIXTURE_MISMATCH: five-model provenance is missing")
    seen = set()
    for entry in models:
        model_id = entry.get("model_id")
        if model_id in seen:
            raise RuntimeError(f"FIXTURE_MISMATCH: duplicate model {model_id}")
        seen.add(model_id)
        if model_id not in expected_models:
            raise RuntimeError(f"FIXTURE_MISMATCH: unexpected model {model_id!r}")
        if entry.get("model_revision") != expected_models[model_id]["model_revision"]:
            raise RuntimeError(f"FIXTURE_MISMATCH: revision mismatch for {model_id}")
        generation = entry.get("generation_config", {})
        if generation.get("path") != "generation_config.json":
            raise RuntimeError(f"FIXTURE_MISMATCH: generation-config path for {model_id}")
        raw_text = generation.get("raw_bytes")
        if not isinstance(raw_text, str):
            raise RuntimeError(f"FIXTURE_MISMATCH: raw bytes missing for {model_id}")
        raw = json.loads(raw_text)
        if raw != generation.get("raw"):
            raise RuntimeError(f"FIXTURE_MISMATCH: raw generation config mismatch for {model_id}")
        digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        if digest != generation.get("sha256"):
            raise RuntimeError(f"FIXTURE_MISMATCH: generation-config hash mismatch for {model_id}")
        talker_do_sample, talker, subtalker_do_sample, subtalker = resolve_sampling_config(raw)
        resolved = {
            "talker": {"do_sample": talker_do_sample, "options": talker},
            "subtalker": {"do_sample": subtalker_do_sample, "options": subtalker},
        }
        if resolved != generation.get("resolved"):
            raise RuntimeError(f"FIXTURE_MISMATCH: resolved sampling mismatch for {model_id}")
    if seen != set(expected_models):
        raise RuntimeError("FIXTURE_MISMATCH: model set differs from pinned P01 models")


def write_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(build_fixture(), indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/validate P02 sampling config matrix fixture")
    parser.add_argument(
        "--check",
        type=Path,
        help="Validate an existing fixture file instead of generating",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FIXTURE_PATH,
        help="Write location for generated fixture",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check is not None:
        validate(args.check)
        print(f"SAMPLING_CONFIG_MATRIX_CHECK_OK {args.check}")
        return 0

    write_fixture(args.output)
    payload = build_fixture()
    print(f"SAMPLING_CONFIG_MATRIX_GENERATED {args.output}")
    print(f"CASES={len(payload['cases'])}")
    print(f"MODELS={len(payload['models'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
