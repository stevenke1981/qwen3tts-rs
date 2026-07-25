#!/usr/bin/env python3
"""Reference command adapter for official Python and qwen-tts CLI outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OFFICIAL_KIND = "official-python"
QWEN_CPP_KIND = "qwentts.cpp"
OFFICIAL_MODE = "official-python"
QWEN_CPP_MODE = "qwentts-cpp"
ARTIFACT_KINDS = ("prompt", "stdout", "stderr", "tensor", "audio", "token")
ARTIFACT_KIND_CANONICAL = {"token": "stdout"}
SCHEMA_VERSION = 1
MAX_DIMS = 32
REFERENCE_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "schemas" / "reference-run.schema.json"
)
DEFAULT_PREFIXES = {
    OFFICIAL_KIND: {
        "prompt": "prompt.txt",
        "stdout": "tokens.bin",
    },
    QWEN_CPP_KIND: {
        "prompt": "prompt.txt",
        "audio": "output.wav",
        "tensor": "tensors",
    },
}

LOG_STDOUT = Path("logs") / "stdout.bin"
LOG_STDERR = Path("logs") / "stderr.txt"


class RefAdapterError(SystemExit):
    def __init__(self, prefix: str, message: str, *, code: int = 1):
        super().__init__(code)
        self.prefix = prefix
        self.message = message


def _error(prefix: str, message: str) -> None:
    raise RefAdapterError(prefix, message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_kind(kind: str) -> str:
    return ARTIFACT_KIND_CANONICAL.get(kind, kind)


def read_text_input(text: str | None, text_file: str | None) -> str:
    if text is None and text_file is None:
        _error("REFERENCE_CONFIG", "either --text or --text-file must be provided")
    if text is not None and text_file is not None:
        _error("REFERENCE_CONFIG", "provide only one of --text or --text-file")

    if text is not None:
        if not text.strip():
            _error("REFERENCE_CONFIG", "input text must not be empty")
        return text

    file_path = Path(text_file)
    if not file_path.is_file():
        _error("REFERENCE_CONFIG", f"invalid --text-file {text_file}")
    value = file_path.read_text(encoding="utf-8")
    if not value.strip():
        _error("REFERENCE_CONFIG", "input text file must not be empty")
    return value


def _safe_ascii_relpath(value: str) -> Path:
    if value != value.strip():
        _error("REFERENCE_OUTPUT", "artifact path contains leading/trailing whitespace")
    if value == "":
        _error("REFERENCE_OUTPUT", "artifact path cannot be empty")
    if os.path.isabs(value):
        _error("REFERENCE_OUTPUT", f"unsafe absolute artifact path: {value}")

    parts = [p for p in re.split(r"[\\\\/]", value) if p != ""]
    if not parts:
        _error("REFERENCE_OUTPUT", f"invalid artifact path: {value}")
    if any(p in (".", "..") for p in parts):
        _error("REFERENCE_OUTPUT", f"unsafe artifact path: {value}")
    if any(re.search(r"[<>:\"|?*]", p) for p in parts):
        _error("REFERENCE_OUTPUT", f"unsafe artifact path: {value}")
    if any(p.endswith(".") for p in parts):
        _error("REFERENCE_OUTPUT", f"unsafe artifact path: {value}")

    return Path(*parts)


def _validate_revision(revision: str) -> str:
    value = revision.strip()
    if not value:
        _error("REFERENCE_CONFIG", "missing --source-revision")
    return value


def _normalize_output_root(output_dir: Path) -> Path:
    if output_dir.exists():
        _error("REFERENCE_OUTPUT", f"output directory already exists: {output_dir}")

    current = output_dir.parent
    root = Path.cwd()
    while current != current.parent:
        if current.exists() and current.is_symlink():
            _error("REFERENCE_OUTPUT", f"output directory parent is a symlink: {current}")
        if current == root:
            break
        current = current.parent

    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _ensure_within_session(output_dir: Path, rel: Path) -> Path:
    session_root = output_dir.resolve()
    candidate = (output_dir / rel).resolve()
    try:
        common = os.path.commonpath([str(candidate), str(session_root)])
    except ValueError:
        _error("REFERENCE_OUTPUT", f"artifact path escapes session: {rel}")
    if common != str(session_root):
        _error("REFERENCE_OUTPUT", f"artifact path escapes session: {rel}")

    parent = candidate.parent
    while parent != output_dir.parent:
        if parent.exists() and parent.is_symlink():
            _error("REFERENCE_OUTPUT", f"artifact path escapes through symlink: {parent}")
        if parent == output_dir:
            break
        parent = parent.parent

    return candidate


def _artifact_map_from_args(values: list[str] | None) -> dict[str, Path]:
    declared: dict[str, Path] = {}
    if not values:
        return declared

    for item in values:
        if "=" not in item:
            _error("REFERENCE_CONFIG", f"invalid --artifact form: {item}")
        kind, path = item.split("=", 1)
        kind = _canonical_kind(kind)
        if kind not in ARTIFACT_KINDS:
            _error("REFERENCE_CONFIG", f"invalid artifact kind: {kind}")
        if kind in declared:
            _error("REFERENCE_OUTPUT", f"duplicate artifact declaration: {kind}")
        declared[kind] = _safe_ascii_relpath(path)
    return declared


@dataclass(frozen=True)
class ArtifactRecord:
    kind: str
    relative_path: str
    byte_length: int
    sha256: str
    shape: list[int] | None = None
    dtype: str | None = None
    layout: str | None = None

    def to_json(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "kind": self.kind,
            "relative_path": self.relative_path,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
        }
        if self.shape is not None:
            obj["shape"] = self.shape
            obj["dtype"] = self.dtype or "f32"
            obj["layout"] = self.layout or "C"
        return obj


def _write_manifest(
    path: Path,
    *,
    kind: str,
    source: str,
    revision: str,
    model: str,
    case_id: str,
    seed: int | None,
    argv: list[str],
    exit_status: int,
    stages: list[dict[str, Any]],
    artifacts: list[ArtifactRecord],
) -> None:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "source": source,
        "revision": revision,
        "model": model,
        "case_id": case_id,
        "seed": seed,
        "argv": argv,
        "exit_status": exit_status,
        "artifacts": [item.to_json() for item in artifacts],
        "stages": stages,
    }
    _write_bytes_file(
        path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _safe_numel(shape: list[int], max_values: int) -> int:
    if max_values < 0:
        _error("REFERENCE_FORMAT", "tensor payload cannot be negative")

    expected = 1
    for dim in shape:
        if dim == 0:
            return 0
        if dim < 0:
            _error("REFERENCE_FORMAT", f"negative tensor dimension: {dim}")
        if max_values and dim > max_values // expected:
            _error("REFERENCE_FORMAT", f"tensor shape overflow in expected size: {shape}")
        expected *= dim
    return expected


def _parse_tensor_dump(path: Path) -> tuple[list[int], bytes]:
    data = path.read_bytes()
    if len(data) < 4:
        _error("REFERENCE_FORMAT", f"malformed tensor dump header: {path.name}")

    ndims = struct.unpack_from("<i", data, 0)[0]
    if ndims < 0:
        _error("REFERENCE_FORMAT", f"negative tensor rank in {path.name}: {ndims}")
    if ndims > MAX_DIMS:
        _error("REFERENCE_FORMAT", f"tensor rank overflow in {path.name}: {ndims}")

    off = 4
    shapes: list[int] = []
    for _ in range(ndims):
        if off + 4 > len(data):
            _error("REFERENCE_FORMAT", f"truncated tensor shape in {path.name}")
        dim = struct.unpack_from("<i", data, off)[0]
        shapes.append(dim)
        off += 4

    data_bytes = len(data) - off
    expected_elements = _safe_numel(shapes, data_bytes // 4)
    expected_bytes = expected_elements * 4
    if data_bytes != expected_bytes:
        if expected_bytes < data_bytes:
            _error("REFERENCE_FORMAT", f"trailing bytes in tensor dump: {path.name}")
        _error("REFERENCE_FORMAT", f"malformed tensor dump (truncated): {path.name}")

    raw = data[off:]
    if expected_elements == 0 and raw:
        _error("REFERENCE_FORMAT", f"malformed tensor payload for zero-sized shape: {path.name}")

    if raw:
        struct.unpack_from("<" + ("f" * expected_elements), raw, 0)

    return shapes, raw


def _run_command(argv: list[str], *, input_text: str | None = None) -> tuple[int, bytes, bytes]:
    try:
        proc = subprocess.run(
            argv,
            input=input_text.encode("utf-8") if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        _error("REFERENCE_EXEC", f"failed to launch child process: {exc}")
    return proc.returncode, proc.stdout, proc.stderr


def _write_bytes_file(path: Path, value: bytes, *, allow_empty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(value)
    except FileExistsError:
        _error("REFERENCE_OUTPUT", f"refusing to overwrite artifact: {path}")
    if (not allow_empty) and path.stat().st_size == 0:
        _error("REFERENCE_OUTPUT", f"artifact is empty: {path}")


def _record_prompt_file(session: Path, rel: Path, text: str) -> ArtifactRecord:
    path = _ensure_within_session(session, rel)
    _write_bytes_file(path, text.encode("utf-8"))
    return ArtifactRecord(
        kind="prompt",
        relative_path=str(rel.as_posix()),
        byte_length=path.stat().st_size,
        sha256=sha256_file(path),
    )


def _record_bytes_file(
    session: Path, rel: Path, value: bytes, kind: str, *, allow_empty: bool = False
) -> ArtifactRecord:
    path = _ensure_within_session(session, rel)
    _write_bytes_file(path, value, allow_empty=allow_empty)
    return ArtifactRecord(
        kind=kind,
        relative_path=str(rel.as_posix()),
        byte_length=path.stat().st_size,
        sha256=sha256_file(path),
    )


def _write_declared_artifacts(
    session: Path,
    stdout_data: bytes,
    stderr_data: bytes,
    declared: dict[str, Path],
) -> list[ArtifactRecord]:
    entries: list[ArtifactRecord] = []
    if "stderr" in declared:
        entries.append(
            _record_bytes_file(
                session,
                declared["stderr"],
                stderr_data,
                kind="stderr",
                allow_empty=False,
            )
        )
    if "stdout" in declared:
        entries.append(
            _record_bytes_file(
                session,
                declared["stdout"],
                stdout_data,
                kind="stdout",
                allow_empty=False,
            )
        )
    return entries


def _collect_log_artifacts(
    session: Path,
    stdout_data: bytes,
    stderr_data: bytes,
    *,
    include_stdout_log: bool,
    include_stderr_log: bool,
) -> list[ArtifactRecord]:
    entries: list[ArtifactRecord] = []
    if include_stdout_log and stdout_data:
        entries.append(
            _record_bytes_file(
                session,
                LOG_STDOUT,
                stdout_data,
                kind="stdout",
                allow_empty=True,
            )
        )
    if include_stderr_log and stderr_data:
        entries.append(
            _record_bytes_file(
                session,
                LOG_STDERR,
                stderr_data,
                kind="stderr",
                allow_empty=True,
            )
        )
    return entries


def _write_failure_manifest(
    *,
    output_root: Path,
    kind: str,
    source: str,
    revision: str,
    model: str,
    case_id: str,
    seed: int | None,
    argv: list[str],
    exit_status: int,
    text: str,
    stdout_data: bytes,
    stderr_data: bytes,
    declared: dict[str, Path],
) -> None:
    artifacts: list[ArtifactRecord] = []
    if kind == OFFICIAL_KIND:
        prompt_rel = declared.get("prompt", Path(DEFAULT_PREFIXES[OFFICIAL_KIND]["prompt"]))
    else:
        prompt_rel = declared.get("prompt", Path(DEFAULT_PREFIXES[QWEN_CPP_KIND]["prompt"]))

    artifacts.append(_record_prompt_file(output_root, prompt_rel, text))
    artifacts.extend(
        _collect_log_artifacts(
            output_root,
            stdout_data,
            stderr_data,
            include_stdout_log=True,
            include_stderr_log=True,
        )
    )

    _write_manifest(
        output_root / "reference-run.json",
        kind=kind,
        source=source,
        revision=revision,
        model=model,
        case_id=case_id,
        seed=seed,
        argv=argv,
        exit_status=exit_status,
        stages=[],
        artifacts=_ordered(artifacts),
    )


def _build_stage_name(path: Path, fallback: str, index: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", path.stem) or fallback
    return f"{clean}_{index:04d}"


def _copy_tensor_artifacts(
    record: list[ArtifactRecord],
    session: Path,
    tensor_dir_rel: Path,
    dump_dir: Path,
    *,
    source: str,
    revision: str,
    model: str,
    case_id: str,
    seed: int | None,
    require_tensor: bool = False,
) -> list[dict[str, Any]]:
    tensor_output_dir = _ensure_within_session(session, tensor_dir_rel)
    tensor_output_dir.mkdir(parents=True, exist_ok=True)
    if not dump_dir.exists():
        _error("REFERENCE_OUTPUT", f"missing tensor dump directory: {dump_dir}")
    if dump_dir.is_symlink():
        _error("REFERENCE_OUTPUT", f"unsafe tensor dump path (symlink): {dump_dir}")

    stage_entries: list[dict[str, Any]] = []
    files = sorted((f for f in dump_dir.iterdir() if f.is_file()), key=lambda p: p.name)
    if require_tensor and not files:
        _error("REFERENCE_OUTPUT", "qwentts-cpp requires at least one tensor dump")

    for idx, file in enumerate(files):
        if file.is_symlink():
            _error(
                "REFERENCE_OUTPUT",
                f"unsafe tensor dump entry is symlink: {file.name}",
            )
        shapes, raw = _parse_tensor_dump(file)
        stage_name = _build_stage_name(file, "tensor", idx)
        out_name = f"{stage_name}.f32.bin"
        out_path = tensor_output_dir / out_name
        _write_bytes_file(out_path, raw, allow_empty=not raw)

        artifact_rel = tensor_dir_rel / out_name
        record.append(
            ArtifactRecord(
                kind="tensor",
                relative_path=str(artifact_rel.as_posix()),
                byte_length=out_path.stat().st_size,
                sha256=sha256_file(out_path),
                shape=shapes,
                dtype="f32",
                layout="C",
            )
        )
        stage_entries.append(
            {
                "name": stage_name,
                "dtype": "f32",
                "shape": shapes,
                "file": out_name,
                "layout": "C",
                "sha256": sha256_file(out_path),
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "revision": revision,
        "model": model,
        "case_id": case_id,
        "seed": seed,
        "stages": stage_entries,
    }
    _write_bytes_file(
        tensor_output_dir / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return stage_entries


def _validate_no_duplicates(outputs: list[ArtifactRecord]) -> None:
    seen = set[str]()
    for item in outputs:
        if item.relative_path in seen:
            _error("REFERENCE_OUTPUT", f"duplicate artifact path: {item.relative_path}")
        seen.add(item.relative_path)

    if any(p.startswith("../") or p.startswith("..\\") or "/../" in p for p in seen):
        _error("REFERENCE_OUTPUT", "artifact path escapes session")


def _ordered(artifacts: list[ArtifactRecord]) -> list[ArtifactRecord]:
    return sorted(artifacts, key=lambda it: it.relative_path)


def _require_declared_output_presence(
    declared: dict[str, Path], artifacts: list[ArtifactRecord]
) -> None:
    produced = {item.kind: item.relative_path for item in artifacts}
    for kind in sorted(declared):
        if kind not in produced:
            _error("REFERENCE_OUTPUT", f"declared artifact missing: {kind}")


def _is_forbidden_extra_arg(arg: str) -> bool:
    forbidden = (
        "--dump",
        "--output",
        "--text",
        "--text-file",
        "-o",
        "--output-wav",
        "--model",
        "--codec",
    )
    if arg in forbidden:
        return True
    if arg.startswith("-o") and arg != "-":
        return True
    return any(arg.startswith(item + "=") for item in forbidden)


def _is_sensitive_extra_arg(arg: str) -> bool:
    normalized = arg.lower().lstrip("-").replace("_", "-")
    return any(
        marker in normalized
        for marker in ("api-key", "authorization", "password", "secret", "token")
    )


def _build_official_argv(
    args: argparse.Namespace,
    text: str,
    *,
    output_root: Path,
    audio_declared: bool,
    tensor_declared: bool,
    audio_rel: Path | None,
) -> list[str]:
    argv = [
        args.python,
        args.runner,
        "--model",
        args.model,
        "--language",
        args.language,
    ]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.max_new is not None:
        argv += ["--max-new-tokens", str(args.max_new)]
    if args.top_k is not None:
        argv += ["--top-k", str(args.top_k)]
    if args.top_p is not None:
        argv += ["--top-p", str(args.top_p)]
    if args.temperature is not None:
        argv += ["--temperature", str(args.temperature)]
    if args.greedy:
        argv += ["--greedy"]
    if args.speaker is not None:
        argv += ["--speaker", args.speaker]
    if args.instruct is not None:
        argv += ["--instruct", args.instruct]
    if args.text is not None:
        argv += ["--text", text]
    else:
        argv += ["--text-file", args.text_file]
    if audio_declared and audio_rel is not None:
        argv += ["--output-wav", str(output_root / audio_rel)]
    if tensor_declared:
        argv += ["--dump", str(output_root / "raw_tensor_dump")]
    argv.extend(args.extra_arg)
    return argv


def _official_plan(args: argparse.Namespace, *, enforce_output_absence: bool) -> dict[str, Any]:
    text = read_text_input(args.text, args.text_file)
    if not args.python:
        _error("REFERENCE_CONFIG", "missing --python")
    if not args.runner:
        _error("REFERENCE_CONFIG", "missing --runner")
    if not args.model.strip():
        _error("REFERENCE_CONFIG", "missing --model")
    if not args.case_id.strip():
        _error("REFERENCE_CONFIG", "missing --case-id")

    revision = _validate_revision(args.source_revision)
    if not Path(args.python).is_file():
        _error("REFERENCE_CONFIG", f"missing python executable: {args.python}")
    if not Path(args.runner).is_file():
        _error("REFERENCE_CONFIG", f"missing runner script: {args.runner}")

    output_root = Path(args.output_dir)
    if enforce_output_absence and output_root.exists():
        _error("REFERENCE_OUTPUT", f"output directory already exists: {args.output_dir}")

    declared = _artifact_map_from_args(args.artifact)
    prompt_rel = declared.get("prompt", Path(DEFAULT_PREFIXES[OFFICIAL_KIND]["prompt"]))
    stdout_rel = declared.get("stdout", Path(DEFAULT_PREFIXES[OFFICIAL_KIND]["stdout"]))
    audio_rel = declared.get("audio")
    tensor_rel = declared.get("tensor")

    for rel in [prompt_rel, stdout_rel]:
        _ensure_within_session(output_root, rel)
    if audio_rel is not None:
        _ensure_within_session(output_root, audio_rel)
    if tensor_rel is not None:
        _ensure_within_session(output_root, tensor_rel)
        _ensure_within_session(output_root, Path("raw_tensor_dump"))

    if prompt_rel == stdout_rel and "prompt" in declared and "stdout" in declared:
        _error("REFERENCE_OUTPUT", "duplicate artifact declarations")

    if any(_is_forbidden_extra_arg(arg) for arg in args.extra_arg):
        _error(
            "REFERENCE_CONFIG",
            "reserved extra argument blocked: --dump, --output, --text, --text-file, --output-wav, --model, --codec, -o",
        )
    if any(_is_sensitive_extra_arg(arg) for arg in args.extra_arg):
        _error("REFERENCE_CONFIG", "sensitive extra arguments are not recorded")

    argv = _build_official_argv(
        args,
        text,
        output_root=output_root,
        audio_declared=("audio" in declared),
        tensor_declared=("tensor" in declared),
        audio_rel=audio_rel,
    )
    return {
        "kind": OFFICIAL_KIND,
        "source": OFFICIAL_KIND,
        "source_revision": revision,
        "model": args.model,
        "case_id": args.case_id,
        "seed": args.seed,
        "output_dir": str(output_root),
        "argv": argv,
        "declared_artifacts": {k: str(v.as_posix()) for k, v in declared.items()},
        "text": text,
        "prompt_rel": prompt_rel,
        "stdout_rel": stdout_rel,
        "audio_rel": audio_rel,
        "tensor_rel": tensor_rel,
    }


def _run_official(args: argparse.Namespace) -> int:
    info = _official_plan(args, enforce_output_absence=True)
    output_root = _normalize_output_root(Path(args.output_dir))
    prompt_rel = info["prompt_rel"]
    stdout_rel = info["stdout_rel"]
    audio_rel = info["audio_rel"]
    tensor_rel = info["tensor_rel"]
    text = info["text"]
    declared = _artifact_map_from_args(args.artifact)

    if audio_rel is not None:
        _ensure_within_session(output_root, audio_rel).parent.mkdir(
            parents=True, exist_ok=True
        )
    if tensor_rel is not None:
        _ensure_within_session(output_root, Path("raw_tensor_dump")).mkdir(
            parents=True, exist_ok=False
        )

    exit_code, stdout_data, stderr_data = _run_command(info["argv"])
    if exit_code != 0:
        _write_failure_manifest(
            output_root=output_root,
            kind=OFFICIAL_KIND,
            source=OFFICIAL_KIND,
            revision=info["source_revision"],
            model=args.model,
            case_id=args.case_id,
            seed=args.seed,
            argv=info["argv"],
            exit_status=exit_code,
            text=text,
            stdout_data=stdout_data,
            stderr_data=stderr_data,
            declared=declared,
        )
        _error(
            "REFERENCE_EXEC",
            f"official-python failed (code={exit_code}): {stderr_data.decode('utf-8', errors='replace').strip()}",
        )

    artifacts: list[ArtifactRecord] = [
        _record_prompt_file(output_root, prompt_rel, text),
        _record_bytes_file(output_root, stdout_rel, stdout_data, kind="stdout"),
    ]

    if audio_rel is not None:
        _build_audio_artifact(artifacts, output_root, audio_rel, output_root / audio_rel)
    stages: list[dict[str, Any]] = []
    if tensor_rel is not None:
        stages = _copy_tensor_artifacts(
            artifacts,
            output_root,
            tensor_rel,
            output_root / "raw_tensor_dump",
            source=OFFICIAL_KIND,
            revision=info["source_revision"],
            model=args.model,
            case_id=args.case_id,
            seed=args.seed,
        )

    if "stdout" in declared and declared["stdout"] != stdout_rel:
        _error("REFERENCE_OUTPUT", "declared stdout path must match token output path")
    if "stderr" in declared:
        artifacts.extend(
            _write_declared_artifacts(output_root, stdout_data, stderr_data, {"stderr": declared["stderr"]})
        )
    artifacts.extend(
        _collect_log_artifacts(
            output_root,
            stdout_data,
            stderr_data,
            include_stdout_log=False,
            include_stderr_log="stderr" not in declared,
        )
    )
    all_artifacts = _ordered(artifacts)
    _require_declared_output_presence(declared, all_artifacts)
    _validate_no_duplicates(all_artifacts)

    manifest_path = output_root / "reference-run.json"
    _write_manifest(
        manifest_path,
        kind=OFFICIAL_KIND,
        source=OFFICIAL_KIND,
        revision=info["source_revision"],
        model=args.model,
        case_id=args.case_id,
        seed=args.seed,
        argv=info["argv"],
        exit_status=exit_code,
        stages=stages,
        artifacts=all_artifacts,
    )
    print(manifest_path)
    return 0


def _build_audio_artifact(
    artifacts: list[ArtifactRecord], output_root: Path, rel: Path, actual_path: Path
) -> None:
    target = _ensure_within_session(output_root, rel)
    if actual_path.is_symlink():
        _error("REFERENCE_OUTPUT", f"unsafe audio path (symlink): {rel}")
    if not actual_path.exists() or not actual_path.is_file() or actual_path.stat().st_size == 0:
        _error("REFERENCE_OUTPUT", f"audio artifact missing or empty: {rel}")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target != actual_path:
        target.write_bytes(actual_path.read_bytes())
    artifacts.append(
        ArtifactRecord(
            kind="audio",
            relative_path=str(rel.as_posix()),
            byte_length=target.stat().st_size,
            sha256=sha256_file(target),
        )
    )


def _build_qwentts_argv(args: argparse.Namespace, text: str, dump_dir: Path, audio_path: Path) -> list[str]:
    exe = str(Path(args.executable))
    argv = [
        exe,
        "--model",
        args.model,
        "--codec",
        args.codec,
    ]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.language is not None:
        argv += ["--lang", args.language]
    if args.max_new is not None:
        argv += ["--max-new", str(args.max_new)]
    if args.greedy:
        argv += ["--greedy"]
    if args.speaker is not None:
        argv += ["--speaker", args.speaker]
    if args.instruct is not None:
        argv += ["--instruct", args.instruct]
    if args.ref_wav is not None:
        argv += ["--ref-wav", args.ref_wav]
    if args.ref_text is not None:
        argv += ["--ref-text", args.ref_text]
    argv.extend(args.extra_arg)
    argv += ["--dump", str(dump_dir), "-o", str(audio_path)]
    return argv


def _qwentts_plan(args: argparse.Namespace, *, enforce_output_absence: bool) -> dict[str, Any]:
    text = read_text_input(args.text, args.text_file)
    if not args.executable:
        _error("REFERENCE_CONFIG", "missing --executable")
    if not args.model.strip():
        _error("REFERENCE_CONFIG", "missing --model")
    if not args.codec.strip():
        _error("REFERENCE_CONFIG", "missing --codec")
    if not args.case_id.strip():
        _error("REFERENCE_CONFIG", "missing --case-id")

    revision = _validate_revision(args.source_revision)
    exe = Path(args.executable)
    if not exe.is_file():
        _error("REFERENCE_CONFIG", f"missing executable: {args.executable}")

    output_root = Path(args.output_dir)
    if enforce_output_absence and output_root.exists():
        _error("REFERENCE_OUTPUT", f"output directory already exists: {args.output_dir}")

    if any(_is_forbidden_extra_arg(arg) for arg in args.extra_arg):
        _error(
            "REFERENCE_CONFIG",
            "reserved extra argument blocked: --dump, --output, --text, --text-file, --output-wav, --model, --codec, -o",
        )
    if any(_is_sensitive_extra_arg(arg) for arg in args.extra_arg):
        _error("REFERENCE_CONFIG", "sensitive extra arguments are not recorded")

    declared = _artifact_map_from_args(args.artifact)
    prompt_rel = declared.get("prompt", Path(DEFAULT_PREFIXES[QWEN_CPP_KIND]["prompt"]))
    audio_rel = declared.get("audio", Path(DEFAULT_PREFIXES[QWEN_CPP_KIND]["audio"]))
    tensor_rel = declared.get("tensor", Path(DEFAULT_PREFIXES[QWEN_CPP_KIND]["tensor"]))

    for rel in [prompt_rel, audio_rel, tensor_rel]:
        _ensure_within_session(output_root, rel)
    _ensure_within_session(output_root, Path("raw_tensor_dump"))

    dump_dir = output_root / "raw_tensor_dump"
    audio_path = output_root / audio_rel
    argv = _build_qwentts_argv(args, text, dump_dir, audio_path)
    return {
        "kind": QWEN_CPP_KIND,
        "source": QWEN_CPP_KIND,
        "source_revision": revision,
        "model": args.model,
        "case_id": args.case_id,
        "seed": args.seed,
        "output_dir": str(output_root),
        "argv": argv,
        "declared_artifacts": {k: str(v.as_posix()) for k, v in declared.items()},
        "text": text,
        "prompt_rel": prompt_rel,
        "audio_rel": audio_rel,
        "tensor_rel": tensor_rel,
        "dump_dir": dump_dir,
        "audio_path": audio_path,
    }


def _run_qwentts(args: argparse.Namespace) -> int:
    info = _qwentts_plan(args, enforce_output_absence=True)
    output_root = _normalize_output_root(Path(args.output_dir))
    prompt_rel = info["prompt_rel"]
    audio_rel = info["audio_rel"]
    tensor_rel = info["tensor_rel"]
    dump_dir = info["dump_dir"]
    audio_path = info["audio_path"]
    text = info["text"]
    declared = _artifact_map_from_args(args.artifact)

    _ensure_within_session(output_root, audio_rel).parent.mkdir(
        parents=True, exist_ok=True
    )
    _ensure_within_session(output_root, Path("raw_tensor_dump")).mkdir(
        parents=True, exist_ok=False
    )

    exit_code, stdout_data, stderr_data = _run_command(info["argv"], input_text=text)
    if exit_code != 0:
        _write_failure_manifest(
            output_root=output_root,
            kind=QWEN_CPP_KIND,
            source=QWEN_CPP_KIND,
            revision=info["source_revision"],
            model=args.model,
            case_id=args.case_id,
            seed=args.seed,
            argv=info["argv"],
            exit_status=exit_code,
            text=text,
            stdout_data=stdout_data,
            stderr_data=stderr_data,
            declared=declared,
        )
        _error(
            "REFERENCE_EXEC",
            f"qwentts.cpp failed (code={exit_code}): {stderr_data.decode('utf-8', errors='replace').strip()}",
        )

    artifacts: list[ArtifactRecord] = [_record_prompt_file(output_root, prompt_rel, text)]
    _build_audio_artifact(artifacts, output_root, audio_rel, audio_path)
    stages = _copy_tensor_artifacts(
        artifacts,
        output_root,
        tensor_rel,
        dump_dir,
        source=QWEN_CPP_KIND,
        revision=info["source_revision"],
        model=args.model,
        case_id=args.case_id,
        seed=args.seed,
        require_tensor=True,
    )

    aux = []
    if "stdout" in declared or "stderr" in declared:
        if "stdout" in declared:
            aux.extend(_write_declared_artifacts(output_root, stdout_data, stderr_data, {"stdout": declared["stdout"]}))
        if "stderr" in declared:
            aux.extend(_write_declared_artifacts(output_root, stdout_data, stderr_data, {"stderr": declared["stderr"]}))
    aux.extend(
        _collect_log_artifacts(
            output_root,
            stdout_data,
            stderr_data,
            include_stdout_log="stdout" not in declared,
            include_stderr_log="stderr" not in declared,
        )
    )
    artifacts.extend(aux)

    _require_declared_output_presence(declared, artifacts)
    _validate_no_duplicates(artifacts)
    artifacts = _ordered(artifacts)

    manifest_path = output_root / "reference-run.json"
    _write_manifest(
        manifest_path,
        kind=QWEN_CPP_KIND,
        source=QWEN_CPP_KIND,
        revision=info["source_revision"],
        model=args.model,
        case_id=args.case_id,
        seed=args.seed,
        argv=info["argv"],
        exit_status=exit_code,
        stages=stages,
        artifacts=artifacts,
    )
    print(manifest_path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="reference_adapter.py")
    sub = ap.add_subparsers(dest="mode", required=True)

    official = sub.add_parser("official-python")
    official.add_argument("--dry-run", action="store_true", help="validate and print execution plan only")
    official.add_argument("--python", required=True)
    official.add_argument("--runner", required=True)
    official.add_argument("--source-revision", required=True)
    official.add_argument("--model", required=True)
    official.add_argument("--case-id", required=True)
    official.add_argument("--output-dir", required=True)
    official.add_argument("--text", default=None)
    official.add_argument("--text-file", default=None)
    official.add_argument("--seed", type=int, default=None)
    official.add_argument("--language", default="auto")
    official.add_argument("--max-new", "--max-new-tokens", default=4096, type=int, dest="max_new")
    official.add_argument("--top-k", type=int, default=50, dest="top_k")
    official.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    official.add_argument("--temperature", type=float, default=0.9, dest="temperature")
    official.add_argument("--greedy", action="store_true")
    official.add_argument("--speaker", default=None)
    official.add_argument("--instruct", default=None)
    official.add_argument("--artifact", action="append", default=[])
    official.add_argument("--extra-arg", action="append", default=[], dest="extra_arg")

    qwen_cpp = sub.add_parser("qwentts-cpp")
    qwen_cpp.add_argument("--dry-run", action="store_true", help="validate and print execution plan only")
    qwen_cpp.add_argument("--executable", required=True)
    qwen_cpp.add_argument("--source-revision", required=True)
    qwen_cpp.add_argument("--model", required=True)
    qwen_cpp.add_argument("--codec", required=True)
    qwen_cpp.add_argument("--case-id", required=True)
    qwen_cpp.add_argument("--output-dir", required=True)
    qwen_cpp.add_argument("--text", default=None)
    qwen_cpp.add_argument("--text-file", default=None)
    qwen_cpp.add_argument("--seed", type=int, default=None)
    qwen_cpp.add_argument("--language", "--lang", default="auto", dest="language")
    qwen_cpp.add_argument("--max-new", type=int, default=4096, dest="max_new")
    qwen_cpp.add_argument("--greedy", action="store_true")
    qwen_cpp.add_argument("--speaker", default=None)
    qwen_cpp.add_argument("--instruct", default=None)
    qwen_cpp.add_argument("--ref-wav", default=None, dest="ref_wav")
    qwen_cpp.add_argument("--ref-text", default=None, dest="ref_text")
    qwen_cpp.add_argument("--artifact", action="append", default=[])
    qwen_cpp.add_argument("--extra-arg", action="append", default=[], dest="extra_arg")
    return ap


def _main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.dry_run:
        if args.mode == OFFICIAL_MODE:
            info = _official_plan(args, enforce_output_absence=True)
            print(
                json.dumps(
                    {
                        "kind": info["kind"],
                        "source": info["source"],
                        "source_revision": info["source_revision"],
                        "model": info["model"],
                        "case_id": info["case_id"],
                        "seed": info["seed"],
                        "output_dir": info["output_dir"],
                        "argv": info["argv"],
                        "declared_artifacts": info["declared_artifacts"],
                        "prompt_rel": str(info["prompt_rel"]),
                        "stdout_rel": str(info["stdout_rel"]),
                        "audio_rel": str(info["audio_rel"]) if info["audio_rel"] is not None else None,
                        "tensor_rel": str(info["tensor_rel"]) if info["tensor_rel"] is not None else None,
                        "dump_dir": str(Path(args.output_dir) / "raw_tensor_dump"),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.mode == QWEN_CPP_MODE:
            info = _qwentts_plan(args, enforce_output_absence=True)
            print(
                json.dumps(
                    {
                        "kind": info["kind"],
                        "source": info["source"],
                        "source_revision": info["source_revision"],
                        "model": info["model"],
                        "case_id": info["case_id"],
                        "seed": info["seed"],
                        "output_dir": info["output_dir"],
                        "argv": info["argv"],
                        "declared_artifacts": info["declared_artifacts"],
                        "prompt_rel": str(info["prompt_rel"]),
                        "audio_rel": str(info["audio_rel"]),
                        "tensor_rel": str(info["tensor_rel"]),
                        "dump_dir": str(info["dump_dir"]),
                        "audio_path": str(info["audio_path"]),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

    if args.mode == OFFICIAL_MODE:
        return _run_official(args)
    if args.mode == QWEN_CPP_MODE:
        return _run_qwentts(args)

    _error("REFERENCE_CONFIG", f"unknown mode: {args.mode}")
    return 1


def main() -> None:
    try:
        raise SystemExit(_main())
    except RefAdapterError as exc:
        print(f"{exc.prefix} {exc.message}", file=sys.stderr)
        raise SystemExit(exc.code)


if __name__ == "__main__":
    main()
