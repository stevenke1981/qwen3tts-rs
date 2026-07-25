"""Unit tests for tools/cpu_f32_smoke.py."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("cpu_f32_smoke.py").resolve()
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "cpu-f32-baseline.schema.json"

RUN_KIND_OFFICIAL = "official-python"
RUN_KIND_QWEN = "qwentts.cpp"
REV_OFFICIAL = "0" * 64
REV_QWEN = "1" * 64


def _run_script(*args: str, cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_artifact(root: Path, rel: str, data: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _validate_report_schema(report: dict[str, object]) -> bool:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        from jsonschema import validate
    except ModuleNotFoundError:
        return False
    validate(instance=report, schema=schema)
    return True


def _build_manifest(
    root: Path,
    *,
    kind: str,
    case_id: str,
    model: str,
    revision: str,
    exit_status: int,
    artifacts: list[tuple[str, str, bytes]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest_items: list[dict] = []
    for kind_name, rel, payload in artifacts:
        _write_artifact(root, rel, payload)
        item = {
            "kind": kind_name,
            "relative_path": rel,
            "byte_length": len(payload),
            "sha256": _sha256(payload),
        }
        if kind_name == "tensor":
            item["shape"] = [1]
            item["dtype"] = "f32"
            item["layout"] = "C"
        manifest_items.append(item)
    manifest = {
        "schema_version": 1,
        "kind": kind,
        "source": kind,
        "revision": revision,
        "model": model,
        "case_id": case_id,
        "seed": 1,
        "argv": [],
        "exit_status": exit_status,
        "artifacts": manifest_items,
        "stages": [],
    }
    path = root / "reference-run.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _build_corpus(root: Path, *, cases: list[dict[str, str]]) -> Path:
    path = root / "corpus.json"
    for case in cases:
        if "text_file" in case:
            text_file = root / case["text_file"]
            text_file.parent.mkdir(parents=True, exist_ok=True)
            text_file.write_text(case.get("text", "text from file"), encoding="utf-8")
    path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return path


class TestCpuF32Smoke(unittest.TestCase):
    def test_report_is_deterministic_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _build_corpus(
                root,
                cases=[
                    {"id": "zh", "language": "zh", "text": "你好"},
                    {"id": "en", "language": "en", "text": "Hello"},
                ],
            )
            run_qwen = _build_manifest(
                root / "qwen",
                kind=RUN_KIND_QWEN,
                case_id="zh",
                model="qwen3tts",
                revision=REV_QWEN,
                exit_status=0,
                artifacts=[("prompt", "prompt.txt", b"p"), ("token", "tokens.bin", b"t")],
            )
            run_official = _build_manifest(
                root / "official",
                kind=RUN_KIND_OFFICIAL,
                case_id="zh",
                model="qwen3tts",
                revision=REV_OFFICIAL,
                exit_status=0,
                artifacts=[("prompt", "prompt.txt", b"p"), ("token", "tokens.bin", b"t")],
            )
            code, out, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--require-adapter",
                RUN_KIND_OFFICIAL,
                "--require-adapter",
                "qwentts-cpp",
                "--require-adapter",
                RUN_KIND_QWEN,
                "--require-case",
                "zh",
                "--require-kind",
                "prompt",
                "--require-kind",
                "token",
                "--run",
                str(run_official.relative_to(root)),
                "--run",
                str(run_qwen.relative_to(root)),
                "--output",
                "report.json",
                cwd=root,
            )
            self.assertEqual(code, 0, err)
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [(r["adapter"], r["case"]) for r in report["runs"]],
                [(RUN_KIND_OFFICIAL, "zh"), (RUN_KIND_QWEN, "zh")],
            )
            self.assertEqual(report["status"], "PASS")

    def test_exact_manifest_artifact_hash_and_size_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _build_corpus(root, cases=[{"id": "case", "language": "en", "text": "a"}])
            run_path = _build_manifest(
                root / "run",
                kind=RUN_KIND_OFFICIAL,
                case_id="case",
                model="qwen3tts",
                revision=REV_OFFICIAL,
                exit_status=0,
                artifacts=[("prompt", "prompt.txt", b"p"), ("token", "tokens.bin", b"t")],
            )
            manifest = json.loads(run_path.read_text(encoding="utf-8"))
            manifest["artifacts"][1]["sha256"] = "0" * 64
            run_path.write_text(json.dumps(manifest), encoding="utf-8")
            code, _, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--run",
                str(run_path.relative_to(root)),
                "--require-kind",
                "token",
                "--output",
                "report.json",
                cwd=root,
            )
            self.assertNotEqual(code, 0)
            self.assertTrue(err.startswith("BASELINE_INPUT"), err)

    def test_full_coverage_passes_for_complete_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _build_corpus(
                root,
                cases=[
                    {"id": "a", "language": "en", "text": "a"},
                    {"id": "b", "language": "en", "text": "b"},
                ],
            )
            runs = [
                _build_manifest(
                    root / f"official_{case}",
                    kind=RUN_KIND_OFFICIAL,
                    case_id=case,
                    model="qwen3tts",
                    revision=REV_OFFICIAL,
                    exit_status=0,
                    artifacts=[
                        ("prompt", "prompt.txt", b"p"),
                        ("token", "tokens.bin", b"t"),
                        ("tensor", "tensor.bin", b"\x00\x00\x00\x00"),
                        ("audio", "audio.wav", b"wav"),
                    ],
                )
                for case in ("a", "b")
            ]
            runs.extend(
                _build_manifest(
                    root / f"qwen_{case}",
                    kind=RUN_KIND_QWEN,
                    case_id=case,
                    model="qwen3tts",
                    revision=REV_QWEN,
                    exit_status=0,
                    artifacts=[
                        ("prompt", "prompt.txt", b"p"),
                        ("token", "tokens.bin", b"t"),
                        ("tensor", "tensor.bin", b"\x00\x00\x00\x00"),
                        ("audio", "audio.wav", b"wav"),
                    ],
                )
                for case in ("a", "b")
            )
            args = [
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--require-adapter",
                RUN_KIND_OFFICIAL,
                "--require-adapter",
                RUN_KIND_QWEN,
                "--require-kind",
                "prompt",
                "--require-kind",
                "token",
                "--require-kind",
                "tensor",
                "--require-kind",
                "audio",
                "--require-case",
                "a",
                "--require-case",
                "b",
            ]
            for run in runs:
                args.extend(["--run", str(run.relative_to(root))])
            args.extend(["--output", "report.json"])
            code, out, err = _run_script(*args, cwd=root)
            self.assertEqual(code, 0, err)
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(
                report["coverage"],
                {
                    "prompt": {"required": 2, "present": 2, "missing": 0, "status": "PASS"},
                    "token": {"required": 2, "present": 2, "missing": 0, "status": "PASS"},
                    "tensor": {"required": 2, "present": 2, "missing": 0, "status": "PASS"},
                    "audio": {"required": 2, "present": 2, "missing": 0, "status": "PASS"},
                },
            )

    def test_split_artifact_coverage_passes_with_adapter_partition(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _build_corpus(
                root,
                cases=[{"id": "case", "language": "en", "text": "hello"}],
            )
            official = _build_manifest(
                root / "official",
                kind=RUN_KIND_OFFICIAL,
                case_id="case",
                model="qwen3tts",
                revision=REV_OFFICIAL,
                exit_status=0,
                artifacts=[("prompt", "prompt.txt", b"p"), ("token", "tokens.bin", b"t")],
            )
            qwen = _build_manifest(
                root / "qwen",
                kind=RUN_KIND_QWEN,
                case_id="case",
                model="qwen3tts",
                revision=REV_QWEN,
                exit_status=0,
                artifacts=[
                    ("prompt", "prompt.txt", b"p"),
                    ("tensor", "tensor.bin", b"\x00\x00\x00\x00"),
                    ("audio", "audio.wav", b"wav"),
                ],
            )
            code, out, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--require-adapter",
                RUN_KIND_OFFICIAL,
                "--require-adapter",
                RUN_KIND_QWEN,
                "--require-case",
                "case",
                "--require-kind",
                "prompt",
                "--require-kind",
                "token",
                "--require-kind",
                "tensor",
                "--require-kind",
                "audio",
                "--run",
                str(official.relative_to(root)),
                "--run",
                str(qwen.relative_to(root)),
                "--output",
                "report.json",
                cwd=root,
            )
            self.assertEqual(code, 0, err)
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(
                report["coverage"],
                {
                    "prompt": {"required": 1, "present": 1, "missing": 0, "status": "PASS"},
                    "token": {"required": 1, "present": 1, "missing": 0, "status": "PASS"},
                    "tensor": {"required": 1, "present": 1, "missing": 0, "status": "PASS"},
                    "audio": {"required": 1, "present": 1, "missing": 0, "status": "PASS"},
                },
            )

    def test_stderr_artifacts_are_retained_and_schema_validates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _build_corpus(
                root,
                cases=[{"id": "case", "language": "en", "text": "hello"}],
            )
            official = _build_manifest(
                root / "official",
                kind=RUN_KIND_OFFICIAL,
                case_id="case",
                model="qwen3tts",
                revision=REV_OFFICIAL,
                exit_status=0,
                artifacts=[
                    ("prompt", "prompt.txt", b"p"),
                    ("token", "tokens.bin", b"t"),
                    ("stderr", "stderr.log", b"oops"),
                ],
            )
            qwen = _build_manifest(
                root / "qwen",
                kind=RUN_KIND_QWEN,
                case_id="case",
                model="qwen3tts",
                revision=REV_QWEN,
                exit_status=0,
                artifacts=[
                    ("prompt", "prompt.txt", b"p"),
                    ("tensor", "tensor.bin", b"\x00\x00\x00\x00"),
                    ("audio", "audio.wav", b"wav"),
                    ("stderr", "stderr.log", b"oops2"),
                ],
            )
            code, out, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--require-adapter",
                RUN_KIND_OFFICIAL,
                "--require-adapter",
                RUN_KIND_QWEN,
                "--require-case",
                "case",
                "--require-kind",
                "prompt",
                "--require-kind",
                "token",
                "--require-kind",
                "tensor",
                "--require-kind",
                "audio",
                "--run",
                str(official.relative_to(root)),
                "--run",
                str(qwen.relative_to(root)),
                "--output",
                "report.json",
                cwd=root,
            )
            self.assertEqual(code, 0, err)
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(
                report["coverage"],
                {
                    "prompt": {"required": 1, "present": 1, "missing": 0, "status": "PASS"},
                    "token": {"required": 1, "present": 1, "missing": 0, "status": "PASS"},
                    "tensor": {"required": 1, "present": 1, "missing": 0, "status": "PASS"},
                    "audio": {"required": 1, "present": 1, "missing": 0, "status": "PASS"},
                },
            )
            for artifact_kinds in [run["artifacts"] for run in report["runs"]]:
                self.assertIn("stderr", {artifact["kind"] for artifact in artifact_kinds})
            if _validate_report_schema(report):
                self.assertEqual(report["schema_version"], 1)
                self.assertEqual(report["backend"], "CPU/F32")
                self.assertIn("runs", report)
                self.assertIn("coverage", report)
                self.assertIn("status", report)
                return
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["backend"], "CPU/F32")
            self.assertIn("runs", report)
            self.assertIn("coverage", report)
            self.assertIn("status", report)

    def test_missing_kind_within_case_causes_coverage_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _build_corpus(
                root,
                cases=[{"id": "case", "language": "en", "text": "hello"}],
            )
            official = _build_manifest(
                root / "official",
                kind=RUN_KIND_OFFICIAL,
                case_id="case",
                model="qwen3tts",
                revision=REV_OFFICIAL,
                exit_status=0,
                artifacts=[("prompt", "prompt.txt", b"p"), ("token", "tokens.bin", b"t")],
            )
            qwen = _build_manifest(
                root / "qwen",
                kind=RUN_KIND_QWEN,
                case_id="case",
                model="qwen3tts",
                revision=REV_QWEN,
                exit_status=0,
                artifacts=[
                    ("prompt", "prompt.txt", b"p"),
                    ("audio", "audio.wav", b"wav"),
                ],
            )
            code, _, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--require-adapter",
                RUN_KIND_OFFICIAL,
                "--require-adapter",
                RUN_KIND_QWEN,
                "--require-case",
                "case",
                "--require-kind",
                "prompt",
                "--require-kind",
                "tensor",
                "--run",
                str(official.relative_to(root)),
                "--run",
                str(qwen.relative_to(root)),
                "--output",
                "report.json",
                cwd=root,
            )
            self.assertNotEqual(code, 0)
            self.assertTrue(err.startswith("BASELINE_COVERAGE"), err)

    def test_missing_required_adapter_fails_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _build_corpus(root, cases=[{"id": "case", "language": "en", "text": "a"}])
            run = _build_manifest(
                root / "official",
                kind=RUN_KIND_OFFICIAL,
                case_id="case",
                model="qwen3tts",
                revision=REV_OFFICIAL,
                exit_status=0,
                artifacts=[("prompt", "prompt.txt", b"p"), ("token", "tokens.bin", b"t")],
            )
            code, _, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--require-adapter",
                RUN_KIND_QWEN,
                "--require-kind",
                "prompt",
                "--run",
                str(run.relative_to(root)),
                "--output",
                "report.json",
                cwd=root,
            )
            self.assertNotEqual(code, 0)
            self.assertTrue(err.startswith("BASELINE_COVERAGE"), err)

    def test_non_zero_reference_run_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _build_corpus(root, cases=[{"id": "case", "language": "en", "text": "a"}])
            run = _build_manifest(
                root / "run",
                kind=RUN_KIND_OFFICIAL,
                case_id="case",
                model="qwen3tts",
                revision=REV_OFFICIAL,
                exit_status=2,
                artifacts=[("prompt", "prompt.txt", b"p"), ("token", "tokens.bin", b"t")],
            )
            code, _, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--run",
                str(run.relative_to(root)),
                "--output",
                "report.json",
                cwd=root,
            )
            self.assertNotEqual(code, 0)
            self.assertTrue(err.startswith("BASELINE_INPUT"), err)

    def test_duplicate_identity_and_empty_required_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _build_corpus(root, cases=[{"id": "case", "language": "en", "text": "a"}])
            first = _build_manifest(
                root / "first",
                kind=RUN_KIND_OFFICIAL,
                case_id="case",
                model="qwen3tts",
                revision=REV_OFFICIAL,
                exit_status=0,
                artifacts=[("prompt", "prompt.txt", b"p"), ("token", "tokens.bin", b"t")],
            )
            second = _build_manifest(
                root / "second",
                kind=RUN_KIND_OFFICIAL,
                case_id="case",
                model="qwen3tts",
                revision=REV_OFFICIAL,
                exit_status=0,
                artifacts=[("prompt", "prompt2.txt", b"p"), ("token", "tokens2.bin", b"t")],
            )
            code, _, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--run",
                str(first.relative_to(root)),
                "--run",
                str(second.relative_to(root)),
                "--require-kind",
                "prompt",
                "--output",
                "report.json",
                cwd=root,
            )
            self.assertNotEqual(code, 0)
            self.assertTrue(err.startswith("BASELINE_INPUT"), err)

            missing_run = _build_manifest(
                root / "empty_token",
                kind=RUN_KIND_QWEN,
                case_id="case",
                model="qwen3tts",
                revision=REV_QWEN,
                exit_status=0,
                artifacts=[("prompt", "prompt.txt", b"p"), ("token", "tokens.bin", b"")],
            )
            code, _, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--require-adapter",
                RUN_KIND_QWEN,
                "--require-kind",
                "token",
                "--run",
                str(missing_run.relative_to(root)),
                "--output",
                "report2.json",
                cwd=root,
            )
            self.assertNotEqual(code, 0)
            self.assertTrue(err.startswith("BASELINE_INPUT"), err)

    def test_unsafe_and_absolute_paths_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _build_corpus(root, cases=[{"id": "case", "language": "en", "text": "a"}])
            run_path = _build_manifest(
                root / "run",
                kind=RUN_KIND_OFFICIAL,
                case_id="case",
                model="qwen3tts",
                revision=REV_OFFICIAL,
                exit_status=0,
                artifacts=[("prompt", "../escape.txt", b"p"), ("token", "tokens.bin", b"t")],
            )
            code, _, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--run",
                str(run_path.relative_to(root)),
                "--output",
                "report.json",
                cwd=root,
            )
            self.assertNotEqual(code, 0)
            self.assertTrue(err.startswith("BASELINE_INPUT"), err)

            run_path = _build_manifest(
                root / "run2",
                kind=RUN_KIND_OFFICIAL,
                case_id="case",
                model="qwen3tts",
                revision=REV_OFFICIAL,
                exit_status=0,
                artifacts=[("prompt", "prompt.txt", b"p"), ("token", "tokens.bin", b"t")],
            )
            manifest = json.loads(run_path.read_text(encoding="utf-8"))
            manifest["artifacts"][1]["relative_path"] = os.path.abspath("tokens.bin")
            manifest["artifacts"][1]["byte_length"] = 1
            run_path.write_text(json.dumps(manifest), encoding="utf-8")
            code, _, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--run",
                str(run_path.relative_to(root)),
                "--require-kind",
                "prompt",
                "--output",
                "report2.json",
                cwd=root,
            )
            self.assertNotEqual(code, 0)
            self.assertTrue(err.startswith("BASELINE_INPUT"), err)

    def test_missing_file_and_output_overwrite_refused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _build_corpus(root, cases=[{"id": "case", "language": "en", "text": "a"}])
            run_dir = root / "run"
            manifest = {
                "schema_version": 1,
                "kind": RUN_KIND_OFFICIAL,
                "source": RUN_KIND_OFFICIAL,
                "revision": REV_OFFICIAL,
                "model": "qwen3tts",
                "case_id": "case",
                "seed": 1,
                "argv": [],
                "exit_status": 0,
                "artifacts": [
                    {"kind": "prompt", "relative_path": "prompt.txt", "byte_length": 1, "sha256": _sha256(b"a")},
                ],
                "stages": [],
            }
            run_path = run_dir / "reference-run.json"
            run_dir.mkdir(parents=True)
            run_path.write_text(json.dumps(manifest), encoding="utf-8")
            code, _, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--run",
                str(run_path.relative_to(root)),
                "--output",
                "report.json",
                cwd=root,
            )
            self.assertNotEqual(code, 0)
            self.assertTrue(err.startswith("BASELINE_INPUT"), err)

            valid_run = _build_manifest(
                root / "run_valid",
                kind=RUN_KIND_OFFICIAL,
                case_id="case",
                model="qwen3tts",
                revision=REV_OFFICIAL,
                exit_status=0,
                artifacts=[("prompt", "prompt.txt", b"p"), ("token", "tokens.bin", b"t")],
            )
            locked = root / "report.json"
            locked.write_text("{}")
            code, _, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--run",
                str(valid_run.relative_to(root)),
                "--output",
                "report.json",
                cwd=root,
            )
            self.assertNotEqual(code, 0)
            self.assertTrue(err.startswith("BASELINE_OUTPUT"), err)

    def test_text_file_case_and_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = _build_corpus(
                root,
                cases=[{"id": "case", "language": "en", "text_file": "texts/en.txt"}],
            )
            run = _build_manifest(
                root / "run",
                kind=RUN_KIND_OFFICIAL,
                case_id="case",
                model="qwen3tts",
                revision=REV_OFFICIAL,
                exit_status=0,
                artifacts=[("prompt", "prompt.txt", b"p"), ("token", "tokens.bin", b"t")],
            )
            code, out, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--require-kind",
                "prompt",
                "--require-kind",
                "token",
                "--run",
                str(run.relative_to(root)),
                "--dry-run",
                "--output",
                "report.json",
                cwd=root,
            )
            self.assertEqual(code, 0, err)
            report = json.loads(out)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(len(report["corpus"]["cases"]), 1)
            self.assertFalse((root / "report.json").exists())

    def test_malformed_corpus_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = root / "bad.json"
            corpus.write_text(json.dumps({"cases": [{"id": "bad", "language": "", "text": "x"}]}), encoding="utf-8")
            code, _, err = _run_script(
                "--repo-root",
                str(root),
                "--corpus",
                str(corpus.relative_to(root)),
                "--target-revision",
                "target",
                "--reference-revision",
                "reference",
                "--official-revision",
                "official",
                "--run",
                str(root / "missing.json"),
                "--output",
                "report.json",
                cwd=root,
            )
            self.assertNotEqual(code, 0)
            self.assertTrue(err.startswith("BASELINE_INPUT"), err)


if __name__ == "__main__":
    unittest.main()
