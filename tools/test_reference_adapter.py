#!/usr/bin/env python3
"""Unit tests for tools/reference_adapter.py."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import reference_adapter

try:
    from jsonschema import validate as validate_json
except Exception:  # pragma: no cover
    validate_json = None


SCRIPT = Path(__file__).with_name("reference_adapter.py").resolve()
SCHEMA = Path(__file__).parent.parent / "schemas" / "reference-run.schema.json"
OFFICIAL_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
QWEN_REVISION = "82cd05b9f3a175612dc89fd6943e610fab096ef5"


def run_adapter(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
    )


def make_fake_qwen_executable(root: Path, script_body: str) -> Path:
    script = root / "qwentts_fake.py"
    script.write_text(script_body, encoding="utf-8")
    if os.name == "nt":
        wrapper = root / "qwentts_fake.cmd"
        wrapper.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
            encoding="utf-8",
        )
        return wrapper

    wrapper = root / "qwentts_fake.sh"
    wrapper.write_text(f"#!/usr/bin/env sh\n\"{sys.executable}\" \"{script}\" \"$@\"\n", encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_manifest(output_dir: Path) -> dict:
    return json.loads((output_dir / "reference-run.json").read_text(encoding="utf-8"))


class _TestBase(unittest.TestCase):
    def run_official(
        self,
        root: Path,
        runner_script: str,
        *,
        output_dir: Path | None = None,
        extra_args: list[str] | None = None,
        text: str = "你好，世界",
        text_file: Path | None = None,
        artifacts: list[tuple[str, str]] | None = None,
        case_id: str = "case-official",
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        runner = root / "official_runner.py"
        runner.write_text(runner_script, encoding="utf-8")
        out_dir = output_dir or (root / "session")

        args = [
            "official-python",
            "--python",
            sys.executable,
            "--runner",
            str(runner),
            "--source-revision",
            OFFICIAL_REVISION,
            "--model",
            "Qwen3TTS",
            "--case-id",
            case_id,
            "--output-dir",
            str(out_dir),
        ]

        if text_file is not None:
            args.extend(["--text-file", str(text_file)])
        else:
            args.extend(["--text", text])

        for kind, rel in artifacts or [("prompt", "prompt.txt"), ("token", "tokens.bin")]:
            args.extend(["--artifact", f"{kind}={rel}"])

        if extra_args:
            args.extend(extra_args)

        return run_adapter(*args, cwd=root), out_dir

    def run_qwentts(
        self,
        root: Path,
        qwen_script: str,
        *,
        output_dir: Path | None = None,
        extra_args: list[str] | None = None,
        text: str = "你好，世界",
        text_file: Path | None = None,
        artifacts: list[tuple[str, str]] | None = None,
        case_id: str = "case-qwen",
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        exe = make_fake_qwen_executable(root, qwen_script)
        out_dir = output_dir or (root / "session")
        args = [
            "qwentts-cpp",
            "--executable",
            str(exe),
            "--source-revision",
            QWEN_REVISION,
            "--model",
            "Qwen3TTS",
            "--codec",
            "12hz",
            "--case-id",
            case_id,
            "--output-dir",
            str(out_dir),
        ]

        if text_file is not None:
            args.extend(["--text-file", str(text_file)])
        else:
            args.extend(["--text", text])

        for kind, rel in artifacts or [("prompt", "prompt.txt"), ("audio", "output.wav"), ("tensor", "tensors")]:
            args.extend(["--artifact", f"{kind}={rel}"])

        if extra_args:
            args.extend(extra_args)

        return run_adapter(*args, cwd=root), out_dir


class ReferenceAdapterArgumentTests(_TestBase):
    def test_official_exact_argv_and_stdout_capture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = root / "argv.json"
            script = r'''
import json
import sys
from pathlib import Path

argv_path = None
if "--emit-argv" in sys.argv:
    idx = sys.argv.index("--emit-argv")
    argv_path = Path(sys.argv[idx + 1])

if argv_path is not None:
    argv_path.write_text(json.dumps(sys.argv[1:]), encoding="utf-8")

sys.stdout.buffer.write(b"\x01\x00\x02\x00")
'''
            result, out_dir = self.run_official(
                root,
                script,
                extra_args=["--extra-arg=--emit-argv", f"--extra-arg={plan}"],
                artifacts=[("prompt", "prompt.txt"), ("stdout", "tokens.bin")],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            argv = json.loads(plan.read_text(encoding="utf-8"))
            self.assertEqual(argv[0], "--model")
            self.assertIn("--text", argv)
            manifest = read_manifest(out_dir)
            self.assertEqual(manifest["argv"][0], sys.executable)
            self.assertEqual(manifest["argv"][1], str(root / "official_runner.py"))
            kinds = [item["kind"] for item in manifest["artifacts"]]
            self.assertEqual(kinds, ["prompt", "stdout"])
            self.assertEqual(manifest["artifacts"][1]["relative_path"], "tokens.bin")
            self.assertEqual(manifest["artifacts"][1]["byte_length"], 4)
            self.assertEqual(manifest["artifacts"][1]["sha256"], sha256(b"\x01\x00\x02\x00"))
            self.assertIn("stages", manifest)
            if validate_json is not None:
                validate_json(manifest, json.loads(SCHEMA.read_text(encoding="utf-8")))

    def test_official_declares_audio_and_tensor_auto_args(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = r'''
import os
import struct
import sys
from pathlib import Path

if "--output-wav" in sys.argv:
    wav_path = Path(sys.argv[sys.argv.index("--output-wav") + 1])
    wav_path.write_bytes(b"audio")

if "--dump" in sys.argv:
    dump_dir = Path(sys.argv[sys.argv.index("--dump") + 1])
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.joinpath("codes.bin").write_bytes(
        struct.pack("<i", 1) + struct.pack("<i", 2) + struct.pack("<ff", 1.0, 2.0)
    )

sys.stdout.buffer.write(b"\x01\x00\x00\x00")
'''
            result, out_dir = self.run_official(
                root,
                script,
                artifacts=[("prompt", "prompt.txt"), ("stdout", "tokens.bin"), ("audio", "final.wav"), ("tensor", "tensor_dir")],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = read_manifest(out_dir)
            self.assertIn("audio", [artifact["kind"] for artifact in manifest["artifacts"]])
            self.assertIn("tensor", [artifact["kind"] for artifact in manifest["artifacts"]])
            self.assertIn("stages", manifest)
            self.assertTrue((out_dir / "tensor_dir" / "manifest.json").is_file())
            manifest_json = json.loads((out_dir / "tensor_dir" / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("schema_version", manifest_json)
            self.assertEqual(manifest_json["source"], "official-python")
            self.assertEqual(manifest_json["revision"], OFFICIAL_REVISION)
            self.assertEqual(manifest_json["seed"], None)
            stage_entries = manifest_json["stages"]
            self.assertEqual(len(stage_entries), 1)
            stage = stage_entries[0]
            self.assertEqual(stage["dtype"], "f32")
            self.assertEqual(stage["layout"], "C")
            self.assertEqual(stage["shape"], [2])
            stage_path = out_dir / "tensor_dir" / stage["file"]
            stage_data = stage_path.read_bytes()
            self.assertEqual(stage["sha256"], sha256(stage_data))


    def test_qwen_cpp_exact_argv_and_utf8_stdin(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stdin_path = root / "stdin.bin"
            script = '''
import struct
import sys
from pathlib import Path

text = sys.stdin.buffer.read()
Path("stdin.bin").write_bytes(text)

dump_dir = Path(sys.argv[sys.argv.index("--dump") + 1])
dump_dir.mkdir(parents=True, exist_ok=True)
dump_dir.joinpath("codes.bin").write_bytes(
    struct.pack("<i", 1) + struct.pack("<i", 2) + struct.pack("<ff", 1.0, 2.0)
)
audio_path = Path(sys.argv[sys.argv.index("-o") + 1])
audio_path.write_bytes(b"audio")
'''
            result, out_dir = self.run_qwentts(root, script)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(stdin_path.read_bytes(), "你好，世界".encode("utf-8"))
            manifest = read_manifest(out_dir)
            self.assertEqual(manifest["kind"], "qwentts.cpp")
            self.assertIn("audio", [artifact["kind"] for artifact in manifest["artifacts"]])
            tensor = [artifact for artifact in manifest["artifacts"] if artifact["kind"] == "tensor"][0]
            self.assertEqual(tensor["shape"], [2])
            self.assertEqual(tensor["dtype"], "f32")
            self.assertIn("stages", manifest)
            tensor_manifest_path = out_dir / "tensors" / "manifest.json"
            self.assertTrue(tensor_manifest_path.is_file())
            tensor_manifest = json.loads(tensor_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(tensor_manifest["source"], "qwentts.cpp")
            self.assertEqual(tensor_manifest["revision"], QWEN_REVISION)
            stage = tensor_manifest["stages"][0]
            self.assertEqual(stage["dtype"], "f32")
            self.assertEqual(stage["layout"], "C")
            self.assertEqual(stage["shape"], [2])
            tensor_file = out_dir / "tensors" / stage["file"]
            self.assertEqual(tensor_file.read_bytes(), struct.pack("<ff", 1.0, 2.0))
            self.assertEqual(stage["sha256"], sha256(tensor_file.read_bytes()))


class ReferenceAdapterFailureTests(_TestBase):
    def test_output_dir_refused_when_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "session"
            out_dir.mkdir()
            script = "import sys\nsys.exit(0)"
            result, _ = self.run_official(root, script, output_dir=out_dir)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr.startswith("REFERENCE_OUTPUT"))

    def test_missing_inputs_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = run_adapter(
                "official-python",
                "--python",
                sys.executable,
                "--runner",
                str(root / "missing.py"),
                "--source-revision",
                OFFICIAL_REVISION,
                "--model",
                "Qwen3TTS",
                "--case-id",
                "case-official",
                "--output-dir",
                str(root / "out"),
                "--text",
                "hello",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr.startswith("REFERENCE_CONFIG"))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = run_adapter(
                "qwentts-cpp",
                "--executable",
                str(root / "missing.exe"),
                "--source-revision",
                QWEN_REVISION,
                "--model",
                "Qwen3TTS",
                "--codec",
                "12hz",
                "--case-id",
                "case-qwen",
                "--output-dir",
                str(root / "out"),
                "--text",
                "hello",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr.startswith("REFERENCE_CONFIG"))

    def test_child_nonzero_exit_propagates_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = "import sys; print('boom', file=sys.stderr); sys.exit(3)"
            result, out_dir = self.run_official(root, script)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr.startswith("REFERENCE_EXEC"))
            self.assertIn("boom", result.stderr)
            manifest = read_manifest(out_dir)
            self.assertEqual(manifest["exit_status"], 3)
            self.assertEqual(manifest["kind"], "official-python")
            artifact_paths = {item["kind"]: item for item in manifest["artifacts"]}
            self.assertIn("stderr", artifact_paths)
            self.assertEqual(artifact_paths["stderr"]["relative_path"], "logs/stderr.txt")
            stderr_bytes = (out_dir / "logs" / "stderr.txt").read_bytes()
            self.assertEqual(stderr_bytes.decode("utf-8").strip(), "boom")
            self.assertEqual(artifact_paths["stderr"]["byte_length"], len(stderr_bytes))
            self.assertEqual(
                artifact_paths["stderr"]["sha256"],
                sha256(stderr_bytes),
            )

    def test_official_declared_outputs_missing_fail(self) -> None:
        script = "import sys\nsys.exit(0)"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result, _ = self.run_official(
                root,
                script,
                artifacts=[("prompt", "prompt.txt"), ("stdout", "tokens.bin"), ("audio", "audio.wav")],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr.startswith("REFERENCE_OUTPUT"))

    def test_qwen_cpp_declared_audio_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = '''
import struct
import sys
from pathlib import Path
dump_dir = Path(sys.argv[sys.argv.index("--dump") + 1])
dump_dir.mkdir(parents=True, exist_ok=True)
dump_dir.joinpath("codes.bin").write_bytes(struct.pack("<i", 1) + struct.pack("<i", 1) + struct.pack("<f", 0.5))
'''
            result, _ = self.run_qwentts(
                root,
                script,
                artifacts=[("audio", "no-audio.wav"), ("prompt", "prompt.txt"), ("tensor", "tensors")],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr.startswith("REFERENCE_OUTPUT"))

    def test_qwentts_cpp_requires_tensor_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = "import sys; from pathlib import Path; audio = Path(sys.argv[sys.argv.index('-o') + 1]); audio.write_bytes(b'audio')"
            result, _ = self.run_qwentts(
                root,
                script,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr.startswith("REFERENCE_OUTPUT"))

    def test_invalid_tensor_headers_are_rejected(self) -> None:
        cases: list[tuple[str, bytes]] = [
            ("truncated", b"\x01\x00\x00\x00"),
            ("shape-missing", struct.pack("<i", 1) + struct.pack("<i", 2)),
            ("trailing", struct.pack("<i", 1) + struct.pack("<i", 1) + struct.pack("<f", 1.0) + b"\x00"),
            ("overflow", struct.pack("<i", 1) + struct.pack("<i", 2_147_483_647)),
        ]

        for name, payload in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    script = f'''
import struct
import sys
from pathlib import Path
dump_dir = Path(sys.argv[sys.argv.index("--dump") + 1])
dump_dir.mkdir(parents=True, exist_ok=True)
dump_dir.joinpath("bad.bin").write_bytes({payload!r})
audio_path = Path(sys.argv[sys.argv.index("-o") + 1])
audio_path.write_bytes(b"audio")
'''
                    result, _ = self.run_qwentts(
                        root,
                        script,
                        artifacts=[("audio", "audio.wav"), ("tensor", "tensors")],
                    )
                    self.assertNotEqual(result.returncode, 0, name)
                    self.assertTrue(result.stderr.startswith("REFERENCE_FORMAT"), result.stderr)

    def test_zero_dim_dump_is_accepted_when_tensor_shape_has_zero(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = """
import struct
import sys
from pathlib import Path

dump_dir = Path(sys.argv[sys.argv.index(\"--dump\") + 1])
dump_dir.mkdir(parents=True, exist_ok=True)
dump_dir.joinpath(\"codes.bin\").write_bytes(
    struct.pack(\"<i\", 2) + struct.pack(\"<i\", 2) + struct.pack(\"<i\", 0) + b\"\"
)
audio_path = Path(sys.argv[sys.argv.index(\"-o\") + 1])
audio_path.write_bytes(b\"audio\")
"""
            result, out_dir = self.run_qwentts(
                root,
                script,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = read_manifest(out_dir)
            tensor = next(item for item in manifest["artifacts"] if item["kind"] == "tensor")
            self.assertEqual(tensor["shape"], [2, 0])
            self.assertEqual(tensor["byte_length"], 0)
            self.assertEqual(tensor["dtype"], "f32")
            tensor_manifest = json.loads((out_dir / "tensors" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(tensor_manifest["stages"][0]["shape"], [2, 0])

    def test_reserved_extra_arg_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = "import sys\nsys.exit(0)"
            result, _ = self.run_official(
                root,
                script,
                extra_args=["--extra-arg=--dump=evil"],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr.startswith("REFERENCE_CONFIG"))

            result, _ = self.run_official(
                root,
                script,
                extra_args=["--extra-arg=-oevil.wav"],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr.startswith("REFERENCE_CONFIG"))

            result, _ = self.run_official(
                root,
                script,
                extra_args=["--extra-arg=--api-key=do-not-record"],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr.startswith("REFERENCE_CONFIG"))


class ReferenceAdapterSafetyTests(_TestBase):
    def test_symlink_guards_are_enforced_without_os_symlink_privilege(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session = root / "session"
            dump_dir = session / "raw_tensor_dump"
            dump_dir.mkdir(parents=True)
            dump_file = dump_dir / "codes.bin"
            dump_file.write_bytes(
                struct.pack("<i", 1)
                + struct.pack("<i", 1)
                + struct.pack("<f", 1.0)
            )

            original_is_symlink = Path.is_symlink

            def fake_is_symlink(path: Path) -> bool:
                if path == dump_file:
                    return True
                return original_is_symlink(path)

            with mock.patch.object(
                Path, "is_symlink", autospec=True, side_effect=fake_is_symlink
            ):
                with self.assertRaises(reference_adapter.RefAdapterError) as raised:
                    reference_adapter._copy_tensor_artifacts(
                        [],
                        session,
                        Path("tensors"),
                        dump_dir,
                        source="qwentts.cpp",
                        revision=QWEN_REVISION,
                        model="Qwen3TTS",
                        case_id="symlink-guard",
                        seed=1,
                        require_tensor=True,
                    )
            self.assertEqual(raised.exception.prefix, "REFERENCE_OUTPUT")

    def test_duplicate_or_unsafe_artifact_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = "import sys\nsys.stdout.buffer.write(b'1234')"
            result, _ = self.run_official(
                root,
                script,
                artifacts=[("prompt", "../escape.txt"), ("stdout", "tokens.bin")],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr.startswith("REFERENCE_OUTPUT"))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = "import sys\nsys.stdout.buffer.write(b'1234')"
            result, _ = self.run_official(
                root,
                script,
                artifacts=[("prompt", "shared.txt"), ("stdout", "shared.txt")],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr.startswith("REFERENCE_OUTPUT"))

    def test_dry_run_no_execution_no_io(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = "import sys\nPath = __import__('pathlib').Path\nPath('must-never-create').write_text('x')"
            runner = root / "official_runner.py"
            runner.write_text(script, encoding="utf-8")
            out_dir = root / "session"
            result = run_adapter(
                "official-python",
                "--dry-run",
                "--python",
                sys.executable,
                "--runner",
                str(runner),
                "--source-revision",
                OFFICIAL_REVISION,
                "--model",
                "Qwen3TTS",
                "--case-id",
                "case-dryrun",
                "--output-dir",
                str(out_dir),
                "--text",
                "dry-run",
                "--artifact",
                "prompt=prompt.txt",
                "--artifact",
                "stdout=tokens.bin",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(out_dir.exists())
            self.assertNotIn("must-never-create", result.stdout)

    def test_symlink_artifact_and_tensor_outputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            probe_target = root / "probe_target.txt"
            probe_link = root / "probe_link.txt"
            probe_target.write_text("x", encoding="utf-8")
            try:
                os.symlink(str(probe_target), str(probe_link))
            except OSError as exc:
                self.skipTest(f"symlink not supported in this environment: {exc}")
            probe_link.unlink()

            script_audio = """
import os
import sys
from pathlib import Path

audio_path = Path(sys.argv[sys.argv.index(\"-o\") + 1])
audio_path.parent.mkdir(parents=True, exist_ok=True)
target = Path("audio_real.wav")
target.write_bytes(b\"audio-real\")
audio_path.symlink_to(target)
"""
            result, _ = self.run_qwentts(
                root,
                script_audio,
                artifacts=[("audio", "audio.wav"), ("prompt", "prompt.txt"), ("tensor", "tensors")],
            )
            self.assertNotEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stderr.startswith("REFERENCE_OUTPUT"))

            script_tensor = """
import os
import sys
import struct
from pathlib import Path

dump_dir = Path(sys.argv[sys.argv.index(\"--dump\") + 1])
dump_dir.mkdir(parents=True, exist_ok=True)
target = dump_dir / \"codes.bin.real\"
target.write_bytes(struct.pack(\"<i\", 1) + struct.pack(\"<i\", 1) + struct.pack(\"<f\", 1.0))
link = dump_dir / \"codes.bin\"
link.symlink_to(target)
audio_path = Path(sys.argv[sys.argv.index(\"-o\") + 1])
audio_path.write_bytes(b\"audio\")
"""
            result, _ = self.run_qwentts(
                root,
                script_tensor,
                artifacts=[("prompt", "prompt.txt"), ("audio", "output.wav"), ("tensor", "tensors")],
            )
            self.assertNotEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stderr.startswith("REFERENCE_OUTPUT"))

            script_dump_dir = """
import os
import sys
from pathlib import Path

dump_dir = Path(sys.argv[sys.argv.index(\"--dump\") + 1])
target_root = dump_dir.parent
target = target_root / \"real_dump\"
target.mkdir(parents=True, exist_ok=True)
dump_dir.rmdir()
dump_dir.symlink_to(target, target_is_directory=True)
"""
            result, _ = self.run_qwentts(
                root,
                script_dump_dir,
                artifacts=[("prompt", "prompt.txt"), ("audio", "output.wav"), ("tensor", "tensors")],
            )
            self.assertNotEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stderr.startswith("REFERENCE_OUTPUT"))

    def test_declared_stderr_artifact_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = "import sys; print('err', file=sys.stderr); sys.stdout.buffer.write(b'1234')"
            result, out_dir = self.run_official(
                root,
                script,
                artifacts=[("prompt", "prompt.txt"), ("stdout", "tokens.bin"), ("stderr", "error.log")],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = read_manifest(out_dir)
            artifact = next(a for a in manifest["artifacts"] if a["kind"] == "stderr")
            self.assertEqual(artifact["relative_path"], "error.log")
            self.assertEqual((out_dir / "error.log").read_text(encoding="utf-8").strip(), "err")


class ReferenceAdapterManifestTests(_TestBase):
    def test_manifest_fields_are_valid_and_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = "import sys\nsys.stdout.buffer.write(b'\\x01\\x00\\x00\\x00')"
            result, out_dir = self.run_official(
                root,
                script,
                artifacts=[("prompt", "zz-prompt.txt"), ("stdout", "aa-token.bin")],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = read_manifest(out_dir)
            if validate_json is not None:
                validate_json(manifest, json.loads(SCHEMA.read_text(encoding="utf-8")))
            artifact_paths = [item["relative_path"] for item in manifest["artifacts"]]
            self.assertEqual(artifact_paths, sorted(artifact_paths))
            self.assertEqual(manifest["kind"], "official-python")
            self.assertEqual(manifest["source"], "official-python")
            self.assertIn("stages", manifest)
            self.assertIsInstance(manifest["stages"], list)
            self.assertIn("seed", manifest)
            self.assertFalse((out_dir / "logs" / "stdout.bin").exists())
            self.assertFalse((out_dir / "logs" / "stderr.txt").exists())


if __name__ == "__main__":
    unittest.main()
