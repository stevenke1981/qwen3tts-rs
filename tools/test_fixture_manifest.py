import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("fixture_manifest.py")


def _run_manifest(manifest: Path, *extra: str):
    cmd = [sys.executable, str(SCRIPT), str(manifest), *extra]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestFixtureManifest(unittest.TestCase):
    def test_valid_manifest_and_matching_hash_passes(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            fixture = root / "fixture.bin"
            fixture.write_bytes(b"qwen3tts")
            manifest = {
                "version": 1,
                "fixtures": [
                    {
                        "id": "base_model",
                        "path": "fixture.bin",
                        "source": "local",
                        "revision": "dev",
                        "sha256": _sha256(b"qwen3tts"),
                        "license": "test",
                        "generated_by": "unittest",
                        "command": "echo test",
                        "required": True,
                    }
                ],
            }
            manifest_path = root / "fixtures.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            code, out = _run_manifest(manifest_path)
            self.assertEqual(code, 0)
            self.assertIn("FIXTURE_VERIFIED", out)

    def test_relative_path_resolves_from_manifest_directory(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            nested = root / "nested"
            nested.mkdir()
            fixture = nested / "artifact.bin"
            fixture.write_bytes(b"relative")
            manifest = {
                "version": 1,
                "fixtures": [
                    {
                        "id": "relative",
                        "path": "artifact.bin",
                        "source": "local",
                        "revision": "rel",
                        "sha256": _sha256(b"relative"),
                        "license": "test",
                        "generated_by": "unittest",
                        "command": "echo test",
                        "required": True,
                    }
                ],
            }
            manifest_path = nested / "fixtures.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            code, out = _run_manifest(manifest_path)
            self.assertEqual(code, 0)
            self.assertIn("FIXTURE_VERIFIED", out)

    def test_empty_required_manifest_fails_with_require_all(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = {"version": 1, "fixtures": []}
            manifest_path = root / "fixtures.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            code, out = _run_manifest(manifest_path, "--require-all")
            self.assertEqual(code, 1)
            self.assertTrue(out.startswith("FIXTURE_MISSING"))

    def test_missing_required_file_fails_with_fixture_missing(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = {
                "version": 1,
                "fixtures": [
                    {
                        "id": "missing",
                        "path": "missing.bin",
                        "source": "local",
                        "revision": "dev",
                        "sha256": _sha256(b"missing"),
                        "license": "test",
                        "generated_by": "unittest",
                        "command": "echo test",
                        "required": True,
                    }
                ],
            }
            manifest_path = root / "fixtures.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            code, out = _run_manifest(manifest_path)
            self.assertEqual(code, 1)
            self.assertEqual(out.split()[0], "FIXTURE_MISSING")

    def test_malformed_metadata_fails_with_fixture_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            fixture = root / "fixture.bin"
            fixture.write_bytes(b"x")
            manifest = {
                "version": 1,
                "fixtures": [
                    {
                        "id": "bad",
                        "path": str(fixture),
                        "source": "local",
                        # missing revision field
                        "sha256": _sha256(b"x"),
                        "license": "test",
                        "generated_by": "unittest",
                        "command": "echo test",
                        "required": True,
                    }
                ],
            }
            manifest_path = root / "fixtures.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            code, out = _run_manifest(manifest_path)
            self.assertEqual(code, 1)
            self.assertEqual(out.split()[0], "FIXTURE_INVALID")

    def test_duplicate_fixture_id_fails(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            fixture_one = root / "a.bin"
            fixture_two = root / "b.bin"
            fixture_one.write_bytes(b"same")
            fixture_two.write_bytes(b"same")
            manifest = {
                "version": 1,
                "fixtures": [
                    {
                        "id": "dup",
                        "path": "a.bin",
                        "source": "local",
                        "revision": "a",
                        "sha256": _sha256(b"same"),
                        "license": "test",
                        "generated_by": "unittest",
                        "command": "echo test",
                        "required": True,
                    },
                    {
                        "id": "dup",
                        "path": "b.bin",
                        "source": "local",
                        "revision": "b",
                        "sha256": _sha256(b"same"),
                        "license": "test",
                        "generated_by": "unittest",
                        "command": "echo test",
                        "required": True,
                    },
                ],
            }
            manifest_path = root / "fixtures.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            code, out = _run_manifest(manifest_path)
            self.assertEqual(code, 1)
            self.assertEqual(out.split()[0], "FIXTURE_INVALID")

    def test_hash_mismatch_fails_with_fixture_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            fixture = root / "fixture.bin"
            fixture.write_bytes(b"good")
            manifest = {
                "version": 1,
                "fixtures": [
                    {
                        "id": "hash",
                        "path": "fixture.bin",
                        "source": "local",
                        "revision": "dev",
                        "sha256": "0" * 64,
                        "license": "test",
                        "generated_by": "unittest",
                        "command": "echo test",
                        "required": True,
                    }
                ],
            }
            manifest_path = root / "fixtures.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            code, out = _run_manifest(manifest_path)
            self.assertEqual(code, 1)
            self.assertEqual(out.split()[0], "FIXTURE_HASH_MISMATCH")

    def test_optional_missing_fixture_is_reported_and_not_failing(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = {
                "version": 1,
                "fixtures": [
                    {
                        "id": "optional_missing",
                        "path": "missing.bin",
                        "source": "local",
                        "revision": "dev",
                        "sha256": _sha256(b"unused"),
                        "license": "test",
                        "generated_by": "unittest",
                        "command": "echo test",
                        "required": False,
                    }
                ],
            }
            manifest_path = root / "fixtures.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            code, out = _run_manifest(manifest_path)
            self.assertEqual(code, 0)
            self.assertIn("FIXTURE_OPTIONAL_MISSING", out)

    def test_absolute_missing_required_fixture_reports_fixture_missing(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = {
                "version": 1,
                "fixtures": [
                    {
                        "id": "abs-missing",
                        "path": str((root / "outside").resolve() / "missing.bin"),
                        "source": "local",
                        "revision": "dev",
                        "sha256": _sha256(b"missing"),
                        "license": "test",
                        "generated_by": "unittest",
                        "command": "echo test",
                        "required": True,
                    }
                ],
            }
            manifest_path = root / "fixtures.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            code, out = _run_manifest(manifest_path)
            self.assertEqual(code, 1)
            self.assertEqual(out.split()[0], "FIXTURE_MISSING")

    def test_directory_path_fails_as_fixture_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            directory = root / "fixture_dir"
            directory.mkdir()
            manifest = {
                "version": 1,
                "fixtures": [
                    {
                        "id": "dir-path",
                        "path": "fixture_dir",
                        "source": "local",
                        "revision": "dev",
                        "sha256": _sha256(b"unused"),
                        "license": "test",
                        "generated_by": "unittest",
                        "command": "echo test",
                        "required": True,
                    }
                ],
            }
            manifest_path = root / "fixtures.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            code, out = _run_manifest(manifest_path)
            self.assertEqual(code, 1)
            self.assertEqual(out.split()[0], "FIXTURE_INVALID")

    def test_provenance_fields_require_non_empty_strings(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            fixture = root / "fixture.bin"
            fixture.write_bytes(b"x")
            manifest = {
                "version": 1,
                "fixtures": [
                    {
                        "id": "bad-prov",
                        "path": "fixture.bin",
                        "source": "",
                        "revision": None,
                        "sha256": _sha256(b"x"),
                        "license": " ",
                        "generated_by": "",
                        "command": "",
                        "required": True,
                    }
                ],
            }
            manifest_path = root / "fixtures.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            code, out = _run_manifest(manifest_path)
            self.assertEqual(code, 1)
            self.assertEqual(out.split()[0], "FIXTURE_INVALID")

if __name__ == "__main__":
    unittest.main()
