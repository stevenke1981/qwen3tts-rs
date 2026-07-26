import json
import pathlib
import struct
import hashlib
import subprocess
import sys

def write_dump(dir_path: pathlib.Path, bad_hash: str | None = None):
    dir_path.mkdir(parents=True, exist_ok=True)
    code = [0.5, 0.6]
    pcm = [0.5, 0.6, 0.7, 0.8]
    code_bytes = struct.pack("<2f", *code)
    pcm_bytes = struct.pack("<4f", *pcm)
    code_file = dir_path / "codec_input_code_matrix_0000.f32.bin"
    pcm_file = dir_path / "codec_final_pcm_0000.f32.bin"
    code_file.write_bytes(code_bytes)
    pcm_file.write_bytes(pcm_bytes)

    def sha(path: pathlib.Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    stages = [
        {
            "name": "codec_input_code_matrix",
            "dtype": "f32",
            "shape": [2],
            "file": code_file.name,
            "layout": "c",
            "sha256": "00badcafe" if bad_hash else sha(code_file),
        },
        {
            "name": "codec_final_pcm",
            "dtype": "f32",
            "shape": [4],
            "file": pcm_file.name,
            "layout": "c",
            "sha256": sha(pcm_file),
        },
    ]

    manifest = {
        "schema_version": 1,
        "source": "qwen3tts-rs",
        "model": "unit-model",
        "case_id": "case-compare",
        "revision": None,
        "seed": None,
        "stages": stages,
    }

    manifest_path = dir_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path

base = pathlib.Path(__file__).resolve().parent / "_tmp_stage_compare"
left = base / "left"
right = base / "right"
if left.exists():
    import shutil
    shutil.rmtree(left)
if right.exists():
    import shutil
    shutil.rmtree(right)

left_manifest = write_dump(left)
right_manifest = write_dump(right)

proc = subprocess.run([sys.executable, "tools/compare_stage_dumps.py", str(left_manifest), str(right_manifest), "--min-cosine", "0.999"])
print(f"IDENTITY_EXIT={proc.returncode}")

# corrupted right hash should fail
write_dump(right, bad_hash=True)
proc2 = subprocess.run([sys.executable, "tools/compare_stage_dumps.py", str(left_manifest), str(right_manifest), "--min-cosine", "0.999"])
print(f"REGRESSION_EXIT={proc2.returncode}")
