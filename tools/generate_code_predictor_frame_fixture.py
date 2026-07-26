#!/usr/bin/env python3
"""Export and verify the pinned official Code Predictor frame oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import struct
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/alignment/p03_code_predictor_frame_real.json"
RAW = ROOT / "artifacts/alignment/P03/P03-T02/qwentts-direct-oracle/raw_tensor_dump"
ORACLE = ROOT / "artifacts/alignment/P03/P03-T03/official-oracle"
SNAPSHOT = "5d83992436eae1d760afd27aff78a71d676296fc"
OFFICIAL_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
QWENTTS_REVISION = "82cd05b9f3a175612dc89fd6943e610fab096ef5"
HIDDEN_SHA256 = "D8C2C441FB0754E50A41225EC9C4F5056F3994E9AC3B77165E62326AF74AD1BA"
MODEL_SHA256 = "180B3B10EB1C9F1B4DB7806D5475BAE3071C0243C299D49926BAB1DA3B6946F6"
CONFIG_SHA256 = "2E714C787C8EDB98B05432685CDDB634ADD2DE4D4E645F653D68251EF72BA011"
GENERATION_CONFIG_SHA256 = "F1B90B4513F3B34C62851049E2492D7B4C5940DAF1276F89C82B8EF04127F3AA"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_dump(path: Path):
    import numpy as np

    payload = path.read_bytes()
    if len(payload) < 12:
        raise RuntimeError(f"invalid tensor header: {path}")
    rank, rows, columns = struct.unpack_from("<III", payload, 0)
    values = np.frombuffer(payload, dtype="<f4", offset=12).copy()
    if rank != 2 or values.size != rows * columns:
        raise RuntimeError(f"invalid tensor shape: {path}")
    return values.reshape(rows, columns)


def write_f32(path: Path, tensor) -> dict[str, Any]:
    array = tensor.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
    path.write_bytes(array.tobytes(order="C"))
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "shape": list(array.shape),
        "dtype": "float32",
        "byte_order": "little-endian",
        "sha256": sha256(path),
    }


def tensor_entry(path: Path, shape: list[int], dtype: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "shape": shape,
        "dtype": dtype,
        "byte_order": "little-endian",
        "sha256": sha256(path),
    }


def cache_tensors(cache):
    keys = []
    values = []
    for layer in cache.layers:
        if layer.keys is None or layer.values is None:
            raise RuntimeError("official DynamicCache contains an uninitialized layer")
        keys.append(layer.keys.detach().cpu().float().contiguous().clone())
        values.append(layer.values.detach().cpu().float().contiguous().clone())
    return keys, values


def export(model_dir: Path, fixture_path: Path) -> None:
    import numpy as np
    import torch
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.models import modeling_qwen3_tts

    required_hashes = {
        model_dir / "model.safetensors": MODEL_SHA256,
        model_dir / "config.json": CONFIG_SHA256,
        model_dir / "generation_config.json": GENERATION_CONFIG_SHA256,
        RAW / "talker-hidden-prefill-final.bin": HIDDEN_SHA256,
    }
    for path, expected in required_hashes.items():
        if not path.exists():
            raise SystemExit(f"FIXTURE_MISSING: {path}")
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"FIXTURE_INVALID sha256 path={path} expected={expected} actual={actual}")

    wrapper = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map="cpu",
        dtype=torch.float32,
        trust_remote_code=True,
    )
    talker = wrapper.model.talker.float().eval()
    predictor = talker.code_predictor.float().eval()
    hidden = torch.from_numpy(read_dump(RAW / "talker-hidden-prefill-final.bin")[-1:].copy())
    hidden = hidden.view(1, 1, -1).float()
    c0 = torch.tensor([[1995]], dtype=torch.long)
    c0_embedding = talker.get_input_embeddings()(c0)
    prefill = torch.cat([hidden, c0_embedding], dim=1)

    logits = []
    normalized_hidden = []
    projected_private_inputs = []
    cache_lengths = []
    prefix_checks = []
    manual_codes = []
    past = None
    previous_keys = None
    previous_values = None

    with torch.inference_mode():
        for group in range(15):
            if group == 0:
                output = predictor(
                    inputs_embeds=prefill,
                    position_ids=torch.tensor([[0, 1]], dtype=torch.long),
                    cache_position=torch.tensor([0, 1], dtype=torch.long),
                    past_key_values=None,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                input_ids = torch.tensor([[manual_codes[-1]]], dtype=torch.long)
                private_input = predictor.model.get_input_embeddings()[group - 1](input_ids)
                projected_private_inputs.append(
                    predictor.small_to_mtp_projection(private_input).detach().cpu().float()
                )
                output = predictor(
                    input_ids=input_ids,
                    position_ids=torch.tensor([[group + 1]], dtype=torch.long),
                    cache_position=torch.tensor([group + 1], dtype=torch.long),
                    past_key_values=past,
                    generation_steps=group,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            step_logits = output.logits[:, -1, :].detach().cpu().float()
            step_hidden = output.hidden_states[-1][:, -1, :].detach().cpu().float()
            next_code = int(step_logits.argmax(dim=-1).item())
            keys, values = cache_tensors(output.past_key_values)
            lengths = {int(key.shape[2]) for key in keys}
            if len(keys) != 5 or len(lengths) != 1:
                raise RuntimeError(f"invalid official cache at group {group}: layers={len(keys)} lengths={lengths}")
            length = lengths.pop()
            expected_length = group + 2
            if length != expected_length:
                raise RuntimeError(
                    f"invalid official cache length at group {group}: expected={expected_length} actual={length}"
                )
            if previous_keys is not None and previous_values is not None:
                unchanged = all(
                    torch.equal(old, new[:, :, : old.shape[2], :])
                    and torch.equal(old_value, new_value[:, :, : old_value.shape[2], :])
                    for old, new, old_value, new_value in zip(
                        previous_keys, keys, previous_values, values, strict=True
                    )
                )
                if not unchanged:
                    raise RuntimeError(f"official cache prefix mutated at group {group}")
                prefix_checks.append(True)
            previous_keys, previous_values = keys, values
            past = output.past_key_values
            logits.append(step_logits)
            normalized_hidden.append(step_hidden)
            manual_codes.append(next_code)
            cache_lengths.append(length)

        generated = predictor.generate(
            inputs_embeds=prefill,
            max_new_tokens=15,
            do_sample=False,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
        generated_codes = generated.sequences.detach().cpu().reshape(-1).tolist()
        if generated_codes != manual_codes:
            raise RuntimeError(
                f"manual/generate code mismatch manual={manual_codes} generated={generated_codes}"
            )
        if len(generated.scores) != 15:
            raise RuntimeError(f"official generate returned {len(generated.scores)} score tensors")
        for group, (manual_logits, generated_logits) in enumerate(
            zip(logits, generated.scores, strict=True)
        ):
            if not torch.equal(manual_logits, generated_logits.detach().cpu().float()):
                max_diff = float(
                    (manual_logits - generated_logits.detach().cpu().float()).abs().max().item()
                )
                raise RuntimeError(
                    f"manual/generate logit mismatch group={group} max_abs={max_diff}"
                )

    ORACLE.mkdir(parents=True, exist_ok=True)
    tensors = {
        "projected_prefill": write_f32(ORACLE / "prefill-input.bin", prefill),
        "projected_private_inputs": write_f32(
            ORACLE / "private-inputs.bin", torch.cat(projected_private_inputs, dim=0)
        ),
        "normalized_hidden": write_f32(
            ORACLE / "normalized-hidden.bin", torch.cat(normalized_hidden, dim=0)
        ),
        "logits": write_f32(ORACLE / "logits.bin", torch.cat(logits, dim=0)),
        "final_cache_k": write_f32(ORACLE / "cache-k.bin", torch.stack(previous_keys, dim=0)),
        "final_cache_v": write_f32(ORACLE / "cache-v.bin", torch.stack(previous_values, dim=0)),
    }
    positions_path = ORACLE / "positions.bin"
    positions_path.write_bytes(np.arange(16, dtype="<i8").tobytes())
    tensors["positions"] = tensor_entry(positions_path, [16], "int64")

    codes = {"c0": 1995, "generated": manual_codes, "all_codebooks": [1995, *manual_codes]}
    codes_path = ORACLE / "codes.json"
    codes_path.write_text(json.dumps(codes, indent=2) + "\n", encoding="utf-8")
    cache_manifest = {
        "layers": 5,
        "lengths": cache_lengths,
        "prefix_bit_identical": all(prefix_checks) and len(prefix_checks) == 14,
        "layout": "LBHSD",
        "final_k": tensors["final_cache_k"],
        "final_v": tensors["final_cache_v"],
    }
    cache_manifest_path = ORACLE / "cache-manifest.json"
    cache_manifest_path.write_text(json.dumps(cache_manifest, indent=2) + "\n", encoding="utf-8")

    source_path = Path(modeling_qwen3_tts.__file__).resolve()
    oracle_manifest = {
        "fixture_id": "p03-code-predictor-frame-real",
        "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "snapshot_revision": SNAPSHOT,
        "official_qwen_revision": OFFICIAL_REVISION,
        "qwentts_cpp_revision": QWENTTS_REVISION,
        "source": {
            "path": str(source_path),
            "sha256": sha256(source_path),
            "qwen_tts_version": importlib.metadata.version("qwen-tts"),
            "transformers_version": importlib.metadata.version("transformers"),
            "torch_version": torch.__version__,
        },
        "model_hashes": {
            "model.safetensors": MODEL_SHA256,
            "config.json": CONFIG_SHA256,
            "generation_config.json": GENERATION_CONFIG_SHA256,
        },
        "external_input": {
            "path": (RAW / "talker-hidden-prefill-final.bin").relative_to(ROOT).as_posix(),
            "sha256": HIDDEN_SHA256,
            "row": -1,
            "c0": 1995,
        },
        "dtype": "float32",
        "device": "cpu",
        "attention_backend": "official-pinned-cpu",
        "cache_lengths": cache_lengths,
        "manual_generate_codes_equal": True,
        "manual_generate_logits_bit_equal": True,
        "tensors": tensors,
        "codes": {
            "path": codes_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(codes_path),
        },
        "cache_manifest": {
            "path": cache_manifest_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(cache_manifest_path),
        },
    }
    oracle_manifest_path = ORACLE / "manifest.json"
    oracle_manifest_path.write_text(json.dumps(oracle_manifest, indent=2) + "\n", encoding="utf-8")

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture.update(
        {
            "oracle_status": "ready",
            "oracle_manifest": oracle_manifest_path.relative_to(ROOT).as_posix(),
            "oracle_manifest_sha256": sha256(oracle_manifest_path),
            "oracle_codes_sha256": sha256(codes_path),
            "oracle_cache_manifest_sha256": sha256(cache_manifest_path),
        }
    )
    fixture.pop("oracle_logits_sha256", None)
    fixture_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(
        f"EXPORTED oracle={ORACLE} codes={len(manual_codes)} "
        f"cache_lengths={cache_lengths[0]}..{cache_lengths[-1]}"
    )


def check(fixture_path: Path) -> None:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    required = {
        "fixture_id",
        "snapshot_revision",
        "official_qwen_revision",
        "qwentts_cpp_revision",
        "talker_hidden_prefill_final_sha256",
        "c0_token",
        "prefill_positions",
        "step_positions",
        "num_predictor_layers",
        "num_code_groups",
        "min_cosine",
        "max_abs_limit",
        "oracle_manifest",
        "oracle_manifest_sha256",
    }
    missing = sorted(required - fixture.keys())
    if missing:
        raise SystemExit("FIXTURE_INVALID missing=" + ",".join(missing))
    if fixture["oracle_status"] != "ready":
        raise SystemExit(f"FIXTURE_INVALID oracle_status={fixture['oracle_status']}")
    if fixture["prefill_positions"] != [0, 1] or fixture["step_positions"] != list(range(2, 16)):
        raise SystemExit("FIXTURE_INVALID positions")
    if fixture["num_predictor_layers"] != 5 or fixture["num_code_groups"] != 16:
        raise SystemExit("FIXTURE_INVALID contract")
    if fixture["min_cosine"] != 0.999 or fixture["max_abs_limit"] != 0.001:
        raise SystemExit("FIXTURE_INVALID thresholds")
    hidden_path = RAW / "talker-hidden-prefill-final.bin"
    if sha256(hidden_path) != HIDDEN_SHA256:
        raise SystemExit("FIXTURE_INVALID hidden sha256")

    manifest_path = ROOT / fixture["oracle_manifest"]
    if not manifest_path.exists() or sha256(manifest_path) != fixture["oracle_manifest_sha256"]:
        raise SystemExit("FIXTURE_INVALID oracle manifest sha256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["cache_lengths"] != list(range(2, 17)):
        raise SystemExit("FIXTURE_INVALID cache lengths")
    if not manifest["manual_generate_codes_equal"] or not manifest["manual_generate_logits_bit_equal"]:
        raise SystemExit("FIXTURE_INVALID official generate cross-check")
    for entry in manifest["tensors"].values():
        path = ROOT / entry["path"]
        if not path.exists() or sha256(path) != entry["sha256"]:
            raise SystemExit(f"FIXTURE_INVALID tensor sha256 path={path}")
        item_size = 8 if entry["dtype"] == "int64" else 4
        expected_size = item_size
        for dimension in entry["shape"]:
            expected_size *= dimension
        if path.stat().st_size != expected_size:
            raise SystemExit(
                f"FIXTURE_INVALID tensor size path={path} expected={expected_size} actual={path.stat().st_size}"
            )
    cache_path = ROOT / manifest["cache_manifest"]["path"]
    cache_manifest = json.loads(cache_path.read_text(encoding="utf-8"))
    if (
        cache_manifest["lengths"] != list(range(2, 17))
        or not cache_manifest["prefix_bit_identical"]
    ):
        raise SystemExit("FIXTURE_INVALID cache evidence")
    codes_path = ROOT / manifest["codes"]["path"]
    codes = json.loads(codes_path.read_text(encoding="utf-8"))
    if len(codes["generated"]) != 15 or len(codes["all_codebooks"]) != 16:
        raise SystemExit("FIXTURE_INVALID codes")
    print(f"FIXTURE_OK id={fixture['fixture_id']} oracle_status=ready tensors={len(manifest['tensors'])}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--check", type=Path, nargs="?", const=FIXTURE)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("QWEN3_TTS_REAL_MODEL_DIR", "")),
    )
    args = parser.parse_args()
    fixture_path = args.check or FIXTURE
    if args.export:
        if not args.model or not (args.model / "config.json").exists():
            raise SystemExit("FIXTURE_MISSING model")
        export(args.model, fixture_path)
    else:
        check(fixture_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
