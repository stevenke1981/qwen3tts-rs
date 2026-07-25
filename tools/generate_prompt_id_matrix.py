#!/usr/bin/env python3
"""Generate and validate prompt-ID oracle across the 5 public Qwen3-TTS variants.

The script uses the official Hugging Face metadata/tokenizer artifacts and records
exact prompt-ID arrays in a checked-in fixture for Rust regression verification.

Usage:
  python tools/generate_prompt_id_matrix.py --help
  python tools/generate_prompt_id_matrix.py --check fixtures/alignment/p01_prompt_id_matrix.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer

from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor


ROOT = Path(__file__).resolve().parent.parent
HF_MODELS = [
    {
        "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "cases": ["x_vector_only", "icl"],
        "speaker": None,
        "include_instruct": False,
    },
    {
        "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "cases": ["custom_voice"],
        "speaker": "Dylan",
        "include_instruct": False,
    },
    {
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "cases": ["x_vector_only", "icl"],
        "speaker": None,
        "include_instruct": False,
    },
    {
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "cases": ["custom_voice"],
        "speaker": "Dylan",
        "include_instruct": True,
    },
    {
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "cases": ["voice_design"],
        "speaker": None,
        "include_instruct": True,
    },
]

TOKENIZER_SOURCE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
FIXTURE_PATH = ROOT / "fixtures" / "alignment" / "p01_prompt_id_matrix.json"

MAIN_TEXT = "請你用自然語調，讀一段簡短的中文介紹。"
REFERENCE_TEXT = "參考文字：這段句子用來建立語者風格與語感。"
INSTRUCTION_TEXT = "請用更溫和、清晰的語氣"

TOKENIZER_CANONICAL_FILES = [
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
]

TOKENIZER_LOCAL_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
]


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


def model_revisions(model_id: str) -> str:
    return HfApi().model_info(model_id).sha


def resolve_snapshot(repo_id: str, revision: str, allow_patterns: List[str]) -> Path:
    path = model_snapshot_path(repo_id, revision)
    if path.exists():
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


def ensure_config_snapshot(repo_id: str, revision: str) -> Path:
    return resolve_snapshot(repo_id, revision, ["config.json"])


def ensure_tokenizer_for_encoding() -> Path:
    revision = model_revisions(TOKENIZER_SOURCE_MODEL)
    snapshot = resolve_snapshot(TOKENIZER_SOURCE_MODEL, revision, TOKENIZER_LOCAL_FILES)
    tokenizer_path = snapshot / "tokenizer.json"
    if tokenizer_path.exists():
        return tokenizer_path

    raise RuntimeError(
        "TOKENIZER_MISSING: could not load tokenizer.json from "
        f"{TOKENIZER_SOURCE_MODEL}@{revision}"
    )


def ensure_official_processor() -> Qwen3TTSProcessor:
    revision = model_revisions(TOKENIZER_SOURCE_MODEL)
    snapshot = resolve_snapshot(TOKENIZER_SOURCE_MODEL, revision, TOKENIZER_LOCAL_FILES)
    tokenizer_path = snapshot / "tokenizer.json"
    if not tokenizer_path.exists():
        raise RuntimeError(
            "TOKENIZER_MISSING: could not load tokenizer.json from "
            f"{TOKENIZER_SOURCE_MODEL}@{revision}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        fix_mistral_regex=True,
    )
    return Qwen3TTSProcessor(tokenizer=tokenizer)


def _blob_id(file_record: Any) -> Optional[str]:
    blob = getattr(file_record, "blob_id", None)
    if blob:
        return blob
    lfs = getattr(file_record, "lfs", None)
    if lfs is not None:
        return getattr(lfs, "oid", None)
    return None


def tokenizer_signature(model_id: str, revision: str) -> Dict[str, str]:
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    found = {}
    for sibling in info.siblings:
        if sibling.rfilename in TOKENIZER_CANONICAL_FILES:
            blob_id = _blob_id(sibling)
            if blob_id is None:
                raise RuntimeError(
                    f"TOKENIZER_IDENTITY_MISSING_BLOB: {model_id}:{sibling.rfilename}"
                )
            found[sibling.rfilename] = blob_id

    for filename in TOKENIZER_CANONICAL_FILES:
        if filename not in found:
            raise RuntimeError(
                f"TOKENIZER_IDENTITY_MISSING_FILE: {filename} not present in {model_id}@{revision}"
            )

    if not found:
        raise RuntimeError(
            f"TOKENIZER_IDENTITY_MISSING_FILES: no canonical tokenizer files for {model_id}@{revision}"
        )

    return found


def verify_shared_tokenizer_identity() -> Dict[str, Any]:
    reference_signature = None
    for item in HF_MODELS:
        model_id = item["model_id"]
        revision = model_revisions(model_id)
        current_signature = tokenizer_signature(model_id, revision)

        if reference_signature is None:
            reference_signature = current_signature
            continue

        for filename, blob_id in reference_signature.items():
            if current_signature.get(filename) != blob_id:
                raise RuntimeError(
                    f"TOKENIZER_IDENTITY_MISMATCH: {filename} differs among models"
                )

    assert reference_signature is not None
    return reference_signature


def encode_ids(processor: Qwen3TTSProcessor, value: str) -> List[int]:
    encoded = processor(text=value, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids[0].tolist()


def build_prompts(processor: Qwen3TTSProcessor) -> Dict[str, List[int]]:
    main_prompt = f"<|im_start|>assistant\n{MAIN_TEXT}<|im_end|>\n<|im_start|>assistant\n"
    reference_prompt = f"<|im_start|>assistant\n{REFERENCE_TEXT}<|im_end|>\n"
    instruction_prompt = f"<|im_start|>user\n{INSTRUCTION_TEXT}<|im_end|>\n"

    main_ids = encode_ids(processor, main_prompt)
    reference_ids = encode_ids(processor, reference_prompt)
    instruction_ids = encode_ids(processor, instruction_prompt)

    reference_body = reference_ids[3 : -2]
    return {
        "main_ids": main_ids,
        "reference_ids": reference_ids,
        "reference_body": reference_body,
        "instruction_ids": instruction_ids,
    }


def find_speaker(metadata: Dict[str, Any], preferred: str | None) -> Dict[str, int] | None:
    talker = metadata.get("talker_config", {})
    speakers = talker.get("spk_id")
    if not isinstance(speakers, dict) or not speakers:
        return None

    normalized = {
        str(name).lower(): (str(name), int(token_id))
        for name, token_id in speakers.items()
    }
    if preferred and preferred.lower() in normalized:
        name, token_id = normalized[preferred.lower()]
        return {"name": name, "token_id": token_id}

    name, token_id = sorted(normalized.values(), key=lambda item: item[0].lower())[0]
    return {"name": name, "token_id": token_id}


def load_config(revision_path: Path) -> Dict[str, Any]:
    config_path = revision_path / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_oracle() -> Dict[str, Any]:
    tokenizer_path = ensure_tokenizer_for_encoding()
    processor = ensure_official_processor()
    prompts = build_prompts(processor)
    tokenizer_revision = model_revisions(TOKENIZER_REPO)
    tokenizer_dir = ensure_config_snapshot(TOKENIZER_REPO, tokenizer_revision)

    tokenizer_signature_dict = verify_shared_tokenizer_identity()
    tokenizer_revision_by_model = {}
    for item in HF_MODELS:
        model_id = item["model_id"]
        tokenizer_revision_by_model[model_id] = model_revisions(model_id)

    models_payload: List[Dict[str, Any]] = []
    for item in HF_MODELS:
        model_id = item["model_id"]
        revision = model_revisions(model_id)
        snapshot_dir = ensure_config_snapshot(model_id, revision)
        config = load_config(snapshot_dir)
        config_path = snapshot_dir / "config.json"
        config_sha256 = sha256_file(config_path)

        tts_type = config.get("tts_model_type")
        speaker = None
        if item["speaker"] and tts_type == "custom_voice":
            speaker = find_speaker(config, item["speaker"])

        def mk_case(case_name: str) -> Dict[str, Any]:
            include_reference = case_name == "icl"
            include_instruction = (
                item["include_instruct"] and case_name in {"custom_voice", "voice_design"}
            )
            return {
                "case": case_name,
                "x_vector_only": case_name == "x_vector_only",
                "main_prompt_ids": prompts["main_ids"],
                "reference_prompt_ids": prompts["reference_ids"] if include_reference else [],
                "reference_body_ids": prompts["reference_body"] if include_reference else [],
                "instruction_prompt_ids": prompts["instruction_ids"] if include_instruction else [],
            }

        models_payload.append(
            {
                "model_id": model_id,
                "model_revision": revision,
                "config_sha256": config_sha256,
                "supports_voice_clone": tts_type == "base",
                "supports_voice_design": tts_type == "voice_design",
                "supports_speaker_presets": tts_type == "custom_voice",
                "expected_speaker": speaker,
                "cases": [mk_case(name) for name in item["cases"]],
            }
        )

    return {
        "version": 1,
        "fixture_id": "p01-prompt-id-matrix",
        "generated_by": "tools/generate_prompt_id_matrix.py official-python",
        "source_repo": "https://huggingface.co/Qwen/Qwen3-TTS",
        "command": "python tools/generate_prompt_id_matrix.py",
        "tokenizer": {
            "model_id": TOKENIZER_REPO,
            "model_revision": tokenizer_revision,
            "config_path": "config.json",
            "config_sha256": sha256_file(tokenizer_dir / "config.json"),
            "tokenizer_path": "tokenizer.json",
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "shared_across_variants": True,
            "tokenizer_signature": tokenizer_signature_dict,
            "tokenizer_model_revisions": tokenizer_revision_by_model,
            "tokenizer_load": {
                "loader": "Qwen3TTSProcessor",
                "tokenizer_source_repo": TOKENIZER_SOURCE_MODEL,
                "local_files_only": True,
            },
        },
        "samples": {
            "main_text": MAIN_TEXT,
            "reference_text": REFERENCE_TEXT,
            "instruction_text": INSTRUCTION_TEXT,
        },
        "models": models_payload,
    }


def write_fixture(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")


def _canonical_for_compare(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=None)


def validate_fixture(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"FIXTURE_MISSING: {path} does not exist")
    current = json.loads(path.read_text(encoding="utf-8"))
    fresh = generate_oracle()
    if _canonical_for_compare(current) != _canonical_for_compare(fresh):
        raise RuntimeError("FIXTURE_MISMATCH: checked-in fixture does not match generated oracle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate or verify prompt-ID oracle for P01-T05")
    p.add_argument(
        "--check",
        type=Path,
        help="Validate an existing fixture file instead of generating",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=FIXTURE_PATH,
        help="Write location for generated fixture",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        validate_fixture(args.check)
        print(f"PROMPT_MATRIX_CHECK_OK {args.check}")
        return 0

    payload = generate_oracle()
    write_fixture(payload, args.output)
    print(f"PROMPT_MATRIX_GENERATED {args.output}")
    print(f"MODELS={len(payload['models'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
