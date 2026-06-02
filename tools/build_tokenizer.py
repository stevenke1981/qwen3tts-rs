#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 Qwen3-TTS 的 BPE 格式（vocab.json + merges.txt + tokenizer_config.json）
合併為 tokenizers crate 可直接載入的 `tokenizer.json`。

用法：
    python tools/build_tokenizer.py \\
        --model-dir <path-to-Qwen3-TTS-0.6B-Base> \\
        --output models/tokenizer.json

若 `qwen-tts` Python 套件未安裝，可改用環境變數指定模型目錄：
    set QWEN3_TTS_MODEL_DIR=C:\\path\\to\\Qwen3-TTS-12Hz-0.6B-Base
    python tools/build_tokenizer.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
except ImportError:
    sys.stderr.write(
        "ERROR: 需要安裝 tokenizers Python 套件以產生 tokenizer.json。\n"
        "    pip install tokenizers\n"
    )
    sys.exit(1)


def find_model_dir(model_id: str | None) -> Path:
    """解析模型目錄路徑。

    優先順序：
      1. 環境變數 QWEN3_TTS_MODEL_DIR
      2. HuggingFace 預設快取 (~/.cache/huggingface/hub)
      3. --model-dir CLI 參數
    """
    env_dir = os.environ.get("QWEN3_TTS_MODEL_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    if model_id is not None:
        home = Path(os.environ.get("USERPROFILE", str(Path.home())))
        candidates = [
            home
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--{model_id.replace('/', '--')}",
            home / ".cache" / "huggingface" / "hub" / model_id,
        ]
        for c in candidates:
            if c.exists():
                snapshots = c / "snapshots"
                if snapshots.exists():
                    for snap in snapshots.iterdir():
                        if (snap / "model.safetensors").exists():
                            return snap
        return candidates[0]

    raise FileNotFoundError(
        "找不到 Qwen3-TTS 模型目錄。請設定 QWEN3_TTS_MODEL_DIR 環境變數，\n"
        "或使用 --model-dir 指定。"
    )


def find_file(model_dir: Path, candidates: list[str]) -> Path:
    """在 snapshot 目錄下尋找檔案（可能是 symlink）。"""
    for name in candidates:
        p = model_dir / name
        if p.is_symlink() or p.exists():
            return p
    raise FileNotFoundError(f"找不到任何 {candidates} 於 {model_dir}")


def build_tokenizer(model_dir: Path) -> Tokenizer:
    """從 vocab.json + merges.txt + tokenizer_config.json 建構 BPE tokenizer。"""
    vocab_path = find_file(model_dir, ["vocab.json"])
    merges_path = find_file(model_dir, ["merges.txt"])
    cfg_path = find_file(model_dir, ["tokenizer_config.json"])

    with open(vocab_path, encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)

    with open(merges_path, encoding="utf-8") as f:
        merges: list[tuple[str, str]] = []
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))

    with open(cfg_path, encoding="utf-8") as f:
        tcfg = json.load(f)

    bpe = BPE(vocab=vocab, merges=merges)
    tok = Tokenizer(bpe)
    # Qwen2 / GPT-2 風格：ByteLevel pretokenizer，add_prefix_space=False。
    tok.pre_tokenizer = ByteLevel(
        add_prefix_space=False, trim_offsets=True, use_regex=True
    )
    tok.decoder = ByteLevelDecoder()

    # 註冊特殊 token：Qwen3-TTS 的 vocab.json 只含 151643 條目
    # （ID 0..151642），33 個特殊 token（ID 151643..151675）來自
    # tokenizer_config.json 的 added_tokens_decoder，必須額外加入。
    special_tokens: list[AddedToken] = []
    for info in tcfg["added_tokens_decoder"].values():
        special_tokens.append(
            AddedToken(
                info["content"],
                normalized=info.get("normalized", False),
                special=info.get("special", True),
            )
        )
    tok.add_special_tokens(special_tokens)
    return tok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="為 Qwen3-TTS 模型產生 tokenizers crate 相容的 tokenizer.json"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="模型目錄路徑；省略時讀 QWEN3_TTS_MODEL_DIR 或 HuggingFace 快取",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="HuggingFace model ID（用於預設快取搜尋）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/tokenizer.json",
        help="輸出 tokenizer.json 路徑",
    )
    args = parser.parse_args()

    model_dir = (
        Path(args.model_dir) if args.model_dir else find_model_dir(args.model_id)
    )
    if not model_dir.exists():
        sys.stderr.write(f"ERROR: 模型目錄不存在：{model_dir}\n")
        return 1

    print(f"[*] Loading vocab/merges/config from: {model_dir}")
    tok = build_tokenizer(model_dir)

    # 驗證：<|im_start|> 必須是 151644
    im_start_id = tok.token_to_id("<|im_start|>")
    im_end_id = tok.token_to_id("<|im_end|>")
    assistant_id = tok.token_to_id("assistant")
    if im_start_id != 151644 or im_end_id != 151645 or assistant_id != 77091:
        sys.stderr.write(
            f"ERROR: 特殊 token ID 對不上預期：\n"
            f"  <|im_start|> = {im_start_id}（預期 151644）\n"
            f"  <|im_end|> = {im_end_id}（預期 151645）\n"
            f"  assistant = {assistant_id}（預期 77091）\n"
        )
        return 1

    # 驗證 "你好" 必須 tokenize 為 [108386]
    ni_hao_ids = tok.encode("你好").ids
    if ni_hao_ids != [108386]:
        sys.stderr.write(f"ERROR: '你好' 編碼不正確：{ni_hao_ids}（預期 [108386]）\n")
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    print(f"[OK] 寫入 {out} (vocab_size={tok.get_vocab_size()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
