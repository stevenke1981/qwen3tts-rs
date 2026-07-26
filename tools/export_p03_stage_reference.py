#!/usr/bin/env python3
"""Export the full 723-stage official-Python reference manifest for P03-T05.

Instruments the official Qwen3-TTS PyTorch Talker and Code Predictor to capture
every intermediate tensor at the same points as the Candle implementation, then
writes a manifest and F32 binary files compatible with compare_stage_dumps.py.

Uses a custom generation loop (not HuggingFace generate()) to intercept every
stage at the exact Candle capture points.

Pins: official source revision, model snapshot revision, case, seed, stage
names/layouts, and file hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official-source", required=True, type=Path)
    ap.add_argument("--official-revision", required=True)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--model-revision", required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--expected-stages", type=int, default=723)
    ap.add_argument(
        "--fixture",
        type=Path,
        default=Path("fixtures/alignment/p02_deterministic_token_sequences_real.json"),
    )
    return ap.parse_args()


class StageWriter:
    """Writes F32 binary stage files and a JSON manifest."""

    def __init__(self, output_dir: Path, metadata: dict):
        if output_dir.exists():
            raise RuntimeError(f"output dir already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        self.output_dir = output_dir
        self.metadata = metadata
        self.stages: list[dict] = []
        self.names: set[str] = set()
        self.count = 0

    def record(self, name: str, tensor, layout: str) -> None:
        import torch

        if name in self.names:
            raise RuntimeError(f"duplicate stage: {name}")
        self.names.add(name)

        t = tensor.detach().to(torch.float32).cpu().contiguous()
        shape = list(t.shape)
        flat = t.flatten().tolist()

        file_name = f"{name}_{self.count:04}.f32.bin"
        self.count += 1
        path = self.output_dir / file_name

        h = hashlib.sha256()
        with open(path, "wb") as f:
            for v in flat:
                b = struct.pack("<f", v)
                h.update(b)
                f.write(b)

        self.stages.append(
            {
                "name": name,
                "dtype": "f32",
                "shape": shape,
                "file": file_name,
                "sha256": h.hexdigest(),
                "layout": layout,
                "byte_order": "little-endian",
            }
        )

    def commit(self) -> Path:
        manifest = {
            "schema_version": 1,
            "source": self.metadata["source"],
            "revision": self.metadata.get("revision"),
            "model": self.metadata["model"],
            "case_id": self.metadata["case_id"],
            "seed": self.metadata.get("seed"),
            "stages": self.stages,
        }
        path = self.output_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return path


def main() -> int:
    args = parse_args()

    official_src = args.official_source
    if not official_src.exists():
        print(f"ERROR: official source not found: {official_src}", file=sys.stderr)
        return 1

    site_packages = official_src / ".venv" / "Lib" / "site-packages"
    if site_packages.exists():
        sys.path.insert(0, str(site_packages))

    import torch

    fixture_path = args.fixture
    if not fixture_path.exists():
        print(f"ERROR: fixture not found: {fixture_path}", file=sys.stderr)
        return 1
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    if fixture.get("snapshot_revision") != args.model_revision:
        print(
            f"ERROR: fixture snapshot_revision {fixture.get('snapshot_revision')} "
            f"!= --model-revision {args.model_revision}",
            file=sys.stderr,
        )
        return 1
    if fixture.get("official_qwen_revision") != args.official_revision:
        print(
            f"ERROR: fixture official_qwen_revision {fixture.get('official_qwen_revision')} "
            f"!= --official-revision {args.official_revision}",
            file=sys.stderr,
        )
        return 1

    # Use fixture seed consistently
    seed = fixture["seed"]
    if args.seed != seed:
        print(
            f"ERROR: --seed {args.seed} != fixture seed {seed}; "
            f"the fixture seed must be used for reproducibility",
            file=sys.stderr,
        )
        return 1

    print(f"Loading model from {args.model} ...")
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        str(args.model),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    pytorch_model = model.model
    pytorch_model.eval()

    talker = pytorch_model.talker
    talker_model = talker.model
    code_predictor = talker.code_predictor
    cp_model = code_predictor.model

    writer = StageWriter(
        args.output_dir,
        {
            "source": "official-python",
            "revision": args.model_revision,
            "model": "Qwen3-TTS-12Hz-0.6B-Base",
            "case_id": args.case_id,
            "seed": seed,
        },
    )

    prompt_ids = fixture["prompt_ids"][0]
    max_new_tokens = 2  # Match the Candle test

    print(
        f"Running custom generation loop: prompt_ids={prompt_ids}, "
        f"seed={seed}, max_new_tokens={max_new_tokens}"
    )

    torch.manual_seed(seed)

    # ── Replicate Candle InputBuilder exactly ──────────────────────────────
    # Special token IDs from TalkerConfig
    TTS_BOS = 151672
    TTS_EOS = 151673
    TTS_PAD = 151671
    CODEC_PAD = 2148
    CODEC_BOS = 2149
    CODEC_THINK = 2154
    CODEC_THINK_BOS = 2156
    CODEC_THINK_EOS = 2157
    CHINESE_LANG_ID = 2055

    prompt_ids = fixture["prompt_ids"][0]
    text_seq_len = len(prompt_ids)
    text_len = text_seq_len - 3 - 5  # role(3) + tail(5)

    input_ids_t = torch.tensor([prompt_ids], dtype=torch.long)

    # Special token embeddings from text_embedding + text_projection
    # (Candle's embed_text applies text_projection: fc1+GELU+fc2, 2048→1024)
    def embed_text(ids):
        return talker.text_projection(talker_model.text_embedding(ids))

    special_ids = torch.tensor([[TTS_BOS, TTS_EOS, TTS_PAD]], dtype=torch.long)
    special_embeds = embed_text(special_ids)  # [1, 3, 1024]
    tts_bos_embed = special_embeds[:, 0:1, :]  # [1, 1, 1024]
    tts_eos_embed = special_embeds[:, 1:2, :]
    tts_pad_embed = special_embeds[:, 2:3, :]

    # Codec conditioning: think + think_bos + language_id + think_eos + pad + bos
    codec_prefill = [CODEC_THINK, CODEC_THINK_BOS, CHINESE_LANG_ID, CODEC_THINK_EOS]
    codec_pad_bos = [CODEC_PAD, CODEC_BOS]
    codec_all = codec_prefill + codec_pad_bos
    codec_all_t = torch.tensor([codec_all], dtype=torch.long)
    codec_emb = talker.get_input_embeddings()(codec_all_t)  # [1, 6, 1024]
    codec_len = codec_emb.shape[1]  # 6

    # Role embedding: first 3 tokens
    role_tokens = input_ids_t[:, 0:3]
    role_emb = embed_text(role_tokens)  # [1, 3, 1024]

    # Codec input: tts_pad * (codec_len-2) + tts_bos, then add codec_emb[:-1]
    pads = tts_pad_embed.expand(1, codec_len - 2, -1)  # [1, 4, 1024]
    with_bos = torch.cat([pads, tts_bos_embed], dim=1)  # [1, 5, 1024]
    codec_prefix = codec_emb[:, :codec_len - 1, :]  # [1, 5, 1024]
    codec_input = with_bos + codec_prefix  # [1, 5, 1024]

    # Full input: role + codec_input
    input_embeds = torch.cat([role_emb, codec_input], dim=1)  # [1, 8, 1024]

    # Text body: tokens 3..3+text_len
    text_body_ids = input_ids_t[:, 3:3 + text_len]
    text_body = embed_text(text_body_ids)  # [1, text_len, 1024]

    # text_with_codec_pad = text_body + codec_emb[-2]
    codec_pad_vec = codec_emb[:, codec_len - 2:codec_len - 1, :]  # [1, 1, 1024]
    text_with_codec_pad = text_body + codec_pad_vec  # [1, text_len, 1024]

    # eos_with_codec_pad = tts_eos + codec_emb[-2]
    eos_with_codec_pad = tts_eos_embed + codec_pad_vec  # [1, 1, 1024]

    # final_pad_bos = tts_pad + codec_emb[-1]
    codec_bos_vec = codec_emb[:, codec_len - 1:codec_len, :]  # [1, 1, 1024]
    final_pad_bos = tts_pad_embed + codec_bos_vec  # [1, 1, 1024]

    # Concatenate all
    input_embeds = torch.cat(
        [input_embeds, text_with_codec_pad, eos_with_codec_pad, final_pad_bos],
        dim=1,
    )  # [1, 11, 1024]

    # trailing_text_hidden = tts_pad_embed
    trailing_text_hidden = tts_pad_embed.clone()  # [1, 1, 1024]

    attention_mask = torch.ones(1, input_embeds.shape[1], dtype=torch.long)

    print(f"Input embeds shape: {input_embeds.shape} (expected [1, 11, 1024])")

    # ── Helper: run Talker model forward and capture all layer stages ──────
    def run_talker_forward(inputs_embeds, attention_mask, position_ids,
                           past_key_values, phase, use_cache=True):
        """Run talker_model.forward with output_hidden_states and capture
        all per-layer hidden states plus position/RoPE stages."""
        # Capture position IDs and RoPE
        if phase == "prefill":
            writer.record("talker-prefill-position-ids", position_ids, "ABT")

        # Compute RoPE
        position_embeddings = talker_model.rotary_emb(inputs_embeds, position_ids)
        cos, sin = position_embeddings
        if phase == "prefill":
            writer.record("talker-prefill-rope-cos", cos.unsqueeze(1), "BBTH")
            writer.record("talker-prefill-rope-sin", sin.unsqueeze(1), "BBTH")
        else:
            writer.record(f"talker-{phase}-rope-cos", cos.unsqueeze(1), "BBTH")
            writer.record(f"talker-{phase}-rope-sin", sin.unsqueeze(1), "BBTH")

        # Capture input
        if phase == "prefill":
            writer.record("talker-input-embed", inputs_embeds, "BTH")
        writer.record(f"talker-input-{phase}", inputs_embeds, "BTH")

        # Run model
        from transformers.cache_utils import DynamicCache
        if past_key_values is None:
            past_key_values = DynamicCache()

        text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids

        from transformers.masking_utils import create_causal_mask
        cache_position = torch.arange(
            past_key_values.get_seq_length(),
            past_key_values.get_seq_length() + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
        causal_mask = create_causal_mask(
            config=talker_model.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        all_hidden = [hidden_states]

        for i, layer in enumerate(talker_model.layers):
            # Capture sub-stages by calling sub-modules individually
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            writer.record(f"talker-{phase}-l{i}-input-norm", normed, "BTH")

            attn_out, _ = layer.self_attn(
                hidden_states=normed,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            writer.record(f"talker-{phase}-l{i}-attention-output", attn_out, "BTH")

            hidden_states = residual + attn_out
            residual = hidden_states
            post_normed = layer.post_attention_layernorm(hidden_states)
            writer.record(f"talker-{phase}-l{i}-post-attention-norm", post_normed, "BTH")

            mlp_out = layer.mlp(post_normed)
            writer.record(f"talker-{phase}-l{i}-mlp-output", mlp_out, "BTH")

            hidden_states = residual + mlp_out
            all_hidden.append(hidden_states)
            writer.record(f"talker-hidden-{phase}-l{i}", hidden_states, "BTH")

        # Final norm
        hidden_states = talker_model.norm(hidden_states)
        writer.record(f"talker-hidden-{phase}-final", hidden_states, "BTH")
        if phase != "prefill":
            writer.record(f"talker-hidden-{phase}", hidden_states, "BTH")

        return hidden_states, past_key_values

    # ── Helper: run Code Predictor and capture all stages ──────────────────
    def run_code_predictor(talker_hidden, codebook_0_embed, frame_index):
        """Run Code Predictor with custom loop capturing all stages."""
        from transformers.cache_utils import DynamicCache

        cp_kv = DynamicCache()
        device = talker_hidden.device
        num_code_groups = talker.config.num_code_groups  # 16

        # Prefill: concat talker_hidden + codebook_0_embed -> [1, 2, 1024]
        prefill = torch.cat([talker_hidden, codebook_0_embed], dim=1)

        writer.record(
            f"code-predictor-prefill-frame{frame_index}-input-embed",
            prefill, "BTH",
        )

        # CP position IDs and RoPE
        cp_positions = torch.tensor([[0, 1]], dtype=torch.long, device=device)
        writer.record(
            f"code-predictor-prefill-frame{frame_index}-position-ids",
            cp_positions, "BT",
        )

        cp_pos_emb = cp_model.rotary_emb(prefill, cp_positions)
        cp_cos, cp_sin = cp_pos_emb
        writer.record(
            f"code-predictor-prefill-frame{frame_index}-rope-cos",
            cp_cos.unsqueeze(1), "BBTH",
        )
        writer.record(
            f"code-predictor-prefill-frame{frame_index}-rope-sin",
            cp_sin.unsqueeze(1), "BBTH",
        )

        # Run CP prefill through layers
        from transformers.masking_utils import create_causal_mask
        cp_cache_pos = torch.arange(2, device=device)
        cp_causal = create_causal_mask(
            config=cp_model.config,
            input_embeds=prefill,
            attention_mask=torch.ones(1, 2, device=device),
            cache_position=cp_cache_pos,
            past_key_values=cp_kv,
            position_ids=cp_positions,
        )

        h = prefill
        for li, layer in enumerate(cp_model.layers):
            residual = h
            normed = layer.input_layernorm(h)
            writer.record(
                f"code-predictor-prefill-frame{frame_index}-l{li}-input-norm",
                normed, "BTH",
            )
            attn_out, _ = layer.self_attn(
                hidden_states=normed,
                attention_mask=cp_causal,
                position_ids=cp_positions,
                past_key_values=cp_kv,
                output_attentions=False,
                use_cache=True,
                cache_position=cp_cache_pos,
                position_embeddings=cp_pos_emb,
            )
            writer.record(
                f"code-predictor-prefill-frame{frame_index}-l{li}-attention-output",
                attn_out, "BTH",
            )
            h = residual + attn_out
            residual = h
            post_normed = layer.post_attention_layernorm(h)
            writer.record(
                f"code-predictor-prefill-frame{frame_index}-l{li}-post-attention-norm",
                post_normed, "BTH",
            )
            mlp_out = layer.mlp(post_normed)
            writer.record(
                f"code-predictor-prefill-frame{frame_index}-l{li}-mlp-output",
                mlp_out, "BTH",
            )
            h = residual + mlp_out
            writer.record(
                f"code-predictor-hidden-prefill-frame{frame_index}-l{li}",
                h, "BTH",
            )

        h = cp_model.norm(h)
        writer.record(
            f"code-predictor-hidden-prefill-frame{frame_index}-final",
            h, "BTH",
        )

        # Prefill logits (step 0) — use lm_head[0]
        last_h = h[:, -1:, :]
        logits = code_predictor.lm_head[0](last_h).squeeze(1)
        writer.record(
            f"code_predictor_step_logits_{frame_index:04}_0000",
            logits, "C",
        )

        # Greedy sample first token
        first_token = logits.argmax(dim=-1).reshape(1, 1)
        code_tokens = [first_token]

        # Steps 1-14
        for step in range(1, num_code_groups - 1):
            emb = code_predictor.get_input_embeddings()[step - 1]
            next_input = emb(code_tokens[step - 1])

            writer.record(
                f"code-predictor-step{step}-frame{frame_index}-codec-embed",
                next_input, "BTH",
            )

            step_pos = torch.tensor([[step + 1]], dtype=torch.long, device=device)
            step_pos_emb = cp_model.rotary_emb(next_input, step_pos)
            step_cache_pos = torch.tensor([step + 1], device=device)

            h = next_input
            for li, layer in enumerate(cp_model.layers):
                residual = h
                normed = layer.input_layernorm(h)
                writer.record(
                    f"code-predictor-step{step}-frame{frame_index}-l{li}-input-norm",
                    normed, "BTH",
                )
                attn_out, _ = layer.self_attn(
                    hidden_states=normed,
                    attention_mask=None,
                    position_ids=step_pos,
                    past_key_values=cp_kv,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=step_cache_pos,
                    position_embeddings=step_pos_emb,
                )
                writer.record(
                    f"code-predictor-step{step}-frame{frame_index}-l{li}-attention-output",
                    attn_out, "BTH",
                )
                h = residual + attn_out
                residual = h
                post_normed = layer.post_attention_layernorm(h)
                writer.record(
                    f"code-predictor-step{step}-frame{frame_index}-l{li}-post-attention-norm",
                    post_normed, "BTH",
                )
                mlp_out = layer.mlp(post_normed)
                writer.record(
                    f"code-predictor-step{step}-frame{frame_index}-l{li}-mlp-output",
                    mlp_out, "BTH",
                )
                h = residual + mlp_out
                writer.record(
                    f"code-predictor-hidden-step{step}-frame{frame_index}-l{li}",
                    h, "BTH",
                )

            h = cp_model.norm(h)
            writer.record(
                f"code-predictor-hidden-step{step}-frame{frame_index}-final",
                h, "BTH",
            )

            logits = code_predictor.lm_head[step](h).squeeze(1)
            writer.record(
                f"code_predictor_step_logits_{frame_index:04}_{step:04}",
                logits, "C",
            )
            code_tokens.append(logits.argmax(dim=-1).reshape(1, 1))

        all_codes = torch.cat(code_tokens, dim=1)
        writer.record(
            f"code_predictor_final_code_matrix_{frame_index:04}",
            all_codes.float(), "C",
        )

        return all_codes

    # ── Custom generation loop ─────────────────────────────────────────────
    try:
        with torch.no_grad():
            # Compute position IDs for prefill
            position_ids = attention_mask.float().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)

            # ── Prefill ──
            hidden, kv_cache = run_talker_forward(
                input_embeds, attention_mask, position_ids, None, "prefill"
            )

            last_hidden = hidden[:, -1:, :]

            # Prefill logits
            logits = talker.codec_head(last_hidden).squeeze(1)
            writer.record("talker-logits-prefill", logits, "BV")
            writer.record("talker_codebook0_logits_0000", logits, "C")

            # Greedy sample codebook 0
            c0_token = logits.argmax(dim=-1).reshape(1, 1)

            # Terminal cap: with max_new_tokens=2, step 0 runs fully,
            # step 1 only produces logits then breaks (matching Candle
            # terminal_cap_step logic).
            # Step 0: full pipeline
            c0_emb = talker.get_input_embeddings()(c0_token)
            writer.record("talker-codec-embed-frame0", c0_emb, "BTH")

            # Code Predictor for frame 0
            codes_1_15 = run_code_predictor(last_hidden, c0_emb, 0)

            # Frame 0 codes
            frame0 = torch.cat([c0_token, codes_1_15], dim=1)

            # Compute next-emb-step0
            sum_emb = c0_emb
            for i in range(talker.config.num_code_groups - 1):
                ci_token = codes_1_15[:, i:i+1]
                ci_emb = code_predictor.get_input_embeddings()[i](ci_token)
                sum_emb = sum_emb + ci_emb

            text_add = trailing_text_hidden[:, 0:1, :]
            next_input = sum_emb + text_add
            writer.record("next-emb-step0", next_input, "BTH")

            # ── Step 1: Talker forward ──
            rope_deltas = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0] + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
            delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
            rope_deltas = rope_deltas - delta0

            seq_len = len(prompt_ids)
            cache_position_start = seq_len
            step1_pos = torch.arange(cache_position_start, cache_position_start + 1, device=attention_mask.device)
            step1_pos = step1_pos.view(1, -1).expand(1, -1)
            step1_pos = step1_pos.add(rope_deltas.squeeze())
            step1_pos = step1_pos.unsqueeze(0).expand(3, -1, -1)

            writer.record("talker-step1-codec-embed", next_input, "BTH")

            hidden1, kv_cache = run_talker_forward(
                next_input, attention_mask, step1_pos, kv_cache, "step1"
            )

            last_hidden1 = hidden1[:, -1:, :]
            logits1 = talker.codec_head(last_hidden1).squeeze(1)
            writer.record("talker-logits-step1", logits1, "BV")
            writer.record("talker_codebook0_logits_0001", logits1, "C")

            # Terminal cap: step 1 stops here (no CP frame1, no step2)
            # Final code matrix uses frame0 only
            writer.record("talker_final_code_matrix", frame0.float(), "C")

    except Exception as e:
        print(f"ERROR during generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    manifest_path = writer.commit()
    stage_count = len(writer.stages)
    print(f"Wrote {stage_count} stages to {manifest_path}")

    if stage_count != args.expected_stages:
        print(
            f"ERROR: stage count {stage_count} != expected {args.expected_stages}",
            file=sys.stderr,
        )
        # List missing stages vs Candle reference
        candle_manifest = Path("artifacts/alignment/P03/P03-T05/runs/candle-full/manifest.json")
        if candle_manifest.exists():
            candle = json.loads(candle_manifest.read_text(encoding="utf-8"))
            candle_names = set(s["name"] for s in candle["stages"])
            python_names = set(s["name"] for s in writer.stages)
            missing = sorted(candle_names - python_names)
            extra = sorted(python_names - candle_names)
            if missing:
                print(f"Missing {len(missing)} stages:", file=sys.stderr)
                for n in missing:
                    print(f"  {n}", file=sys.stderr)
            if extra:
                print(f"Extra {len(extra)} stages:", file=sys.stderr)
                for n in extra[:10]:
                    print(f"  {n}", file=sys.stderr)
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
