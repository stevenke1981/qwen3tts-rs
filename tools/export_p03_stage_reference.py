#!/usr/bin/env python3
"""Export the full 723-stage official-Python reference manifest for P03-T05.

Instruments the official Qwen3-TTS PyTorch Talker and Code Predictor to capture
every intermediate tensor at the same points as the Candle implementation, then
writes a manifest and F32 binary files compatible with compare_stage_dumps.py.

Pins: official source revision, model snapshot revision, case, seed, stage
names/layouts, and file hashes.

Usage:
    python tools/export_p03_stage_reference.py \
        --official-source C:\\Users\\steven\\Qwen3-TTS \
        --official-revision 022e286b98fbec7e1e916cb940cdf532cd9f488e \
        --model C:\\Users\\steven\\.cache\\huggingface\\hub\\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\\snapshots\\5d83992436eae1d760afd27aff78a71d676296fc \
        --model-revision 5d83992436eae1d760afd27aff78a71d676296fc \
        --case-id p03-t01-real \
        --seed 42 \
        --output-dir artifacts/alignment/P03/P03-T05/runs/official-python-full \
        --expected-stages 723
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Stage writer
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Instrumentation hooks
# ---------------------------------------------------------------------------

class TalkerInstrumentor:
    """Monkey-patches the official Talker to capture all 723 stages."""

    def __init__(self, writer: StageWriter, torch_mod: Any):
        self.w = writer
        self.torch = torch_mod
        self.generation_step = -1  # -1 = prefill
        self.frame_index = 0
        self.cp_frame_index = 0
        self._hooks: list = []

    def _record_tensor(self, name: str, t, layout: str) -> None:
        self.w.record(name, t, layout)

    # ── Talker decoder layer hooks ──────────────────────────────────────────

    def _make_talker_layer_hook(self, layer_idx: int):
        """Returns a forward hook for a Talker decoder layer."""
        w = self.w
        torch_mod = self.torch

        def hook(module, args, kwargs, output):
            # Determine phase
            step = self.generation_step
            if step == -1:
                phase = "prefill"
            else:
                phase = f"step{step + 1}"

            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return output

            # The layer output is (hidden_states,) or (hidden_states, attn_weights)
            layer_out = output[0] if isinstance(output, tuple) else output

            # We need to capture sub-stages. Since we can't easily hook into
            # the middle of the layer's forward, we recompute the sub-stages
            # from the layer's internal state.
            # Instead, we use pre/post hooks on the sub-modules.
            return output

        return hook

    def _make_talker_sublayer_hooks(self, layer, layer_idx: int):
        """Register hooks on a Talker decoder layer's sub-modules."""
        w = self.w
        torch_mod = self.torch
        instr = self

        # input_layernorm output = input-norm stage
        def input_norm_hook(module, args, output):
            step = instr.generation_step
            phase = "prefill" if step == -1 else f"step{step + 1}"
            name = f"talker-{phase}-l{layer_idx}-input-norm"
            w.record(name, output, "BTH")

        # self_attn output = attention-output stage
        def attn_output_hook(module, args, kwargs, output):
            step = instr.generation_step
            phase = "prefill" if step == -1 else f"step{step + 1}"
            name = f"talker-{phase}-l{layer_idx}-attention-output"
            attn_out = output[0] if isinstance(output, tuple) else output
            w.record(name, attn_out, "BTH")

        # post_attention_layernorm output = post-attention-norm stage
        def post_norm_hook(module, args, output):
            step = instr.generation_step
            phase = "prefill" if step == -1 else f"step{step + 1}"
            name = f"talker-{phase}-l{layer_idx}-post-attention-norm"
            w.record(name, output, "BTH")

        # mlp output = mlp-output stage
        def mlp_output_hook(module, args, output):
            step = instr.generation_step
            phase = "prefill" if step == -1 else f"step{step + 1}"
            name = f"talker-{phase}-l{layer_idx}-mlp-output"
            w.record(name, output, "BTH")

        self._hooks.append(layer.input_layernorm.register_forward_hook(input_norm_hook))
        self._hooks.append(
            layer.self_attn.register_forward_hook(attn_output_hook, with_kwargs=True)
        )
        self._hooks.append(
            layer.post_attention_layernorm.register_forward_hook(post_norm_hook)
        )
        self._hooks.append(layer.mlp.register_forward_hook(mlp_output_hook))

    # ── Code Predictor decoder layer hooks ─────────────────────────────────

    def _make_cp_sublayer_hooks(self, layer, layer_idx: int):
        """Register hooks on a Code Predictor decoder layer's sub-modules."""
        w = self.w
        instr = self

        def input_norm_hook(module, args, output):
            step = instr.generation_step
            cp_step = instr._cp_step
            frame = instr.cp_frame_index
            if cp_step == 0:
                phase = f"prefill-frame{frame}"
            else:
                phase = f"step{cp_step}-frame{frame}"
            name = f"code-predictor-{phase}-l{layer_idx}-input-norm"
            w.record(name, output, "BTH")

        def attn_output_hook(module, args, kwargs, output):
            step = instr.generation_step
            cp_step = instr._cp_step
            frame = instr.cp_frame_index
            if cp_step == 0:
                phase = f"prefill-frame{frame}"
            else:
                phase = f"step{cp_step}-frame{frame}"
            name = f"code-predictor-{phase}-l{layer_idx}-attention-output"
            attn_out = output[0] if isinstance(output, tuple) else output
            w.record(name, attn_out, "BTH")

        def post_norm_hook(module, args, output):
            step = instr.generation_step
            cp_step = instr._cp_step
            frame = instr.cp_frame_index
            if cp_step == 0:
                phase = f"prefill-frame{frame}"
            else:
                phase = f"step{cp_step}-frame{frame}"
            name = f"code-predictor-{phase}-l{layer_idx}-post-attention-norm"
            w.record(name, output, "BTH")

        def mlp_output_hook(module, args, output):
            step = instr.generation_step
            cp_step = instr._cp_step
            frame = instr.cp_frame_index
            if cp_step == 0:
                phase = f"prefill-frame{frame}"
            else:
                phase = f"step{cp_step}-frame{frame}"
            name = f"code-predictor-{phase}-l{layer_idx}-mlp-output"
            w.record(name, output, "BTH")

        self._hooks.append(layer.input_layernorm.register_forward_hook(input_norm_hook))
        self._hooks.append(
            layer.self_attn.register_forward_hook(attn_output_hook, with_kwargs=True)
        )
        self._hooks.append(
            layer.post_attention_layernorm.register_forward_hook(post_norm_hook)
        )
        self._hooks.append(layer.mlp.register_forward_hook(mlp_output_hook))

    # ── Talker model forward wrapper ───────────────────────────────────────

    def wrap_talker_model(self, talker_model):
        """Wrap Qwen3TTSTalkerModel.forward to capture hidden states."""
        orig_forward = talker_model.forward
        instr = self
        w = self.w

        def wrapped_forward(*args, **kwargs):
            output = orig_forward(*args, **kwargs)

            step = instr.generation_step
            phase = "prefill" if step == -1 else f"step{step + 1}"

            # Capture hidden states per layer from output_hidden_states
            if hasattr(output, "hidden_states") and output.hidden_states is not None:
                hs = output.hidden_states
                # hs[0] = input embed, hs[1..N] = after each layer, hs[-1] = after norm
                for i in range(1, len(hs) - 1):
                    name = f"talker-hidden-{phase}-l{i - 1}"
                    w.record(name, hs[i], "BTH")
                # Final norm output
                w.record(f"talker-hidden-{phase}-final", hs[-1], "BTH")
                if phase != "prefill":
                    w.record(f"talker-hidden-{phase}", hs[-1], "BTH")

            return output

        talker_model.forward = wrapped_forward

    def wrap_talker_for_gen(self, talker_gen):
        """Wrap Qwen3TTSTalkerForConditionalGeneration.forward to capture
        input embeddings, position IDs, RoPE, logits, and codec embeds."""
        orig_forward = talker_gen.forward
        instr = self
        w = self.w
        torch_mod = self.torch

        def wrapped_forward(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            cache_position=None,
            past_hidden=None,
            trailing_text_hidden=None,
            tts_pad_embed=None,
            generation_step=None,
            **kwargs,
        ):
            # Determine phase
            if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
                instr.generation_step = -1
            else:
                if generation_step is not None:
                    instr.generation_step = generation_step
                else:
                    instr.generation_step += 1

            step = instr.generation_step
            phase = "prefill" if step == -1 else f"step{step + 1}"

            # Capture input embeddings
            if inputs_embeds is not None:
                if step == -1:
                    w.record("talker-input-embed", inputs_embeds, "BTH")
                w.record(f"talker-input-{phase}", inputs_embeds, "BTH")

            # Capture position IDs and RoPE before calling the model
            # We need to compute them the same way the model does
            if attention_mask is not None:
                if (
                    cache_position is None
                    or (cache_position is not None and cache_position[0] == 0)
                    or talker_gen.rope_deltas is None
                ):
                    delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                    pos_ids, rope_deltas = talker_gen.get_rope_index(attention_mask)
                    rope_deltas = rope_deltas - delta0
                    talker_gen.rope_deltas = rope_deltas
                else:
                    batch_size, seq_length = input_ids.shape if input_ids is not None else (1, 1)
                    delta = cache_position[0] + talker_gen.rope_deltas if cache_position is not None else 0
                    pos_ids = torch_mod.arange(seq_length, device=attention_mask.device)
                    pos_ids = pos_ids.view(1, -1).expand(batch_size, -1)
                    pos_ids = pos_ids.add(delta)
                    pos_ids = pos_ids.unsqueeze(0).expand(3, -1, -1)

                if step == -1:
                    w.record("talker-prefill-position-ids", pos_ids, "ABT")

                # Compute RoPE
                talker_model = talker_gen.model
                if inputs_embeds is not None:
                    position_embeddings = talker_model.rotary_emb(inputs_embeds, pos_ids)
                    cos, sin = position_embeddings
                    if step == -1:
                        w.record("talker-prefill-rope-cos", cos.unsqueeze(1), "BBTH")
                        w.record("talker-prefill-rope-sin", sin.unsqueeze(1), "BBTH")
                    else:
                        w.record(f"talker-{phase}-rope-cos", cos.unsqueeze(1), "BBTH")
                        w.record(f"talker-{phase}-rope-sin", sin.unsqueeze(1), "BBTH")

            # Call original forward
            output = orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                past_hidden=past_hidden,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                generation_step=generation_step,
                **kwargs,
            )

            # Capture logits
            if hasattr(output, "logits") and output.logits is not None:
                logits = output.logits
                # Take last position logits
                last_logits = logits[:, -1, :]
                if step == -1:
                    w.record("talker-logits-prefill", last_logits, "BV")
                else:
                    w.record(f"talker-logits-step{step}", last_logits, "BV")

            return output

        talker_gen.forward = wrapped_forward

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Validate official source
    official_src = args.official_source
    if not official_src.exists():
        print(f"ERROR: official source not found: {official_src}", file=sys.stderr)
        return 1

    # Add official source to path so we can import qwen_tts
    site_packages = official_src / ".venv" / "Lib" / "site-packages"
    if site_packages.exists():
        sys.path.insert(0, str(site_packages))

    import torch

    # Load fixture
    fixture_path = args.fixture
    if not fixture_path.exists():
        print(f"ERROR: fixture not found: {fixture_path}", file=sys.stderr)
        return 1
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    # Validate fixture revisions
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

    # Load model
    print(f"Loading model from {args.model} ...")
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        str(args.model),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    # Qwen3TTSModel is a high-level wrapper; the actual nn.Module is model.model
    pytorch_model = model.model
    pytorch_model.eval()

    # Get the Talker
    talker = pytorch_model.talker  # Qwen3TTSTalkerForConditionalGeneration
    talker_model = talker.model  # Qwen3TTSTalkerModel
    code_predictor = talker.code_predictor  # Qwen3TTSTalkerCodePredictorModelForConditionalGeneration

    # Set up stage writer
    writer = StageWriter(
        args.output_dir,
        {
            "source": "official-python",
            "revision": args.model_revision,
            "model": "Qwen3-TTS-12Hz-0.6B-Base",
            "case_id": args.case_id,
            "seed": args.seed,
        },
    )

    # Set up instrumentor
    instr = TalkerInstrumentor(writer, torch)
    instr._cp_step = 0

    # Register hooks on Talker decoder layers
    for i, layer in enumerate(talker_model.layers):
        instr._make_talker_sublayer_hooks(layer, i)

    # Register hooks on Code Predictor decoder layers
    cp_model = code_predictor.model  # Qwen3TTSTalkerCodePredictorModel
    for i, layer in enumerate(cp_model.layers):
        instr._make_cp_sublayer_hooks(layer, i)

    # Wrap Talker model forward for hidden states
    instr.wrap_talker_model(talker_model)

    # Wrap Talker generation forward for logits, position IDs, RoPE
    instr.wrap_talker_for_gen(talker)

    # Wrap Code Predictor to track step and frame index
    orig_cp_generate = code_predictor.generate
    orig_cp_model_forward = cp_model.forward
    instr._cp_forward_count = 0  # counts forward calls within one generate()

    def wrapped_cp_model_forward(*cp_args, **cp_kwargs):
        # Track step: first call is prefill (step 0), subsequent are steps 1..14
        inputs_embeds = cp_kwargs.get("inputs_embeds")
        if inputs_embeds is None and cp_args:
            inputs_embeds = cp_args[0]
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            instr._cp_step = 0
            instr._cp_forward_count = 0
        else:
            instr._cp_forward_count += 1
            instr._cp_step = instr._cp_forward_count

        frame = instr.cp_frame_index
        cp_step = instr._cp_step

        # Capture input embed
        if inputs_embeds is not None:
            if cp_step == 0:
                writer.record(
                    f"code-predictor-prefill-frame{frame}-input-embed",
                    inputs_embeds,
                    "BTH",
                )
            else:
                writer.record(
                    f"code-predictor-step{cp_step}-frame{frame}-codec-embed",
                    inputs_embeds,
                    "BTH",
                )

        result = orig_cp_model_forward(*cp_args, **cp_kwargs)

        # Capture hidden states from output
        if hasattr(result, "hidden_states") and result.hidden_states is not None:
            hs = result.hidden_states
            # hs[0] = input embed, hs[1..N] = after each layer, hs[-1] = after norm
            if cp_step == 0:
                for li in range(1, len(hs) - 1):
                    writer.record(
                        f"code-predictor-hidden-prefill-frame{frame}-l{li - 1}",
                        hs[li],
                        "BTH",
                    )
                if len(hs) > 1:
                    writer.record(
                        f"code-predictor-hidden-prefill-frame{frame}-final",
                        hs[-1],
                        "BTH",
                    )
            else:
                for li in range(1, len(hs) - 1):
                    writer.record(
                        f"code-predictor-hidden-step{cp_step}-frame{frame}-l{li - 1}",
                        hs[li],
                        "BTH",
                    )
                if len(hs) > 1:
                    writer.record(
                        f"code-predictor-hidden-step{cp_step}-frame{frame}-final",
                        hs[-1],
                        "BTH",
                    )

        return result

    cp_model.forward = wrapped_cp_model_forward

    def wrapped_cp_generate(*gen_args, **gen_kwargs):
        gen_kwargs["output_hidden_states"] = True
        gen_kwargs["return_dict_in_generate"] = True
        instr._cp_forward_count = 0
        instr._cp_step = 0
        result = orig_cp_generate(*gen_args, **gen_kwargs)
        instr.cp_frame_index += 1
        return result

    code_predictor.generate = wrapped_cp_generate

    # Build input from fixture
    prompt_ids = fixture["prompt_ids"][0]
    language = fixture["language"]
    seed = fixture["seed"]
    max_new_tokens = 2  # Match the Candle test

    print(f"Running generation: prompt_ids={prompt_ids}, seed={seed}, max_new_tokens={max_new_tokens}")

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Build input embeddings
    # The Talker expects inputs_embeds constructed from prompt_ids
    # We need to replicate the InputBuilder logic
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    # Get text embeddings and project
    text_emb = talker_model.text_embedding(input_ids)
    # Apply text projection
    text_proj = talker.text_projection(text_emb)

    # Get codec embedding for the BOS token
    # The Talker uses codec_embedding for codec tokens
    # For prefill, we use the text projection as input

    # Create attention mask
    attention_mask = torch.ones(1, len(prompt_ids), dtype=torch.long)

    # Trailing text hidden and tts_pad_embed
    trailing_text_hidden = text_proj  # Use text projection as trailing
    tts_pad_embed = torch.zeros(1, 1, talker.config.hidden_size)

    # Run generation
    try:
        with torch.no_grad():
            output = talker.generate(
                inputs_embeds=text_proj,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=fixture["talker"]["do_sample"],
                temperature=fixture["talker"]["temperature"],
                top_k=fixture["talker"]["top_k"],
                top_p=fixture["talker"]["top_p"],
                repetition_penalty=fixture["talker"]["repetition_penalty"],
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
    except Exception as e:
        print(f"ERROR during generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Remove hooks
    instr.remove_hooks()

    # Commit manifest
    manifest_path = writer.commit()
    stage_count = len(writer.stages)
    print(f"Wrote {stage_count} stages to {manifest_path}")

    if stage_count != args.expected_stages:
        print(
            f"ERROR: stage count {stage_count} != expected {args.expected_stages}",
            file=sys.stderr,
        )
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
