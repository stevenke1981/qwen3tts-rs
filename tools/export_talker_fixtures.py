#!/usr/bin/env python3
"""Export small PyTorch talker reference fixtures for Rust alignment tests."""

import argparse
import json
from pathlib import Path

import torch


def load_model(model_id: str):
    from qwen_tts import Qwen3TTSModel

    wrapper = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cpu",
        dtype=torch.float32,
        trust_remote_code=True,
    )
    wrapper.model.eval()
    return wrapper


def flatten(tensor: torch.Tensor):
    return tensor.detach().cpu().reshape(-1).tolist()


def export_text_projection(model, out_dir: Path):
    talker = model.talker
    input_ids = torch.tensor(
        [[
            model.config.tts_bos_token_id,
            model.config.tts_eos_token_id,
            model.config.tts_pad_token_id,
            198,
        ]],
        dtype=torch.long,
        device=talker.device,
    )

    with torch.no_grad():
        text_emb = talker.get_text_embeddings()(input_ids)
        output = talker.text_projection(text_emb)

    fixture = {
        "name": "talker_text_projection",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "input_ids": input_ids.cpu().tolist(),
        "shape": list(output.shape),
        "output": flatten(output),
    }
    out_path = out_dir / "talker_text_projection.json"
    out_path.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"wrote {out_path}")


def export_codec_embedding_and_head(model, out_dir: Path):
    talker = model.talker
    codec_ids = torch.tensor(
        [[
            model.config.talker_config.codec_bos_id,
            model.config.talker_config.codec_pad_id,
            model.config.talker_config.codec_eos_token_id,
            model.config.talker_config.codec_language_id["chinese"],
        ]],
        dtype=torch.long,
        device=talker.device,
    )
    hidden = (
        torch.arange(2 * model.config.talker_config.hidden_size, dtype=torch.float32, device=talker.device)
        .reshape(1, 2, model.config.talker_config.hidden_size)
        / 1024.0
    )

    with torch.no_grad():
        codec_embed = talker.get_input_embeddings()(codec_ids)
        codec_logits = talker.codec_head(hidden)

    fixture = {
        "name": "talker_codec_embedding_head",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "codec_ids": codec_ids.cpu().tolist(),
        "codec_embed_shape": list(codec_embed.shape),
        "codec_embed": flatten(codec_embed),
        "hidden_shape": list(hidden.shape),
        "hidden": flatten(hidden),
        "codec_logits_shape": list(codec_logits.shape),
        "codec_logits": flatten(codec_logits),
    }
    out_path = out_dir / "talker_codec_embedding_head.json"
    out_path.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"wrote {out_path}")


def causal_mask(seq_len: int, device):
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.reshape(1, 1, seq_len, seq_len)


def export_talker_attention_layer0(model, out_dir: Path):
    talker = model.talker
    cfg = model.config.talker_config
    seq_len = 3
    hidden = (
        torch.arange(seq_len * cfg.hidden_size, dtype=torch.float32, device=talker.device)
        .reshape(1, seq_len, cfg.hidden_size)
        / 1024.0
    )
    position_ids = torch.arange(seq_len, dtype=torch.long, device=talker.device)
    position_ids = position_ids.view(1, 1, seq_len).expand(3, 1, seq_len)
    mask = causal_mask(seq_len, talker.device)

    with torch.no_grad():
        position_embeddings = talker.model.rotary_emb(hidden, position_ids)
        output, _ = talker.model.layers[0].self_attn(
            hidden_states=hidden,
            position_embeddings=position_embeddings,
            attention_mask=mask,
            past_key_values=None,
            cache_position=torch.arange(seq_len, dtype=torch.long, device=talker.device),
        )

    fixture = {
        "name": "talker_attention_layer0",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "hidden_shape": list(hidden.shape),
        "hidden": flatten(hidden),
        "position_ids_shape": list(position_ids.shape),
        "position_ids": position_ids.cpu().reshape(-1).tolist(),
        "output_shape": list(output.shape),
        "output": flatten(output),
    }
    out_path = out_dir / "talker_attention_layer0.json"
    out_path.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"wrote {out_path}")


def export_talker_decoder_layer0(model, out_dir: Path):
    talker = model.talker
    cfg = model.config.talker_config
    seq_len = 3
    hidden = (
        torch.arange(seq_len * cfg.hidden_size, dtype=torch.float32, device=talker.device)
        .reshape(1, seq_len, cfg.hidden_size)
        / 1024.0
    )
    position_ids = torch.arange(seq_len, dtype=torch.long, device=talker.device)
    position_ids = position_ids.view(1, 1, seq_len).expand(3, 1, seq_len)
    mask = causal_mask(seq_len, talker.device)

    with torch.no_grad():
        position_embeddings = talker.model.rotary_emb(hidden, position_ids)
        output = talker.model.layers[0](
            hidden_states=hidden,
            attention_mask=mask,
            position_ids=position_ids[0],
            past_key_values=None,
            cache_position=torch.arange(seq_len, dtype=torch.long, device=talker.device),
            position_embeddings=position_embeddings,
        )[0]

    fixture = {
        "name": "talker_decoder_layer0",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "hidden_shape": list(hidden.shape),
        "hidden": flatten(hidden),
        "position_ids_shape": list(position_ids.shape),
        "position_ids": position_ids.cpu().reshape(-1).tolist(),
        "output_shape": list(output.shape),
        "output": flatten(output),
    }
    out_path = out_dir / "talker_decoder_layer0.json"
    out_path.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"wrote {out_path}")


def export_talker_model_prefill(model, out_dir: Path):
    talker = model.talker
    cfg = model.config.talker_config
    seq_len = 3
    hidden = (
        torch.arange(seq_len * cfg.hidden_size, dtype=torch.float32, device=talker.device)
        .reshape(1, seq_len, cfg.hidden_size)
        / 1024.0
    )
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=talker.device)

    with torch.no_grad():
        output = talker.model(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state

    fixture = {
        "name": "talker_model_prefill",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "hidden_shape": list(hidden.shape),
        "hidden": flatten(hidden),
        "output_shape": list(output.shape),
        "output": flatten(output),
    }
    out_path = out_dir / "talker_model_prefill.json"
    out_path.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"wrote {out_path}")


def export_code_predictor_first_step(model, out_dir: Path):
    talker = model.talker
    cfg = model.config.talker_config
    talker_hidden = (
        torch.arange(cfg.hidden_size, dtype=torch.float32, device=talker.device)
        .reshape(1, 1, cfg.hidden_size)
        / 1024.0
    )
    c0_token = torch.tensor([[cfg.codec_bos_id]], dtype=torch.long, device=talker.device)
    c0_embed = talker.get_input_embeddings()(c0_token)
    inputs_embeds = torch.cat([talker_hidden, c0_embed], dim=1)

    with torch.no_grad():
        output = talker.code_predictor(
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        logits = output.logits[:, -1, :]
        next_token = logits.argmax(dim=-1)

    fixture = {
        "name": "code_predictor_first_step",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "talker_hidden_shape": list(talker_hidden.shape),
        "talker_hidden": flatten(talker_hidden),
        "c0_token": c0_token.cpu().tolist(),
        "logits_shape": list(logits.shape),
        "logits": flatten(logits),
        "next_token": next_token.cpu().tolist(),
    }
    out_path = out_dir / "code_predictor_first_step.json"
    out_path.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"wrote {out_path}")


def export_code_predictor_greedy(model, out_dir: Path):
    talker = model.talker
    cfg = model.config.talker_config
    talker_hidden = (
        torch.arange(cfg.hidden_size, dtype=torch.float32, device=talker.device)
        .reshape(1, 1, cfg.hidden_size)
        / 1024.0
    )
    c0_token = torch.tensor([[cfg.codec_bos_id]], dtype=torch.long, device=talker.device)
    c0_embed = talker.get_input_embeddings()(c0_token)
    inputs_embeds = torch.cat([talker_hidden, c0_embed], dim=1)

    with torch.no_grad():
        output = talker.code_predictor.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=cfg.num_code_groups - 1,
            do_sample=False,
            use_cache=True,
        )

    fixture = {
        "name": "code_predictor_greedy",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "talker_hidden_shape": list(talker_hidden.shape),
        "talker_hidden": flatten(talker_hidden),
        "c0_token": c0_token.cpu().tolist(),
        "generated_shape": list(output.shape),
        "generated": output.cpu().tolist(),
    }
    out_path = out_dir / "code_predictor_greedy.json"
    out_path.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"wrote {out_path}")


def export_talker_single_frame(model, out_dir: Path):
    talker = model.talker
    cfg = model.config.talker_config
    seq_len = 3
    inputs_embeds = (
        torch.arange(seq_len * cfg.hidden_size, dtype=torch.float32, device=talker.device)
        .reshape(1, seq_len, cfg.hidden_size)
        / 1024.0
    )
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=talker.device)
    trailing_text_hidden = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float32, device=talker.device)
    tts_pad_embed = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float32, device=talker.device)

    with torch.no_grad():
        model_out = talker.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
        )
        last_hidden = model_out.last_hidden_state[:, -1:, :]
        c0_logits = talker.codec_head(last_hidden).squeeze(1)
        c0_token = c0_logits.argmax(dim=-1, keepdim=True)
        c0_embed = talker.get_input_embeddings()(c0_token)
        c1_15 = talker.code_predictor.generate(
            inputs_embeds=torch.cat([last_hidden, c0_embed], dim=1),
            max_new_tokens=cfg.num_code_groups - 1,
            do_sample=False,
            use_cache=True,
        )
        full_codes = torch.cat([c0_token, c1_15], dim=-1)

    fixture = {
        "name": "talker_single_frame",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "inputs_embeds_shape": list(inputs_embeds.shape),
        "inputs_embeds": flatten(inputs_embeds),
        "attention_mask_shape": list(attention_mask.shape),
        "attention_mask": attention_mask.cpu().reshape(-1).tolist(),
        "trailing_text_hidden_shape": list(trailing_text_hidden.shape),
        "trailing_text_hidden": flatten(trailing_text_hidden),
        "tts_pad_embed_shape": list(tts_pad_embed.shape),
        "tts_pad_embed": flatten(tts_pad_embed),
        "last_hidden_shape": list(last_hidden.shape),
        "last_hidden": flatten(last_hidden),
        "c0_token": c0_token.cpu().tolist(),
        "generated_shape": list(full_codes.shape),
        "generated": full_codes.cpu().tolist(),
    }
    out_path = out_dir / "talker_single_frame.json"
    out_path.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"wrote {out_path}")


def export_talker_two_frame(model, out_dir: Path):
    talker = model.talker
    cfg = model.config.talker_config
    seq_len = 3
    inputs_embeds = (
        torch.arange(seq_len * cfg.hidden_size, dtype=torch.float32, device=talker.device)
        .reshape(1, seq_len, cfg.hidden_size)
        / 1024.0
    )
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=talker.device)
    trailing_text_hidden = torch.zeros((1, 2, cfg.hidden_size), dtype=torch.float32, device=talker.device)
    tts_pad_embed = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float32, device=talker.device)
    frames = []

    with torch.no_grad():
        output = talker(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            use_cache=True,
        )
        next_c0 = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for step in range(2):
            output = talker(
                input_ids=next_c0,
                past_key_values=output.past_key_values,
                past_hidden=output.past_hidden,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                generation_step=output.generation_step,
                use_cache=True,
                cache_position=torch.tensor([seq_len + step], dtype=torch.long, device=talker.device),
            )
            frames.append(output.hidden_states[1])
            next_c0 = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        full_codes = torch.cat(frames, dim=0)

    fixture = {
        "name": "talker_two_frame",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "inputs_embeds_shape": list(inputs_embeds.shape),
        "inputs_embeds": flatten(inputs_embeds),
        "attention_mask_shape": list(attention_mask.shape),
        "attention_mask": attention_mask.cpu().reshape(-1).tolist(),
        "trailing_text_hidden_shape": list(trailing_text_hidden.shape),
        "trailing_text_hidden": flatten(trailing_text_hidden),
        "tts_pad_embed_shape": list(tts_pad_embed.shape),
        "tts_pad_embed": flatten(tts_pad_embed),
        "generated_shape": list(full_codes.shape),
        "generated": full_codes.cpu().tolist(),
    }
    out_path = out_dir / "talker_two_frame.json"
    out_path.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"wrote {out_path}")


def export_talker_prompt_input_builder(wrapper, out_dir: Path):
    model = wrapper.model
    talker = model.talker
    cfg = model.config.talker_config
    text = "你好"
    language = "Chinese"
    input_text = wrapper._build_assistant_text(text)
    input_id = wrapper._tokenize_texts([input_text])[0]
    language_id = cfg.codec_language_id[language.lower()]

    with torch.no_grad():
        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            talker.get_text_embeddings()(
                torch.tensor(
                    [[
                        model.config.tts_bos_token_id,
                        model.config.tts_eos_token_id,
                        model.config.tts_pad_token_id,
                    ]],
                    device=talker.device,
                    dtype=input_id.dtype,
                )
            )
        ).chunk(3, dim=1)

        codec_prefill = [[
            cfg.codec_think_id,
            cfg.codec_think_bos_id,
            language_id,
            cfg.codec_think_eos_id,
        ]]
        codec_input_embedding_0 = talker.get_input_embeddings()(
            torch.tensor(codec_prefill, device=talker.device, dtype=input_id.dtype)
        )
        codec_input_embedding_1 = talker.get_input_embeddings()(
            torch.tensor(
                [[cfg.codec_pad_id, cfg.codec_bos_id]],
                device=talker.device,
                dtype=input_id.dtype,
            )
        )
        codec_input_embedding = torch.cat([codec_input_embedding_0, codec_input_embedding_1], dim=1)

        role_embed = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
        codec_input = torch.cat(
            [
                tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
                tts_bos_embed,
            ],
            dim=1,
        ) + codec_input_embedding[:, :-1]
        inputs_embeds = torch.cat([role_embed, codec_input], dim=1)
        first_text = (
            talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:4]))
            + codec_input_embedding[:, -1:]
        )
        inputs_embeds = torch.cat([inputs_embeds, first_text], dim=1)
        trailing_text_hidden = torch.cat(
            [
                talker.text_projection(talker.get_text_embeddings()(input_id[:, 4:-5])),
                tts_eos_embed,
            ],
            dim=1,
        )
        attention_mask = torch.ones((1, inputs_embeds.shape[1]), dtype=torch.long, device=talker.device)

    fixture = {
        "name": "talker_prompt_input_builder",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "text": text,
        "language": language,
        "input_text": input_text,
        "input_ids_shape": list(input_id.shape),
        "input_ids": input_id.cpu().reshape(-1).tolist(),
        "inputs_embeds_shape": list(inputs_embeds.shape),
        "inputs_embeds": flatten(inputs_embeds),
        "attention_mask_shape": list(attention_mask.shape),
        "attention_mask": attention_mask.cpu().reshape(-1).tolist(),
        "trailing_text_hidden_shape": list(trailing_text_hidden.shape),
        "trailing_text_hidden": flatten(trailing_text_hidden),
        "tts_pad_embed_shape": list(tts_pad_embed.shape),
        "tts_pad_embed": flatten(tts_pad_embed),
    }
    out_path = out_dir / "talker_prompt_input_builder.json"
    out_path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Hugging Face model id or local model path",
    )
    parser.add_argument(
        "--out-dir",
        default="tests/fixtures",
        help="Directory for JSON fixture output",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = load_model(args.model)
    model = wrapper.model
    export_text_projection(model, out_dir)
    export_codec_embedding_and_head(model, out_dir)
    export_talker_attention_layer0(model, out_dir)
    export_talker_decoder_layer0(model, out_dir)
    export_talker_model_prefill(model, out_dir)
    export_code_predictor_first_step(model, out_dir)
    export_code_predictor_greedy(model, out_dir)
    export_talker_single_frame(model, out_dir)
    export_talker_two_frame(model, out_dir)
    export_talker_prompt_input_builder(wrapper, out_dir)


if __name__ == "__main__":
    main()
