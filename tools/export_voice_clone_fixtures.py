#!/usr/bin/env python3
"""Export PyTorch voice-clone fixtures for Rust native alignment tests."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def flatten(tensor: torch.Tensor):
    return tensor.detach().cpu().float().reshape(-1).tolist()


def deterministic_waveform(samples: int = 24000, sr: int = 24000) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / float(sr)
    wav = 0.08 * np.sin(2.0 * np.pi * 220.0 * t)
    wav += 0.03 * np.sin(2.0 * np.pi * 440.0 * t + 0.25)
    wav += 0.01 * np.sin(2.0 * np.pi * 880.0 * t + 0.5)
    return wav.astype(np.float32)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Base model used for speaker encoder and speech tokenizer",
    )
    parser.add_argument(
        "--out",
        default="tests/fixtures/voice_clone_native.json",
        help="Output fixture JSON path",
    )
    args = parser.parse_args()

    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    model = load_model(args.model)
    wav = deterministic_waveform()

    with torch.inference_mode():
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        mels = mel_spectrogram(
            wav_t,
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        speaker = model.model.speaker_encoder(mels.to(model.model.device).to(model.model.dtype))[0]
        encoded = model.model.speech_tokenizer.encode(wav, sr=24000)
        codes = encoded.audio_codes[0].detach().cpu().to(torch.int64)

    fixture = {
        "name": "voice_clone_native",
        "model": args.model,
        "sample_rate": 24000,
        "waveform_shape": [1, int(wav.shape[0])],
        "waveform": wav.reshape(-1).tolist(),
        "mel_shape": list(mels.shape),
        "mel": flatten(mels),
        "speaker_shape": list(speaker.shape),
        "speaker": flatten(speaker),
        "codes_shape": list(codes.shape),
        "codes": codes.tolist(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
