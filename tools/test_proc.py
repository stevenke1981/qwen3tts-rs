"""Quick test of processor output."""

from qwen_tts import Qwen3TTSModel
import torch

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cpu",
    dtype=torch.float32,
    trust_remote_code=True,
)

inputs = model.processor(text=["hello world"], return_tensors="pt")
print(f"input_ids shape: {inputs['input_ids'].shape}")
print(f"input_ids: {inputs['input_ids']}")
print(f"Decoded: {model.processor.tokenizer.decode(inputs['input_ids'][0])}")
print(f"Num tokens: {inputs['input_ids'].shape[1]}")
