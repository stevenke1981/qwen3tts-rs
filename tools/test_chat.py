"""Test chat template formatting."""

from qwen_tts import Qwen3TTSModel
import torch

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cpu",
    dtype=torch.float32,
    trust_remote_code=True,
)

tokenizer = model.processor.tokenizer

# Check the chat template
print(f"Chat template: {tokenizer.chat_template[:200]}")
print()

# Try applying the template
messages = [{"role": "assistant", "content": "hello world"}]
encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
print(f"Encoded shape: {encoded.shape}")
print(f"Encoded: {encoded}")
print(f"Decoded: {tokenizer.decode(encoded[0])}")
