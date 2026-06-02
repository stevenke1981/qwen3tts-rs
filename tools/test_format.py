"""Test the correct input format for model.generate()."""

from qwen_tts import Qwen3TTSModel
import torch

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cpu",
    dtype=torch.float32,
    trust_remote_code=True,
)
mm = model.model
tokenizer = model.processor.tokenizer
device = next(mm.parameters()).device

# Check special tokens in tokenizer
print("Special tokens:")
for name, tid in sorted(tokenizer.special_tokens_map.items()):
    print(f"  {name}: {tid}")
print()

# Build the full conversation format manually
# Format: <|im_start|>assistant\nTEXT
im_start = tokenizer("<|im_start|>", return_tensors="pt")["input_ids"][0]
im_end = tokenizer("<|im_end|>", return_tensors="pt")["input_ids"][0]
assistant_tokens = tokenizer("assistant\n", return_tensors="pt")["input_ids"][0]
newline = tokenizer("\n", return_tensors="pt")["input_ids"][0]

print(f"im_start: {im_start} ({tokenizer.decode(im_start)})")
print(f"assistant: {assistant_tokens} ({tokenizer.decode(assistant_tokens)})")
print(f"im_end: {im_end} ({tokenizer.decode(im_end)})")
print()

# Full format: <|im_start|>assistant\ntext<|im_end|>\n<|im_start|>assistant\n
text_tokens = tokenizer("Hello world", return_tensors="pt")["input_ids"][0]
full_ids = torch.cat(
    [
        im_start,
        assistant_tokens,
        text_tokens,
        im_end,
        newline,
        im_start,
        assistant_tokens,
    ]
)
print(f"Full {len(full_ids)} tokens: {full_ids}")
print(f"Decoded: {tokenizer.decode(full_ids)}")
print()

# Call generate with the right format
input_ids = [full_ids.unsqueeze(0).to(device)]
languages = ["english"]
speakers = [None]

print("Calling model.generate()...")
with torch.no_grad():
    codes, hiddens = mm.generate(
        input_ids=input_ids,
        languages=languages,
        speakers=speakers,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        max_new_tokens=128,
        repetition_penalty=1.05,
    )

print(f"Codes shape: {codes[0].shape}")
print(f"First frame: {codes[0][0]}")
print(f"Num frames: {len(codes[0])}")
