"""Python Qwen3-TTS 對照測試"""
import time, os, sys
import torch
from transformers import AutoProcessor, Qwen2TTSTalker, Qwen2TTSForConditionalGeneration

model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
text = "主人您好，今天要測試語音合成會不會把後面的句子截斷。這是一段比較長的文字，包含好幾個句子。第一句是開場白。第二句是繼續說明。第三句是更多內容。第四句是測試看看後面的文字是否都能夠順利合成出來。第五句是最後一句，如果能夠完整聽到這一句，就代表沒有被截斷。"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load
t0 = time.time()
print(f"Loading model {model_id}...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = Qwen2TTSForConditionalGeneration.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.float16
).to(device)
load_time = time.time() - t0
print(f"Load time: {load_time:.2f}s")

# Generate
t1 = time.time()
print("Generating...")
inputs = processor(text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300)
gen_time = time.time() - t1
print(f"Generation time: {gen_time:.2f}s")

# Save
output_path = sys.argv[1] if len(sys.argv) > 1 else "/d/qwen3tts-rs/py_output.wav"
if hasattr(outputs, 'waveform'):
    import soundfile as sf
    waveform = outputs.waveform.cpu().numpy()
    sf.write(output_path, waveform[0], samplerate=24000)
    size = os.path.getsize(output_path)
    duration = waveform.shape[1] / 24000
    print(f"Output: {output_path}")
    print(f"Duration: {duration:.2f}s")
    print(f"Size: {size/1024:.1f}KB")
else:
    print(f"Output keys: {outputs.keys() if hasattr(outputs, 'keys') else dir(outputs)[:10]}")
    print(f"Output type: {type(outputs)}")

total = time.time() - t0
print(f"Total: {total:.2f}s")
