"""Python Qwen3-TTS 對照測試 v2"""
import time, os, sys, warnings
warnings.filterwarnings("ignore")

import torch
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

text = "主人您好，今天要測試語音合成會不會把後面的句子截斷。這是一段比較長的文字，包含好幾個句子。第一句是開場白。第二句是繼續說明。第三句是更多內容。第四句是測試看看後面的文字是否都能夠順利合成出來。第五句是最後一句，如果能夠完整聽到這一句，就代表沒有被截斷。"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load
t0 = time.time()
print("Loading Python model...")
model_dir = "C:/Users/steven/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3"
model = Qwen3TTSModel.from_pretrained(model_dir, torch_dtype=torch.float16).to(device)
tokenizer = Qwen3TTSTokenizer.from_pretrained(model_dir)
load_time = time.time() - t0
print(f"Load time: {load_time:.2f}s")

# Generate
t1 = time.time()
print("Generating...")
outputs = model.generate(
    text=text,
    tokenizer=tokenizer,
    language="chinese",
    device=device,
    max_new_tokens=300,
)
gen_time = time.time() - t1
print(f"Generation time: {gen_time:.2f}s")

# Save
output_path = sys.argv[1] if len(sys.argv) > 1 else "/d/qwen3tts-rs/py_output.wav"
waveform = None
if isinstance(outputs, dict):
    waveform = outputs.get('waveform')
    frames = outputs.get('frames', 'N/A')
else:
    waveform = getattr(outputs, 'waveform', None)
    frames = 'N/A'

if waveform is not None:
    import soundfile as sf
    if torch.is_tensor(waveform):
        waveform = waveform.cpu().numpy()
    sf.write(output_path, waveform[0] if waveform.ndim > 1 else waveform, samplerate=24000)
    size = os.path.getsize(output_path)
    duration = waveform.shape[-1] / 24000
    print(f"Output: {output_path}")
    print(f"Duration: {duration:.2f}s")
    print(f"Size: {size/1024:.1f}KB")
    print(f"Frames: {frames}")
else:
    print(f"Output type: {type(outputs)}")
    print(f"Keys: {outputs.keys() if hasattr(outputs, 'keys') else dir(outputs)[:10]}")

total = time.time() - t0
print(f"Total: {total:.2f}s")
