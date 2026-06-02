"""Test: save tokens and verify binary format."""

import sys, struct

sys.path.insert(0, "tools")
from generate_tokens import load_model, generate_codes

text = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "Hello, this is a test of the Qwen3 TTS system."
)
out_path = sys.argv[2] if len(sys.argv) > 2 else "tokens.bin"
lang = sys.argv[3] if len(sys.argv) > 3 else "english"

model = load_model("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
codes_list = generate_codes(model, text, language=lang, max_new_tokens=2048)

with open(out_path, "wb") as f:
    for codes in codes_list:
        num = codes.shape[0]
        f.write(struct.pack("<I", num))
        for frame in codes:
            for t in frame:
                f.write(struct.pack("<H", int(t)))

n = codes_list[0].shape[0]
print(f"Saved {n} frames ({n * 32} bytes) -> {out_path}", file=sys.stderr)

# Verify
with open(out_path, "rb") as f:
    data = f.read()
n2 = struct.unpack("<I", data[:4])[0]
print(f"Verified: {n2} frames, total {len(data)} bytes", file=sys.stderr)
