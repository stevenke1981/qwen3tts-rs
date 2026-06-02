"""Generate a spectrogram-like ASCII plot of the audio."""

import struct, sys, numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "output.wav"

with open(path, "rb") as f:
    d = f.read()
rate = struct.unpack("<I", d[24:28])[0]
data_start = d.find(b"data") + 8
data_size = struct.unpack("<I", d[40:44])[0]
samples = np.frombuffer(d[data_start : data_start + data_size], dtype=np.int16).astype(
    float
)

# Normalize to [-1, 1]
samples /= 32768.0

# Time-domain waveform (downsampled for display)
downsample = int(rate / 100)  # 100 points per second
if downsample < 1:
    downsample = 1
peaks = []
for i in range(0, len(samples), downsample):
    frame = samples[i : i + downsample]
    peaks.append(np.max(np.abs(frame)) if len(frame) > 0 else 0)

print("=== Time Domain Envelope ===")
h = 40
for p in peaks:
    bar = int(p * h)
    if bar > 0:
        print("█" * min(bar, h))
    else:
        print("·")

# Energy per second
print()
print("=== Per-Second RMS Energy ===")
sec_samples = rate
for i in range(0, min(len(samples), 6 * sec_samples), sec_samples):
    frame = samples[i : i + sec_samples]
    rms = np.sqrt(np.mean(frame**2))
    print(f"  sec {i // sec_samples}: RMS={rms:.4f} | {'█' * int(rms * 80)}")
