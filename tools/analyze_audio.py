"""Analyze the WAV output quality.

Focus on whether it contains actual speech or just noise.
"""

import struct
import sys
import numpy as np
import math

path = sys.argv[1] if len(sys.argv) > 1 else "output_norm.wav"

with open(path, "rb") as f:
    d = f.read()

# Parse WAV
rate = struct.unpack("<I", d[24:28])[0]
channels = struct.unpack("<H", d[22:24])[0]
bits = struct.unpack("<H", d[34:36])[0]
data_size = struct.unpack("<I", d[40:44])[0]
data_start = d.find(b"data") + 8

# Read samples
samples = np.frombuffer(d[data_start : data_start + data_size], dtype=np.int16)
samples_f32 = samples.astype(np.float32) / 32768.0

print(f"=== Audio Quality Report ===")
print(f"Duration: {len(samples_f32) / rate:.2f}s")
print(f"Sample rate: {rate}Hz")
print(f"Channels: {channels}")
print()

# Basic stats
print("=== Waveform Stats ===")
print(f"Peak positive: {samples_f32.max():.6f}")
print(f"Peak negative: {samples_f32.min():.6f}")
print(f"RMS: {np.sqrt(np.mean(samples_f32**2)):.6f}")
print(f"DC offset: {samples_f32.mean():.6f}")
print(
    f"Zero crossing rate: {np.sum(np.abs(np.diff(np.signbit(samples_f32).astype(int)))) / len(samples_f32):.4f}"
)
print()

# Spectral analysis (simple FFT)
fft = np.fft.rfft(samples_f32 * np.hanning(len(samples_f32)))
freqs = np.fft.rfftfreq(len(samples_f32), 1.0 / rate)
power = np.abs(fft) ** 2

# Find dominant frequencies
peak_bins = np.argsort(power)[-5:]
print("=== Top 5 Frequency Peaks ===")
for b in reversed(peak_bins):
    print(f"  {freqs[b]:6.1f} Hz: power={power[b]:.1f}")

# Energy in speech bands
speech_low = power[(freqs >= 80) & (freqs < 300)].sum()
speech_mid = power[(freqs >= 300) & (freqs < 3000)].sum()
speech_high = power[(freqs >= 3000) & (freqs < 8000)].sum()
total = power.sum()
print()
print("=== Spectral Energy Distribution ===")
print(f"  Low   (80-300Hz):   {speech_low / total * 100:.1f}%  (vocal fundamental)")
print(
    f"  Mid   (300-3kHz):   {speech_mid / total * 100:.1f}%  (formants, intelligibility)"
)
print(f"  High  (3k-8kHz):    {speech_high / total * 100:.1f}%  (sibilance, detail)")
print(
    f"  Above 8kHz:         {(total - speech_low - speech_mid - speech_high) / total * 100:.1f}%"
)

# Check for silence/emptiness
silent = np.sum(np.abs(samples_f32) < 0.001)
print(
    f"Silent samples (<0.001): {silent}/{len(samples_f32)} ({silent / len(samples_f32) * 100:.1f}%)"
)

# Audio quality assessment
print()
print("=== Quality Assessment ===")
rms = np.sqrt(np.mean(samples_f32**2))
if rms < 0.01:
    print("❌ Very quiet — almost silence")
elif rms < 0.05:
    print("⚠️  Low volume")
else:
    print("✅ Good volume level")

# Spectral tilt
tilt = np.polyfit(np.log10(freqs[1:] + 1), np.log10(power[1:] + 1), 1)[0]
print(f"Spectral tilt: {tilt:.2f} dB/decade")
if tilt < -3:
    print("⚠️  Heavy low-pass (possibly muffled speech)")
elif tilt < 0:
    print("✅ Speech-like spectral slope")
else:
    print("⚠️  Rising spectrum (possible high-frequency noise)")

# ZCR-based speech detection
zcr = np.sum(np.abs(np.diff(np.signbit(samples_f32).astype(int)))) / len(samples_f32)
if 0.02 < zcr < 0.25:
    print("✅ ZCR in speech range")
elif zcr > 0.3:
    print("⚠️  High ZCR (possible noise)")
else:
    print("⚠️  Low ZCR (possible silence or tonal)")
