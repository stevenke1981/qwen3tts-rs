"""Check WAV file info."""

import struct, os, sys

path = sys.argv[1] if len(sys.argv) > 1 else "output.wav"
with open(path, "rb") as f:
    d = f.read()
    if d[:4] != b"RIFF":
        print(f"Not a WAV file: {path}")
        sys.exit(1)
    rate = struct.unpack("<I", d[24:28])[0]
    channels = struct.unpack("<H", d[22:24])[0]
    bits = struct.unpack("<H", d[34:36])[0]
    data_size = struct.unpack("<I", d[40:44])[0]
    dur = data_size / rate / channels * 8 / bits
    print(f"WAV: {rate}Hz, {channels}ch, {bits}bit")
    print(f"Data: {data_size} bytes = {dur:.2f}s")
    print(f"File: {os.path.getsize(path)} bytes")
