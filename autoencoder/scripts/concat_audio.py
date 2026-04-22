"""Concatenate short audio files into ~30s chunks for training.

Usage:
  python -u -m autoencoder.scripts.concat_audio
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path


def main():
    sr = 22050
    stride = 512
    target_duration = 30.0  # seconds
    target_samples = int(target_duration * sr)
    # Align to stride
    target_samples = (target_samples // stride) * stride
    silence_samples = int(0.1 * sr)  # 0.1s silence between clips

    with open("data/ljspeech_train_splits.json") as f:
        items = json.load(f)

    out_dir = Path("data/ljspeech_concat_30s")
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_idx = 0
    buffer = np.zeros(0, dtype=np.float32)
    silence = np.zeros(silence_samples, dtype=np.float32)
    manifest = []

    for i, item in enumerate(items):
        try:
            data, file_sr = sf.read(item["audio_path"])
            audio = data.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
        except Exception:
            continue

        # Add silence separator if buffer not empty
        if len(buffer) > 0:
            buffer = np.concatenate([buffer, silence])

        buffer = np.concatenate([buffer, audio])

        # Flush when buffer exceeds target
        while len(buffer) >= target_samples:
            chunk = buffer[:target_samples]
            # Align to stride
            chunk_len = (len(chunk) // stride) * stride
            chunk = chunk[:chunk_len]

            out_path = out_dir / f"chunk_{chunk_idx:04d}.wav"
            sf.write(str(out_path), chunk, sr)
            manifest.append({"audio_path": str(out_path)})
            chunk_idx += 1

            buffer = buffer[target_samples:]

    # Handle remaining buffer (pad to stride-aligned)
    if len(buffer) > stride * 10:  # at least ~0.2s
        chunk_len = (len(buffer) // stride) * stride
        chunk = buffer[:chunk_len]
        out_path = out_dir / f"chunk_{chunk_idx:04d}.wav"
        sf.write(str(out_path), chunk, sr)
        manifest.append({"audio_path": str(out_path)})
        chunk_idx += 1

    # Save manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_duration = sum(sf.info(m["audio_path"]).duration for m in manifest)
    print(f"Created {len(manifest)} chunks in {out_dir}/")
    print(f"Total duration: {total_duration/3600:.1f}h ({total_duration:.0f}s)")
    print(f"Target chunk: {target_samples/sr:.1f}s ({target_samples} samples)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
