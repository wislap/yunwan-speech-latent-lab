"""Pack individual .pt latent files into a single memory-mapped file.

Usage:
  python -u -m autoencoder.scripts.pack_latents
"""

import json
import torch
import numpy as np
from pathlib import Path


def main():
    src_dir = Path("data/ljspeech_latents_v14_2")
    manifest_path = src_dir / "manifest.json"

    with open(manifest_path) as f:
        paths = json.load(f)
    
    N = len(paths)
    print(f"Packing {N} segments...")

    # Peek at first file to get shapes
    sample = torch.load(paths[0], map_location="cpu", weights_only=True)
    audio_len = sample["audio"].shape[0]      # 51200
    latent_shape = sample["latent"].shape      # [128, 100]
    print(f"  audio: [{audio_len}], latent: {list(latent_shape)}")

    # Allocate packed arrays
    out_path = src_dir / "packed"
    out_path.mkdir(exist_ok=True)

    audio_all = np.memmap(str(out_path / "audio.bin"), dtype=np.float32,
                          mode='w+', shape=(N, audio_len))
    latent_all = np.memmap(str(out_path / "latent.bin"), dtype=np.float16,
                           mode='w+', shape=(N, latent_shape[0], latent_shape[1]))

    for i, p in enumerate(paths):
        data = torch.load(p, map_location="cpu", weights_only=True)
        audio_all[i] = data["audio"].numpy()
        latent_all[i] = data["latent"].half().numpy()
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{N}")

    audio_all.flush()
    latent_all.flush()

    # Save metadata
    meta = {
        "n_segments": N,
        "audio_len": audio_len,
        "latent_channels": latent_shape[0],
        "latent_frames": latent_shape[1],
    }
    with open(out_path / "meta.json", "w") as f:
        json.dump(meta, f)

    print(f"\nDone: {out_path}/")
    print(f"  audio.bin: {N * audio_len * 4 / 1e9:.2f} GB")
    print(f"  latent.bin: {N * latent_shape[0] * latent_shape[1] * 2 / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
