"""Extract latents from V14.2 encoder for V14.3 decoder-only training.

Saves paired (audio_segment, latent) as .pt files for fast loading.

Usage:
  python -u -m autoencoder.scripts.extract_latents_v14_2
"""

import json
import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from autoencoder.models.autoencoder import WavVAE


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = 22050
    stride = 512
    segment_length = 512 * 100  # 51200

    # Load V14.2 model
    ckpt_path = "outputs/models/wav_vae_v14.2_full_remote_main/best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, snr={ckpt.get('best_snr', 0):.2f}")

    model = WavVAE(
        latent_dim=128,
        encoder_channels=(128, 256, 512, 512, 1024, 1024),
        strides=(2, 4, 4, 4, 4),
        dilations=(1, 3, 9),
    )
    # Only load encoder + bottleneck (decoder structure differs between V14.2 and V14.3)
    enc_keys = {k: v for k, v in ckpt["model"].items()
                if k.startswith("encoder.") or k.startswith("bottleneck.")}
    model.load_state_dict(enc_keys, strict=False)
    model = model.to(device).eval()
    print(f"Loaded {len(enc_keys)} encoder/bottleneck keys")

    # Load data index
    with open("data/ljspeech_train_splits.json") as f:
        items = json.load(f)
    print(f"Total audio files: {len(items)}")

    out_dir = Path("data/ljspeech_latents_v14_2")
    out_dir.mkdir(parents=True, exist_ok=True)

    total_segments = 0
    for i, item in enumerate(items):
        audio_path = item["audio_path"]
        try:
            data, file_sr = sf.read(audio_path)
        except Exception as e:
            print(f"  Skip {audio_path}: {e}")
            continue

        audio = torch.from_numpy(data.astype(np.float32))
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)

        T = audio.shape[0]
        T_aligned = (T // stride) * stride
        if T_aligned < segment_length:
            # Pad short audio
            audio = torch.nn.functional.pad(audio, (0, segment_length - T))
            T_aligned = segment_length

        audio = audio[:T_aligned]

        # Split into non-overlapping segments
        n_segments = T_aligned // segment_length
        for seg_idx in range(n_segments):
            start = seg_idx * segment_length
            seg = audio[start:start + segment_length].unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                enc_out = model.encoder(seg)
                z, mean, std = model.bottleneck.encode(enc_out, training=False)

            # Save: audio segment (fp32) + latent mean (fp16 to save space)
            file_id = Path(audio_path).stem
            save_path = out_dir / f"{file_id}_seg{seg_idx}.pt"
            torch.save({
                "audio": seg.squeeze().cpu(),          # [51200] fp32
                "latent": mean.squeeze().cpu().half(),  # [128, 100] fp16
            }, save_path)
            total_segments += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(items)} files, {total_segments} segments")

    print(f"\nDone: {total_segments} segments saved to {out_dir}/")

    # Save manifest
    manifest = sorted(str(p) for p in out_dir.glob("*.pt"))
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"Manifest: {manifest_path} ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
