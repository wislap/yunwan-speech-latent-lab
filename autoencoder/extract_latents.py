"""V14 Wav-VAE latent extraction tool.

Loads a trained WavVAE checkpoint, runs the encoder in eval mode (z=μ),
and saves .pt latent files + latent_stats.json + train/val index files.

Usage:
  python -m autoencoder.extract_latents \
    --checkpoint outputs/models/best.pt \
    --audio_dir data/LJSpeech-1.1/wavs \
    --output_dir data/ljspeech_v14 \
    --val_ratio 0.05 \
    --sr 22050
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torchaudio

from autoencoder.models.autoencoder import WavVAE


def load_model(
    checkpoint_path: str,
    device: torch.device,
    latent_dim: int = 64,
    encoder_channels: tuple[int, ...] = (128, 256, 512, 1024),
    strides: tuple[int, ...] = (7, 7, 9),
    dilations: tuple[int, ...] = (1, 3, 9),
) -> WavVAE:
    """Load WavVAE from checkpoint."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}", file=sys.stderr)
        sys.exit(1)

    model = WavVAE(
        latent_dim=latent_dim,
        encoder_channels=encoder_channels,
        strides=strides,
        dilations=dilations,
    )

    # Support both full checkpoint and state_dict-only
    state_dict = ckpt.get("model", ckpt)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"ERROR: Incompatible checkpoint format: {e}", file=sys.stderr)
        sys.exit(1)

    model = model.to(device)
    model.eval()
    return model


def extract_latents(
    checkpoint_path: str,
    audio_dir: str,
    output_dir: str,
    val_ratio: float = 0.05,
    sr: int = 22050,
    device_str: str = "cuda",
    latent_dim: int = 64,
    encoder_channels: tuple[int, ...] = (128, 256, 512, 1024),
    strides: tuple[int, ...] = (7, 7, 9),
    dilations: tuple[int, ...] = (1, 3, 9),
) -> None:
    """Extract latents from all audio files in a directory."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(
        checkpoint_path, device,
        latent_dim=latent_dim,
        encoder_channels=encoder_channels,
        strides=strides,
        dilations=dilations,
    )
    total_stride = model.total_stride
    print(f"Model loaded. Total stride: {total_stride}")

    # Find audio files
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        print(f"ERROR: Audio directory not found: {audio_dir}", file=sys.stderr)
        sys.exit(1)

    audio_files = sorted(audio_path.glob("*.wav"))
    if len(audio_files) == 0:
        print(f"ERROR: No .wav files found in {audio_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files")

    # Create output directories
    out_path = Path(output_dir)
    latent_dir = out_path / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    all_latents = []
    items = []
    skipped = 0
    resampler = None

    for i, audio_file in enumerate(audio_files):
        file_id = audio_file.stem

        try:
            audio, file_sr = torchaudio.load(str(audio_file))
        except Exception as e:
            print(f"  WARNING: Cannot read {audio_file}: {e}")
            skipped += 1
            continue

        # To mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample if needed
        if file_sr != sr:
            if resampler is None or resampler.orig_freq != file_sr:
                resampler = torchaudio.transforms.Resample(file_sr, sr)
            audio = resampler(audio)

        # [1, T] → [1, 1, T]
        audio = audio.unsqueeze(0).to(device)

        with torch.inference_mode():
            mu, _ = model.encoder(audio)
            z = mu  # eval mode: use mean directly

        # Save latent [latent_dim, T'] (squeeze batch dim)
        latent = z.squeeze(0).cpu()
        latent_path = latent_dir / f"{file_id}.pt"
        torch.save(latent, latent_path)

        # Collect for stats
        all_latents.append(latent)

        items.append({
            "id": file_id,
            "latent_path": str(latent_path.relative_to(out_path.parent) if out_path.parent != Path(".") else latent_path),
            "audio_path": str(audio_file),
        })

        if (i + 1) % 100 == 0 or i == len(audio_files) - 1:
            print(f"  Processed {i + 1}/{len(audio_files)} files")

    if skipped > 0:
        print(f"  Skipped {skipped} unreadable files")

    if len(all_latents) == 0:
        print("ERROR: No latents extracted", file=sys.stderr)
        sys.exit(1)

    # Compute global stats
    all_cat = torch.cat([l.flatten() for l in all_latents])
    global_mean = float(all_cat.mean().item())
    global_std = float(all_cat.std().item())

    # Warnings for unusual stats
    if abs(global_mean) > 1.0:
        print(f"  WARNING: Latent mean ({global_mean:.4f}) deviates significantly from 0")
    if global_std < 0.5 or global_std > 2.0:
        print(f"  WARNING: Latent std ({global_std:.4f}) deviates significantly from 1.0")

    # Save latent_stats.json
    stats = {"mean": global_mean, "std": global_std}
    stats_path = out_path / "latent_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f)
    print(f"  Latent stats: mean={global_mean:.4f}, std={global_std:.4f}")

    # Split into train/val
    n_val = max(1, int(len(items) * val_ratio))
    val_items = items[:n_val]
    train_items = items[n_val:]

    train_path = out_path / "train.json"
    val_path = out_path / "val.json"
    with open(train_path, "w") as f:
        json.dump(train_items, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_items, f, indent=2)

    print(f"\nDone! Extracted {len(items)} latents")
    print(f"  Train: {len(train_items)}, Val: {len(val_items)}")
    print(f"  Output: {out_path}")
    print(f"  Stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="V14 Wav-VAE Latent Extraction")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--audio_dir", required=True, help="Directory containing .wav files")
    parser.add_argument("--output_dir", required=True, help="Output directory for latents")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--sr", type=int, default=22050, help="Target sample rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension")
    args = parser.parse_args()

    extract_latents(
        checkpoint_path=args.checkpoint,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        sr=args.sr,
        device_str=args.device,
        latent_dim=args.latent_dim,
    )


if __name__ == "__main__":
    main()
