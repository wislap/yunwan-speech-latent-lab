"""V18 latent export: produce packed latents compatible with FM-DiT pipeline.

Output format: same as `autoencoder/extract_latents.py` V16.3 path:
  packed/audio.bin    [N, audio_len] float32
  packed/latent.bin   [N, latent_channels, latent_frames] float16
  packed/meta.json    {n_segments, audio_len, latent_channels, latent_frames,
                       phoneme_dim, residual_dim, frame_rate, ...}
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from autoencoder.models.v18_encoder import V18Autoencoder


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def resolve_audio_path(item: dict, roots: list[Path]) -> Path | None:
    for key in ("audio_path", "path", "wav_path", "file"):
        if key in item:
            p = Path(item[key])
            if p.exists():
                return p
            for root in roots:
                candidate = root / p
                if candidate.exists():
                    return candidate
    return None


def load_audio(path: Path, sr: int, segment_length: int, total_stride: int) -> torch.Tensor | None:
    try:
        data, file_sr = sf.read(path, always_2d=False)
    except Exception:
        return None
    if data.ndim == 2:
        data = data.mean(axis=1)
    audio = torch.from_numpy(data.astype(np.float32))
    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, file_sr, sr)

    # Segment-crop or pad to exactly segment_length
    if audio.numel() < segment_length:
        audio = torch.nn.functional.pad(audio, (0, segment_length - audio.numel()))
    else:
        # Center crop aligned to total_stride
        extra = audio.numel() - segment_length
        start = (extra // 2 // total_stride) * total_stride
        audio = audio[start : start + segment_length]
    return audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--segment-length", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    packed_dir = args.out_dir / "packed"
    packed_dir.mkdir(exist_ok=True)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = V18Autoencoder(sr=args.sr)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded V18 checkpoint; missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(device).eval()

    total_stride = model.total_stride
    latent_frames = args.segment_length // total_stride
    latent_channels = model.latent_dim
    phoneme_dim = model.phoneme_dim
    residual_dim = model.residual_dim

    items = read_manifest(args.manifest)
    roots = [Path.cwd(), args.path_root, args.manifest.parent]

    # First pass: count valid items
    valid_items = []
    for item in items:
        p = resolve_audio_path(item, roots)
        if p is None:
            continue
        valid_items.append((item, p))
    n_total = len(valid_items)
    print(f"Valid items: {n_total} / {len(items)}")

    # Pre-allocate memmap arrays
    audio_path = packed_dir / "audio.bin"
    latent_path = packed_dir / "latent.bin"
    audio_mmap = np.memmap(
        audio_path, dtype=np.float32, mode="w+",
        shape=(n_total, args.segment_length),
    )
    latent_mmap = np.memmap(
        latent_path, dtype=np.float16, mode="w+",
        shape=(n_total, latent_channels, latent_frames),
    )

    manifest_items = []
    with torch.no_grad():
        for i, (item, p) in enumerate(valid_items):
            audio = load_audio(p, args.sr, args.segment_length, total_stride)
            if audio is None:
                continue
            a = audio.view(1, 1, -1).to(device)
            enc = model.encode(a)
            z = enc["z"]  # [1, latent_dim, latent_frames]
            audio_mmap[i] = audio.numpy().astype(np.float32)
            latent_mmap[i] = z.squeeze(0).cpu().numpy().astype(np.float16)
            manifest_items.append({
                "index": i,
                "id": item.get("id", i),
                "text_id": str(item.get("text_id", "")),
                "speaker_id": str(item.get("speaker_id", "")),
                "original_item": item,
            })
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{n_total}]")

    audio_mmap.flush()
    latent_mmap.flush()

    meta = {
        "n_segments": n_total,
        "audio_len": args.segment_length,
        "latent_channels": latent_channels,
        "latent_frames": latent_frames,
        "phoneme_dim": phoneme_dim,
        "residual_dim": residual_dim,
        "frame_rate_hz": args.sr / total_stride,
        "total_stride": total_stride,
        "sr": args.sr,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
    }
    with (packed_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with (args.out_dir / "items.json").open("w") as f:
        json.dump(manifest_items, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {n_total} items to {args.out_dir}")
    print(f"  audio.bin: {audio_path}")
    print(f"  latent.bin: {latent_path}")
    print(f"  meta.json: {packed_dir / 'meta.json'}")
    print(f"  Meta: {json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
