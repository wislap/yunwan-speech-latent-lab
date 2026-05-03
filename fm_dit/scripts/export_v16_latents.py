"""Export V16 autoencoder latents into an FM-DiT manifest.

This exporter is intentionally simple for the first DiT smoke test:
- V16 encoder produces deterministic latent means.
- Text is tokenized with a lightweight character-level fallback.
- Phoneme durations are assigned uniformly over latent frames.
- Pitch is set to zero; energy is frame RMS at the latent hop.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf
from scipy.signal import resample_poly

from autoencoder.models.autoencoder import WavVAE


def load_ljspeech_metadata(root: Path, max_samples: int | None) -> list[dict]:
    metadata = root / "metadata.csv"
    items = []
    with metadata.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 2:
                continue
            audio_id, text = parts[0], parts[1]
            audio_path = root / "wavs" / f"{audio_id}.wav"
            if audio_path.exists():
                items.append({"id": audio_id, "text": text, "audio_path": str(audio_path)})
            if max_samples is not None and len(items) >= max_samples:
                break
    return items


def tokenize_text(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z']+|[.,!?;:;-]", text.lower())
    out = []
    for token in tokens:
        if re.fullmatch(r"[.,!?;:;-]", token):
            out.append("<sil>")
        else:
            out.extend(list(token))
            out.append("<sp>")
    return out[:-1] if out and out[-1] == "<sp>" else out or ["<sil>"]


def build_vocab(items: list[dict]) -> dict[str, int]:
    vocab = {"<pad>": 0, "<sil>": 1, "<unk>": 2, "<sp>": 3}
    for item in items:
        for token in tokenize_text(item["text"]):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def uniform_durations(n_tokens: int, n_frames: int) -> list[int]:
    if n_tokens <= 0:
        return [n_frames]
    base = n_frames // n_tokens
    rem = n_frames % n_tokens
    return [base + (1 if i < rem else 0) for i in range(n_tokens)]


def frame_energy(audio: torch.Tensor, n_frames: int, hop: int) -> list[float]:
    x = audio.squeeze(0)
    need = n_frames * hop
    if x.numel() < need:
        x = F.pad(x, (0, need - x.numel()))
    else:
        x = x[:need]
    frames = x.view(n_frames, hop)
    energy = torch.sqrt((frames * frames).mean(dim=1) + 1e-12)
    if energy.std() > 1e-6:
        energy = (energy - energy.mean()) / energy.std()
    return energy.cpu().tolist()


def load_model(checkpoint: Path, latent_dim: int, strides: list[int], device: torch.device) -> WavVAE:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = WavVAE(
        latent_dim=latent_dim,
        encoder_channels=[128, 256, 512, 1024, 1024],
        strides=strides,
        dilations=[1, 3, 9],
        sample_latent=False,
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ljspeech-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--strides", type=str, default="2,4,4,8")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    strides = [int(x) for x in args.strides.split(",")]
    total_stride = 1
    for stride in strides:
        total_stride *= stride

    items = load_ljspeech_metadata(args.ljspeech_root, args.max_samples)
    if not items:
        raise RuntimeError(f"No LJSpeech items found under {args.ljspeech_root}")
    vocab = build_vocab(items)

    args.out.mkdir(parents=True, exist_ok=True)
    latent_dir = args.out / "latents"
    latent_dir.mkdir(exist_ok=True)

    model = load_model(args.checkpoint, args.latent_dim, strides, device)
    rows = []
    stat_sum = torch.zeros(args.latent_dim, dtype=torch.float64)
    stat_sumsq = torch.zeros(args.latent_dim, dtype=torch.float64)
    stat_count = 0
    global_values = []

    for idx, item in enumerate(items):
        audio_np, file_sr = sf.read(item["audio_path"], always_2d=False)
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)
        if file_sr != args.sr:
            audio_np = resample_poly(audio_np, args.sr, file_sr)
        audio = torch.from_numpy(audio_np).float().unsqueeze(0)

        n = (audio.shape[-1] // total_stride) * total_stride
        if n < total_stride:
            continue
        audio = audio[:, :n]

        with torch.inference_mode():
            z, mean, _ = model.encode(audio.unsqueeze(0).to(device))
        latent = mean.squeeze(0).detach().cpu().float()
        n_frames = int(latent.shape[-1])

        latent_path = latent_dir / f"{item['id']}.pt"
        torch.save(latent, latent_path)

        frames = latent.T.double()
        stat_sum += frames.sum(dim=0)
        stat_sumsq += (frames * frames).sum(dim=0)
        stat_count += frames.shape[0]
        global_values.append(latent.flatten())

        tokens = tokenize_text(item["text"])
        phoneme_ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
        durations = uniform_durations(len(phoneme_ids), n_frames)
        frame_ids = []
        for pid, dur in zip(phoneme_ids, durations):
            frame_ids.extend([pid] * dur)
        frame_ids = frame_ids[:n_frames]
        if len(frame_ids) < n_frames:
            frame_ids.extend([phoneme_ids[-1] if phoneme_ids else vocab["<sil>"]] * (n_frames - len(frame_ids)))

        rows.append({
            "id": item["id"],
            "latent_path": str(latent_path),
            "phoneme_ids": phoneme_ids,
            "phoneme_durations": durations,
            "frame_phoneme_ids": frame_ids,
            "pitch": [0.0] * n_frames,
            "energy": frame_energy(audio, n_frames, total_stride),
            "duration": n_frames,
            "text": item["text"],
            "audio_path": item["audio_path"],
        })

        if (idx + 1) % 25 == 0 or idx + 1 == len(items):
            print(f"processed {idx + 1}/{len(items)} rows={len(rows)}", flush=True)

    if not rows:
        raise RuntimeError("No latents exported")

    mean = stat_sum / stat_count
    var = (stat_sumsq / stat_count) - mean * mean
    std = torch.sqrt(var.clamp_min(1e-12))
    flat = torch.cat(global_values).double()
    stats = {
        "mean": mean.float().tolist(),
        "std": std.float().tolist(),
        "global_mean": float(flat.mean().item()),
        "global_std": float(flat.std().item()),
        "latent_dim": args.latent_dim,
        "frame_rate": args.sr / total_stride,
        "sample_rate": args.sr,
        "total_stride": total_stride,
        "strides": strides,
    }

    n_val = max(1, int(len(rows) * args.val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    (args.out / "train.json").write_text(json.dumps(train_rows, indent=2))
    (args.out / "val.json").write_text(json.dumps(val_rows, indent=2))
    (args.out / "latent_stats.json").write_text(json.dumps(stats, indent=2))
    (args.out / "vocab.json").write_text(json.dumps(vocab, indent=2))

    print(f"done total={len(rows)} train={len(train_rows)} val={len(val_rows)} out={args.out}")
    print(f"global_mean={stats['global_mean']:.6f} global_std={stats['global_std']:.6f}")


if __name__ == "__main__":
    main()
