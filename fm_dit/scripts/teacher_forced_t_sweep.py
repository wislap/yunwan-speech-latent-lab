"""Teacher-forced flow quality sweep for FM-DiT V16-AE checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fm_dit.train_v1_v16ae import FMDiTV16AE, lengths_to_valid_mask, stats_vectors


def to_time_channel(latent: torch.Tensor, latent_dim: int) -> torch.Tensor:
    if latent.dim() != 2:
        raise ValueError(f"expected 2D latent, got {tuple(latent.shape)}")
    return latent.T if latent.shape[0] == latent_dim else latent


def normalize_latent(latent: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (latent - mean.view(1, -1)) / std.clamp_min(1e-6).view(1, -1)


def masked_flat(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return x[valid.squeeze(-1)]


def load_model(checkpoint: Path, device: torch.device) -> tuple[FMDiTV16AE, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = FMDiTV16AE(
        vocab_size=256,
        latent_dim=int(args.get("latent_dim", 384)),
        model_dim=args.get("model_dim", 512),
        dit_dim_ff=int(args.get("dit_dim_ff", 2048)),
        dit_n_heads=int(args.get("dit_heads", 8)),
        cond_drop_prob=float(args.get("cond_drop_prob", 0.1)),
        use_duration_predictor=not bool(args.get("no_duration_predictor", False)),
        use_condition_encoder=not bool(args.get("no_condition_encoder", False)),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), args


def configure_pca(model: FMDiTV16AE, pca_path: Path, stats_path: Path, variance_scale: float, use_mean: bool, device: torch.device) -> None:
    data = np.load(pca_path)
    components = torch.from_numpy(data["components"].astype("float32")).to(device)
    if "pca_score_std" in data:
        score_std = torch.from_numpy(data["pca_score_std"].astype("float32")).to(device)
    else:
        score_std = torch.from_numpy(np.sqrt(data["eigenvalues"].astype("float32"))).to(device)
    pca_mean = torch.from_numpy(data["mean"].astype("float32")).to(device)
    latent_mean, latent_std = stats_vectors(str(stats_path), model.latent_dim)
    model.configure_pca_noise(
        components=components,
        score_std=score_std,
        pca_mean=pca_mean,
        latent_mean=latent_mean.to(device),
        latent_std=latent_std.to(device),
        variance_scale=variance_scale,
        use_mean=use_mean,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pca-path", type=Path, default=None)
    parser.add_argument("--pca-variance-scale", type=float, default=1.0)
    parser.add_argument("--pca-no-mean", action="store_true")
    parser.add_argument("--indices", type=str, default="0,1,2,3,4")
    parser.add_argument("--t-values", type=str, default="0,0.025,0.05,0.1,0.15,0.25,0.5,0.8")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    items = json.loads(args.manifest.read_text())
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    t_values = [float(x) for x in args.t_values.split(",") if x.strip()]
    mean, std = stats_vectors(str(args.stats), latent_dim=384)
    mean = mean.to(device)
    std = std.to(device)

    model, ckpt_args = load_model(args.checkpoint, device)
    if args.pca_path is not None:
        configure_pca(model, args.pca_path, args.stats, args.pca_variance_scale, not args.pca_no_mean, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with torch.inference_mode():
        for idx in indices:
            item = items[idx]
            latent = torch.load(item["latent_path"], map_location="cpu", weights_only=True)
            latent = to_time_channel(latent, model.latent_dim).to(device)
            z = normalize_latent(latent, mean, std).unsqueeze(0)
            S = z.shape[1]
            lengths = torch.tensor([S], dtype=torch.long, device=device)
            valid = lengths_to_valid_mask(lengths, S).unsqueeze(-1)
            key_padding_mask = ~valid.squeeze(-1)
            frame_phoneme_ids = torch.tensor([item["frame_phoneme_ids"][:S]], dtype=torch.long, device=device)
            pitch = torch.tensor([item["pitch"][:S]], dtype=torch.float32, device=device)
            energy = torch.tensor([item["energy"][:S]], dtype=torch.float32, device=device)
            cond = model.encode_cond(frame_phoneme_ids, pitch, energy, drop_cond=False)
            cond = cond * valid.to(dtype=cond.dtype)

            for t_val in t_values:
                metrics = []
                for repeat in range(args.repeats):
                    noise = model.sample_base_noise_like(z, valid_mask=valid)
                    t = torch.full((1, 1, 1), t_val, device=device, dtype=z.dtype)
                    x_t = (1.0 - t) * noise + t * z
                    v_target = z - noise
                    v_pred = model.latent_out_proj(
                        model.dit(
                            model.latent_in_proj(x_t),
                            torch.full((1,), t_val, device=device, dtype=z.dtype),
                            cond,
                            key_padding_mask=key_padding_mask,
                        )
                    )
                    z_hat = x_t + (1.0 - t) * v_pred

                    target_flat = masked_flat(z, valid)
                    pred_flat = masked_flat(z_hat, valid)
                    v_target_flat = masked_flat(v_target, valid)
                    v_pred_flat = masked_flat(v_pred, valid)
                    err = pred_flat - target_flat
                    v_err = v_pred_flat - v_target_flat
                    mse = float((err * err).mean().item())
                    v_mse = float((v_err * v_err).mean().item())
                    snr = float((10.0 * torch.log10((target_flat * target_flat).mean() / ((err * err).mean() + 1e-12))).item())
                    v_cos = float(F.cosine_similarity(v_pred_flat.flatten().unsqueeze(0), v_target_flat.flatten().unsqueeze(0)).item())
                    cos = float(F.cosine_similarity(pred_flat.flatten().unsqueeze(0), target_flat.flatten().unsqueeze(0)).item())
                    metrics.append((mse, v_mse, snr, cos, v_cos, float(pred_flat.std().item()), float(target_flat.std().item())))

                arr = np.asarray(metrics, dtype=np.float64)
                row = {
                    "index": idx,
                    "id": item.get("id", str(idx)),
                    "frames": S,
                    "t": t_val,
                    "repeats": args.repeats,
                    "mse_mean": float(arr[:, 0].mean()),
                    "mse_p05": float(np.percentile(arr[:, 0], 5)),
                    "mse_p95": float(np.percentile(arr[:, 0], 95)),
                    "v_mse_mean": float(arr[:, 1].mean()),
                    "snr_db_mean": float(arr[:, 2].mean()),
                    "snr_db_p05": float(np.percentile(arr[:, 2], 5)),
                    "snr_db_p95": float(np.percentile(arr[:, 2], 95)),
                    "cos_mean": float(arr[:, 3].mean()),
                    "v_cos_mean": float(arr[:, 4].mean()),
                    "pred_std_mean": float(arr[:, 5].mean()),
                    "target_std_mean": float(arr[:, 6].mean()),
                }
                rows.append(row)
                print(
                    f"idx={idx} t={t_val:g} snr={row['snr_db_mean']:.2f}dB "
                    f"mse={row['mse_mean']:.5f} vcos={row['v_cos_mean']:.3f}",
                    flush=True,
                )

    csv_path = args.out_dir / "teacher_forced_t_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "pca_path": str(args.pca_path) if args.pca_path else None,
        "pca_variance_scale": args.pca_variance_scale,
        "pca_use_mean": not args.pca_no_mean,
        "checkpoint_args": ckpt_args,
        "rows": rows,
    }
    (args.out_dir / "teacher_forced_t_sweep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
