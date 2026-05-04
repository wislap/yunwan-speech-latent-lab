"""Decode FM-DiT V16-AE latent samples to audio for listening checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from autoencoder.models.autoencoder import WavVAE
from fm_dit.train_v1_v16ae import FMDiTV16AE, stats_vectors


def load_audio(path: str, sr: int) -> torch.Tensor:
    audio_np, file_sr = sf.read(path, always_2d=False)
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    if file_sr != sr:
        audio_np = resample_poly(audio_np, sr, file_sr)
    return torch.from_numpy(audio_np).float()


def save_audio(path: Path, audio: torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = audio.detach().cpu().float().flatten().clamp(-1.0, 1.0)
    sf.write(path, x.numpy(), sr)


def audio_metrics(x: torch.Tensor, y: torch.Tensor | None = None) -> dict:
    x = x.detach().cpu().float().flatten()
    out = {
        "rms": float(torch.sqrt((x * x).mean() + 1e-12).item()),
        "peak": float(x.abs().max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "seconds": float(x.numel() / 22050),
    }
    if y is not None:
        y = y.detach().cpu().float().flatten()
        n = min(x.numel(), y.numel())
        x = x[:n]
        y = y[:n]
        err = x - y
        out["mse_to_ref"] = float((err * err).mean().item())
        out["snr_to_ref_db"] = float((10.0 * torch.log10((y * y).mean() / ((err * err).mean() + 1e-12))).item())
        out["cos_to_ref"] = float(F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item())
    return out


def load_ae(checkpoint: Path, latent_dim: int, strides: list[int], device: torch.device) -> WavVAE:
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


def load_dit(checkpoint: Path, device: torch.device) -> tuple[FMDiTV16AE, dict]:
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
        use_coarse_predictor=bool(args.get("coarse_latent", False)),
        coarse_hidden_dim=int(args.get("coarse_hidden_dim", 1024)),
        coarse_layers=int(args.get("coarse_layers", 4)),
        coarse_heads=int(args.get("coarse_heads", 8)),
        coarse_dim_ff=int(args.get("coarse_dim_ff", 4096)),
        residual_flow=bool(args.get("residual_flow", False)),
        uncond_residual_flow=bool(args.get("uncond_residual_flow", False)),
        detach_coarse_for_flow=not bool(args.get("no_detach_coarse_for_flow", False)),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), args


def configure_pca_noise(
    model: FMDiTV16AE,
    pca_path: Path,
    stats_path: Path,
    variance_scale: float,
    use_mean: bool,
    device: torch.device,
) -> None:
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
    parser.add_argument("--dit-checkpoint", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--indices", type=str, default="0,1,2")
    parser.add_argument("--cfg-scales", type=str, default="1.0,1.5,2.0")
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--sample-solver", type=str, default="dpm2", choices=["euler", "heun", "dpm2"])
    parser.add_argument("--sample-t-start", type=float, default=0.0)
    parser.add_argument("--pca-path", type=Path, default=None)
    parser.add_argument("--pca-variance-scale", type=float, default=1.0)
    parser.add_argument("--pca-no-mean", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--strides", type=str, default="2,4,4,8")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    strides = [int(x) for x in args.strides.split(",")]

    items = json.loads(args.manifest.read_text())
    stats = json.loads(args.stats.read_text())
    latent_mean = stats.get("mean", stats.get("global_mean", 0.0))
    latent_std = stats.get("std", stats.get("global_std", 1.0))
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    cfg_scales = [float(x) for x in args.cfg_scales.split(",") if x.strip()]

    ae = load_ae(args.ae_checkpoint, args.latent_dim, strides, device)
    dit, dit_args = load_dit(args.dit_checkpoint, device)
    pca_path = args.pca_path
    if pca_path is None and dit_args.get("noise_init") == "pca" and dit_args.get("pca_path"):
        pca_path = Path(dit_args["pca_path"])
    if pca_path is not None:
        configure_pca_noise(
            dit,
            pca_path,
            args.stats,
            args.pca_variance_scale,
            not args.pca_no_mean,
            device,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "manifest": str(args.manifest),
        "dit_checkpoint": str(args.dit_checkpoint),
        "ae_checkpoint": str(args.ae_checkpoint),
        "sample_steps": args.sample_steps,
        "sample_solver": args.sample_solver,
        "sample_t_start": args.sample_t_start,
        "pca_path": str(pca_path) if pca_path else None,
        "seed": args.seed,
        "items": [],
    }

    with torch.inference_mode():
        for idx in indices:
            item = items[idx]
            item_dir = args.out_dir / f"{idx:03d}_{item['id']}"
            item_dir.mkdir(parents=True, exist_ok=True)

            original = load_audio(item["audio_path"], args.sr)
            output_length = original.numel()
            save_audio(item_dir / "original.wav", original, args.sr)

            latent = torch.load(item["latent_path"], map_location="cpu", weights_only=True)
            if latent.shape[0] != args.latent_dim:
                latent = latent.T
            z = latent.unsqueeze(0).to(device)
            ae_recon = ae.decode(z, output_length=output_length).squeeze(0).squeeze(0)
            save_audio(item_dir / "ae_recon.wav", ae_recon, args.sr)

            frame_phoneme_ids = torch.tensor([item["frame_phoneme_ids"]], dtype=torch.long, device=device)
            pitch = torch.tensor([item["pitch"]], dtype=torch.float32, device=device)
            energy = torch.tensor([item["energy"]], dtype=torch.float32, device=device)
            frame_lengths = torch.tensor([len(item["frame_phoneme_ids"])], dtype=torch.long, device=device)

            item_report = {
                "index": idx,
                "id": item["id"],
                "text": item.get("text", ""),
                "frames": int(frame_lengths.item()),
                "audio_path": item["audio_path"],
                "original": audio_metrics(original),
                "ae_recon": audio_metrics(ae_recon, original),
                "dit": {},
            }

            for cfg in cfg_scales:
                pred_latent = dit.sample(
                    frame_phoneme_ids,
                    pitch,
                    energy,
                    n_steps=args.sample_steps,
                    cfg_scale=cfg,
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                    solver=args.sample_solver,
                    frame_lengths=frame_lengths,
                    t_start=args.sample_t_start,
                )
                pred_z = pred_latent.transpose(1, 2).to(device)
                pred_audio = ae.decode(pred_z, output_length=output_length).squeeze(0).squeeze(0)
                wav_name = f"dit_cfg{cfg:g}.wav"
                save_audio(item_dir / wav_name, pred_audio, args.sr)
                item_report["dit"][f"cfg_{cfg:g}"] = audio_metrics(pred_audio, original)

            report["items"].append(item_report)
            print(f"decoded {idx}: {item['id']} -> {item_dir}", flush=True)

    (args.out_dir / "decode_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {args.out_dir / 'decode_report.json'}")


if __name__ == "__main__":
    main()
