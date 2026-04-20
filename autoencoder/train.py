"""V14 Wav-VAE training script.

Two-stage training:
  Phase 1 (epoch < adv_start_epoch): reconstruction losses only
  Phase 2 (epoch >= adv_start_epoch): + adversarial + feature matching

Usage:
  python -m autoencoder.train experiment=v14_wav_vae runtime=local
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from autoencoder.losses_stft import MultiResolutionSTFTLoss
from autoencoder.losses import MultiScaleMelLoss, compute_reconstruction_loss
from autoencoder.models.autoencoder import WavVAE
from autoencoder.models.discriminator import (
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
)
from autoencoder.models.discriminator_v14 import V14Discriminator


# ─── Dataset ─────────────────────────────────────────────────


class AudioDataset(Dataset):
    """Simple audio dataset: loads wav files, random crops to segment_length."""

    def __init__(
        self,
        data_path: str,
        segment_length: int = 65536,
        sr: int = 22050,
    ):
        self.segment_length = segment_length
        self.sr = sr
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

        # data_path can be a JSON file (list of dicts with audio_path)
        # or a directory of wav files
        data_path = Path(data_path)
        if data_path.suffix == ".json":
            with open(data_path) as f:
                items = json.load(f)
            self.audio_paths = []
            for item in items:
                if isinstance(item, dict):
                    # Try common keys
                    for key in ("audio_path", "path", "wav_path", "file"):
                        if key in item:
                            self.audio_paths.append(str(item[key]))
                            break
                elif isinstance(item, str):
                    self.audio_paths.append(item)
        elif data_path.is_dir():
            self.audio_paths = sorted(
                str(p) for p in data_path.glob("*.wav")
            )
        else:
            raise ValueError(f"data_path must be a .json file or directory: {data_path}")

        if len(self.audio_paths) == 0:
            raise ValueError(f"No audio files found at {data_path}")

    def __len__(self) -> int:
        return len(self.audio_paths)

    def _resolve_path(self, path: str) -> str:
        """Try to resolve relative paths."""
        p = Path(path)
        if p.exists():
            return str(p)
        # Try relative to cwd
        cwd_p = Path.cwd() / p
        if cwd_p.exists():
            return str(cwd_p)
        return path

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self._resolve_path(self.audio_paths[idx])
        try:
            import soundfile as sf
            data, file_sr = sf.read(path)
            audio = torch.from_numpy(data.astype('float32'))
            if audio.dim() > 1:
                audio = audio.mean(dim=-1)  # to mono
        except Exception:
            # Return silence on error
            return torch.zeros(self.segment_length)

        # Resample if needed
        if file_sr != self.sr:
            if file_sr not in self._resamplers:
                self._resamplers[file_sr] = torchaudio.transforms.Resample(file_sr, self.sr)
            audio = self._resamplers[file_sr](audio)

        # Random crop or pad
        T = audio.shape[0]
        if T >= self.segment_length:
            start = torch.randint(0, T - self.segment_length + 1, (1,)).item()
            audio = audio[start : start + self.segment_length]
        else:
            audio = F.pad(audio, (0, self.segment_length - T))

        return audio


def collate_audio(batch: list[torch.Tensor]) -> torch.Tensor:
    """Stack audio tensors into [B, 1, T]."""
    return torch.stack(batch, dim=0).unsqueeze(1)


# ─── Utilities ───────────────────────────────────────────────


def compute_snr(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    """Signal-to-noise ratio in dB."""
    noise = x_hat - x
    signal_power = (x ** 2).mean()
    noise_power = (noise ** 2).mean()
    if noise_power < 1e-10:
        return 100.0
    return 10.0 * math.log10(signal_power.item() / noise_power.item())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def safe_save(obj: dict, path: Path) -> None:
    """Save checkpoint with atomic write."""
    tmp_path = path.with_suffix(".tmp")
    torch.save(obj, tmp_path)
    tmp_path.rename(path)


# ─── Evaluation ──────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: WavVAE,
    dataloader: DataLoader,
    device: torch.device,
    mrstft_loss_fn: nn.Module,
    mel_loss_fn: nn.Module,
    max_batches: int = 8,
    output_dir: Path | None = None,
    epoch: int = 0,
    sr: int = 22050,
) -> dict[str, float]:
    """Run evaluation: compute SNR, STFT loss, KL, optional PESQ, export wavs."""
    was_training = model.training
    model.eval()

    snr_total = 0.0
    stft_total = 0.0
    kl_total = 0.0
    pesq_total = 0.0
    pesq_count = 0
    count = 0

    # Try importing pesq
    try:
        from pesq import pesq as pesq_fn
        has_pesq = True
    except ImportError:
        has_pesq = False
        print("  [eval] pesq library not available, skipping PESQ metric")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        audio = batch if isinstance(batch, torch.Tensor) else batch[0]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        audio = audio.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(audio)
            x_hat = out["x_hat"]

        # SNR
        snr_total += compute_snr(x_hat, audio)

        # STFT loss
        stft_total += mrstft_loss_fn(x_hat, audio).item()

        # Regularization loss
        kl_total += out["reg_loss"].item()

        # PESQ (first sample only per batch, CPU)
        if has_pesq and batch_idx < 4:
            try:
                ref = audio[0, 0].cpu().numpy()
                deg = x_hat[0, 0].cpu().numpy()
                score = pesq_fn(sr, ref, deg, "wb")
                pesq_total += score
                pesq_count += 1
            except Exception:
                pass

        # Export wav files (first batch only)
        if output_dir is not None and batch_idx == 0:
            wav_dir = output_dir / "eval_wavs" / f"epoch_{epoch:04d}"
            wav_dir.mkdir(parents=True, exist_ok=True)
            for i in range(min(4, audio.shape[0])):
                torchaudio.save(
                    str(wav_dir / f"original_{i}.wav"),
                    audio[i].cpu(), sr,
                )
                torchaudio.save(
                    str(wav_dir / f"recon_{i}.wav"),
                    x_hat[i].cpu(), sr,
                )

        count += 1

    if was_training:
        model.train()

    if count == 0:
        return {"snr": 0.0, "stft_loss": 0.0, "kl": 0.0, "pesq": 0.0}

    metrics = {
        "snr": snr_total / count,
        "stft_loss": stft_total / count,
        "kl": kl_total / count,
    }
    if pesq_count > 0:
        metrics["pesq"] = pesq_total / pesq_count
    else:
        metrics["pesq"] = 0.0

    return metrics


# ─── Training ────────────────────────────────────────────────


def train(args: DictConfig) -> None:
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sr = int(args.sr)
    segment_length = int(args.segment_length)

    # ── Model ────────────────────────────────────────────────
    model = WavVAE(
        latent_dim=int(args.latent_dim),
        encoder_channels=list(args.encoder_channels),
        strides=list(args.strides),
        dilations=list(args.dilations),
        reg_weight=float(args.get("reg_weight", 1e-2)),
        margin_mean=float(args.get("margin_mean", 3.0)),
        std_min=float(args.get("std_min", 0.1)),
        std_max=float(args.get("std_max", 1.5)),
        reg_warmup_steps=int(args.get("reg_warmup_steps", 1000)),
    ).to(device)

    print(f"WavVAE parameters: {count_parameters(model):,}")
    print(f"  Encoder: {count_parameters(model.encoder):,}")
    print(f"  Decoder: {count_parameters(model.decoder):,}")
    print(f"  Total stride: {model.total_stride}")

    # ── torch.compile ────────────────────────────────────────
    use_compile = bool(args.get("use_compile", False))
    if use_compile and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # ── Discriminator ────────────────────────────────────────
    adv_enable = bool(args.adv_enable)
    disc = None
    if adv_enable:
        disc = V14Discriminator(
            n_ffts=tuple(args.disc_n_ffts),
            hop_lengths=tuple(args.disc_hop_lengths),
            win_lengths=tuple(args.disc_win_lengths),
            filters=int(args.get("disc_filters", 32)),
        ).to(device)
        print(f"Discriminator parameters: {count_parameters(disc):,}")
        if use_compile and hasattr(torch, "compile"):
            disc = torch.compile(disc)

    # ── Loss functions ───────────────────────────────────────
    mrstft_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=list(args.stft_fft_sizes),
        hop_sizes=list(args.stft_hop_sizes),
        win_sizes=list(args.stft_win_sizes),
        sr=sr,
    ).to(device)

    mel_loss_fn = MultiScaleMelLoss(
        sr=sr,
        fft_sizes=tuple(args.mel_fft_sizes),
        n_mels_list=tuple(args.mel_n_mels),
    ).to(device)

    # ── Data ─────────────────────────────────────────────────
    train_dataset = AudioDataset(args.data_path, segment_length, sr)
    eval_dataset = AudioDataset(
        args.get("eval_data_path", args.data_path), segment_length, sr,
    )

    num_workers = int(args.get("num_workers", 4))
    prefetch_factor = int(args.get("prefetch_factor", 2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_audio,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=min(num_workers, 2),
        prefetch_factor=prefetch_factor if min(num_workers, 2) > 0 else None,
        pin_memory=True,
        collate_fn=collate_audio,
    )

    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # ── Optimizers ───────────────────────────────────────────
    g_optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), betas=(0.8, 0.99))
    d_optimizer = None
    if disc is not None:
        d_optimizer = torch.optim.Adam(disc.parameters(), lr=float(args.adv_d_lr), betas=(0.8, 0.99))

    # LR warmup + cosine decay
    total_steps = int(args.epochs) * len(train_loader)
    warmup_steps = int(args.get("lr_warmup_steps", 500))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda)
    d_scheduler = None
    if d_optimizer is not None:
        d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda)

    # ── Checkpoint resume ────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_snr = float("-inf")

    ckpt_path = output_dir / "latest.pt"
    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        g_optimizer.load_state_dict(ckpt["g_optimizer"])
        if disc is not None and "disc" in ckpt:
            disc.load_state_dict(ckpt["disc"])
        if d_optimizer is not None and "d_optimizer" in ckpt:
            d_optimizer.load_state_dict(ckpt["d_optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_snr = ckpt.get("best_snr", float("-inf"))
        print(f"  Resumed at epoch {start_epoch}, step {global_step}")

    # ── Training config ──────────────────────────────────────
    epochs = int(args.epochs)
    adv_start_epoch = int(args.adv_start_epoch)
    grad_accum_steps = int(args.get("grad_accum_steps", 1))
    grad_clip = float(args.get("grad_clip", 1.0))
    log_interval = int(args.get("log_interval", 16))
    eval_every_epochs = int(args.get("eval_every_epochs", 5))

    l1_weight = float(args.l1_time_weight)
    stft_weight = float(args.stft_weight)
    mel_weight = float(args.mel_weight)
    adv_g_weight = float(args.adv_g_weight)
    adv_fm_weight = float(args.adv_fm_weight)

    nan_streak = 0
    max_nan_streak = 20

    print(f"\nStarting training: {epochs} epochs, adv starts at epoch {adv_start_epoch}")
    print(f"  grad_accum={grad_accum_steps}, grad_clip={grad_clip}")
    print(f"  Loss weights: l1={l1_weight}, stft={stft_weight}, mel={mel_weight}")
    if adv_enable:
        print(f"  Adv weights: g={adv_g_weight}, fm={adv_fm_weight}")
    print()

    # ── Training loop ────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        if disc is not None:
            disc.train()

        use_adv = adv_enable and epoch >= adv_start_epoch
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_steps = 0
        t_start = time.time()

        g_optimizer.zero_grad()
        if d_optimizer is not None:
            d_optimizer.zero_grad()

        for batch_idx, audio in enumerate(train_loader):
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
            audio = audio.to(device, non_blocking=True)

            # ── Generator step ───────────────────────────────
            model.set_step(global_step)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(audio)
                x_hat = out["x_hat"]
                reg_loss = out["reg_loss"]

                recon_losses = compute_reconstruction_loss(
                    audio, x_hat, mrstft_loss_fn, mel_loss_fn, reg_loss,
                    l1_weight=l1_weight, stft_weight=stft_weight, mel_weight=mel_weight,
                )
                g_loss = recon_losses["total"]

                # Adversarial losses (Phase 2)
                adv_g_loss_val = 0.0
                fm_loss_val = 0.0
                d_loss_val = 0.0

                if use_adv and disc is not None:
                    fake_logits, fake_features = disc(x_hat)
                    with torch.no_grad():
                        _, real_features = disc(audio)

                    g_adv = generator_loss(fake_logits)
                    fm_loss = feature_matching_loss(real_features, fake_features)
                    g_loss = g_loss + adv_g_weight * g_adv + adv_fm_weight * fm_loss
                    adv_g_loss_val = g_adv.item()
                    fm_loss_val = fm_loss.item()

            # NaN/Inf check
            if not torch.isfinite(g_loss):
                nan_streak += 1
                if nan_streak >= max_nan_streak:
                    print(f"ERROR: {max_nan_streak} consecutive NaN/Inf losses, stopping.")
                    return
                print(f"  [step {global_step}] NaN/Inf G loss, skipping (streak={nan_streak})")
                g_optimizer.zero_grad()
                if d_optimizer is not None:
                    d_optimizer.zero_grad()
                continue
            nan_streak = 0

            # Gradient accumulation
            (g_loss / grad_accum_steps).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # Skip only if gradient is NaN/Inf
                if torch.isfinite(g_norm):
                    g_optimizer.step()
                else:
                    print(f"  [step {global_step}] NaN gradient (norm={g_norm:.1f}), skipping")
                g_optimizer.zero_grad()
                g_scheduler.step()

            # ── Discriminator step (Phase 2) ─────────────────
            if use_adv and disc is not None and d_optimizer is not None:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        x_hat_d = model(audio)["x_hat"]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    real_logits, _ = disc(audio)
                    fake_logits, _ = disc(x_hat_d.detach())
                    d_loss = discriminator_loss(real_logits, fake_logits)

                if torch.isfinite(d_loss):
                    (d_loss / grad_accum_steps).backward()

                    if (batch_idx + 1) % grad_accum_steps == 0:
                        d_norm = torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
                        if torch.isfinite(d_norm):
                            d_optimizer.step()
                        d_optimizer.zero_grad()
                        d_scheduler.step()

                    d_loss_val = d_loss.item()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss_val
            epoch_steps += 1
            global_step += 1

            # Logging
            if global_step % log_interval == 0:
                lr_g = g_optimizer.param_groups[0]["lr"]
                msg = (
                    f"  [epoch {epoch}/{epochs}] step {global_step} | "
                    f"G={g_loss.item():.4f} "
                    f"(l1={recon_losses['l1'].item():.4f} "
                    f"stft={recon_losses['stft'].item():.4f} "
                    f"mel={recon_losses['mel'].item():.4f} "
                    f"nrg={recon_losses['energy'].item():.4f} "
                    f"reg={recon_losses['reg'].item():.6f})"
                )
                if use_adv:
                    msg += f" | adv_g={adv_g_loss_val:.4f} fm={fm_loss_val:.4f} D={d_loss_val:.4f}"
                msg += f" | lr={lr_g:.2e}"
                print(msg, flush=True)

        # Epoch summary
        elapsed = time.time() - t_start
        avg_g = epoch_g_loss / max(epoch_steps, 1)
        avg_d = epoch_d_loss / max(epoch_steps, 1)
        phase = "recon" if not use_adv else "adv"
        print(
            f"Epoch {epoch}/{epochs} [{phase}] | "
            f"avg_G={avg_g:.4f} avg_D={avg_d:.4f} | "
            f"{elapsed:.1f}s ({epoch_steps} steps)",
            flush=True,
        )

        # ── Evaluation ───────────────────────────────────────
        if (epoch + 1) % eval_every_epochs == 0 or epoch == epochs - 1:
            print(f"\n  Running evaluation (epoch {epoch})...")
            metrics = evaluate(
                model, eval_loader, device,
                mrstft_loss_fn, mel_loss_fn,
                max_batches=int(args.eval_batches),
                output_dir=output_dir,
                epoch=epoch,
                sr=sr,
            )
            print(
                f"  Eval: SNR={metrics['snr']:.2f}dB | "
                f"STFT={metrics['stft_loss']:.4f} | "
                f"KL={metrics['kl']:.6f} | "
                f"PESQ={metrics['pesq']:.3f}",
                flush=True,
            )

            # Save best
            if metrics["snr"] > best_snr:
                best_snr = metrics["snr"]
                save_dict = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_snr": best_snr,
                    "metrics": metrics,
                }
                safe_save(save_dict, output_dir / "best.pt")
                print(f"  New best SNR: {best_snr:.2f}dB")

            print()

        # ── Save checkpoint ──────────────────────────────────
        save_dict = {
            "model": model.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_snr": best_snr,
        }
        if disc is not None:
            save_dict["disc"] = disc.state_dict()
        if d_optimizer is not None:
            save_dict["d_optimizer"] = d_optimizer.state_dict()
        safe_save(save_dict, output_dir / "latest.pt")

    print(f"\nTraining complete. Best SNR: {best_snr:.2f}dB")


# ─── Hydra entry point ──────────────────────────────────────


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("V14 Wav-VAE Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg.experiment.train))
    print("=" * 60)

    try:
        train(cfg.experiment.train)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(
                f"\nOOM Error: {e}\n"
                "Try reducing batch_size or increasing grad_accum_steps."
            )
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
