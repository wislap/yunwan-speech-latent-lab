"""Generation-friendly latent training script.

Multi-phase training pipeline:
  Phase 1: Low-pass data + mel/STFT magnitude + masked predictor
  Phase 2: Gradually raise frequency + stronger losses
  Phase 3: Full frequency + L1/L2 + discriminator

Usage:
    python -m autoencoder.train_gen_friendly --config autoencoder/conf/experiment/v19_gen_friendly_phase1.yaml
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from autoencoder.datasets import SynthesizedAudioDataset
from autoencoder.losses import MultiScaleMelLoss
from autoencoder.losses_gen_friendly import PhaseScheduler, TemporalSmoothnessLoss
from autoencoder.losses_stft import MultiResolutionSTFTLoss
from autoencoder.masking import BlockMaskGenerator
from autoencoder.models.autoencoder import WavVAE
from autoencoder.models.masked_predictor import MaskedLatentPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def compute_effective_rank(z: torch.Tensor, threshold: float = 0.95) -> float:
    """Compute effective rank of latent representations.

    Args:
        z: [B, C, T] or [N, C] latent tensor.
        threshold: Variance explained threshold (0.95 for rank95).

    Returns:
        Number of dimensions needed to explain `threshold` of variance.
    """
    if z.dim() == 3:
        # [B, C, T] -> [B*T, C]
        B, C, T = z.shape
        z_flat = z.permute(0, 2, 1).reshape(-1, C)
    else:
        z_flat = z

    # Center
    z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)

    # SVD
    try:
        _, s, _ = torch.linalg.svd(z_centered, full_matrices=False)
    except Exception:
        return float(z_flat.shape[1])

    # Cumulative variance
    var = s ** 2
    cumvar = var.cumsum(0) / var.sum()

    # Find rank at threshold
    rank = (cumvar < threshold).sum().item() + 1
    return float(rank)


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device) -> tuple[WavVAE, MaskedLatentPredictor]:
    """Build WavVAE and MaskedLatentPredictor."""
    model_cfg = cfg.get("model", {})

    ae = WavVAE(
        latent_dim=model_cfg.get("latent_dim", 384),
        strides=model_cfg.get("strides", [2, 4, 4, 8]),
        encoder_channels=model_cfg.get("encoder_channels", [128, 256, 512, 1024, 1024]),
        dilations=model_cfg.get("dilations", [1, 3, 9]),
        sample_latent=model_cfg.get("sample_latent", False),
        reg_weight=model_cfg.get("reg_weight", 1e-4),
    ).to(device)

    pred_cfg = cfg.get("predictor", {})
    predictor = MaskedLatentPredictor(
        latent_dim=model_cfg.get("latent_dim", 384),
        hidden_dim=pred_cfg.get("hidden_dim", 512),
        n_heads=pred_cfg.get("n_heads", 8),
        n_layers=pred_cfg.get("n_layers", 6),
        phoneme_vocab_size=pred_cfg.get("phoneme_vocab_size", 512),
        max_seq_len=pred_cfg.get("max_seq_len", 4096),
        dropout=pred_cfg.get("dropout", 0.1),
    ).to(device)

    return ae, predictor


def load_checkpoint(
    ae: WavVAE,
    predictor: MaskedLatentPredictor,
    checkpoint_path: str,
    device: torch.device,
    pretrained_ae_path: str | None = None,
) -> dict:
    """Load checkpoint or initialize from pretrained AE.

    Returns state dict with training metadata (epoch, step, etc).
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ae.load_state_dict(ckpt["ae_state_dict"])
        predictor.load_state_dict(ckpt["predictor_state_dict"])
        return ckpt.get("meta", {})

    # Initialize from pretrained V16.3
    if pretrained_ae_path and os.path.exists(pretrained_ae_path):
        logger.info(f"Loading pretrained AE from: {pretrained_ae_path}")
        ckpt = torch.load(pretrained_ae_path, map_location=device, weights_only=False)
        # Handle different checkpoint formats
        if "model_state_dict" in ckpt:
            ae.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "state_dict" in ckpt:
            ae.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            ae.load_state_dict(ckpt, strict=False)
        logger.info("Pretrained AE loaded. Predictor initialized randomly.")
    else:
        logger.info("No pretrained AE found. Training from scratch.")

    return {}


def save_checkpoint(
    ae: WavVAE,
    predictor: MaskedLatentPredictor,
    optimizers: dict,
    meta: dict,
    save_path: str,
):
    """Save full training state."""
    torch.save(
        {
            "ae_state_dict": ae.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "optimizer_states": {k: v.state_dict() for k, v in optimizers.items()},
            "meta": meta,
        },
        save_path,
    )


def train_step(
    ae: WavVAE,
    predictor: MaskedLatentPredictor,
    batch: dict,
    masker: BlockMaskGenerator,
    mel_loss_fn: MultiScaleMelLoss,
    stft_loss_fn: MultiResolutionSTFTLoss,
    smooth_loss_fn: TemporalSmoothnessLoss,
    phase_config,
    step: int,
    total_steps: int,
    device: torch.device,
    predictor_grad_to_encoder: bool = False,
    var_floor: float = 0.0,
) -> dict:
    """Single training step.

    Args:
        predictor_grad_to_encoder: If False, predictor input z is detached
            from encoder (warmup mode). If True, gradient flows through.
        var_floor: Minimum latent temporal variance. If > 0, adds penalty.

    Returns dict of loss components.
    """
    audio = batch["audio"].to(device)  # [B, 1, T]
    phoneme_ids = batch["phoneme_ids"].to(device)  # [B, T_ph]
    phoneme_durations = batch["phoneme_durations"].to(device)  # [B, T_ph]

    # --- Encode ---
    enc_out = ae.encoder(audio)
    z, mean, std = ae.bottleneck.encode(enc_out)  # z: [B, C, T_frames]

    # --- Decode & reconstruction loss ---
    recon = ae.decoder(z, output_length=audio.shape[-1])

    # Mel loss
    loss_mel = mel_loss_fn(recon, audio)

    # STFT magnitude loss
    loss_stft = stft_loss_fn(recon.squeeze(1), audio.squeeze(1))

    # --- Variance floor penalty ---
    # Encourage latent to have temporal variation
    loss_var = torch.tensor(0.0, device=device)
    if var_floor > 0:
        # Variance across time dimension per channel, then mean
        temporal_var = z.var(dim=2).mean()  # scalar
        loss_var = F.relu(var_floor - temporal_var)

    # --- Masked prediction loss ---
    B, C, T_frames = z.shape
    z_t = z.permute(0, 2, 1)  # [B, T_frames, C] for predictor

    # Generate mask
    mask = masker.generate_mask(T_frames, B).to(device)  # [B, T_frames], True=visible

    # Predictor input: detach from encoder during warmup
    if predictor_grad_to_encoder:
        z_pred_input = z_t  # gradient flows to encoder
    else:
        z_pred_input = z_t.detach()  # no gradient to encoder

    # Predict
    z_pred = predictor(z_pred_input, mask, phoneme_ids, phoneme_durations)

    # MSE loss only at masked positions
    masked_positions = ~mask  # [B, T_frames]
    if masked_positions.any():
        z_target = z_t.detach()  # target always detached
        # The gradient flows through the z_t that goes INTO the predictor (visible frames)
        # NOT through the target. This is the JEPA-style approach.
        pred_masked = z_pred[masked_positions]  # [N_masked, C]
        target_masked = z_target[masked_positions]  # [N_masked, C]
        loss_pred = F.mse_loss(pred_masked, target_masked)
    else:
        loss_pred = torch.tensor(0.0, device=device)

    # --- Temporal smoothness ---
    loss_smooth = smooth_loss_fn(z)
    smooth_weight = smooth_loss_fn.get_weight(step, total_steps)

    # --- Total loss ---
    loss_total = (
        phase_config.mel_weight * loss_mel
        + phase_config.stft_mag_weight * loss_stft
        + phase_config.predictor_weight * loss_pred
        + smooth_weight * loss_smooth
        + 10.0 * loss_var  # variance floor penalty
    )

    # L1 time domain (Phase 2/3)
    if phase_config.l1_time_weight > 0:
        loss_l1 = F.l1_loss(recon, audio)
        loss_total = loss_total + phase_config.l1_time_weight * loss_l1
    else:
        loss_l1 = torch.tensor(0.0, device=device)

    return {
        "total": loss_total,
        "mel": loss_mel.item(),
        "stft": loss_stft.item(),
        "pred": loss_pred.item(),
        "smooth": loss_smooth.item(),
        "var": loss_var.item(),
        "l1": loss_l1.item() if isinstance(loss_l1, torch.Tensor) else loss_l1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build models ---
    ae, predictor = build_model(cfg, device)

    # --- Load weights ---
    meta = load_checkpoint(
        ae, predictor,
        checkpoint_path=args.resume,
        device=device,
        pretrained_ae_path=cfg.get("pretrained_ae_path"),
    )

    # --- Dataset ---
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    dataset = SynthesizedAudioDataset(
        data_jsonl=data_cfg["train_manifest"],
        sr=data_cfg.get("sr", 22050),
        segment_length=train_cfg.get("segment_length", 65536),
        lpf_cutoff_hz=data_cfg.get("lpf_cutoff_hz", 4000.0),
        total_stride=cfg["model"].get("total_stride", 256),
        path_root=data_cfg.get("path_root"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    # --- Losses ---
    mel_loss_fn = MultiScaleMelLoss(
        sr=data_cfg.get("sr", 22050),
    ).to(device)

    stft_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048],
    ).to(device)

    smooth_loss_fn = TemporalSmoothnessLoss(
        initial_weight=train_cfg.get("smooth_initial_weight", 1e-5),
        max_weight=train_cfg.get("smooth_max_weight", 1e-3),
    )

    # --- Masking ---
    mask_cfg = cfg.get("masking", {})
    masker = BlockMaskGenerator(
        mask_ratio=mask_cfg.get("mask_ratio", 0.6),
        min_block_frames=mask_cfg.get("min_block_frames", 50),
        max_block_frames=mask_cfg.get("max_block_frames", 150),
    )

    # --- Optimizers ---
    optimizer_ae = torch.optim.AdamW(
        list(ae.encoder.parameters()) + list(ae.decoder.parameters()),
        lr=train_cfg.get("ae_lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    optimizer_pred = torch.optim.AdamW(
        predictor.parameters(),
        lr=train_cfg.get("predictor_lr", 5e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    optimizers = {"ae": optimizer_ae, "pred": optimizer_pred}

    # --- Phase scheduler ---
    scheduler = PhaseScheduler.default_3phase(
        phase1_epochs=train_cfg.get("phase1_epochs", 30),
        phase2_epochs=train_cfg.get("phase2_epochs", 20),
        phase3_epochs=train_cfg.get("phase3_epochs", 20),
    )

    # --- Training loop ---
    start_epoch = meta.get("epoch", 0)
    global_step = meta.get("global_step", 0)
    total_epochs = train_cfg.get("epochs", scheduler.total_epochs)
    total_steps = total_epochs * len(dataloader)
    grad_accum = train_cfg.get("grad_accum_steps", 16)
    log_interval = train_cfg.get("log_interval", 50)
    eval_interval = train_cfg.get("eval_interval", 500)
    save_interval = train_cfg.get("save_interval", 1000)
    output_dir = Path(train_cfg.get("output_dir", "outputs/models/v19_gen_friendly"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Warmup config: predictor doesn't affect encoder for first N steps
    predictor_warmup_steps = train_cfg.get("predictor_warmup_steps", 500)
    var_floor = train_cfg.get("var_floor", 0.01)

    logger.info(f"Training: {total_epochs} epochs, {len(dataset)} samples, device={device}")
    logger.info(f"AE params: {sum(p.numel() for p in ae.parameters()):,}")
    logger.info(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")
    logger.info(f"Predictor warmup: {predictor_warmup_steps} steps (no grad to encoder)")
    logger.info(f"Variance floor: {var_floor}")

    ae.train()
    predictor.train()

    for epoch in range(start_epoch, total_epochs):
        phase_config = scheduler.get_current_config(epoch)
        epoch_losses = {"mel": 0, "stft": 0, "pred": 0, "smooth": 0}
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Determine if predictor gradient flows to encoder
            pred_grad = global_step >= predictor_warmup_steps

            losses = train_step(
                ae, predictor, batch, masker,
                mel_loss_fn, stft_loss_fn, smooth_loss_fn,
                phase_config, global_step, total_steps, device,
                predictor_grad_to_encoder=pred_grad,
                var_floor=var_floor,
            )

            loss = losses["total"] / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

                optimizer_ae.step()
                optimizer_pred.step()
                optimizer_ae.zero_grad()
                optimizer_pred.zero_grad()
                global_step += 1

                # Logging
                if global_step % log_interval == 0:
                    warmup_status = "WARMUP" if global_step < predictor_warmup_steps else "JOINT"
                    logger.info(
                        f"[Phase {phase_config.phase_id}|{warmup_status}] "
                        f"Epoch {epoch} Step {global_step} | "
                        f"mel={losses['mel']:.4f} stft={losses['stft']:.4f} "
                        f"pred={losses['pred']:.4f} smooth={losses['smooth']:.6f} "
                        f"var={losses['var']:.6f}"
                    )

                # Eval
                if global_step % eval_interval == 0:
                    ae.eval()
                    with torch.no_grad():
                        # Quick effective rank on last batch
                        enc_out = ae.encoder(batch["audio"].to(device))
                        z_eval, _, _ = ae.bottleneck.encode(enc_out)
                        rank95 = compute_effective_rank(z_eval, 0.95)
                        rank99 = compute_effective_rank(z_eval, 0.99)
                    logger.info(f"  [Eval] rank95={rank95:.1f} rank99={rank99:.1f}")
                    ae.train()

                # Save
                if global_step % save_interval == 0:
                    save_checkpoint(
                        ae, predictor, optimizers,
                        {"epoch": epoch, "global_step": global_step, "phase": phase_config.phase_id},
                        str(output_dir / "latest.pt"),
                    )

            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k] += losses[k]
            n_batches += 1

        # End of epoch
        if n_batches > 0:
            avg = {k: v / n_batches for k, v in epoch_losses.items()}
            logger.info(
                f"Epoch {epoch}/{total_epochs} | Phase {phase_config.phase_id} | "
                f"avg mel={avg['mel']:.4f} stft={avg['stft']:.4f} "
                f"pred={avg['pred']:.4f} smooth={avg['smooth']:.6f}"
            )

        # Save epoch checkpoint
        save_checkpoint(
            ae, predictor, optimizers,
            {"epoch": epoch + 1, "global_step": global_step, "phase": phase_config.phase_id},
            str(output_dir / f"epoch_{epoch}.pt"),
        )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
