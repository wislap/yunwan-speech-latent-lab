"""Decompose gradients: reconstruction loss vs discriminator loss per block.

Runs on CPU to avoid competing with GPU training.

Usage:
  python -u -m autoencoder.scripts.gradient_decompose
"""

import torch
import torch.nn.functional as F
import json
import numpy as np
import soundfile as sf
from autoencoder.models.autoencoder import WavVAE
from autoencoder.models.discriminator_v14 import V14Discriminator
from autoencoder.models.discriminator import generator_loss, feature_matching_loss
from autoencoder.losses_stft import MultiResolutionSTFTLoss
from autoencoder.losses import MultiScaleMelLoss


def main():
    device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(
        "outputs/models/wav_vae_v14.4_phase2_remote_main/best.pt",
        map_location="cpu", weights_only=False,
    )
    print(f"Checkpoint: epoch={ckpt.get('epoch')}, snr={ckpt.get('best_snr', 0):.2f}")

    model = WavVAE(
        latent_dim=128, encoder_channels=(128, 256, 512, 512, 1024, 1024),
        strides=(2, 4, 4, 4, 4), dilations=(1, 3, 9),
    )
    model.load_state_dict(ckpt["model"])
    model.train()

    disc = V14Discriminator()
    if "disc" in ckpt:
        disc.load_state_dict(ckpt["disc"])
    disc.train()

    mrstft = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048], sr=22050,
    )
    mel_fn = MultiScaleMelLoss(sr=22050)

    with open("data/ljspeech_train_splits.json") as f:
        items = json.load(f)

    # Load one sample
    data, _ = sf.read(items[0]["audio_path"])
    audio = torch.from_numpy(data.astype(np.float32))[:51200]
    audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]

    components = []
    for i, block in enumerate(model.encoder.blocks):
        components.append((f"enc.block[{i}]", block))
    for i, block in enumerate(model.decoder.blocks):
        components.append((f"dec.block[{i}]", block))

    # ── Pass 1: reconstruction loss only ──
    print("\n=== Reconstruction loss gradients ===")
    model.zero_grad()
    out = model(audio)
    l1 = F.l1_loss(out["x_hat"], audio)
    stft = mrstft(out["x_hat"], audio)
    mel = mel_fn(out["x_hat"], audio)
    recon_loss = 10 * l1 + 0.5 * stft + 0.1 * mel
    recon_loss.backward()

    recon_grads = {}
    print(f"{'Component':<20} {'grad_norm':>10} {'grad_std':>12}")
    print("-" * 45)
    for name, module in components:
        grads = [p.grad.flatten() for p in module.parameters() if p.grad is not None]
        if grads:
            g = torch.cat(grads)
            recon_grads[name] = g.clone()
            print(f"{name:<20} {g.norm().item():>10.4f} {g.std().item():>12.6f}")

    # ── Pass 2: adversarial loss only ──
    print("\n=== Adversarial loss gradients ===")
    model.zero_grad()
    out2 = model(audio)
    fake_logits, fake_feats = disc(out2["x_hat"])
    with torch.no_grad():
        _, real_feats = disc(audio)
    adv_loss = 0.1 * generator_loss(fake_logits) + 1.0 * feature_matching_loss(real_feats, fake_feats)
    adv_loss.backward()

    adv_grads = {}
    print(f"{'Component':<20} {'grad_norm':>10} {'grad_std':>12}")
    print("-" * 45)
    for name, module in components:
        grads = [p.grad.flatten() for p in module.parameters() if p.grad is not None]
        if grads:
            g = torch.cat(grads)
            adv_grads[name] = g.clone()
            print(f"{name:<20} {g.norm().item():>10.4f} {g.std().item():>12.6f}")

    # ── Compare: cosine similarity, magnitude ratio ──
    print("\n=== Recon vs Adv gradient comparison ===")
    print(f"{'Component':<20} {'cos_sim':>8} {'recon/adv':>10} {'adv/total':>10}")
    print("-" * 52)
    for name, _ in components:
        if name in recon_grads and name in adv_grads:
            r = recon_grads[name]
            a = adv_grads[name]
            cos = F.cosine_similarity(r.unsqueeze(0), a.unsqueeze(0)).item()
            ratio = r.norm().item() / max(a.norm().item(), 1e-10)
            adv_frac = a.norm().item() / (r.norm().item() + a.norm().item())
            print(f"{name:<20} {cos:>8.4f} {ratio:>10.2f} {adv_frac:>10.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
