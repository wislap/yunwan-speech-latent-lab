"""Gradient flow analysis per block.

Analyzes: gradient magnitude, direction consistency, signal stability.
Run when GPU is free (pause training first).

Usage:
  python -u -m autoencoder.scripts.gradient_analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import math
import soundfile as sf
from autoencoder.models.autoencoder import WavVAE
from autoencoder.losses_stft import MultiResolutionSTFTLoss
from autoencoder.losses import MultiScaleMelLoss, compute_reconstruction_loss


def main():
    device = torch.device("cuda")
    sr = 22050
    stride = 512

    # Load latest V14.4 checkpoint
    ckpt_path = "outputs/models/wav_vae_v14.4_full_remote_main/latest.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint: epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')}")

    model = WavVAE(
        latent_dim=128, encoder_channels=(128, 256, 512, 512, 1024, 1024),
        strides=(2, 4, 4, 4, 4), dilations=(1, 3, 9),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).train()

    mrstft = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048], sr=sr,
    ).to(device)
    mel_fn = MultiScaleMelLoss(sr=sr).to(device)

    # Load multiple different audio samples
    with open("data/ljspeech_train_splits.json") as f:
        items = json.load(f)

    seg_len = stride * 100

    # ═══════════════════════════════════════════════════════════
    # Part 1: Gradient magnitude per block (single batch)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Part 1: Gradient magnitude per block")
    print("=" * 70)

    batch = []
    for item in items[:4]:
        data, _ = sf.read(item["audio_path"])
        audio = torch.from_numpy(data.astype(np.float32))[:seg_len]
        if audio.shape[0] < seg_len:
            audio = F.pad(audio, (0, seg_len - audio.shape[0]))
        batch.append(audio)
    x = torch.stack(batch).unsqueeze(1).to(device)

    model.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
        losses = compute_reconstruction_loss(
            x, out["x_hat"], mrstft, mel_fn, out["reg_loss"],
            l1_weight=10, l2_weight=5, stft_weight=0.5, mel_weight=0.1,
        )
    losses["total"].backward()

    print(f"\n{'Component':<30} {'grad_norm':>10} {'grad_mean':>10} {'grad_std':>10} {'params':>8}")
    print("-" * 75)

    components = [
        ("encoder.stem", model.encoder.stem),
        ("encoder.proj", model.encoder.proj),
        ("decoder.proj", model.decoder.proj),
        ("decoder.tail", model.decoder.tail),
    ]
    for i, block in enumerate(model.encoder.blocks):
        components.append((f"encoder.block[{i}]", block))
    for i, block in enumerate(model.decoder.blocks):
        components.append((f"decoder.block[{i}]", block))

    for name, module in components:
        grads = [p.grad.float() for p in module.parameters() if p.grad is not None]
        if not grads:
            print(f"{name:<30} {'no grad':>10}")
            continue
        all_g = torch.cat([g.flatten() for g in grads])
        n_params = all_g.numel()
        print(f"{name:<30} {all_g.norm().item():>10.4f} {all_g.mean().item():>10.6f} {all_g.std().item():>10.6f} {n_params:>8}")

    # ═══════════════════════════════════════════════════════════
    # Part 2: Gradient direction consistency across batches
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Part 2: Gradient direction consistency (cosine sim across 5 batches)")
    print("=" * 70)

    grad_snapshots = {name: [] for name, _ in components}

    for batch_idx in range(5):
        batch = []
        start = batch_idx * 4
        for item in items[start:start + 4]:
            data, _ = sf.read(item["audio_path"])
            audio = torch.from_numpy(data.astype(np.float32))[:seg_len]
            if audio.shape[0] < seg_len:
                audio = F.pad(audio, (0, seg_len - audio.shape[0]))
            batch.append(audio)
        x = torch.stack(batch).unsqueeze(1).to(device)

        model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            losses = compute_reconstruction_loss(
                x, out["x_hat"], mrstft, mel_fn, out["reg_loss"],
                l1_weight=10, l2_weight=5, stft_weight=0.5, mel_weight=0.1,
            )
        losses["total"].backward()

        for name, module in components:
            grads = [p.grad.float().flatten() for p in module.parameters() if p.grad is not None]
            if grads:
                grad_snapshots[name].append(torch.cat(grads).detach().cpu())

    print(f"\n{'Component':<30} {'mean_cos':>10} {'min_cos':>10} {'std_cos':>10}")
    print("-" * 65)

    for name, _ in components:
        snaps = grad_snapshots[name]
        if len(snaps) < 2:
            continue
        cosines = []
        for i in range(len(snaps)):
            for j in range(i + 1, len(snaps)):
                cos = F.cosine_similarity(snaps[i].unsqueeze(0), snaps[j].unsqueeze(0)).item()
                cosines.append(cos)
        cosines = np.array(cosines)
        print(f"{name:<30} {cosines.mean():>10.4f} {cosines.min():>10.4f} {cosines.std():>10.4f}")

    # ═══════════════════════════════════════════════════════════
    # Part 3: LayerScale values (how much do ResUnits contribute?)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Part 3: LayerScale values (DilatedResUnit contribution)")
    print("=" * 70)

    for prefix, blocks in [("encoder", model.encoder.blocks), ("decoder", model.decoder.blocks)]:
        for i, block in enumerate(blocks):
            scales = []
            for name, param in block.named_parameters():
                if "scale" in name:
                    scales.append(param.data.float().mean().item())
            if scales:
                print(f"  {prefix}.block[{i}]: LayerScale mean={np.mean(scales):.4f} "
                      f"min={np.min(scales):.4f} max={np.max(scales):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
