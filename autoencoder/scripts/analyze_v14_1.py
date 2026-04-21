"""Analyze V14.1 best checkpoint latent space."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import os
import numpy as np
import soundfile as sf
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


# Monkey-patch to load old checkpoint with 1x1 conv shortcuts
import autoencoder.models.autoencoder as ae_mod

def pixel_unshuffle_1d(x, factor):
    B, C, T = x.shape
    T_out = T // factor
    x = x[:, :, :T_out * factor]
    return x.view(B, C, T_out, factor).permute(0, 1, 3, 2).contiguous().view(B, C * factor, T_out)

def pixel_shuffle_1d(x, factor):
    B, C, T = x.shape
    C_out = C // factor
    return x.view(B, C_out, factor, T).permute(0, 1, 3, 2).contiguous().view(B, C_out, T * factor)

class OldDownsampleShortcut(nn.Module):
    def __init__(self, in_ch, out_ch, factor):
        super().__init__()
        self.factor = factor
        self.proj = WNConv1d(in_ch * factor, out_ch, kernel_size=1, bias=False)
    def forward(self, x):
        return self.proj(pixel_unshuffle_1d(x, self.factor))

class OldUpsampleShortcut(nn.Module):
    def __init__(self, in_ch, out_ch, factor):
        super().__init__()
        self.factor = factor
        self.proj = WNConv1d(in_ch, out_ch * factor, kernel_size=1, bias=False)
    def forward(self, x):
        return pixel_shuffle_1d(self.proj(x), self.factor)

ae_mod.DownsampleShortcut = OldDownsampleShortcut
ae_mod.UpsampleShortcut = OldUpsampleShortcut

from autoencoder.models.autoencoder import WavVAE


def main():
    device = torch.device("cuda")
    ckpt = torch.load(
        "outputs/models/wav_vae_v14.1_full_remote_main/best.pt",
        map_location="cpu", weights_only=False,
    )
    epoch = ckpt["epoch"]
    best_snr = ckpt["best_snr"]
    print(f"Checkpoint: epoch={epoch}, snr={best_snr:.2f}")

    model = WavVAE(
        latent_dim=256, encoder_channels=(128, 256, 512, 1024),
        strides=(7, 7, 9), dilations=(1, 3, 9),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    stride = 441

    with open("data/ljspeech_train_splits.json") as f:
        items = json.load(f)

    n_eval = 200
    all_z_mean = []
    all_std_vals = []
    all_z_frames = []
    snrs = []
    ratios = []

    for i, item in enumerate(items[:n_eval]):
        data, sr = sf.read(item["audio_path"])
        audio = torch.from_numpy(data.astype(np.float32))
        T_aligned = (audio.shape[0] // stride) * stride
        if T_aligned < stride:
            continue
        audio = audio[:T_aligned].unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(audio)

        z = out["z"].squeeze(0).cpu()
        std_v = out["std"].squeeze(0).cpu()
        xh = out["x_hat"].float()
        audio_f = audio.float()

        noise_pow = ((xh - audio_f) ** 2).mean().item()
        sig_pow = (audio_f ** 2).mean().item()
        snr = 10 * math.log10(sig_pow / max(noise_pow, 1e-10))
        snrs.append(snr)
        ratios.append(xh.std().item() / audio_f.std().item())

        all_z_mean.append(z.mean(dim=-1).numpy())
        all_std_vals.append(std_v.mean(dim=-1).numpy())

        if i < 50:
            all_z_frames.append(z[:, ::4].T.numpy())

    # SNR
    snrs_arr = np.array(snrs)
    print(f"\n=== SNR ({len(snrs)} samples) ===")
    print(f"  mean={snrs_arr.mean():.2f}, std={snrs_arr.std():.2f}, median={np.median(snrs_arr):.2f}")
    print(f"  min={snrs_arr.min():.2f}, max={snrs_arr.max():.2f}")
    pcts = [0, 10, 25, 50, 75, 90, 100]
    pvals = np.percentile(snrs_arr, pcts)
    for p, v in zip(pcts, pvals):
        print(f"    p{p}={v:.2f}")

    # Ratio
    ratios_arr = np.array(ratios)
    print(f"\n=== Ratio ===")
    print(f"  mean={ratios_arr.mean():.3f}, std={ratios_arr.std():.3f}")

    # Latent
    Z = np.stack(all_z_mean)
    S = np.stack(all_std_vals)
    print(f"\n=== Latent Space ===")
    print(f"  z: mean={Z.mean():.4f}, std={Z.std():.4f}")
    dim_stds = Z.std(axis=0)
    print(f"  per-dim std: mean={dim_stds.mean():.4f}, min={dim_stds.min():.4f}, max={dim_stds.max():.4f}")
    print(f"  encoder std: mean={S.mean():.6f}")
    print(f"  dead dims (std<0.01): {(dim_stds < 0.01).sum()}/256")
    print(f"  low dims (std<0.05): {(dim_stds < 0.05).sum()}/256")
    print(f"  active dims (std>0.1): {(dim_stds > 0.1).sum()}/256")

    # PCA
    from numpy.linalg import svd
    Z_c = Z - Z.mean(axis=0)
    _, sigma, _ = svd(Z_c, full_matrices=False)
    ev = (sigma ** 2) / (sigma ** 2).sum()
    cumvar = np.cumsum(ev)

    print(f"\n=== PCA ===")
    print(f"  top 10 singular values: {sigma[:10].round(3)}")
    for k in [1, 2, 5, 10, 20, 50, 100, 128, 200, 256]:
        if k <= len(cumvar):
            print(f"    {k:3d} dims: {cumvar[k-1]*100:.1f}%")
    eff_rank = np.exp(-np.sum(ev * np.log(ev + 1e-10)))
    print(f"  effective rank: {eff_rank:.1f}")

    # t-SNE
    Z_frames = np.concatenate(all_z_frames, axis=0)
    print(f"\n=== t-SNE ({Z_frames.shape[0]} frames, using 2000) ===")
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500)
    Z_2d = tsne.fit_transform(Z_frames[:2000])
    print(f"  range: x=[{Z_2d[:,0].min():.1f},{Z_2d[:,0].max():.1f}], y=[{Z_2d[:,1].min():.1f},{Z_2d[:,1].max():.1f}]")
    print(f"  spread: x_std={Z_2d[:,0].std():.1f}, y_std={Z_2d[:,1].std():.1f}")

    # Correlation matrix analysis
    corr = np.corrcoef(Z.T)  # [256, 256]
    off_diag = corr[np.triu_indices(256, k=1)]
    print(f"\n=== Dimension Correlations ===")
    print(f"  mean |corr|: {np.abs(off_diag).mean():.4f}")
    print(f"  max |corr|: {np.abs(off_diag).max():.4f}")
    print(f"  highly correlated pairs (|r|>0.5): {(np.abs(off_diag) > 0.5).sum()}")

    # Save
    os.makedirs("outputs/analysis_v14_1", exist_ok=True)
    np.savez(
        "outputs/analysis_v14_1/latent_analysis.npz",
        Z=Z, S=S, snrs=snrs_arr, ratios=ratios_arr,
        sigma=sigma, explained_var=ev, Z_frames_tsne=Z_2d,
        dim_stds=dim_stds, corr_offdiag_abs_mean=np.abs(off_diag).mean(),
    )
    print(f"\nSaved to outputs/analysis_v14_1/")


if __name__ == "__main__":
    main()
