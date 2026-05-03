"""Create report figures for exported reconstruction wavs.

The analysis is CPU-only. It reads eval_wavs/epoch_xxxx/original_*.wav and
recon_*.wav pairs, computes band SNR, magnitude response error, and phase
coherence/error from STFT bins, then writes CSV and PNG figures.

Usage:
  python -m autoencoder.scripts.plot_recon_diagnostics \
    --run V16.2=outputs/models/wav_vae_v16.2_256x_time_remote_main/eval_wavs \
    --run V16.1=outputs/models/wav_vae_v16.1_hf_time_remote_main/eval_wavs \
    --out docs/v16.2/figures
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import get_window, stft


BANDS_HZ = (
    (0, 250),
    (250, 500),
    (500, 1000),
    (1000, 1500),
    (1500, 2500),
    (2500, 3500),
    (3500, 5000),
    (5000, 7000),
    (7000, 9000),
    (9000, 11025),
)


@dataclass
class PairMetric:
    run: str
    epoch: int
    sample: str
    seconds: float
    full_snr: float
    lsd: float
    band_lo: int
    band_hi: int
    band_snr: float
    mag_ratio_db: float
    mag_l1_db: float
    phase_mae_deg: float
    phase_coherence: float
    complex_corr: float


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float64), int(sr)


def find_pairs(eval_wavs: Path) -> list[tuple[int, str, Path, Path]]:
    pairs = []
    for epoch_dir in sorted(eval_wavs.glob("epoch_*")):
        if not epoch_dir.is_dir():
            continue
        epoch = int(epoch_dir.name.split("_")[-1])
        for orig in sorted(epoch_dir.glob("original_*.wav")):
            sample = orig.name.removeprefix("original_").removesuffix(".wav")
            recon = epoch_dir / f"recon_{sample}.wav"
            if recon.exists():
                pairs.append((epoch, sample, orig, recon))
    return pairs


def snr_db(ref: np.ndarray, pred: np.ndarray) -> float:
    err = pred - ref
    return 10.0 * math.log10((np.mean(ref * ref) + 1e-12) / (np.mean(err * err) + 1e-12))


def lsd_db(ref: np.ndarray, pred: np.ndarray) -> float:
    n_fft = 2048
    hop = 512
    n = min(len(ref), len(pred))
    if n < n_fft:
        return float("nan")
    win = get_window("hann", n_fft, fftbins=True)
    vals = []
    for start in range(0, n - n_fft + 1, hop):
        r = np.fft.rfft(ref[start:start + n_fft] * win)
        p = np.fft.rfft(pred[start:start + n_fft] * win)
        vals.append(np.sqrt(np.mean((20 * np.log10(np.abs(r) + 1e-7) - 20 * np.log10(np.abs(p) + 1e-7)) ** 2)))
    return float(np.mean(vals))


def weighted_angle_mae_deg(delta_phase: np.ndarray, weight: np.ndarray) -> float:
    total = float(np.sum(weight)) + 1e-12
    return float(np.sum(np.abs(np.angle(np.exp(1j * delta_phase))) * weight) / total * 180.0 / math.pi)


def pair_metrics(run: str, epoch: int, sample: str, orig: Path, recon: Path) -> list[PairMetric]:
    ref, sr = load_mono(orig)
    pred, sr2 = load_mono(recon)
    if sr != sr2:
        raise ValueError(f"Sample-rate mismatch: {orig}={sr}, {recon}={sr2}")
    n = min(len(ref), len(pred))
    ref = ref[:n]
    pred = pred[:n]
    err = pred - ref
    full_snr = snr_db(ref, pred)
    full_lsd = lsd_db(ref, pred)

    n_fft = 2048
    hop = 512
    _, _, ref_stft = stft(ref, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    freqs, _, pred_stft = stft(pred, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    _, _, err_stft = stft(err, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)

    ref_mag = np.abs(ref_stft)
    pred_mag = np.abs(pred_stft)
    err_mag = np.abs(err_stft)
    phase_delta = np.angle(pred_stft * np.conj(ref_stft))
    phase_vec = np.exp(1j * phase_delta)
    phase_weight = ref_mag * pred_mag

    rows = []
    for lo, hi in BANDS_HZ:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        r_energy = float(np.mean(ref_mag[mask] ** 2))
        e_energy = float(np.mean(err_mag[mask] ** 2))
        band_snr = 10.0 * math.log10((r_energy + 1e-12) / (e_energy + 1e-12))
        mag_ratio_db = float(np.mean(20.0 * np.log10((pred_mag[mask] + 1e-7) / (ref_mag[mask] + 1e-7))))
        mag_l1_db = float(np.mean(np.abs(20.0 * np.log10(pred_mag[mask] + 1e-7) - 20.0 * np.log10(ref_mag[mask] + 1e-7))))
        w = phase_weight[mask]
        coherence = float(np.abs(np.sum(phase_vec[mask] * w)) / (np.sum(w) + 1e-12))
        phase_mae = weighted_angle_mae_deg(phase_delta[mask], w)
        complex_corr = float(
            np.abs(np.sum(pred_stft[mask] * np.conj(ref_stft[mask])))
            / (np.sqrt(np.sum(np.abs(pred_stft[mask]) ** 2) * np.sum(np.abs(ref_stft[mask]) ** 2)) + 1e-12)
        )
        rows.append(PairMetric(
            run=run,
            epoch=epoch,
            sample=sample,
            seconds=n / sr,
            full_snr=full_snr,
            lsd=full_lsd,
            band_lo=lo,
            band_hi=hi,
            band_snr=band_snr,
            mag_ratio_db=mag_ratio_db,
            mag_l1_db=mag_l1_db,
            phase_mae_deg=phase_mae,
            phase_coherence=coherence,
            complex_corr=complex_corr,
        ))
    return rows


def write_csv(rows: list[PairMetric], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(PairMetric.__dataclass_fields__)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def aggregate(rows: list[PairMetric], metric: str) -> dict[str, dict[int, list[float]]]:
    out: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        label = f"{row.band_lo}-{row.band_hi}"
        out.setdefault(row.run, {}).setdefault(row.epoch, [])
    return out


def rows_for(rows: list[PairMetric], run: str, epoch: int | None = None) -> list[PairMetric]:
    return [r for r in rows if r.run == run and (epoch is None or r.epoch == epoch)]


def latest_epoch(rows: list[PairMetric], run: str) -> int:
    return max(r.epoch for r in rows if r.run == run)


def band_centers() -> list[float]:
    return [(lo + hi) / 2 for lo, hi in BANDS_HZ]


def band_labels() -> list[str]:
    return [f"{lo/1000:g}-{hi/1000:g}k" if hi >= 1000 else f"{lo}-{hi}" for lo, hi in BANDS_HZ]


def plot_latest_bands(rows: list[PairMetric], out_dir: Path, metric: str, ylabel: str, title: str, filename: str) -> None:
    runs = sorted({r.run for r in rows})
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    x = np.arange(len(BANDS_HZ))
    width = 0.8 / max(len(runs), 1)
    for idx, run in enumerate(runs):
        ep = latest_epoch(rows, run)
        vals = [getattr(r, metric) for r in sorted(rows_for(rows, run, ep), key=lambda r: r.band_lo)]
        ax.bar(x - 0.4 + width / 2 + idx * width, vals, width=width, label=f"{run} epoch {ep}")
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels(), rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def plot_epoch_curves(rows: list[PairMetric], out_dir: Path) -> None:
    high_bands = {(3500, 5000), (5000, 7000), (7000, 9000), (9000, 11025)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150, sharex=True)
    axes = axes.ravel()
    for run in sorted({r.run for r in rows}):
        epochs = sorted({r.epoch for r in rows if r.run == run})
        full = []
        hf_snr = []
        phase = []
        coh = []
        for ep in epochs:
            ep_rows = rows_for(rows, run, ep)
            full.append(np.mean([r.full_snr for r in ep_rows]))
            hf_rows = [r for r in ep_rows if (r.band_lo, r.band_hi) in high_bands]
            hf_snr.append(np.mean([r.band_snr for r in hf_rows]))
            phase.append(np.mean([r.phase_mae_deg for r in hf_rows]))
            coh.append(np.mean([r.phase_coherence for r in hf_rows]))
        axes[0].plot(epochs, full, marker="o", label=run)
        axes[1].plot(epochs, hf_snr, marker="o", label=run)
        axes[2].plot(epochs, phase, marker="o", label=run)
        axes[3].plot(epochs, coh, marker="o", label=run)
    axes[0].set_ylabel("Full-band SNR (dB)")
    axes[1].set_ylabel("HF band SNR (dB)")
    axes[2].set_ylabel("HF phase MAE (deg)")
    axes[3].set_ylabel("HF phase coherence")
    axes[2].set_xlabel("Epoch")
    axes[3].set_xlabel("Epoch")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Reconstruction Diagnostics Over Training")
    fig.tight_layout()
    fig.savefig(out_dir / "epoch_curves.png")
    plt.close(fig)


def plot_heatmap(rows: list[PairMetric], out_dir: Path, run: str, metric: str, title: str, filename: str) -> None:
    epochs = sorted({r.epoch for r in rows if r.run == run})
    mat = []
    for ep in epochs:
        ep_rows = sorted(rows_for(rows, run, ep), key=lambda r: r.band_lo)
        mat.append([getattr(r, metric) for r in ep_rows])
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    im = ax.imshow(np.array(mat), aspect="auto", origin="lower", interpolation="nearest")
    ax.set_yticks(np.arange(len(epochs)))
    ax.set_yticklabels(epochs)
    ax.set_xticks(np.arange(len(BANDS_HZ)))
    ax.set_xticklabels(band_labels(), rotation=25, ha="right")
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("Epoch")
    ax.set_title(f"{run}: {title}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(title)
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="NAME=eval_wavs_path")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: list[PairMetric] = []
    for run, path in args.run:
        pairs = find_pairs(path)
        if not pairs:
            raise SystemExit(f"No wav pairs found for {run}: {path}")
        for epoch, sample, orig, recon in pairs:
            all_rows.extend(pair_metrics(run, epoch, sample, orig, recon))

    write_csv(all_rows, args.out / "recon_diagnostics_detail.csv")
    plot_epoch_curves(all_rows, args.out)
    plot_latest_bands(all_rows, args.out, "band_snr", "SNR (dB)", "Latest Multi-Band SNR", "latest_band_snr.png")
    plot_latest_bands(all_rows, args.out, "mag_ratio_db", "Recon / Original (dB)", "Latest Frequency Response Bias", "latest_frequency_response_bias.png")
    plot_latest_bands(all_rows, args.out, "phase_mae_deg", "Weighted phase MAE (deg)", "Latest Phase Error", "latest_phase_error.png")
    plot_latest_bands(all_rows, args.out, "phase_coherence", "Coherence (0-1)", "Latest Phase Coherence", "latest_phase_coherence.png")
    for run, _ in args.run:
        plot_heatmap(all_rows, args.out, run, "band_snr", "Band SNR (dB)", f"{run}_band_snr_heatmap.png")
        plot_heatmap(all_rows, args.out, run, "phase_coherence", "Phase Coherence", f"{run}_phase_coherence_heatmap.png")

    print(f"Wrote {args.out / 'recon_diagnostics_detail.csv'}")
    for png in sorted(args.out.glob("*.png")):
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()
