"""Diagnose exported reconstruction wavs.

This script compares eval_wavs/epoch_xxxx/original_*.wav and recon_*.wav
pairs without loading the model. It is CPU-only and safe to run while a GPU
training job is active.

Usage:
  python -m autoencoder.scripts.diagnose_recon_wavs \
    outputs/models/wav_vae_v16.2_256x_time_remote_main/eval_wavs
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import get_window


BANDS_HZ = (
    (0, 500),
    (500, 1500),
    (1500, 3500),
    (3500, 6000),
    (6000, 11025),
)


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float64), int(sr)


def snr_db(ref: np.ndarray, pred: np.ndarray) -> float:
    err = pred - ref
    ref_power = float(np.mean(ref * ref))
    err_power = float(np.mean(err * err))
    return 10.0 * math.log10((ref_power + 1e-12) / (err_power + 1e-12))


def band_energies(ref: np.ndarray, pred: np.ndarray, sr: int) -> dict[str, tuple[float, float]]:
    n = min(len(ref), len(pred))
    ref = ref[:n]
    pred = pred[:n]
    err = pred - ref

    window = get_window("hann", n, fftbins=True)
    ref_spec = np.fft.rfft(ref * window)
    err_spec = np.fft.rfft(err * window)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    out: dict[str, tuple[float, float]] = {}
    for lo, hi in BANDS_HZ:
        mask = (freqs >= lo) & (freqs < hi)
        ref_energy = float(np.mean(np.abs(ref_spec[mask]) ** 2)) if np.any(mask) else 0.0
        err_energy = float(np.mean(np.abs(err_spec[mask]) ** 2)) if np.any(mask) else 0.0
        band_snr = 10.0 * math.log10((ref_energy + 1e-12) / (err_energy + 1e-12))
        err_ratio = err_energy / (ref_energy + 1e-12)
        out[f"{lo}-{hi}"] = (band_snr, err_ratio)
    return out


def spectral_lsd(ref: np.ndarray, pred: np.ndarray, sr: int) -> float:
    del sr
    n_fft = 2048
    hop = 512
    n = min(len(ref), len(pred))
    ref = ref[:n]
    pred = pred[:n]
    if n < n_fft:
        return float("nan")

    win = get_window("hann", n_fft, fftbins=True)
    vals = []
    for start in range(0, n - n_fft + 1, hop):
        r = np.fft.rfft(ref[start:start + n_fft] * win)
        p = np.fft.rfft(pred[start:start + n_fft] * win)
        lr = 20.0 * np.log10(np.abs(r) + 1e-7)
        lp = 20.0 * np.log10(np.abs(p) + 1e-7)
        vals.append(float(np.sqrt(np.mean((lr - lp) ** 2))))
    return float(np.mean(vals)) if vals else float("nan")


def find_pairs(eval_wavs: Path) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    for epoch_dir in sorted(eval_wavs.glob("epoch_*")):
        if not epoch_dir.is_dir():
            continue
        for orig in sorted(epoch_dir.glob("original_*.wav")):
            suffix = orig.name.removeprefix("original_")
            recon = epoch_dir / f"recon_{suffix}"
            if recon.exists():
                pairs.append((epoch_dir.name, orig, recon))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_wavs", type=Path)
    parser.add_argument("--csv", action="store_true", help="Print machine-readable CSV.")
    args = parser.parse_args()

    pairs = find_pairs(args.eval_wavs)
    if not pairs:
        raise SystemExit(f"No original/recon wav pairs found under {args.eval_wavs}")

    rows = []
    for epoch, orig_path, recon_path in pairs:
        ref, sr = load_mono(orig_path)
        pred, sr2 = load_mono(recon_path)
        if sr != sr2:
            raise ValueError(f"Sample-rate mismatch: {orig_path}={sr}, {recon_path}={sr2}")
        n = min(len(ref), len(pred))
        ref = ref[:n]
        pred = pred[:n]
        bands = band_energies(ref, pred, sr)
        high_band = bands["6000-11025"]
        low_mid = np.mean([bands[k][0] for k in ("0-500", "500-1500", "1500-3500")])
        high_gap = low_mid - high_band[0]
        rows.append({
            "epoch": epoch,
            "sample": orig_path.stem.removeprefix("original_"),
            "sr": sr,
            "seconds": n / sr,
            "snr": snr_db(ref, pred),
            "lsd": spectral_lsd(ref, pred, sr),
            "b0": bands["0-500"][0],
            "b1": bands["500-1500"][0],
            "b2": bands["1500-3500"][0],
            "b3": bands["3500-6000"][0],
            "b4": bands["6000-11025"][0],
            "hf_err_ratio": high_band[1],
            "hf_gap": high_gap,
        })

    header = (
        "epoch,sample,seconds,snr,lsd,"
        "snr_0_500,snr_500_1500,snr_1500_3500,snr_3500_6000,snr_6000_11025,"
        "hf_err_ratio,hf_gap"
    )
    print(header)
    for r in rows:
        print(
            f"{r['epoch']},{r['sample']},{r['seconds']:.2f},"
            f"{r['snr']:.2f},{r['lsd']:.2f},"
            f"{r['b0']:.2f},{r['b1']:.2f},{r['b2']:.2f},{r['b3']:.2f},{r['b4']:.2f},"
            f"{r['hf_err_ratio']:.4f},{r['hf_gap']:.2f}"
        )


if __name__ == "__main__":
    main()
