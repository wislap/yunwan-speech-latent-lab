"""V18 AE diagnostics: evaluate V18 checkpoint against acceptance gate.

Metrics produced:
  - Recon SNR / STFT (full val set)
  - Latent rank95 on z, z_phoneme, z_residual
  - Phoneme CTC frame accuracy (via argmax)
  - Cross-speaker text retrieval top1 on z_phoneme
  - Linear speaker probe accuracy on z_phoneme and z_residual
  - Recon SNR with z_residual zeroed
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from autoencoder.losses_stft import MultiResolutionSTFTLoss
from autoencoder.models.v18_encoder import V18Autoencoder


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def resolve_audio_path(item: dict, roots: list[Path]) -> Path:
    for key in ("audio_path", "path", "wav_path", "file"):
        if key in item:
            p = Path(item[key])
            if p.exists():
                return p
            for root in roots:
                candidate = root / p
                if candidate.exists():
                    return candidate
    raise FileNotFoundError(f"Cannot find audio for {item.get('id', item)}")


def load_audio(path: Path, sr: int, total_stride: int) -> torch.Tensor:
    data, file_sr = sf.read(path, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    audio = torch.from_numpy(data.astype(np.float32))
    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, file_sr, sr)
    aligned = (audio.numel() // total_stride) * total_stride
    return audio[:aligned]


def compute_snr(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    noise = x_hat - x
    signal_power = (x ** 2).mean()
    noise_power = (noise ** 2).mean()
    if noise_power < 1e-10:
        return 100.0
    return 10.0 * math.log10(signal_power.item() / noise_power.item())


def effective_rank(values: np.ndarray) -> dict[str, float]:
    if values.shape[0] < 2:
        return {"rank95": 0.0, "rank99": 0.0, "top1": 0.0}
    x = values.astype(np.float64) - values.mean(axis=0, keepdims=True)
    _, sigma, _ = np.linalg.svd(x, full_matrices=False)
    power = sigma ** 2
    ev = power / (power.sum() + 1e-30)
    cum = np.cumsum(ev)
    return {
        "rank95": float(np.searchsorted(cum, 0.95) + 1),
        "rank99": float(np.searchsorted(cum, 0.99) + 1),
        "top1": float(ev[0]),
    }


def retrieval_top1(features: np.ndarray, rows: list[dict]) -> float:
    ranks = []
    for i, row in enumerate(rows):
        candidates = [j for j, r in enumerate(rows) if r["speaker_id"] != row["speaker_id"]]
        positives = [j for j in candidates if rows[j]["text_id"] == row["text_id"]]
        if not candidates or not positives:
            continue
        f = features[i : i + 1]
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-30)
        fc = features[candidates]
        fc = fc / (np.linalg.norm(fc, axis=1, keepdims=True) + 1e-30)
        sims = (f @ fc.T)[0]
        ordered = [candidates[k] for k in np.argsort(-sims)]
        rank = min(ordered.index(p) + 1 for p in positives)
        ranks.append(rank)
    if not ranks:
        return 0.0
    return float(sum(1 for r in ranks if r == 1) / len(ranks))


def speaker_probe_accuracy(features: np.ndarray, rows: list[dict]) -> float:
    """Leave-one-out nearest centroid classification by speaker_id."""
    speakers = sorted({r["speaker_id"] for r in rows})
    if len(speakers) < 2:
        return 0.0
    correct = 0
    for i, row in enumerate(rows):
        scores = []
        for spk in speakers:
            idx = [j for j, r in enumerate(rows) if r["speaker_id"] == spk and j != i]
            if not idx:
                scores.append(-np.inf)
                continue
            centroid = features[idx].mean(axis=0, keepdims=True)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-30)
            f = features[i : i + 1]
            f = f / (np.linalg.norm(f) + 1e-30)
            scores.append(float((f @ centroid.T)[0, 0]))
        pred = speakers[int(np.argmax(scores))]
        correct += int(pred == row["speaker_id"])
    return correct / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--max-items", type=int, default=0, help="0 = all")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) or {}

    # Use defaults matching v18_base.yaml if not in checkpoint
    model = V18Autoencoder(
        sr=args.sr,
        phoneme_dim=int(ckpt_args.get("v18_phoneme_dim", 128)),
        residual_dim=int(ckpt_args.get("v18_residual_dim", 256)),
        phoneme_hidden_dim=int(ckpt_args.get("v18_phoneme_hidden_dim", 256)),
        phoneme_layers=int(ckpt_args.get("v18_phoneme_layers", 4)),
        phoneme_heads=int(ckpt_args.get("v18_phoneme_heads", 4)),
        phoneme_ffn_dim=int(ckpt_args.get("v18_phoneme_ffn_dim", 1024)),
        phoneme_ctc_vocab_size=int(ckpt_args.get("v18_ctc_vocab_size", 512)),
        use_checkpoint=False,
    )
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    items = read_manifest(args.manifest)
    if args.max_items > 0:
        items = items[: args.max_items]
    roots = [Path.cwd(), args.path_root, args.manifest.parent]

    mrstft = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048],
        sr=args.sr,
    ).to(device)

    rows = []
    snr_full = []
    snr_residual_zero = []
    stft_full = []
    z_phoneme_pool = []
    z_residual_pool = []
    z_pool = []
    ctc_acc_total = 0.0
    ctc_count = 0

    print(f"Evaluating {len(items)} items...")
    for i, item in enumerate(items):
        try:
            path = resolve_audio_path(item, roots)
            audio = load_audio(path, args.sr, model.total_stride)
        except (FileNotFoundError, RuntimeError) as e:
            continue
        if audio.numel() < model.total_stride * 4:
            continue

        a = audio.view(1, 1, -1).to(device)
        with torch.no_grad():
            out = model(a)
            x_hat_full = out["x_hat"].float()
            # With residual zeroed
            z_pz = torch.cat([out["z_phoneme"], torch.zeros_like(out["z_residual"])], dim=1)
            x_hat_pz = model.decode(z_pz, output_length=a.shape[-1]).float()

        snr_full.append(compute_snr(x_hat_full, a.float()))
        snr_residual_zero.append(compute_snr(x_hat_pz, a.float()))
        stft_full.append(mrstft(x_hat_full, a.float()).item())

        # Pooled features for probes
        z_phoneme_pool.append(out["z_phoneme"].mean(dim=-1).squeeze(0).cpu().numpy())
        z_residual_pool.append(out["z_residual"].mean(dim=-1).squeeze(0).cpu().numpy())
        z_pool.append(out["z"].mean(dim=-1).squeeze(0).cpu().numpy())

        # CTC frame-argmax "accuracy": we cannot map CTC argmax frame-to-label
        # without alignment, but we can measure the fraction of non-blank
        # predictions as a proxy for "CTC is doing something meaningful".
        log_probs = out["ctc_log_probs"]  # [1, T, V]
        preds = log_probs.argmax(dim=-1).squeeze(0)  # [T]
        ctc_acc_total += float((preds != 0).float().mean().item())
        ctc_count += 1

        rows.append({
            "id": item.get("id", i),
            "text_id": str(item.get("text_id") or item.get("same_text_group") or i),
            "speaker_id": str(item.get("speaker_id") or item.get("same_speaker_group") or ""),
        })

    print(f"  Evaluated {len(rows)} items")

    if not rows:
        print("No items evaluated.")
        return

    z_phoneme_np = np.stack(z_phoneme_pool)
    z_residual_np = np.stack(z_residual_pool)
    z_np = np.stack(z_pool)

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "n_items": len(rows),
        "recon_snr_mean": float(np.mean(snr_full)),
        "recon_snr_median": float(np.median(snr_full)),
        "recon_stft_mean": float(np.mean(stft_full)),
        "recon_snr_residual_zero_mean": float(np.mean(snr_residual_zero)),
        "recon_snr_drop_when_residual_zero": float(
            np.mean(snr_full) - np.mean(snr_residual_zero)
        ),
        "ctc_non_blank_frac_mean": float(ctc_acc_total / max(ctc_count, 1)),
        "z_rank": effective_rank(z_np),
        "z_phoneme_rank": effective_rank(z_phoneme_np),
        "z_residual_rank": effective_rank(z_residual_np),
        "z_phoneme_cross_speaker_retrieval_top1": retrieval_top1(z_phoneme_np, rows),
        "z_residual_cross_speaker_retrieval_top1": retrieval_top1(z_residual_np, rows),
        "z_phoneme_speaker_probe_acc": speaker_probe_accuracy(z_phoneme_np, rows),
        "z_residual_speaker_probe_acc": speaker_probe_accuracy(z_residual_np, rows),
    }

    # Acceptance gate check
    gates = {
        "recon_snr_ge_18": summary["recon_snr_mean"] >= 18.0,
        "z_rank95_ge_200": summary["z_rank"]["rank95"] >= 200,
        "ctc_non_blank_ge_0p30": summary["ctc_non_blank_frac_mean"] >= 0.30,
        "z_phoneme_speaker_probe_le_60": summary["z_phoneme_speaker_probe_acc"] <= 0.60,
        "z_residual_speaker_probe_ge_90": summary["z_residual_speaker_probe_acc"] >= 0.90,
        "residual_zero_drops_snr_gt_10dB": summary["recon_snr_drop_when_residual_zero"] > 10.0,
    }
    summary["acceptance_gate"] = gates
    summary["acceptance_gate_pass"] = all(gates.values())

    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print concise report
    print("\n=== V18 AE Diagnostics ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Items: {summary['n_items']}")
    print(f"Recon SNR (mean): {summary['recon_snr_mean']:.2f} dB")
    print(f"Recon STFT (mean): {summary['recon_stft_mean']:.4f}")
    print(f"Recon SNR with z_residual=0: {summary['recon_snr_residual_zero_mean']:.2f} dB")
    print(f"  -> drop: {summary['recon_snr_drop_when_residual_zero']:.2f} dB")
    print(f"CTC non-blank fraction: {summary['ctc_non_blank_frac_mean']:.4f}")
    print(f"Latent rank95: z={summary['z_rank']['rank95']:.0f}, "
          f"z_phoneme={summary['z_phoneme_rank']['rank95']:.0f}, "
          f"z_residual={summary['z_residual_rank']['rank95']:.0f}")
    print(f"z_phoneme cross-speaker text retrieval top1: "
          f"{summary['z_phoneme_cross_speaker_retrieval_top1']:.4f}")
    print(f"z_residual cross-speaker text retrieval top1: "
          f"{summary['z_residual_cross_speaker_retrieval_top1']:.4f}")
    print(f"z_phoneme speaker probe acc: "
          f"{summary['z_phoneme_speaker_probe_acc']:.4f} (target <= 0.60)")
    print(f"z_residual speaker probe acc: "
          f"{summary['z_residual_speaker_probe_acc']:.4f} (target >= 0.90)")
    print("\n--- Acceptance Gate ---")
    for k, v in gates.items():
        marker = "PASS" if v else "FAIL"
        print(f"  [{marker}] {k}")
    print(f"\nOverall: {'PASS' if summary['acceptance_gate_pass'] else 'FAIL'}")

    print(f"\nWrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
