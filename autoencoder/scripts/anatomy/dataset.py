"""Anatomy dataset: encode audio to latents and align with per-frame labels.

Entry points:
- `prepare_samples(manifest, n_samples, ...)` selects a deterministic subset of
  manifest items with a fixed seed.
- `encode_and_label(items, model, vocab, ...)` runs the encoder on each item,
  produces (latent_frames [T, D], FrameLabels) for each, and returns two
  aligned lists.
- `stack_frames(entries)` concatenates frame-level latents and labels into
  flat arrays, also returning utterance-index arrays for per-utterance
  operations later.

Assumes V16.3 AE frame rate (1 AE frame per total_stride audio samples).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .common import (
    ANATOMY_SEED,
    ModelBundle,
    TARGET_SR,
    V16_3_TOTAL_STRIDE,
    load_audio_tensor,
    resolve_audio_path,
)
from .labels import FrameLabels, ae_frame_count, compute_frame_labels, phoneme_vocabulary


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def prepare_samples(
    items: Sequence[dict[str, Any]],
    n_samples: int,
    seed: int = ANATOMY_SEED,
) -> list[dict[str, Any]]:
    """Deterministically pick a subset of manifest items."""
    if n_samples <= 0 or n_samples >= len(items):
        return list(items)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=n_samples, replace=False)
    idx = np.sort(idx)
    return [items[int(i)] for i in idx]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@dataclass
class LatentLabelPair:
    """Frame-level latents with aligned FrameLabels for one utterance."""

    latent: np.ndarray  # [T, D]
    labels: FrameLabels
    item: dict[str, Any] = field(default_factory=dict)
    audio_path: str = ""
    snr_db: float | None = None


def encode_and_label(
    items: Sequence[dict[str, Any]],
    bundle: ModelBundle,
    path_roots: Sequence[Path],
    segment_samples: int = 0,
    crop_mode: str = "center",
    build_phoneme_vocab: bool = True,
    device: torch.device | None = None,
    target_sr: int = TARGET_SR,
    verbose: bool = False,
    compute_snr: bool = False,
) -> tuple[list[LatentLabelPair], dict[str, int]]:
    """Encode audio for each item and compute frame labels.

    Returns a list of LatentLabelPair and the phoneme vocab used.
    """
    device = device or next(bundle.model.parameters()).device
    vocab = phoneme_vocabulary(items) if build_phoneme_vocab else {"<sil>": 0}

    pairs: list[LatentLabelPair] = []
    for idx, item in enumerate(items):
        audio_path = resolve_audio_path(item, path_roots)
        if not Path(audio_path).exists():
            if verbose:
                print(f"[anatomy.dataset] missing {audio_path}, skipping")
            continue
        try:
            audio_t = load_audio_tensor(
                audio_path,
                target_sr=target_sr,
                segment_samples=segment_samples,
                total_stride=bundle.total_stride,
                crop_mode=crop_mode,
            )
        except ValueError as e:
            if verbose:
                print(f"[anatomy.dataset] {audio_path}: {e}")
            continue
        audio_np = audio_t.squeeze().numpy().astype(np.float32)
        x = audio_t.to(device)
        with torch.no_grad():
            z, _, _ = bundle.model.encode(x)
        frames = z.squeeze(0).detach().cpu().float().numpy().T  # [T, D]

        n_frames = frames.shape[0]
        expected = ae_frame_count(len(audio_np), bundle.total_stride)
        if n_frames != expected and abs(n_frames - expected) > 1:
            # AE frame count can differ by 1 due to padding; tolerate small diff
            if verbose:
                print(
                    f"[anatomy.dataset] warn {audio_path}: encode_frames={n_frames} vs audio_frames={expected}"
                )
        # Align label arrays to encoder frame count
        labels = compute_frame_labels(
            audio_np[: n_frames * bundle.total_stride],
            item,
            phoneme_vocab=vocab if build_phoneme_vocab else None,
            sr=target_sr,
            total_stride=bundle.total_stride,
            sample_id=f"sample_{idx:04d}",
        )
        # Trim to common length
        common = min(n_frames, labels.n_frames)
        frames = frames[:common]
        labels = FrameLabels(
            phoneme=labels.phoneme[:common],
            speaker=labels.speaker,
            f0=labels.f0[:common],
            voiced=labels.voiced[:common],
            energy=labels.energy[:common],
            centroid=labels.centroid[:common],
            n_frames=common,
            sample_id=labels.sample_id,
            meta=labels.meta,
        )

        snr_db = None
        if compute_snr:
            with torch.no_grad():
                x_hat = bundle.model.decode(z, output_length=x.shape[-1])
            err = torch.mean((x_hat - x) ** 2).item()
            sig = torch.mean(x**2).item()
            snr_db = 10.0 * math.log10((sig + 1e-12) / (err + 1e-12))

        pairs.append(
            LatentLabelPair(
                latent=frames.astype(np.float32),
                labels=labels,
                item=item,
                audio_path=str(audio_path),
                snr_db=snr_db,
            )
        )
        if verbose and (idx + 1) % 16 == 0:
            print(f"[anatomy.dataset] encoded {idx + 1}/{len(items)}")

    return pairs, vocab


# ---------------------------------------------------------------------------
# Flat stacking
# ---------------------------------------------------------------------------


@dataclass
class StackedFrames:
    """Frame-level latents and labels from multiple utterances concatenated.

    latent: [N_frames_total, D]
    utterance_index: [N_frames_total] — int index into the original pair list
    labels_frame: dict[str, np.ndarray], each [N_frames_total]
    per_utt_speaker: [N_utt] — speaker ids per utterance
    per_utt_latent_mean: [N_utt, D]
    vocab: phoneme vocab used
    """

    latent: np.ndarray
    utterance_index: np.ndarray
    labels_frame: dict[str, np.ndarray]
    per_utt_speaker: list[str | None]
    per_utt_latent_mean: np.ndarray
    vocab: dict[str, int]
    n_utts: int


def stack_frames(pairs: Sequence[LatentLabelPair], vocab: dict[str, int]) -> StackedFrames:
    latents = []
    utt_idx = []
    phoneme = []
    f0 = []
    voiced = []
    energy = []
    centroid = []
    per_utt_speaker: list[str | None] = []
    per_utt_means = []

    for i, pair in enumerate(pairs):
        z = pair.latent
        T = z.shape[0]
        latents.append(z)
        utt_idx.append(np.full(T, i, dtype=np.int64))
        phoneme.append(pair.labels.phoneme)
        f0.append(pair.labels.f0)
        voiced.append(pair.labels.voiced.astype(np.int8))
        energy.append(pair.labels.energy)
        centroid.append(pair.labels.centroid)
        per_utt_speaker.append(pair.labels.speaker)
        per_utt_means.append(z.mean(axis=0))

    latent_flat = np.concatenate(latents, axis=0)
    labels_frame = {
        "phoneme": np.concatenate(phoneme, axis=0),
        "f0": np.concatenate(f0, axis=0),
        "voiced": np.concatenate(voiced, axis=0),
        "energy": np.concatenate(energy, axis=0),
        "centroid": np.concatenate(centroid, axis=0),
    }
    utt_idx_flat = np.concatenate(utt_idx, axis=0)
    per_utt_means_arr = np.stack(per_utt_means, axis=0) if per_utt_means else np.zeros((0, latent_flat.shape[1]))

    return StackedFrames(
        latent=latent_flat,
        utterance_index=utt_idx_flat,
        labels_frame=labels_frame,
        per_utt_speaker=per_utt_speaker,
        per_utt_latent_mean=per_utt_means_arr,
        vocab=vocab,
        n_utts=len(pairs),
    )


__all__ = [
    "LatentLabelPair",
    "StackedFrames",
    "prepare_samples",
    "encode_and_label",
    "stack_frames",
]
