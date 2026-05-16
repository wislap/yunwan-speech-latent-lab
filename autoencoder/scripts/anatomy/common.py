"""Shared utilities for the latent anatomy probe suite.

Includes:
- Deterministic seed setup
- Manifest loading (JSON or JSONL) and path resolution
- Audio loading with AE-stride alignment
- V16.3 / V18 checkpoint loading
- Bootstrap helpers for CI estimation
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Constants matching the plan
# ---------------------------------------------------------------------------

ANATOMY_SEED = 20260510
TARGET_SR = 22050
V16_3_STRIDES = (2, 4, 4, 8)
V16_3_TOTAL_STRIDE = 256
V16_3_ENCODER_CHANNELS = (128, 256, 512, 1024, 1024)
V16_3_LATENT_DIM = 384


def set_global_seed(seed: int = ANATOMY_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Manifest handling
# ---------------------------------------------------------------------------

_AUDIO_PATH_KEYS = ("audio_path", "path", "wav_path", "file")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a JSON or JSONL manifest. Returns a list of dicts."""
    path = Path(path)
    if path.suffix == ".jsonl":
        items = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Prefer "val" split if present, else "train"
        for key in ("val", "valid", "validation", "eval", "train"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError(f"Dict manifest at {path} has no list-valued split key")
    if not isinstance(data, list):
        raise ValueError(f"Unsupported manifest format at {path}: {type(data).__name__}")
    return data


def item_audio_key(item: dict[str, Any]) -> str:
    for key in _AUDIO_PATH_KEYS:
        if item.get(key):
            return str(item[key])
    raise KeyError(f"No audio path key found in item with id={item.get('id', '<unknown>')}")


def resolve_audio_path(item: dict[str, Any], path_roots: Sequence[Path]) -> Path:
    raw = Path(item_audio_key(item))
    if raw.is_absolute() and raw.exists():
        return raw
    for root in path_roots:
        p = (root / raw).resolve()
        if p.exists():
            return p
    # Last resort: return raw; caller decides how to report missing
    return raw


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_audio_tensor(
    path: Path,
    target_sr: int = TARGET_SR,
    segment_samples: int = 0,
    total_stride: int = V16_3_TOTAL_STRIDE,
    crop_mode: str = "center",
) -> torch.Tensor:
    """Load audio as a mono float32 tensor, aligned to `total_stride`.

    Returns tensor of shape [1, 1, N] where N is a multiple of total_stride.
    """
    audio_np, sr = sf.read(path, always_2d=False)
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    audio = torch.from_numpy(audio_np.astype(np.float32))
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    if segment_samples > 0:
        if audio.numel() >= segment_samples:
            max_start = audio.numel() - segment_samples
            if crop_mode == "first":
                start = 0
            elif crop_mode == "last":
                start = max_start
            else:
                start = max_start // 2
            start = (start // total_stride) * total_stride
            audio = audio[start : start + segment_samples]
        else:
            pad = segment_samples - audio.numel()
            audio = torch.nn.functional.pad(audio, (0, pad))
    aligned = (audio.numel() // total_stride) * total_stride
    if aligned < total_stride * 2:
        raise ValueError(f"Audio too short after alignment: {path}")
    return audio[:aligned].view(1, 1, -1)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


@dataclass
class ModelBundle:
    model: Any  # WavVAE or v18
    checkpoint: dict[str, Any]
    latent_dim: int
    total_stride: int
    strides: tuple[int, ...]
    kind: str  # "v16_3" or "v18"


def load_v16_3(checkpoint: Path, device: torch.device) -> ModelBundle:
    """Load a V16.3-style WavVAE from a best.pt checkpoint."""
    from autoencoder.models.autoencoder import WavVAE

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = WavVAE(
        latent_dim=V16_3_LATENT_DIM,
        encoder_channels=list(V16_3_ENCODER_CHANNELS),
        strides=list(V16_3_STRIDES),
        dilations=[1, 3, 9],
        sample_latent=False,
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return ModelBundle(
        model=model,
        checkpoint=ckpt,
        latent_dim=V16_3_LATENT_DIM,
        total_stride=V16_3_TOTAL_STRIDE,
        strides=V16_3_STRIDES,
        kind="v16_3",
    )


# ---------------------------------------------------------------------------
# V18 loader
# ---------------------------------------------------------------------------


def load_v18(checkpoint: Path, device: torch.device) -> ModelBundle:
    """Load a V18 autoencoder from a best.pt checkpoint.

    V18 uses a phoneme_dim=128 + residual_dim=256 concat to 384-d latent,
    with identical strides [2, 4, 4, 8] and total stride 256 matching V16.3.
    The `encode` method returns a dict; we wrap it to match the V16.3
    encode() signature expected by anatomy.dataset.encode_and_label.
    """
    from autoencoder.models.v18_encoder import V18Autoencoder

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Read configuration from checkpoint if available, else V18 defaults
    args = ckpt.get("args") or {}
    model = V18Autoencoder(
        sr=args.get("sr", 22050),
        phoneme_dim=args.get("v18_phoneme_dim", 128),
        residual_dim=args.get("v18_residual_dim", 256),
        strides=tuple(args.get("strides", V16_3_STRIDES)),
        dilations=tuple(args.get("dilations", (1, 3, 9))),
        residual_channels=tuple(args.get("v18_residual_channels", (96, 192, 384, 768, 768))),
        phoneme_hidden_dim=args.get("v18_phoneme_hidden_dim", 256),
        phoneme_layers=args.get("v18_phoneme_layers", 4),
        phoneme_heads=args.get("v18_phoneme_heads", 4),
        phoneme_ffn_dim=args.get("v18_phoneme_ffn_dim", 1024),
        phoneme_conv_kernel=args.get("v18_phoneme_conv_kernel", 31),
        phoneme_dropout=args.get("v18_phoneme_dropout", 0.0),
        phoneme_n_mels=args.get("v18_phoneme_n_mels", 80),
        phoneme_n_fft=args.get("v18_phoneme_n_fft", 1024),
        phoneme_hop_length=args.get("v18_phoneme_hop_length", 256),
        phoneme_ctc_vocab_size=args.get("v18_ctc_vocab_size", 512),
        reg_weight=args.get("reg_weight", 1e-4),
        margin_mean=args.get("margin_mean", 6.0),
        std_min=args.get("std_min", 0.05),
        std_max=args.get("std_max", 3.0),
        reg_warmup_steps=args.get("reg_warmup_steps", 5000),
    )
    model.load_state_dict(ckpt["model"])
    # Attach a V16.3-style encode() so callers in anatomy.dataset do not
    # need to branch on model kind.
    orig_encode = model.encode

    def v16_like_encode(x: torch.Tensor):
        out = orig_encode(x)
        return out["z"], out["mean"], out["std"]

    model.encode = v16_like_encode  # type: ignore[assignment]
    model = model.to(device).eval()
    return ModelBundle(
        model=model,
        checkpoint=ckpt,
        latent_dim=model.latent_dim,
        total_stride=model.total_stride,
        strides=tuple(model._strides),
        kind="v18",
    )


def load_checkpoint_auto(checkpoint: Path, device: torch.device, kind: str = "auto") -> ModelBundle:
    """Dispatch to the correct loader. `kind` in {"auto", "v16_3", "v18"}."""
    if kind == "v16_3":
        return load_v16_3(checkpoint, device)
    if kind == "v18":
        return load_v18(checkpoint, device)
    # Auto-detect by state_dict key presence
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", {})
    keys = list(state.keys()) if isinstance(state, dict) else []
    if any("phoneme_encoder" in k for k in keys):
        return load_v18(checkpoint, device)
    return load_v16_3(checkpoint, device)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def bootstrap_ci(
    data: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n_resamples: int = 100,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (point_estimate, lower, upper) bootstrap CI on `stat`."""
    rng = rng or np.random.default_rng(ANATOMY_SEED)
    values = np.asarray(data)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(stat(values))
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = stat(values[idx])
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Rank / spectrum helpers
# ---------------------------------------------------------------------------


def svd_spectrum(values: np.ndarray) -> dict[str, np.ndarray | float]:
    """Centered SVD of a [N, D] matrix. Returns singular values and rank metrics."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"svd_spectrum expects [N, D] with N>=2, got {arr.shape}")
    arr = arr - arr.mean(axis=0, keepdims=True)
    _, sigma, _ = np.linalg.svd(arr, full_matrices=False)
    power = sigma**2
    total = float(power.sum() + 1e-30)
    ev = power / total
    cum = np.cumsum(ev)
    entropy_rank = float(np.exp(-np.sum(ev * np.log(ev + 1e-30))))
    participation = float(total**2 / (np.sum(power**2) + 1e-30))
    return {
        "sigma": sigma,
        "explained_variance_ratio": ev,
        "cumulative_variance": cum,
        "entropy_rank": entropy_rank,
        "participation_rank": participation,
        "rank_90": int(np.searchsorted(cum, 0.90) + 1),
        "rank_95": int(np.searchsorted(cum, 0.95) + 1),
        "rank_99": int(np.searchsorted(cum, 0.99) + 1),
    }


__all__ = [
    "ANATOMY_SEED",
    "TARGET_SR",
    "V16_3_STRIDES",
    "V16_3_TOTAL_STRIDE",
    "V16_3_ENCODER_CHANNELS",
    "V16_3_LATENT_DIM",
    "set_global_seed",
    "load_manifest",
    "item_audio_key",
    "resolve_audio_path",
    "load_audio_tensor",
    "ModelBundle",
    "load_v16_3",
    "load_v18",
    "load_checkpoint_auto",
    "bootstrap_ci",
    "svd_spectrum",
]
