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
import soundfile as sf_io
import numpy as np
import torchaudio
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from autoencoder.losses_stft import MultiResolutionSTFTLoss, MultiResolutionMultiBandSTFTLoss
from autoencoder.losses import MultiScaleMelLoss, compute_reconstruction_loss
from autoencoder.models.autoencoder import WavVAE
from autoencoder.models.discriminator import (
    MultiScaleSubBandDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
)
from autoencoder.models.discriminator_v14 import V14Discriminator


# ─── Dataset ─────────────────────────────────────────────────


class AudioDataset(Dataset):
    """Audio dataset with stride-aligned cropping.
    
    All returned audio lengths are exact multiples of total_stride,
    eliminating boundary padding artifacts in encoder/decoder.
    
    Modes:
      - segment_length > 0: random crop to fixed length (legacy)
      - segment_length = 0: full-length, stride-aligned (dynamic batching)
    """

    def __init__(
        self,
        data_path: str,
        segment_length: int = 0,
        sr: int = 22050,
        total_stride: int = 441,
        augment: bool = False,
    ):
        self.segment_length = segment_length
        self.sr = sr
        self.total_stride = total_stride
        self.augment = augment
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

        if segment_length > 0:
            assert segment_length % total_stride == 0, (
                f"segment_length={segment_length} must be divisible by "
                f"total_stride={total_stride}"
            )

        # data_path can be a JSON file (list of dicts with audio_path)
        # or a directory of wav files
        data_path = Path(data_path)
        if data_path.suffix == ".json":
            with open(data_path) as f:
                items = json.load(f)
            self.audio_paths = []
            for item in items:
                if isinstance(item, dict):
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

        # Pre-scan lengths for bucket sampler (only in full-length mode)
        self._lengths = None
        if segment_length == 0:
            self._scan_lengths()

    def _scan_lengths(self):
        """Pre-scan audio file lengths for bucket sampling."""
        import soundfile as sf
        lengths = []
        for p in self.audio_paths:
            try:
                info = sf.info(p)
                T = int(info.frames)
                T_aligned = (T // self.total_stride) * self.total_stride
                lengths.append(max(T_aligned, self.total_stride))
            except Exception:
                lengths.append(self.total_stride)
        self._lengths = lengths
        print(f"  Audio lengths: min={min(lengths)/self.sr:.1f}s, "
              f"max={max(lengths)/self.sr:.1f}s, "
              f"mean={sum(lengths)/len(lengths)/self.sr:.1f}s")

    def __len__(self) -> int:
        return len(self.audio_paths)

    def _resolve_path(self, path: str) -> str:
        p = Path(path)
        if p.exists():
            return str(p)
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
                audio = audio.mean(dim=-1)
        except Exception:
            fallback_len = self.segment_length if self.segment_length > 0 else self.total_stride * 100
            return torch.zeros(fallback_len)

        # Resample if needed
        if file_sr != self.sr:
            if file_sr not in self._resamplers:
                self._resamplers[file_sr] = torchaudio.transforms.Resample(file_sr, self.sr)
            audio = self._resamplers[file_sr](audio)

        T = audio.shape[0]
        s = self.total_stride

        if self.segment_length > 0:
            # Fixed-length crop mode
            if T >= self.segment_length:
                max_start = T - self.segment_length
                start = torch.randint(0, max_start + 1, (1,)).item()
                start = (start // s) * s
                audio = audio[start : start + self.segment_length]
            else:
                audio = F.pad(audio, (0, self.segment_length - T))
        else:
            # Full-length mode: just align to stride
            T_aligned = (T // s) * s
            if T_aligned < s:
                audio = F.pad(audio, (0, s - T))
            else:
                audio = audio[:T_aligned]

        # Data augmentation
        if self.augment:
            gain_db = (torch.rand(1).item() - 0.5) * 12.0
            audio = audio * (10.0 ** (gain_db / 20.0))
            noise_snr_db = 30.0 + torch.rand(1).item() * 20.0
            sig_power = (audio ** 2).mean().clamp(min=1e-10)
            noise_power = sig_power / (10.0 ** (noise_snr_db / 10.0))
            audio = audio + torch.randn_like(audio) * noise_power.sqrt()

        return audio


class BucketSampler(torch.utils.data.Sampler):
    """Dynamic batching: groups by length, each batch has ~constant total samples.
    
    Short audio → larger batch, long audio → smaller batch.
    Maximizes GPU utilization while avoiding OOM.
    """

    def __init__(self, lengths: list[int], max_total_samples: int = 200000, drop_last: bool = True):
        self.lengths = lengths
        self.max_total_samples = max_total_samples
        self.drop_last = drop_last
        # Sort indices by length
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

    def _build_batches(self):
        batches = []
        current_batch = []
        current_samples = 0

        for idx in self.sorted_indices:
            sample_len = self.lengths[idx]
            # Would adding this sample exceed budget?
            new_total = (len(current_batch) + 1) * max(current_samples // max(len(current_batch), 1), sample_len)
            # Simpler: just track max_len_in_batch * batch_size
            if current_batch:
                max_len = max(sample_len, max(self.lengths[i] for i in current_batch))
                projected = max_len * (len(current_batch) + 1)
            else:
                projected = sample_len

            if projected > self.max_total_samples and current_batch:
                batches.append(current_batch)
                current_batch = [idx]
            else:
                current_batch.append(idx)

        if current_batch and not self.drop_last:
            batches.append(current_batch)
        elif current_batch and len(current_batch) > 0:
            batches.append(current_batch)

        return batches

    def __iter__(self):
        batches = self._build_batches()
        perm = torch.randperm(len(batches)).tolist()
        for i in perm:
            yield from batches[i]

    def __len__(self):
        return len(self.sorted_indices)


def collate_audio(batch: list[torch.Tensor]) -> torch.Tensor:
    """Stack audio tensors into [B, 1, T]. Pads to longest in batch."""
    max_len = max(x.shape[0] for x in batch)
    padded = [F.pad(x, (0, max_len - x.shape[0])) if x.shape[0] < max_len else x for x in batch]
    return torch.stack(padded, dim=0).unsqueeze(1)


class LatentDataset(Dataset):
    """Memory-mapped dataset of pre-extracted (audio, latent) pairs.
    
    Reads from packed binary files for zero-overhead random access.
    Falls back to individual .pt files if packed format not found.
    """

    def __init__(self, manifest_path: str, max_samples: int = 0):
        manifest_dir = Path(manifest_path).parent
        packed_dir = manifest_dir / "packed"
        meta_path = packed_dir / "meta.json"

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.n_total = meta["n_segments"]
            self.audio_mmap = np.memmap(
                str(packed_dir / "audio.bin"), dtype=np.float32, mode='r',
                shape=(self.n_total, meta["audio_len"]))
            self.latent_mmap = np.memmap(
                str(packed_dir / "latent.bin"), dtype=np.float16, mode='r',
                shape=(self.n_total, meta["latent_channels"], meta["latent_frames"]))
            self._packed = True
        else:
            with open(manifest_path) as f:
                self.paths = json.load(f)
            self.n_total = len(self.paths)
            self._packed = False

        # Limit epoch size (randomly sample each epoch via __getitem__ with modulo)
        self.n = min(max_samples, self.n_total) if max_samples > 0 else self.n_total
        print(f"  LatentDataset: {self.n}/{self.n_total} segments per epoch"
              f" ({'mmap' if self._packed else '.pt'})")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        # Random remap so each epoch sees different segments
        real_idx = (idx + torch.randint(0, self.n_total, (1,)).item()) % self.n_total
        if self._packed:
            return {
                "audio": torch.from_numpy(self.audio_mmap[real_idx].copy()),
                "latent": torch.from_numpy(self.latent_mmap[real_idx].copy()).float(),
            }
        else:
            data = torch.load(self.paths[real_idx], map_location="cpu", weights_only=True)
            return {"audio": data["audio"], "latent": data["latent"].float()}


def collate_latent(batch: list[dict]) -> dict:
    """Collate latent dataset batch."""
    return {
        "audio": torch.stack([b["audio"] for b in batch]).unsqueeze(1),  # [B, 1, T]
        "latent": torch.stack([b["latent"] for b in batch]),             # [B, C, T']
    }


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


def apply_freeze_policy(model: WavVAE, args: DictConfig) -> None:
    """Apply configured module freezes after any global requires_grad reset."""
    freeze_encoder = bool(args.get("freeze_encoder", False))
    if not freeze_encoder:
        return
    model = getattr(model, "_orig_mod", model)

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.bottleneck.parameters():
        p.requires_grad = False

    if bool(args.get("use_phase_refiner", False)) and bool(args.get("freeze_decoder", True)):
        for p in model.decoder.parameters():
            p.requires_grad = False

    if bool(args.get("use_hybrid_decoder", False)) and bool(args.get("freeze_decoder_blocks", True)):
        for p in model.decoder.proj.parameters():
            p.requires_grad = False
        for p in model.decoder.block0.parameters():
            p.requires_grad = False

    if bool(args.get("use_dual_path_decoder", False)):
        for p in model.decoder.proj.parameters():
            p.requires_grad = False
        for p in model.decoder.block0.parameters():
            p.requires_grad = False
        for p in model.decoder.time_blocks.parameters():
            p.requires_grad = False
        for p in model.decoder.tail.parameters():
            p.requires_grad = False


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

        out = model(audio)
        x_hat = out["x_hat"].float()
        audio = audio.float()

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
                sf_io.write(
                    str(wav_dir / f"original_{i}.wav"),
                    audio[i, 0].cpu().numpy(), sr,
                )
                sf_io.write(
                    str(wav_dir / f"recon_{i}.wav"),
                    x_hat[i, 0].cpu().numpy(), sr,
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
        sample_latent=bool(args.get("sample_latent", True)),
        use_istft_decoder=bool(args.get("use_istft_decoder", False)),
        istft_n_fft=int(args.get("istft_n_fft", 2048)),
        istft_hop=int(args.get("istft_hop", 512)),
        sr=sr,
        use_phase_refiner=bool(args.get("use_phase_refiner", False)),
        phase_n_fft=int(args.get("phase_n_fft", 2048)),
        phase_hop=int(args.get("phase_hop", 512)),
        use_vocos_decoder=bool(args.get("use_vocos_decoder", False)),
        vocos_hidden_dim=int(args.get("vocos_hidden_dim", 1024)),
        vocos_n_layers=int(args.get("vocos_n_layers", 8)),
        vocos_n_fft=int(args.get("vocos_n_fft", 1024)),
        vocos_hop=int(args.get("vocos_hop", 512)),
        use_hybrid_decoder=bool(args.get("use_hybrid_decoder", False)),
        hybrid_vocos_hidden=int(args.get("hybrid_vocos_hidden", 512)),
        hybrid_vocos_layers=int(args.get("hybrid_vocos_layers", 4)),
        hybrid_n_fft=int(args.get("hybrid_n_fft", 512)),
        hybrid_hop=int(args.get("hybrid_hop", 128)),
        use_dual_path_decoder=bool(args.get("use_dual_path_decoder", False)),
        dual_vocos_hidden=int(args.get("dual_vocos_hidden", 512)),
        dual_vocos_layers=int(args.get("dual_vocos_layers", 6)),
        dual_n_fft=int(args.get("dual_n_fft", 1024)),
        dual_hop=int(args.get("dual_hop", 256)),
    ).to(device)

    print(f"WavVAE parameters: {count_parameters(model):,}")
    print(f"  Encoder: {count_parameters(model.encoder):,}")
    print(f"  Decoder: {count_parameters(model.decoder):,}")
    print(f"  Total stride: {model.total_stride}")

    # ── Teacher / encoder freeze ─────────────────────────────
    teacher_ckpt_path = args.get("teacher_checkpoint", None)
    freeze_encoder = bool(args.get("freeze_encoder", False))

    if teacher_ckpt_path and Path(teacher_ckpt_path).exists():
        print(f"  Loading teacher checkpoint: {teacher_ckpt_path}")
        teacher_ckpt = torch.load(teacher_ckpt_path, map_location=device, weights_only=False)
        teacher_state = teacher_ckpt["model"]
        # Try full model load first, fall back to partial with key remapping
        try:
            model.load_state_dict(teacher_state)
            print(f"  Loaded full model ({len(teacher_state)} keys)")
        except RuntimeError as e:
            print(f"  Full load failed, trying partial with remapping")
            # Remap V14.4 decoder keys for DualPathDecoder
            remapped = {}
            for k, v in teacher_state.items():
                new_k = k
                if k.startswith("decoder.blocks.0."):
                    new_k = k.replace("decoder.blocks.0.", "decoder.block0.")
                elif k.startswith("decoder.blocks."):
                    # decoder.blocks.N -> decoder.time_blocks.(N-1)
                    parts = k.split(".")
                    idx = int(parts[2])
                    if idx >= 1:
                        parts[1] = "time_blocks"
                        parts[2] = str(idx - 1)
                        new_k = ".".join(parts)
                remapped[new_k] = v
            
            model_keys = set(model.state_dict().keys())
            partial = {k: v for k, v in remapped.items() if k in model_keys}
            missing, unexpected = model.load_state_dict(partial, strict=False)
            print(f"  Loaded {len(partial)} matching keys, "
                  f"{len(missing)} missing (new modules)")

    if freeze_encoder:
        print("  Freezing encoder and bottleneck")
        apply_freeze_policy(model, args)
        # Also report the extra freezes that apply to this run.
        if bool(args.get("use_phase_refiner", False)) and bool(args.get("freeze_decoder", True)):
            print("  Freezing decoder (phase refiner only training)")
        if bool(args.get("use_hybrid_decoder", False)) and bool(args.get("freeze_decoder_blocks", True)):
            print("  Freezing decoder.proj + block0 (vocos head only training)")
        if bool(args.get("use_dual_path_decoder", False)):
            print("  Freezing time-domain path (vocos head only training)")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {trainable:,}")

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

    # ── Multi-band STFT loss (V14.6+) ───────────────────────
    mb_stft_enable = bool(args.get("mb_stft_enable", False))
    mb_stft_loss_fn = None
    mb_stft_weight = float(args.get("mb_stft_weight", 1.0))
    if mb_stft_enable:
        mb_stft_loss_fn = MultiResolutionMultiBandSTFTLoss(
            fft_sizes=list(args.get("mb_stft_fft_sizes", [1024, 2048])),
            hop_sizes=list(args.get("mb_stft_hop_sizes", [256, 512])),
            win_sizes=list(args.get("mb_stft_win_sizes", [1024, 2048])),
            sr=sr,
            band_edges_hz=list(args.get("mb_stft_band_edges", [0, 500, 1500, 3500, 6000, 11025])),
            band_weights=list(args.get("mb_stft_band_weights", [1.0, 1.0, 1.5, 2.5, 3.0])),
            complex_weight=float(args.get("mb_stft_complex_weight", 0.5)),
        ).to(device)
        print(f"  Multi-band STFT loss: weight={mb_stft_weight}, "
              f"bands={list(args.get('mb_stft_band_edges', [0, 500, 1500, 3500, 6000, 11025]))}, "
              f"band_weights={list(args.get('mb_stft_band_weights', [1.0, 1.0, 1.5, 2.5, 3.0]))}")

    # ── Sub-band discriminator (V14.6+) ─────────────────────
    subband_disc_enable = bool(args.get("subband_disc_enable", False))
    subband_disc = None
    if subband_disc_enable:
        subband_disc = MultiScaleSubBandDiscriminator(
            n_ffts=list(args.get("subband_disc_n_ffts", [1024, 2048])),
            hop_lengths=list(args.get("subband_disc_hop_lengths", [256, 512])),
            win_lengths=list(args.get("subband_disc_win_lengths", [1024, 2048])),
            sr=sr,
            freq_range_hz=tuple(args.get("subband_disc_freq_range", [4000, 11025])),
            filters=int(args.get("subband_disc_filters", 32)),
        ).to(device)
        print(f"  Sub-band discriminator: {count_parameters(subband_disc):,} params, "
              f"freq_range={list(args.get('subband_disc_freq_range', [4000, 11025]))}Hz")

    # ── Data ─────────────────────────────────────────────────
    latent_manifest = args.get("latent_manifest", None)
    use_latent_dataset = latent_manifest is not None and Path(latent_manifest).exists()

    total_stride = model.total_stride
    segment_length = (segment_length // total_stride) * total_stride
    print(f"  Segment length (stride-aligned): {segment_length} ({segment_length // total_stride} frames)")

    num_workers = int(args.get("num_workers", 4))
    prefetch_factor = int(args.get("prefetch_factor", 2))

    if use_latent_dataset:
        print(f"  Using pre-extracted latents: {latent_manifest}")
        train_dataset = LatentDataset(latent_manifest, max_samples=12000)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_latent,
        )
    else:
        augment = bool(args.get("augment", False))
        train_dataset = AudioDataset(args.data_path, segment_length, sr, total_stride, augment=augment)

        if segment_length == 0 and train_dataset._lengths is not None:
            # Full-length mode: dynamic batch size based on total samples budget
            max_total_samples = int(args.get("max_total_samples", 200000))
            bucket_sampler = BucketSampler(train_dataset._lengths, max_total_samples=max_total_samples)
            # Use batch_sampler (yields list of indices per batch)
            batch_lists = bucket_sampler._build_batches()
            print(f"  Dynamic batching: {len(batch_lists)} batches, "
                  f"bs range [{min(len(b) for b in batch_lists)}-{max(len(b) for b in batch_lists)}], "
                  f"max_total_samples={max_total_samples}")

            class _BatchSampler:
                def __init__(self, batches):
                    self.batches = batches
                def __iter__(self):
                    perm = torch.randperm(len(self.batches)).tolist()
                    for i in perm:
                        yield self.batches[i]
                def __len__(self):
                    return len(self.batches)

            train_loader = DataLoader(
                train_dataset,
                batch_sampler=_BatchSampler(batch_lists),
                num_workers=num_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                pin_memory=True,
                collate_fn=collate_audio,
            )
        else:
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

    eval_dataset = AudioDataset(
        args.get("eval_data_path", args.data_path), segment_length, sr, total_stride,
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
        d_params = list(disc.parameters())
        if subband_disc is not None:
            d_params += list(subband_disc.parameters())
        d_optimizer = torch.optim.Adam(d_params, lr=float(args.adv_d_lr), betas=(0.8, 0.99))

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
        if "g_optimizer" in ckpt:
            g_optimizer.load_state_dict(ckpt["g_optimizer"])
        if disc is not None and "disc" in ckpt:
            disc.load_state_dict(ckpt["disc"])
        if subband_disc is not None and "subband_disc" in ckpt:
            subband_disc.load_state_dict(ckpt["subband_disc"])
        if d_optimizer is not None and "d_optimizer" in ckpt:
            d_optimizer.load_state_dict(ckpt["d_optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_snr = ckpt.get("best_snr", float("-inf"))
        if "g_scheduler" in ckpt:
            g_scheduler.load_state_dict(ckpt["g_scheduler"])
        if d_scheduler is not None and "d_scheduler" in ckpt:
            d_scheduler.load_state_dict(ckpt["d_scheduler"])
        print(f"  Resumed at epoch {start_epoch}, step {global_step}")

    # ── Training config ──────────────────────────────────────
    epochs = int(args.epochs)
    adv_start_epoch = int(args.adv_start_epoch)
    grad_accum_steps = int(args.get("grad_accum_steps", 1))
    grad_clip = float(args.get("grad_clip", 1.0))
    log_interval = int(args.get("log_interval", 16))
    eval_every_epochs = int(args.get("eval_every_epochs", 5))

    l1_weight = float(args.l1_time_weight)
    l2_weight = float(args.get("l2_time_weight", 5.0))
    stft_weight = float(args.stft_weight)
    mel_weight = float(args.mel_weight)
    adv_g_weight = float(args.adv_g_weight)
    adv_fm_weight = float(args.adv_fm_weight)

    nan_streak = 0
    max_nan_streak = 20
    disc_warmup_epochs = int(args.get("disc_warmup_epochs", 0))

    print(f"\nStarting training: {epochs} epochs, adv starts at epoch {adv_start_epoch}")
    if disc_warmup_epochs > 0:
        print(f"  Discriminator warmup: {disc_warmup_epochs} epochs (AE frozen)")
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
        if subband_disc is not None:
            subband_disc.train()

        use_adv = adv_enable and epoch >= adv_start_epoch
        # Disc warmup: freeze AE, only train D
        disc_warming = use_adv and disc_warmup_epochs > 0 and epoch < (adv_start_epoch + disc_warmup_epochs)
        if disc_warming:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
        else:
            model.train()
            for p in model.parameters():
                p.requires_grad = True
            apply_freeze_policy(model, args)
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_steps = 0
        t_start = time.time()

        g_optimizer.zero_grad()
        if d_optimizer is not None:
            d_optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(train_loader):
            # Handle both audio-only and latent dataset formats
            if use_latent_dataset and isinstance(batch_data, dict):
                audio = batch_data["audio"].to(device, non_blocking=True)
                latent = batch_data["latent"].to(device, non_blocking=True)
            else:
                audio = batch_data
                if audio.dim() == 2:
                    audio = audio.unsqueeze(1)
                audio = audio.to(device, non_blocking=True)
                latent = None

            # ── Generator step ───────────────────────────────
            model.set_step(global_step)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if latent is not None:
                    # Decoder-only: skip encoder, decode from pre-extracted latent
                    x_hat = model.decode(latent, output_length=audio.shape[-1])
                    reg_loss = torch.tensor(0.0, device=device)
                    out = {"x_hat": x_hat, "reg_loss": reg_loss}
                else:
                    out = model(audio)
                x_hat = out["x_hat"]
                reg_loss = out["reg_loss"]

                recon_losses = compute_reconstruction_loss(
                    audio, x_hat, mrstft_loss_fn, mel_loss_fn, reg_loss,
                    l1_weight=l1_weight, l2_weight=l2_weight,
                    stft_weight=stft_weight, mel_weight=mel_weight,
                    multiband_stft_loss_fn=mb_stft_loss_fn,
                    multiband_stft_weight=mb_stft_weight,
                )
                g_loss = recon_losses["total"]

                # Adversarial losses (Phase 2)
                adv_g_loss_val = 0.0
                fm_loss_val = 0.0
                d_loss_val = 0.0
                sb_adv_g_val = 0.0

                if use_adv and disc is not None:
                    fake_logits, fake_features = disc(x_hat)
                    with torch.no_grad():
                        _, real_features = disc(audio)

                    g_adv = generator_loss(fake_logits)
                    fm_loss = feature_matching_loss(real_features, fake_features)
                    g_loss = g_loss + adv_g_weight * g_adv + adv_fm_weight * fm_loss
                    adv_g_loss_val = g_adv.item()
                    fm_loss_val = fm_loss.item()
                    
                    # Sub-band discriminator (高频专注)
                    if subband_disc is not None:
                        sb_fake_logits, sb_fake_features = subband_disc(x_hat)
                        with torch.no_grad():
                            _, sb_real_features = subband_disc(audio)
                        sb_g_adv = generator_loss(sb_fake_logits)
                        sb_fm = feature_matching_loss(sb_real_features, sb_fake_features)
                        sb_weight = float(args.get("subband_adv_g_weight", 0.1))
                        sb_fm_weight = float(args.get("subband_adv_fm_weight", 2.0))
                        g_loss = g_loss + sb_weight * sb_g_adv + sb_fm_weight * sb_fm
                        sb_adv_g_val = sb_g_adv.item()

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
            if not disc_warming:
                (g_loss / grad_accum_steps).backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
                    
                    # Sub-band discriminator loss
                    if subband_disc is not None:
                        sb_real_logits, _ = subband_disc(audio)
                        sb_fake_logits, _ = subband_disc(x_hat_d.detach())
                        sb_d_loss = discriminator_loss(sb_real_logits, sb_fake_logits)
                        d_loss = d_loss + sb_d_loss

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
                    f"l2={recon_losses['l2'].item():.4f} "
                    f"stft={recon_losses['stft'].item():.4f} "
                    f"mel={recon_losses['mel'].item():.4f} "
                    f"reg={recon_losses['reg'].item():.6f})"
                )
                if "mb_stft" in recon_losses:
                    msg += f" mb_stft={recon_losses['mb_stft'].item():.4f}"
                if use_adv:
                    msg += f" | adv_g={adv_g_loss_val:.4f} fm={fm_loss_val:.4f} D={d_loss_val:.4f}"
                    if sb_adv_g_val > 0:
                        msg += f" sb_g={sb_adv_g_val:.4f}"
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
            "g_scheduler": g_scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_snr": best_snr,
        }
        if disc is not None:
            save_dict["disc"] = disc.state_dict()
        if subband_disc is not None:
            save_dict["subband_disc"] = subband_disc.state_dict()
        if d_optimizer is not None:
            save_dict["d_optimizer"] = d_optimizer.state_dict()
        if d_scheduler is not None:
            save_dict["d_scheduler"] = d_scheduler.state_dict()
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
