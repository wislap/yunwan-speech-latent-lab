"""Losses and scheduling for generation-friendly latent training.

Contains:
- TemporalSmoothnessLoss: frame-to-frame L2 regularizer
- PhaseConfig / PhaseScheduler: multi-phase training orchestration
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


class TemporalSmoothnessLoss(nn.Module):
    """Frame-to-frame L2 smoothness regularizer.

    Computes mean(||z_t - z_{t-1}||^2) over time dimension.
    Weight linearly increases from initial_weight to max_weight over training.

    Args:
        initial_weight: Starting loss weight (very small).
        max_weight: Maximum loss weight at end of training.
    """

    def __init__(self, initial_weight: float = 1e-5, max_weight: float = 1e-3):
        super().__init__()
        self.initial_weight = initial_weight
        self.max_weight = max_weight

    def get_weight(self, step: int, total_steps: int) -> float:
        """Linear interpolation from initial to max weight."""
        if total_steps <= 0:
            return self.initial_weight
        progress = min(step / total_steps, 1.0)
        return self.initial_weight + progress * (self.max_weight - self.initial_weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute temporal smoothness loss.

        Args:
            z: Latent tensor [B, C, T] or [B, T, C].
               Assumes time is the last dim if 3D with C < T, else dim=2.

        Returns:
            Scalar loss (unweighted - caller applies weight).
        """
        # Assume z is [B, C, T] (encoder output format)
        if z.dim() == 3 and z.shape[1] > z.shape[2]:
            # Likely [B, T, C], transpose
            z = z.transpose(1, 2)

        # z is [B, C, T]
        if z.shape[2] < 2:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        diff = z[:, :, 1:] - z[:, :, :-1]  # [B, C, T-1]
        loss = (diff ** 2).mean()
        return loss


@dataclass
class PhaseConfig:
    """Configuration for a single training phase."""

    phase_id: int
    epochs: int
    lpf_cutoff_hz: float | None  # None = no filtering

    # Loss weights
    mel_weight: float = 1.0
    stft_mag_weight: float = 0.5
    l1_time_weight: float = 0.0
    l2_time_weight: float = 0.0
    predictor_weight: float = 1.0
    smooth_weight: float = 1e-4

    # Discriminator
    adv_enable: bool = False
    adv_g_weight: float = 0.0
    adv_fm_weight: float = 0.0

    # Learning rates
    encoder_lr: float = 1e-4
    decoder_lr: float = 1e-4
    predictor_lr: float = 5e-4

    # Masking
    mask_ratio: float = 0.6
    min_block_frames: int = 400
    max_block_frames: int = 800


class PhaseScheduler:
    """Manages multi-phase training configuration.

    Phase 1: Low-pass data + mel/STFT magnitude only + masked predictor
    Phase 2: Gradually raise LPF cutoff + stronger STFT loss
    Phase 3: Full frequency + L1/L2 + discriminator

    Args:
        phase_configs: List of PhaseConfig for each phase.
    """

    def __init__(self, phase_configs: list[PhaseConfig]):
        assert len(phase_configs) >= 1
        self.phases = phase_configs
        # Compute cumulative epoch boundaries
        self._boundaries = []
        cumulative = 0
        for pc in phase_configs:
            cumulative += pc.epochs
            self._boundaries.append(cumulative)

    @classmethod
    def default_3phase(
        cls,
        phase1_epochs: int = 30,
        phase2_epochs: int = 20,
        phase3_epochs: int = 20,
    ) -> "PhaseScheduler":
        """Create default 3-phase scheduler."""
        configs = [
            PhaseConfig(
                phase_id=1,
                epochs=phase1_epochs,
                lpf_cutoff_hz=4000.0,
                mel_weight=1.0,
                stft_mag_weight=0.5,
                l1_time_weight=0.0,
                predictor_weight=1.0,
                smooth_weight=1e-4,
                adv_enable=False,
                encoder_lr=1e-4,
                decoder_lr=1e-4,
                predictor_lr=5e-4,
                mask_ratio=0.6,
                min_block_frames=400,
                max_block_frames=800,
            ),
            PhaseConfig(
                phase_id=2,
                epochs=phase2_epochs,
                lpf_cutoff_hz=8000.0,  # Will be interpolated
                mel_weight=1.0,
                stft_mag_weight=1.0,
                l1_time_weight=0.0,
                predictor_weight=0.8,
                smooth_weight=5e-4,
                adv_enable=False,
                encoder_lr=5e-5,
                decoder_lr=1e-4,
                predictor_lr=3e-4,
                mask_ratio=0.6,
                min_block_frames=400,
                max_block_frames=800,
            ),
            PhaseConfig(
                phase_id=3,
                epochs=phase3_epochs,
                lpf_cutoff_hz=None,  # Full frequency
                mel_weight=1.0,
                stft_mag_weight=1.0,
                l1_time_weight=5.0,
                predictor_weight=0.5,
                smooth_weight=1e-3,
                adv_enable=True,
                adv_g_weight=1.0,
                adv_fm_weight=2.0,
                encoder_lr=2e-5,
                decoder_lr=5e-5,
                predictor_lr=1e-4,
                mask_ratio=0.5,
                min_block_frames=300,
                max_block_frames=600,
            ),
        ]
        return cls(configs)

    def get_phase_index(self, epoch: int) -> int:
        """Get phase index (0-based) for given epoch."""
        for i, boundary in enumerate(self._boundaries):
            if epoch < boundary:
                return i
        return len(self.phases) - 1

    def get_current_config(self, epoch: int) -> PhaseConfig:
        """Get configuration for the given epoch."""
        idx = self.get_phase_index(epoch)
        return self.phases[idx]

    def get_lpf_cutoff(self, epoch: int) -> float | None:
        """Get low-pass filter cutoff for the given epoch.

        In Phase 2, interpolates between Phase 1 and Phase 2 cutoff values.
        """
        config = self.get_current_config(epoch)
        if config.phase_id == 1:
            return config.lpf_cutoff_hz
        elif config.phase_id == 2:
            # Interpolate from phase1 cutoff to phase2 cutoff
            phase1_cutoff = self.phases[0].lpf_cutoff_hz or 4000.0
            phase2_cutoff = config.lpf_cutoff_hz or 11025.0
            # Progress within phase 2
            phase2_start = self._boundaries[0]
            phase2_end = self._boundaries[1]
            progress = (epoch - phase2_start) / max(phase2_end - phase2_start, 1)
            progress = min(max(progress, 0.0), 1.0)
            return phase1_cutoff + progress * (phase2_cutoff - phase1_cutoff)
        else:
            return None  # Full frequency

    @property
    def total_epochs(self) -> int:
        return self._boundaries[-1] if self._boundaries else 0
