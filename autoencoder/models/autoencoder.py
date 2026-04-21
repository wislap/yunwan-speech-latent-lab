"""V14.4 Wav-VAE: DAC-style encoder (no shortcut) + sub-pixel decoder.

Key changes from V14.3:
- Removed all shortcuts (DownsampleShortcut/UpsampleShortcut) to eliminate
  aliasing from pixel_unshuffle in encoder path.
- Encoder: pure sequential (ResUnits → Snake → StridedConv), matching DAC.
- Decoder: pure sequential (Snake → SubPixelConv → ResUnits), no shortcut.
- Kept: SnakeBeta, weight_norm, LayerScale, dilated residual units.
"""

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ─── Pixel Shuffle 1D (decoder only) ────────────────────────


def pixel_shuffle_1d(x: torch.Tensor, factor: int) -> torch.Tensor:
    """[B, C, T] → [B, C//factor, T*factor]  (channel-to-space)."""
    B, C, T = x.shape
    C_out = C // factor
    return x.view(B, C_out, factor, T).permute(0, 1, 3, 2).contiguous().view(B, C_out, T * factor)


# ─── Activation ──────────────────────────────────────────────


def _snake_beta(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


class SnakeBeta(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(channels))
        self.log_beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.log_alpha.exp().unsqueeze(0).unsqueeze(-1)
        beta = self.log_beta.exp().unsqueeze(0).unsqueeze(-1)
        return _snake_beta(x, alpha, beta)


# ─── Weight-Normed Conv Factories ────────────────────────────


def WNConv1d(*args, **kwargs) -> nn.Conv1d:
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs) -> nn.ConvTranspose1d:
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# ─── Dilated Residual Unit ───────────────────────────────────


class DilatedResUnit(nn.Module):
    """Pre-activation residual with warmup LayerScale.
    
    Scale warms up from init_scale to target_scale over warmup_steps,
    then becomes a learnable parameter at target_scale.
    """

    def __init__(self, channels: int, dilation: int = 1, kernel_size: int = 7,
                 init_scale: float = 0.01, target_scale: float = 1.0):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            SnakeBeta(channels),
            WNConv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            SnakeBeta(channels),
            WNConv1d(channels, channels, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.ones(1, channels, 1) * init_scale)
        self.init_scale = init_scale
        self.target_scale = target_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.block(x)


# ─── Encoder Block (DAC-style, no shortcut) ─────────────────


class EncoderBlock(nn.Module):
    """3x DilatedResUnit → SnakeBeta → strided Conv.

    Pure sequential, no shortcut. Matches DAC's EncoderBlock.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilations: Sequence[int] = (1, 3, 9),
        kernel_size: int = 7,
    ):
        super().__init__()
        layers = []
        for d in dilations:
            layers.append(DilatedResUnit(in_channels, d, kernel_size))
        layers.append(SnakeBeta(in_channels))
        layers.append(WNConv1d(
            in_channels, out_channels,
            kernel_size=2 * stride, stride=stride,
            padding=math.ceil(stride / 2),
        ))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─── Decoder Block (sub-pixel, no shortcut) ─────────────────


class DecoderBlock(nn.Module):
    """SnakeBeta → sub-pixel upsample (Conv + pixel_shuffle) → 3x DilatedResUnit.

    No shortcut, no ConvTranspose. Zero aliasing from upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilations: Sequence[int] = (1, 3, 9),
        kernel_size: int = 7,
    ):
        super().__init__()
        expanded = out_channels * stride
        self.stride = stride

        self.block = nn.Sequential(
            SnakeBeta(in_channels),
            WNConv1d(in_channels, expanded, kernel_size=3, padding=1),
        )
        self.res_units = nn.Sequential(*[
            DilatedResUnit(out_channels, d, kernel_size) for d in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x)  # [B, out_ch*stride, T]
        h = pixel_shuffle_1d(h, self.stride)  # [B, out_ch, T*stride]
        return self.res_units(h)


# ─── Encoder ─────────────────────────────────────────────────


class WavVAEEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        channels: Sequence[int] = (128, 256, 512, 1024),
        strides: Sequence[int] = (7, 7, 9),
        dilations: Sequence[int] = (1, 3, 9),
    ):
        super().__init__()
        self.stem = WNConv1d(1, channels[0], kernel_size=7, padding=3)
        self.blocks = nn.ModuleList([
            EncoderBlock(channels[i], channels[i + 1], strides[i], dilations)
            for i in range(len(strides))
        ])
        self.proj = WNConv1d(channels[-1], latent_dim * 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.proj(h)


# ─── Decoder ─────────────────────────────────────────────────


class WavVAEDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        channels: Sequence[int] = (1024, 512, 256, 128),
        strides: Sequence[int] = (9, 7, 7),
        dilations: Sequence[int] = (1, 3, 9),
    ):
        super().__init__()
        self.proj = WNConv1d(latent_dim, channels[0], kernel_size=7, padding=3)
        self.blocks = nn.ModuleList([
            DecoderBlock(channels[i], channels[i + 1], strides[i], dilations)
            for i in range(len(strides))
        ])
        self.tail = nn.Sequential(
            SnakeBeta(channels[-1]),
            WNConv1d(channels[-1], 1, kernel_size=7, padding=3, bias=False),
        )

    def forward(self, z: torch.Tensor, output_length: int | None = None) -> torch.Tensor:
        h = self.proj(z)
        for block in self.blocks:
            h = block(h)
        x = self.tail(h)
        if output_length is not None:
            if x.shape[-1] > output_length:
                x = x[:, :, :output_length]
            elif x.shape[-1] < output_length:
                x = F.pad(x, (0, output_length - x.shape[-1]))
        return x


# ─── VAE Bottleneck (softplus, matching LongCat) ────────────


class VAEBottleneck(nn.Module):
    """Soft-constrained bottleneck: reparameterization + range penalty.

    Instead of KL divergence (which forces N(0,1) and kills reconstruction),
    uses soft range penalties on mean and std:
      - mean: penalize |μ| > margin_mean
      - std:  penalize σ < std_min or σ > std_max

    This keeps latents bounded and well-scaled for downstream DiT
    without forcing a specific distribution shape.
    """

    def __init__(
        self,
        reg_weight: float = 1e-2,
        margin_mean: float = 3.0,
        std_min: float = 0.1,
        std_max: float = 1.5,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.reg_weight = reg_weight
        self.margin_mean = margin_mean
        self.std_min = std_min
        self.std_max = std_max
        self.warmup_steps = warmup_steps
        self._step = 0

    def set_step(self, step: int) -> None:
        self._step = step

    @property
    def effective_weight(self) -> float:
        if self.warmup_steps <= 0:
            return self.reg_weight
        return self.reg_weight * min(self._step / self.warmup_steps, 1.0)

    def encode(
        self, encoder_output: torch.Tensor, training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split encoder output into mean/std, sample z."""
        mean, scale_param = encoder_output.chunk(2, dim=1)
        std = F.softplus(scale_param) + 1e-4
        if training:
            z = mean + std * torch.randn_like(mean)
        else:
            z = mean
        return z, mean, std

    def reg_loss(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Soft range penalty on mean and std."""
        # Mean penalty: penalize |μ| > margin
        mean_penalty = torch.mean(F.relu(mean.abs() - self.margin_mean) ** 2)
        # Std penalty: penalize σ < std_min or σ > std_max
        std_lo_penalty = torch.mean(F.relu(self.std_min - std) ** 2)
        std_hi_penalty = torch.mean(F.relu(std - self.std_max) ** 2)
        return self.effective_weight * (mean_penalty + std_lo_penalty + std_hi_penalty)


# ─── WavVAE Main Model ──────────────────────────────────────


class WavVAE(nn.Module):
    """Complete Wav-VAE aligned with LongCat-AudioDiT architecture."""

    def __init__(
        self,
        latent_dim: int = 64,
        encoder_channels: Sequence[int] = (128, 256, 512, 1024),
        decoder_channels: Sequence[int] | None = None,
        strides: Sequence[int] = (7, 7, 9),
        dilations: Sequence[int] = (1, 3, 9),
        reg_weight: float = 1e-2,
        margin_mean: float = 3.0,
        std_min: float = 0.1,
        std_max: float = 1.5,
        reg_warmup_steps: int = 1000,
    ):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = tuple(reversed(encoder_channels))

        self.encoder = WavVAEEncoder(
            latent_dim=latent_dim,
            channels=encoder_channels,
            strides=strides,
            dilations=dilations,
        )
        self.decoder = WavVAEDecoder(
            latent_dim=latent_dim,
            channels=decoder_channels,
            strides=list(reversed(strides)),
            dilations=dilations,
        )
        self.bottleneck = VAEBottleneck(
            reg_weight=reg_weight,
            margin_mean=margin_mean,
            std_min=std_min,
            std_max=std_max,
            warmup_steps=reg_warmup_steps,
        )
        self._strides = list(strides)

    def set_step(self, step: int) -> None:
        """Update global step for regularization warmup + LayerScale warmup."""
        self.bottleneck.set_step(step)
        self._warmup_layer_scales(step)

    def _warmup_layer_scales(self, step: int, warmup_steps: int = 1000) -> None:
        """Warmup LayerScale from init_scale to target_scale."""
        progress = min(step / max(warmup_steps, 1), 1.0)
        for module in self.modules():
            if isinstance(module, DilatedResUnit):
                target = module.target_scale
                init = module.init_scale
                current = init + (target - init) * progress
                # Clamp the parameter to at least current warmup value
                with torch.no_grad():
                    module.scale.clamp_(min=current)

    @property
    def total_stride(self) -> int:
        return math.prod(self._strides)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z, mean, std)."""
        enc_out = self.encoder(x)
        return self.bottleneck.encode(enc_out, self.training)

    def decode(self, z: torch.Tensor, output_length: int | None = None) -> torch.Tensor:
        return self.decoder(z, output_length)

    def forward(self, x: torch.Tensor) -> dict:
        z, mean, std = self.encode(x)
        x_hat = self.decode(z, output_length=x.shape[-1])
        reg_loss = self.bottleneck.reg_loss(mean, std)
        return {
            "x_hat": x_hat,
            "z": z,
            "mean": mean,
            "std": std,
            "reg_loss": reg_loss,
        }
