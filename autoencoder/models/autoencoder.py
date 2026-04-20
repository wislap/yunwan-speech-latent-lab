"""V14 Wav-VAE: Oobleck-style fully convolutional VAE on raw waveforms.

Architecture aligned with LongCat-AudioDiT's Wav-VAE implementation:
- SnakeBeta activation (log-scale alpha + beta)
- pixel_unshuffle/shuffle shortcuts (not adaptive_avg_pool)
- softplus VAE bottleneck (not exp(log_var))
- pre-activation before strided conv/transposed conv
- weight_norm on all convolutions, no LayerNorm/BatchNorm
"""

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ─── Pixel Shuffle 1D ───────────────────────────────────────


def pixel_unshuffle_1d(x: torch.Tensor, factor: int) -> torch.Tensor:
    """[B, C, T] → [B, C*factor, T//factor]  (space-to-channel)."""
    B, C, T = x.shape
    T_out = T // factor
    x = x[:, :, :T_out * factor]  # trim
    return x.view(B, C, T_out, factor).permute(0, 1, 3, 2).contiguous().view(B, C * factor, T_out)


def pixel_shuffle_1d(x: torch.Tensor, factor: int) -> torch.Tensor:
    """[B, C, T] → [B, C//factor, T*factor]  (channel-to-space)."""
    B, C, T = x.shape
    C_out = C // factor
    return x.view(B, C_out, factor, T).permute(0, 1, 3, 2).contiguous().view(B, C_out, T * factor)


# ─── Shortcuts (from LongCat-AudioDiT) ──────────────────────


class DownsampleShortcut(nn.Module):
    """Non-parametric downsample: pixel_unshuffle + adaptive grouped mean."""

    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.out_channels = out_channels
        # After pixel_unshuffle: C_expanded = in_channels * factor
        # We need to map C_expanded → out_channels via grouped averaging
        self.c_expanded = in_channels * factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pixel_unshuffle_1d(x, self.factor)  # [B, C*s, T/s]
        B, C, T = x.shape
        # Use simple linear interpolation on channel dim if not evenly divisible
        # This is equivalent to adaptive average pooling but deterministic
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = F.adaptive_avg_pool1d(x, self.out_channels)  # [B, T, out_ch]
        return x.permute(0, 2, 1)  # [B, out_ch, T]


class UpsampleShortcut(nn.Module):
    """Non-parametric upsample: channel interpolation + pixel_shuffle."""

    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_channels = out_channels * factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        repeats = math.ceil(self.target_channels / C)
        x_rep = x.repeat_interleave(repeats, dim=1)  # [B, C*repeats, T]
        x_rep = x_rep[:, :self.target_channels, :]  # trim to exact target
        return pixel_shuffle_1d(x_rep, self.factor)  # [B, out_ch, T*factor]


# ─── Activation ──────────────────────────────────────────────


def _snake_beta(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


class SnakeBeta(nn.Module):
    """Snake-beta activation with log-scale learnable alpha and beta.

    Matches LongCat-AudioDiT's AudioDiTSnakeBeta.
    """

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
    """Pre-activation residual: Act → Conv(dilation=d) → Act → Conv(1x1) → LayerScale → add.

    Matches LongCat's _VaeResidualUnit + LayerScale for training stability.
    """

    def __init__(self, channels: int, dilation: int = 1, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            SnakeBeta(channels),
            WNConv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            SnakeBeta(channels),
            WNConv1d(channels, channels, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.ones(1, channels, 1) * 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.block(x)


# ─── Oobleck Encoder Block ──────────────────────────────────


class OobleckEncoderBlock(nn.Module):
    """3x DilatedResUnit → SnakeBeta → strided Conv + shortcut.

    Matches LongCat's _VaeEncoderBlock.
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
        self.layers = nn.Sequential(*layers)
        self.shortcut = DownsampleShortcut(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layers(x)
        s = self.shortcut(x)
        min_len = min(h.shape[-1], s.shape[-1])
        return h[:, :, :min_len] + s[:, :, :min_len]


# ─── Oobleck Decoder Block ──────────────────────────────────


class OobleckDecoderBlock(nn.Module):
    """SnakeBeta → transposed Conv → 3x DilatedResUnit + shortcut.

    Matches LongCat's _VaeDecoderBlock.
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
        layers = [
            SnakeBeta(in_channels),
            WNConvTranspose1d(
                in_channels, out_channels,
                kernel_size=2 * stride, stride=stride,
                padding=math.ceil(stride / 2),
            ),
        ]
        for d in dilations:
            layers.append(DilatedResUnit(out_channels, d, kernel_size))
        self.layers = nn.Sequential(*layers)
        self.shortcut = UpsampleShortcut(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layers(x)
        s = self.shortcut(x)
        # Align lengths (transposed conv may differ by ±1)
        min_len = min(h.shape[-1], s.shape[-1])
        return h[:, :, :min_len] + s[:, :, :min_len]


# ─── Encoder ─────────────────────────────────────────────────


class WavVAEEncoder(nn.Module):
    """Conv1d stem → N OobleckEncoderBlocks → Conv1d projection."""

    def __init__(
        self,
        latent_dim: int = 64,
        channels: Sequence[int] = (128, 256, 512, 1024),
        strides: Sequence[int] = (7, 7, 9),
        dilations: Sequence[int] = (1, 3, 9),
    ):
        super().__init__()
        # encoder_latent_dim = latent_dim * 2 for (mean, scale_param)
        self.stem = WNConv1d(1, channels[0], kernel_size=7, padding=3)
        self.blocks = nn.ModuleList([
            OobleckEncoderBlock(channels[i], channels[i + 1], strides[i], dilations)
            for i in range(len(strides))
        ])
        self.proj = WNConv1d(channels[-1], latent_dim * 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input [B, 1, T], output [B, latent_dim*2, T']."""
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.proj(h)


# ─── Decoder ─────────────────────────────────────────────────


class WavVAEDecoder(nn.Module):
    """Conv1d proj → N OobleckDecoderBlocks → SnakeBeta → Conv1d tail → Tanh."""

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
            OobleckDecoderBlock(channels[i], channels[i + 1], strides[i], dilations)
            for i in range(len(strides))
        ])
        self.tail = nn.Sequential(
            SnakeBeta(channels[-1]),
            WNConv1d(channels[-1], 1, kernel_size=7, padding=3, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, output_length: int | None = None) -> torch.Tensor:
        """Input [B, latent_dim, T'], output [B, 1, T]."""
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
    """VAE bottleneck with softplus std parameterization.

    Matches LongCat-AudioDiT: mean + softplus(scale_param) + 1e-4.
    Much more numerically stable than exp(0.5 * log_var).
    """

    def __init__(
        self,
        kl_weight: float = 1e-4,
        free_bits: float = 0.25,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.free_bits = free_bits

    def encode(
        self, encoder_output: torch.Tensor, training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split encoder output into mean/std, sample z.

        Returns (z, mean, std).
        """
        mean, scale_param = encoder_output.chunk(2, dim=1)
        std = F.softplus(scale_param) + 1e-4
        if training:
            z = mean + std * torch.randn_like(mean)
        else:
            z = mean
        return z, mean, std

    def kl_loss(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0,I)) with free-bits, using (mean, std) parameterization."""
        var = std.pow(2)
        log_var = var.log()
        kl_per_dim = -0.5 * (1 + log_var - mean.pow(2) - var)
        kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        return self.kl_weight * kl_per_dim.mean()


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
        kl_weight: float = 1e-4,
        free_bits: float = 0.25,
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
            kl_weight=kl_weight,
            free_bits=free_bits,
        )
        self._strides = list(strides)

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
        kl_loss = self.bottleneck.kl_loss(mean, std)
        return {
            "x_hat": x_hat,
            "z": z,
            "mean": mean,
            "std": std,
            "kl_loss": kl_loss,
        }
