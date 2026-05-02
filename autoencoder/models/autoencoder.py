"""V14.4 Wav-VAE: Pure sequential (DAC-style) + sub-pixel decoder.

V14.4 architecture (no skip, no LayerScale):
- Encoder: Conv stem → N × (DilatedResUnit×3 + SnakeBeta + StridedConv) → projection
- Decoder: projection → N × (SnakeBeta + Conv + pixel_shuffle + DilatedResUnit×3) → tail
- VAE bottleneck: softplus std, soft range penalty
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
    """Pre-activation residual. No LayerScale — matches DAC."""

    def __init__(self, channels: int, dilation: int = 1, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            SnakeBeta(channels),
            WNConv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            SnakeBeta(channels),
            WNConv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ─── Encoder Block (with anti-aliased skip) ─────────────────


class EncoderBlock(nn.Module):
    """3x DilatedResUnit → SnakeBeta → strided Conv. Pure sequential (DAC-style).

    No skip connections — avoids pixel_unshuffle aliasing and AvgPool explosion.
    V14.4 architecture: proven stable, clean frequency response.
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


# ─── Decoder Block (sub-pixel + parameter-free skip) ────────


class DecoderBlock(nn.Module):
    """SnakeBeta → sub-pixel upsample → 3x DilatedResUnit. Pure sequential.

    Sub-pixel upsample (Conv + pixel_shuffle) instead of ConvTranspose1d.
    No skip connections — V14.4 architecture.
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

        # block[0] = SnakeBeta, block[1] = upsample conv
        self.block = nn.Sequential(
            SnakeBeta(in_channels),
            WNConv1d(in_channels, expanded, kernel_size=3, padding=1),
        )
        self.res_units = nn.Sequential(*[
            DilatedResUnit(out_channels, d, kernel_size) for d in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x)  # act + upsample_conv
        h = pixel_shuffle_1d(h, self.stride)
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


# ─── iSTFT Decoder (V14.8) ───────────────────────────────────


# ─── Vocos-style iSTFT Decoder (V14.7f) ─────────────────────


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block adapted for 1D audio (from Vocos).

    Depthwise conv → LayerNorm → pointwise expand → GELU → pointwise shrink.
    """

    def __init__(self, dim: int, intermediate_dim: int, kernel_size: int = 7):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)
        return residual + x


class VocosDecoder(nn.Module):
    """Vocos-style isotropic decoder with iSTFT synthesis.

    Maintains the same temporal resolution (latent frame rate) across all layers.
    No transposed convolutions — upsampling is done entirely by iSTFT.
    Each frequency bin has a direct parameter path for magnitude and phase.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 1024,
        n_layers: int = 8,
        kernel_size: int = 7,
        n_fft: int = 1024,
        hop_length: int = 512,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_fft // 2 + 1

        self.input_proj = nn.Conv1d(latent_dim, hidden_dim, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(hidden_dim, hidden_dim * 2, kernel_size)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Conv1d(hidden_dim, self.n_bins * 2, kernel_size=1)
        # Initialize output near silence to prevent gradient explosion
        nn.init.normal_(self.output_proj.weight, std=0.001)
        nn.init.constant_(self.output_proj.bias[:self.n_bins], -5.0)
        nn.init.zeros_(self.output_proj.bias[self.n_bins:])
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, z: torch.Tensor, output_length: int | None = None) -> torch.Tensor:
        h = self.input_proj(z)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        out = self.output_proj(h)
        out = out.float()
        mag_param, phase_param = out.chunk(2, dim=1)
        mag = F.softplus(mag_param)
        x = torch.cos(phase_param)
        y = torch.sin(phase_param)
        stft_complex = mag * torch.complex(x, y)
        wav = torch.istft(
            stft_complex, self.n_fft, self.hop_length, self.n_fft,
            window=self.window, length=output_length,
        )
        return wav.unsqueeze(1)


class ISTFTHead(nn.Module):
    """iSTFT synthesis head: predicts STFT magnitude + phase from features.

    Input: [B, in_ch, T_samples] at full sample rate (from decoder blocks)
    Output: [B, 1, T_audio] waveform via iSTFT

    Uses strided conv to aggregate per-sample features into STFT frames,
    then predicts magnitude and phase per frequency bin.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 256,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_fft // 2 + 1

        # Aggregate per-sample features into STFT frames
        # Conv with kernel=hop*2, stride=hop acts like a learned STFT
        self.frame_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels,
                      kernel_size=hop_length * 2, stride=hop_length,
                      padding=hop_length // 2),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # Project to mag + cos(phase) + sin(phase)
        self.proj = nn.Conv1d(hidden_channels, self.n_bins * 3, kernel_size=1)
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x: torch.Tensor, output_length: int | None = None) -> torch.Tensor:
        h = self.frame_conv(x)  # [B, hidden, T_frames]
        h = self.proj(h)        # [B, n_bins*3, T_frames]

        mag, cos_ph, sin_ph = h.chunk(3, dim=1)
        mag = F.softplus(mag)
        phase = torch.atan2(sin_ph, cos_ph)
        stft_complex = mag * torch.exp(1j * phase)

        wav = torch.istft(
            stft_complex, self.n_fft, self.hop_length, self.n_fft,
            window=self.window, length=output_length,
        )
        return wav.unsqueeze(1)


class WavVAEDecoderISTFT(nn.Module):
    """V14.8 decoder: full time-domain decoder + parallel iSTFT head.

    Keeps all 5 decoder blocks + tail from V14.4 (loaded from checkpoint).
    Adds a parallel iSTFT head that takes the same block[4] output.
    Final output = time-domain waveform + iSTFT waveform (learned blend).

    The time-domain path handles low-frequency content well.
    The iSTFT path provides phase-accurate high-frequency reconstruction.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        channels: Sequence[int] = (1024, 1024, 512, 512, 256, 128),
        strides: Sequence[int] = (4, 4, 4, 4, 2),
        dilations: Sequence[int] = (1, 3, 9),
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        super().__init__()
        assert len(strides) == len(channels) - 1

        self.proj = WNConv1d(latent_dim, channels[0], kernel_size=7, padding=3)
        self.blocks = nn.ModuleList([
            DecoderBlock(channels[i], channels[i + 1], strides[i], dilations)
            for i in range(len(strides))
        ])
        # Time-domain tail (from V14.4, loaded from checkpoint)
        self.tail = nn.Sequential(
            SnakeBeta(channels[-1]),
            WNConv1d(channels[-1], 1, kernel_size=7, padding=3, bias=False),
        )
        # Parallel iSTFT head (trained from scratch)
        self.istft_head = ISTFTHead(
            in_channels=channels[-1],
            hidden_channels=256,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        # Learnable blend weight (sigmoid -> 0=time-domain, 1=iSTFT)
        self.blend_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, z: torch.Tensor, output_length: int | None = None) -> torch.Tensor:
        h = self.proj(z)
        for block in self.blocks:
            h = block(h)

        # Time-domain path
        x_time = self.tail(h)

        # iSTFT path
        x_istft = self.istft_head(h, output_length=x_time.shape[-1])

        # Blend
        alpha = torch.sigmoid(self.blend_logit)
        x = (1 - alpha) * x_time + alpha * x_istft

        if output_length is not None:
            if x.shape[-1] > output_length:
                x = x[:, :, :output_length]
            elif x.shape[-1] < output_length:
                x = F.pad(x, (0, output_length - x.shape[-1]))
        return x


class VocosHead(nn.Module):
    """Vocos-style iSTFT head for hybrid decoder.

    Takes intermediate features from time-domain decoder block[0],
    predicts STFT magnitude + phase, synthesizes waveform via iSTFT.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        hidden_dim: int = 512,
        n_layers: int = 4,
        kernel_size: int = 7,
        n_fft: int = 512,
        hop_length: int = 128,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_fft // 2 + 1

        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(hidden_dim, hidden_dim * 2, kernel_size)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Conv1d(hidden_dim, self.n_bins * 2, kernel_size=1)
        # Safe init: mag bias=-5 → softplus(-5)≈0.007, near silence
        nn.init.normal_(self.output_proj.weight, std=0.001)
        nn.init.constant_(self.output_proj.bias[:self.n_bins], -5.0)
        nn.init.zeros_(self.output_proj.bias[self.n_bins:])
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, features: torch.Tensor, output_length: int | None = None) -> torch.Tensor:
        h = self.input_proj(features)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        out = self.output_proj(h).float()

        mag_param, phase_param = out.chunk(2, dim=1)
        mag = F.softplus(mag_param)
        x = torch.cos(phase_param)
        y = torch.sin(phase_param)
        stft_complex = mag * torch.complex(x, y)

        wav = torch.istft(
            stft_complex, self.n_fft, self.hop_length, self.n_fft,
            window=self.window, length=output_length,
        )
        return wav.unsqueeze(1)


class HybridDecoder(nn.Module):
    """V14.7g: Time-domain block[0] + Vocos iSTFT head.

    Uses one time-domain DecoderBlock to expand latent from 43Hz to 172Hz,
    then VocosHead predicts STFT magnitude+phase and synthesizes via iSTFT.

    block[0] does "abstract → concrete" mapping (it's good at this).
    VocosHead does "concrete features → frequency domain" (it's good at this).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        first_channels: tuple[int, int] = (1024, 1024),
        first_stride: int = 4,
        dilations: Sequence[int] = (1, 3, 9),
        vocos_hidden: int = 512,
        vocos_layers: int = 4,
        n_fft: int = 512,
        hop_length: int = 128,
    ):
        super().__init__()
        self.proj = WNConv1d(latent_dim, first_channels[0], kernel_size=7, padding=3)
        self.block0 = DecoderBlock(
            first_channels[0], first_channels[1], first_stride, dilations,
        )
        self.vocos_head = VocosHead(
            in_channels=first_channels[1],
            hidden_dim=vocos_hidden,
            n_layers=vocos_layers,
            n_fft=n_fft,
            hop_length=hop_length,
        )

    def forward(self, z: torch.Tensor, output_length: int | None = None) -> torch.Tensor:
        h = self.proj(z)
        h = self.block0(h)
        return self.vocos_head(h, output_length=output_length)


class DualPathDecoder(nn.Module):
    """V15: Dual-path decoder — time-domain for magnitude, Vocos for phase.

    Architecture:
      z → proj → block[0] → shared features [1024ch, 172Hz]
                                    │
                    ┌────────────────┤
                    │                │
          time-domain path     Vocos phase path
          (block[1..4]+tail)   (ConvNeXt+iSTFT)
                    │                │
               wav_time          wav_freq
                    │                │
                    └── freq-domain fusion ──→ wav_final
                        (time mag + freq phase)

    Time-domain path produces accurate magnitude (proven 22.3 dB).
    Vocos path produces accurate phase (block[0] features have R²=0.96).
    Fusion takes magnitude from time-domain, phase from Vocos.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        channels: Sequence[int] = (1024, 1024, 512, 512, 256, 128),
        strides: Sequence[int] = (4, 4, 4, 4, 2),
        dilations: Sequence[int] = (1, 3, 9),
        vocos_hidden: int = 512,
        vocos_layers: int = 6,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        super().__init__()
        assert len(strides) == len(channels) - 1
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Shared: proj + block[0]
        self.proj = WNConv1d(latent_dim, channels[0], kernel_size=7, padding=3)
        self.block0 = DecoderBlock(channels[0], channels[1], strides[0], dilations)

        # Time-domain path: block[1..N] + tail
        self.time_blocks = nn.ModuleList([
            DecoderBlock(channels[i], channels[i + 1], strides[i], dilations)
            for i in range(1, len(strides))
        ])
        self.tail = nn.Sequential(
            SnakeBeta(channels[-1]),
            WNConv1d(channels[-1], 1, kernel_size=7, padding=3, bias=False),
        )

        # Vocos phase path: from block[0] features predict phase
        self.vocos_head = VocosHead(
            in_channels=channels[1],
            hidden_dim=vocos_hidden,
            n_layers=vocos_layers,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        # Learnable blend: 0=pure time-domain, 1=pure vocos
        self.blend_logit = nn.Parameter(torch.tensor(0.0))  # init 50/50

    def forward(self, z: torch.Tensor, output_length: int | None = None) -> torch.Tensor:
        # Shared path
        h = self.proj(z)
        h_shared = self.block0(h)  # [B, C1, T*stride0]

        # Time-domain path → waveform (auxiliary, for training block0)
        h_time = h_shared
        for block in self.time_blocks:
            h_time = block(h_time)
        wav_time = self.tail(h_time)  # [B, 1, T]

        if output_length is not None:
            if wav_time.shape[-1] > output_length:
                wav_time = wav_time[:, :, :output_length]
            elif wav_time.shape[-1] < output_length:
                wav_time = F.pad(wav_time, (0, output_length - wav_time.shape[-1]))

        T = wav_time.shape[-1]

        # Vocos phase path → waveform (primary output)
        wav_vocos = self.vocos_head(h_shared, output_length=T)

        # Simple blend in waveform domain
        alpha = torch.sigmoid(self.blend_logit)
        wav_out = (1 - alpha) * wav_time + alpha * wav_vocos

        return wav_out


# ─── Phase Refiner (V14.9) ───────────────────────────────────


class PhaseRefiner(nn.Module):
    """Post-decoder phase correction via STFT domain.

    Takes the coarse waveform from decoder + latent z,
    predicts per-bin phase delta, applies it via iSTFT.
    
    The decoder already produces reasonable magnitude but wrong phase.
    This module only corrects phase, keeping magnitude intact.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_fft // 2 + 1

        # Predict phase delta from latent (latent is at STFT frame rate)
        self.phase_net = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, self.n_bins * 2, kernel_size=1),  # cos_delta, sin_delta
        )
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(
        self, x_coarse: torch.Tensor, z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_coarse: [B, 1, T] coarse waveform from decoder
            z: [B, latent_dim, T_lat] latent features
        Returns:
            [B, 1, T] phase-corrected waveform
        """
        T = x_coarse.shape[-1]

        # Predict phase delta from latent
        pd = self.phase_net(z)  # [B, n_bins*2, T_lat]
        cos_d, sin_d = pd.chunk(2, dim=1)  # each [B, n_bins, T_lat]
        phase_delta = torch.atan2(sin_d, cos_d)

        # STFT of coarse waveform
        stft = torch.stft(
            x_coarse.squeeze(1).float(), self.n_fft, self.hop_length,
            self.n_fft, window=self.window, return_complex=True,
        )  # [B, n_bins, T_frames]

        # Align time dimensions
        T_frames = stft.shape[-1]
        T_lat = phase_delta.shape[-1]
        if T_lat != T_frames:
            phase_delta = F.interpolate(
                phase_delta, size=T_frames, mode="linear", align_corners=False,
            )

        # Apply phase correction: keep magnitude, rotate phase
        mag = stft.abs()
        phase_orig = stft.angle()
        phase_new = phase_orig + phase_delta
        stft_corrected = mag * torch.exp(1j * phase_new)

        # iSTFT
        wav = torch.istft(
            stft_corrected, self.n_fft, self.hop_length, self.n_fft,
            window=self.window, length=T,
        )
        return wav.unsqueeze(1)


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
    """Complete Wav-VAE aligned with LongCat-AudioDiT architecture.

    Args:
        use_istft_decoder: If True, use iSTFT decoder (V14.8).
            Drops block[4]+tail, replaces with ISTFTHead.
        istft_n_fft: FFT size for iSTFT head.
        istft_hop: Hop size for iSTFT head.
    """

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
        use_istft_decoder: bool = False,
        istft_n_fft: int = 2048,
        istft_hop: int = 512,
        sr: int = 22050,
        use_phase_refiner: bool = False,
        phase_n_fft: int = 2048,
        phase_hop: int = 512,
        use_vocos_decoder: bool = False,
        vocos_hidden_dim: int = 1024,
        vocos_n_layers: int = 8,
        vocos_n_fft: int = 1024,
        vocos_hop: int = 512,
        use_hybrid_decoder: bool = False,
        hybrid_vocos_hidden: int = 512,
        hybrid_vocos_layers: int = 4,
        hybrid_n_fft: int = 512,
        hybrid_hop: int = 128,
        use_dual_path_decoder: bool = False,
        dual_vocos_hidden: int = 512,
        dual_vocos_layers: int = 6,
        dual_n_fft: int = 1024,
        dual_hop: int = 256,
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

        if use_dual_path_decoder:
            # V15: dual-path — time for magnitude, vocos for phase
            rev_strides = list(reversed(strides))
            self.decoder = DualPathDecoder(
                latent_dim=latent_dim,
                channels=decoder_channels,
                strides=rev_strides,
                dilations=dilations,
                vocos_hidden=dual_vocos_hidden,
                vocos_layers=dual_vocos_layers,
                n_fft=dual_n_fft,
                hop_length=dual_hop,
            )
        elif use_hybrid_decoder:
            # V14.7g: block[0] + Vocos head
            rev_strides = list(reversed(strides))
            self.decoder = HybridDecoder(
                latent_dim=latent_dim,
                first_channels=(decoder_channels[0], decoder_channels[1]),
                first_stride=rev_strides[0],
                dilations=dilations,
                vocos_hidden=hybrid_vocos_hidden,
                vocos_layers=hybrid_vocos_layers,
                n_fft=hybrid_n_fft,
                hop_length=hybrid_hop,
            )
        elif use_vocos_decoder:
            # V14.7f: Vocos-style isotropic decoder with iSTFT
            self.decoder = VocosDecoder(
                latent_dim=latent_dim,
                hidden_dim=vocos_hidden_dim,
                n_layers=vocos_n_layers,
                n_fft=vocos_n_fft,
                hop_length=vocos_hop,
            )
        elif use_istft_decoder:
            # V14.8: full decoder + parallel iSTFT head
            rev_strides = list(reversed(strides))
            self.decoder = WavVAEDecoderISTFT(
                latent_dim=latent_dim,
                channels=decoder_channels,
                strides=rev_strides,
                dilations=dilations,
                n_fft=istft_n_fft,
                hop_length=istft_hop,
            )
        else:
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

        # Phase refiner (V14.9)
        self.phase_refiner = None
        if use_phase_refiner:
            self.phase_refiner = PhaseRefiner(
                latent_dim=latent_dim,
                n_fft=phase_n_fft,
                hop_length=phase_hop,
            )

    def set_step(self, step: int) -> None:
        """Update global step for regularization warmup."""
        self.bottleneck.set_step(step)

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
        if self.phase_refiner is not None:
            x_hat = self.phase_refiner(x_hat, z)
        reg_loss = self.bottleneck.reg_loss(mean, std)
        return {
            "x_hat": x_hat,
            "z": z,
            "mean": mean,
            "std": std,
            "reg_loss": reg_loss,
        }
