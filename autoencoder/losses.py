"""V14 Wav-VAE loss functions.

Multi-scale mel spectrogram loss + aggregation helper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _create_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> torch.Tensor:
    """Create a mel filterbank matrix without torchaudio dependency."""
    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    import math
    n_freqs = n_fft // 2 + 1
    f_max = sr / 2.0

    mel_min = hz_to_mel(0.0)
    mel_max = hz_to_mel(f_max)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = torch.tensor([mel_to_hz(m) for m in mel_points])
    bin_points = (hz_points * n_fft / sr).long().clamp(0, n_freqs - 1)

    fb = torch.zeros(n_freqs, n_mels)
    for i in range(n_mels):
        lo, mid, hi = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        if lo < mid:
            fb[lo:mid, i] = torch.linspace(0, 1, mid - lo)
        if mid < hi:
            fb[mid:hi, i] = torch.linspace(1, 0, hi - mid)
    return fb


class MultiScaleMelLoss(nn.Module):
    """Multi-scale mel spectrogram L1 loss.

    Pure PyTorch implementation — no torchaudio dependency.
    """

    def __init__(
        self,
        sr: int = 22050,
        fft_sizes: tuple[int, ...] = (512, 1024, 2048),
        n_mels_list: tuple[int, ...] = (64, 128, 256),
    ):
        super().__init__()
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = [f // 4 for f in fft_sizes]
        self.n_mels_list = list(n_mels_list)

        # Pre-compute mel filterbanks and windows
        for i, (fft_size, n_mels) in enumerate(zip(fft_sizes, n_mels_list)):
            self.register_buffer(f"mel_fb_{i}", _create_mel_filterbank(sr, fft_size, n_mels))
            self.register_buffer(f"window_{i}", torch.hann_window(fft_size))

    def _mel_spec(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Compute mel spectrogram for scale idx."""
        fft_size = self.fft_sizes[idx]
        hop_size = self.hop_sizes[idx]
        window = getattr(self, f"window_{idx}")
        mel_fb = getattr(self, f"mel_fb_{idx}")

        # STFT
        spec = torch.stft(
            x, fft_size, hop_size, fft_size,
            window=window, return_complex=True,
        )
        mag = spec.abs()  # [B, F, T]
        # Apply mel filterbank: [B, F, T] @ [F, M] -> [B, M, T]
        mel = torch.matmul(mag.transpose(-2, -1), mel_fb).transpose(-2, -1)
        return mel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_in = x.squeeze(1).float()
        y_in = y.squeeze(1).float()
        loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        for i in range(len(self.fft_sizes)):
            loss = loss + F.l1_loss(self._mel_spec(x_in, i), self._mel_spec(y_in, i))
        return loss / len(self.fft_sizes)


def compute_reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mrstft_loss_fn: nn.Module,
    mel_loss_fn: nn.Module,
    reg_loss: torch.Tensor,
    l1_weight: float = 10.0,
    l2_weight: float = 5.0,
    stft_weight: float = 0.5,
    mel_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Aggregate all reconstruction losses.

    L1+L2 time-domain losses dominate to directly optimize waveform accuracy.
    """
    l1_loss = F.l1_loss(x_hat, x)
    l2_loss = F.mse_loss(x_hat, x)
    stft_loss = mrstft_loss_fn(x_hat, x)
    mel_loss = mel_loss_fn(x_hat, x)

    total = (
        l1_weight * l1_loss
        + l2_weight * l2_loss
        + stft_weight * stft_loss
        + mel_weight * mel_loss
        + reg_loss
    )
    return {
        "total": total,
        "l1": l1_loss,
        "l2": l2_loss,
        "stft": stft_loss,
        "mel": mel_loss,
        "reg": reg_loss,
    }
