"""V14 Wav-VAE loss functions.

Multi-scale mel spectrogram loss + aggregation helper.
Reuses MultiResolutionSTFTLoss from pqmf_ae/losses.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MultiScaleMelLoss(nn.Module):
    """Multi-scale mel spectrogram L1 loss.

    Computes mel spectrograms at multiple FFT sizes and returns the
    average L1 distance across scales.
    """

    def __init__(
        self,
        sr: int = 22050,
        fft_sizes: tuple[int, ...] = (512, 1024, 2048),
        n_mels_list: tuple[int, ...] = (64, 128, 256),
    ):
        super().__init__()
        self.mel_specs = nn.ModuleList()
        for fft_size, n_mels in zip(fft_sizes, n_mels_list):
            hop = fft_size // 4
            self.mel_specs.append(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr,
                    n_fft=fft_size,
                    hop_length=hop,
                    n_mels=n_mels,
                    power=1.0,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, T] predicted audio
            y: [B, 1, T] target audio
        Returns:
            Scalar average L1 mel loss across scales.
        """
        loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        # MelSpectrogram expects [B, T] or [B, 1, T]; squeeze channel dim
        x_in = x.squeeze(1)
        y_in = y.squeeze(1)
        for mel_spec in self.mel_specs:
            mel_spec = mel_spec.to(x.device)
            loss = loss + F.l1_loss(mel_spec(x_in), mel_spec(y_in))
        return loss / len(self.mel_specs)


def compute_reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mrstft_loss_fn: nn.Module,
    mel_loss_fn: nn.Module,
    kl_loss: torch.Tensor,
    l1_weight: float = 1.0,
    stft_weight: float = 1.0,
    mel_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Aggregate all reconstruction losses.

    Returns a dict with individual loss terms and the total.
    """
    l1_loss = F.l1_loss(x_hat, x)
    stft_loss = mrstft_loss_fn(x_hat, x)
    mel_loss = mel_loss_fn(x_hat, x)

    total = (
        l1_weight * l1_loss
        + stft_weight * stft_loss
        + mel_weight * mel_loss
        + kl_loss
    )
    return {
        "total": total,
        "l1": l1_loss,
        "stft": stft_loss,
        "mel": mel_loss,
        "kl": kl_loss,
    }
