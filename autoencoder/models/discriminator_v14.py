"""V14 MS-STFT Discriminator wrapper.

Wraps the existing MultiScaleSTFTDiscriminator with V14-specific config
(smaller FFT sizes for finer temporal resolution).
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from autoencoder.models.discriminator import (
    MultiScaleSTFTDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
)


class V14Discriminator(nn.Module):
    """V14 MS-STFT discriminator with 3 resolutions (256, 512, 1024)."""

    def __init__(
        self,
        n_ffts: tuple[int, ...] = (256, 512, 1024),
        hop_lengths: tuple[int, ...] = (64, 128, 256),
        win_lengths: tuple[int, ...] = (256, 512, 1024),
        filters: int = 32,
    ):
        super().__init__()
        self.disc = MultiScaleSTFTDiscriminator(
            filters=filters,
            n_ffts=list(n_ffts),
            hop_lengths=list(hop_lengths),
            win_lengths=list(win_lengths),
        )

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: [B, 1, T] waveform
        Returns:
            (logits_list, features_list) from multi-scale STFT discriminator
        """
        return self.disc(x)
