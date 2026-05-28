"""Masked Latent Predictor for generation-friendly latent training.

A Transformer that predicts masked latent frames given:
- Visible latent frames (from the encoder)
- Phoneme sequence expanded to frame level via durations

Gradient flows from predictor loss back to the encoder, creating
implicit information bottleneck pressure.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, D]

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:, :seq_len]


class MaskedLatentPredictor(nn.Module):
    """Transformer-based masked latent predictor.

    Takes visible latent frames + phoneme context, predicts masked frames.

    Args:
        latent_dim: Dimension of latent space (384 for V16.3).
        hidden_dim: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer layers.
        phoneme_vocab_size: Size of phoneme vocabulary.
        max_seq_len: Maximum sequence length.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        phoneme_vocab_size: int = 512,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Phoneme embedding
        self.phoneme_embed = nn.Embedding(phoneme_vocab_size, hidden_dim)

        # Project latent to hidden dim
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # Learnable mask token (for masked positions)
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(hidden_dim, max_seq_len)

        # Transformer encoder (bidirectional)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Output projection back to latent dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def expand_phonemes_to_frames(
        self,
        phoneme_ids: torch.Tensor,
        phoneme_durations: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """Expand phoneme embeddings to frame-level using durations.

        Args:
            phoneme_ids: [B, T_ph] phoneme indices.
            phoneme_durations: [B, T_ph] frames per phoneme.
            target_len: Target frame count.

        Returns:
            [B, target_len, hidden_dim] frame-level phoneme features.
        """
        B = phoneme_ids.shape[0]
        ph_emb = self.phoneme_embed(phoneme_ids)  # [B, T_ph, D]

        frame_features = []
        for b in range(B):
            # Repeat each phoneme embedding by its duration
            expanded = torch.repeat_interleave(
                ph_emb[b], phoneme_durations[b], dim=0
            )  # [sum(dur), D]
            # Truncate or pad to target_len
            if expanded.shape[0] >= target_len:
                expanded = expanded[:target_len]
            else:
                pad_len = target_len - expanded.shape[0]
                expanded = F.pad(expanded, (0, 0, 0, pad_len))
            frame_features.append(expanded)

        return torch.stack(frame_features)  # [B, target_len, D]

    def forward(
        self,
        z: torch.Tensor,
        visible_mask: torch.Tensor,
        phoneme_ids: torch.Tensor,
        phoneme_durations: torch.Tensor,
    ) -> torch.Tensor:
        """Predict masked latent frames.

        Args:
            z: Full latent sequence [B, T, latent_dim] (gradient flows through).
            visible_mask: [B, T] bool, True = visible, False = masked.
            phoneme_ids: [B, T_ph] phoneme indices.
            phoneme_durations: [B, T_ph] frames per phoneme.

        Returns:
            z_predicted: [B, T, latent_dim] predictions for ALL positions.
                         Loss should only be computed at masked positions.
        """
        B, T, _ = z.shape

        # Project latent to hidden dim
        z_proj = self.latent_proj(z)  # [B, T, hidden_dim]

        # Replace masked positions with mask token
        mask_expanded = visible_mask.unsqueeze(-1)  # [B, T, 1]
        z_input = torch.where(mask_expanded, z_proj, self.mask_token.expand(B, T, -1))

        # Add phoneme context
        ph_frames = self.expand_phonemes_to_frames(
            phoneme_ids, phoneme_durations, T
        )  # [B, T, hidden_dim]
        z_input = z_input + ph_frames

        # Add positional encoding
        z_input = z_input + self.pos_enc(T)

        # Transformer forward
        hidden = self.transformer(z_input)  # [B, T, hidden_dim]

        # Project to latent dim
        z_predicted = self.output_proj(hidden)  # [B, T, latent_dim]

        return z_predicted
