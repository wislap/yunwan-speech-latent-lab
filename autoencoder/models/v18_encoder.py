"""V18: Generation-Aware Autoencoder.

Splits the encoder into a supervised phoneme stream and a free residual stream.
Concatenates to a 384-d latent that matches the V16.3 decoder interface.

Structure:
    audio -> V18PhonemeEncoder  -> z_phoneme  [B, 128, T_frames]  (CTC-supervised)
    audio -> V18ResidualEncoder -> z_residual [B, 256, T_frames]  (unsupervised)
    z = concat([z_phoneme, z_residual], dim=1)  -> [B, 384, T_frames]
    z -> V16.3 decoder -> audio_hat

Frame rate matches V16.3: AE total stride = 256 at sr=22050 -> ~86 Hz frames.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder.models.autoencoder import (
    EncoderBlock,
    VAEBottleneck,
    WNConv1d,
    WavVAEDecoder,
)


# ─── Phoneme Encoder ─────────────────────────────────────────


class V18PhonemeEncoder(nn.Module):
    """LogMel + Conformer phoneme encoder at AE frame rate.

    Unlike the V17.3 Conformer teacher which subsamples mel features by 4x
    (matching stride 1024), V18 keeps the mel hop at 256 and does NOT
    subsample. That yields frame-level embeddings at the AE's native
    stride-256 rate, so z_phoneme is exactly frame-aligned with z_residual.

    Input:  audio [B, 1, T_samples] or [B, T_samples]
    Output: z_phoneme [B, out_dim, T_frames] with T_frames = T_samples // 256
    """

    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        conv_kernel: int = 31,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hop_length = hop_length
        self.out_dim = out_dim

        import torchaudio
        from torchaudio.models import Conformer

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            n_mels=n_mels,
            f_min=40.0,
            f_max=min(sr / 2.0, 11000.0),
            power=2.0,
            normalized=False,
        )
        # Project mel (n_mels) -> hidden_dim at the same frame rate.
        # No stride -> no subsampling; keeps native stride-256 alignment.
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.conformer = Conformer(
            input_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel,
            dropout=dropout,
            use_group_norm=False,
            convolution_first=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute frame-aligned phoneme embeddings.

        Args:
            audio: [B, 1, T_samples] or [B, T_samples]
        Returns:
            z_phoneme: [B, out_dim, T_frames]
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        mel = self.mel(audio.float()).clamp_min(1e-6).log()  # [B, n_mels, T_mel]
        h = self.input_proj(mel)  # [B, hidden_dim, T_mel]
        # Conformer wants [B, T, C] + lengths
        h_t = h.transpose(1, 2)  # [B, T, hidden_dim]
        lengths = torch.full(
            (h_t.shape[0],),
            h_t.shape[1],
            device=h_t.device,
            dtype=torch.long,
        )
        h_t, _ = self.conformer(h_t, lengths)  # [B, T, hidden_dim]
        z = self.head(h_t)  # [B, T, out_dim]
        return z.transpose(1, 2).contiguous()  # [B, out_dim, T]


class PhonemeCTCHead(nn.Module):
    """Shallow CTC head reading from z_phoneme frames.

    Input:  [B, in_dim, T]
    Output: log-probs [B, T, vocab_size] for CTC loss
    """

    def __init__(self, in_dim: int = 128, vocab_size: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, vocab_size, kernel_size=1),
        )

    def forward(self, z_phoneme: torch.Tensor) -> torch.Tensor:
        logits = self.net(z_phoneme)  # [B, vocab, T]
        return F.log_softmax(logits.transpose(1, 2), dim=-1)  # [B, T, vocab]


# ─── Residual Encoder ────────────────────────────────────────


class V18ResidualEncoder(nn.Module):
    """Narrower V16.3-style time-domain encoder for the residual stream.

    Architecture matches WavVAEEncoder but with reduced channels.
    Stride chain: stem -> [2, 4, 4, 8] -> total stride 256.

    Input:  [B, 1, T_samples]
    Output: [B, out_dim, T_frames] with T_frames = T_samples // 256
    """

    def __init__(
        self,
        out_dim: int = 256,
        channels: Sequence[int] = (96, 192, 384, 768, 768),
        strides: Sequence[int] = (2, 4, 4, 8),
        dilations: Sequence[int] = (1, 3, 9),
        use_checkpoint: bool = False,
    ):
        super().__init__()
        assert len(channels) == len(strides) + 1, (
            f"channels len {len(channels)} must be strides len {len(strides)} + 1"
        )
        self.use_checkpoint = use_checkpoint
        self.stem = WNConv1d(1, channels[0], kernel_size=7, padding=3)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(channels[i], channels[i + 1], strides[i], dilations)
                for i in range(len(strides))
            ]
        )
        # Single-sided projection (not *2 for VAE mean/std) because the residual
        # stream feeds the composite bottleneck from a concatenated latent, not
        # a per-stream VAE.
        self.proj = WNConv1d(channels[-1], out_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        return self.proj(h)


# ─── Composite V18 Autoencoder ───────────────────────────────


class V18Autoencoder(nn.Module):
    """Composite AE:
        phoneme_encoder -> z_phoneme [B, phoneme_dim, T]
        residual_encoder -> z_residual [B, residual_dim, T]
        concat -> z [B, phoneme_dim + residual_dim, T]  (default 384)
        bottleneck range penalty on z
        decoder -> audio_hat

    The phoneme sub-latent always occupies the first `phoneme_dim` channels
    of z so downstream consumers can split it deterministically.
    """

    def __init__(
        self,
        sr: int = 22050,
        phoneme_dim: int = 128,
        residual_dim: int = 256,
        decoder_channels: Sequence[int] | None = None,
        strides: Sequence[int] = (2, 4, 4, 8),
        dilations: Sequence[int] = (1, 3, 9),
        residual_channels: Sequence[int] = (96, 192, 384, 768, 768),
        phoneme_hidden_dim: int = 256,
        phoneme_layers: int = 4,
        phoneme_heads: int = 4,
        phoneme_ffn_dim: int = 1024,
        phoneme_conv_kernel: int = 31,
        phoneme_dropout: float = 0.0,
        phoneme_n_mels: int = 80,
        phoneme_n_fft: int = 1024,
        phoneme_hop_length: int = 256,
        phoneme_ctc_vocab_size: int = 512,
        reg_weight: float = 1.0e-4,
        margin_mean: float = 6.0,
        std_min: float = 0.05,
        std_max: float = 3.0,
        reg_warmup_steps: int = 5000,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.phoneme_dim = phoneme_dim
        self.residual_dim = residual_dim
        self.latent_dim = phoneme_dim + residual_dim  # composite z dim

        if decoder_channels is None:
            # Mirror V16.3: encoder channels reversed for decoder.
            # Here the decoder expects latent_dim input channels, then
            # decoder channels are the V16.3 defaults reversed.
            decoder_channels = (1024, 1024, 512, 256, 128)
        rev_strides = list(reversed(list(strides)))

        self.phoneme_encoder = V18PhonemeEncoder(
            sr=sr,
            n_fft=phoneme_n_fft,
            hop_length=phoneme_hop_length,
            n_mels=phoneme_n_mels,
            hidden_dim=phoneme_hidden_dim,
            out_dim=phoneme_dim,
            num_layers=phoneme_layers,
            num_heads=phoneme_heads,
            ffn_dim=phoneme_ffn_dim,
            conv_kernel=phoneme_conv_kernel,
            dropout=phoneme_dropout,
        )
        self.residual_encoder = V18ResidualEncoder(
            out_dim=residual_dim,
            channels=residual_channels,
            strides=strides,
            dilations=dilations,
            use_checkpoint=use_checkpoint,
        )
        self.decoder = WavVAEDecoder(
            latent_dim=self.latent_dim,
            channels=decoder_channels,
            strides=rev_strides,
            dilations=dilations,
            use_checkpoint=use_checkpoint,
        )
        # Range-penalty bottleneck on the concatenated latent. V18 does not
        # use VAE reparameterization (sample_latent=False) because the phoneme
        # stream is deterministic and speaker/detail content should be
        # preserved, matching V16.3's choice.
        self.bottleneck = VAEBottleneck(
            reg_weight=reg_weight,
            margin_mean=margin_mean,
            std_min=std_min,
            std_max=std_max,
            warmup_steps=reg_warmup_steps,
            sample_latent=False,
        )
        self.ctc_head = PhonemeCTCHead(
            in_dim=phoneme_dim,
            vocab_size=phoneme_ctc_vocab_size,
        )
        # For downstream speaker/content decomposition callers.
        self._strides = list(strides)
        self.total_stride = 1
        for s in strides:
            self.total_stride *= s

        # Flag used by autoencoder.train.set_step to propagate step counts
        # to the bottleneck.
        self._step = 0

    def set_step(self, step: int) -> None:
        self._step = step
        self.bottleneck.set_step(step)

    # ─── Fluid API matching WavVAE for train.py compatibility ────────

    def encode(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode audio and return split latents plus bottleneck outputs.

        For V18 we use a deterministic latent: the bottleneck range penalty
        is computed on z as mean with a tiny dummy std, but no sampling.
        """
        z_phoneme = self.phoneme_encoder(audio)  # [B, Dp, T]
        z_residual = self.residual_encoder(audio)  # [B, Dr, T]
        # Align frame counts defensively — they SHOULD match.
        T_common = min(z_phoneme.shape[-1], z_residual.shape[-1])
        z_phoneme = z_phoneme[..., :T_common]
        z_residual = z_residual[..., :T_common]
        z = torch.cat([z_phoneme, z_residual], dim=1)  # [B, D, T]

        # Bottleneck range penalty on z.
        # VAEBottleneck.encode expects [B, 2*D, T] to split mean/std.
        # We fake this with a constant dummy log-std so the range penalty
        # acts on mean = z only.
        dummy_scale = torch.zeros_like(z)
        enc_out = torch.cat([z, dummy_scale], dim=1)
        z_sampled, mean, std = self.bottleneck.encode(enc_out, training=self.training)
        reg_loss = self.bottleneck.reg_loss(mean, std)

        return {
            "z": z_sampled,  # equal to mean because sample_latent=False
            "z_phoneme": z_phoneme,
            "z_residual": z_residual,
            "mean": mean,
            "std": std,
            "reg_loss": reg_loss,
        }

    def decode(self, z: torch.Tensor, output_length: int | None = None) -> torch.Tensor:
        return self.decoder(z, output_length=output_length)

    def ctc_log_probs(self, z_phoneme: torch.Tensor) -> torch.Tensor:
        """Return [B, T, vocab] log-probs for CTC loss."""
        return self.ctc_head(z_phoneme)

    def forward(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        enc = self.encode(audio)
        x_hat = self.decode(enc["z"], output_length=audio.shape[-1])
        ctc_lp = self.ctc_log_probs(enc["z_phoneme"])
        return {
            "x_hat": x_hat,
            "z": enc["z"],
            "z_phoneme": enc["z_phoneme"],
            "z_residual": enc["z_residual"],
            "mean": enc["mean"],
            "std": enc["std"],
            "reg_loss": enc["reg_loss"],
            "ctc_log_probs": ctc_lp,
        }

    # ─── Pretrained weight loading ──────────────────────────────────

    def load_phoneme_pretrained(self, checkpoint_path: str, strict: bool = False) -> dict:
        """Load V17.3 Conformer teacher weights into the phoneme encoder.

        V17.3 teacher subsamples by 4x with strided convs; V18's phoneme
        encoder has a non-strided mel-to-hidden projection, so direct key
        transfer is limited to the Conformer blocks, LayerNorm, and mel
        buffers. We remap what we can and leave the rest at random init.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        remapped: dict = {}
        own_state = self.phoneme_encoder.state_dict()
        for k, v in state.items():
            # V17.3 teacher keys vs V18 phoneme encoder keys
            # Mel buffer keys share prefix 'mel.'
            if k.startswith("mel."):
                target = k  # e.g. 'mel.mel_scale.fb'
            elif k.startswith("conformer."):
                target = k  # Conformer layers
            else:
                continue
            if target in own_state and own_state[target].shape == v.shape:
                remapped[target] = v
        missing, unexpected = self.phoneme_encoder.load_state_dict(
            remapped, strict=False
        )
        return {
            "attempted": len(state),
            "remapped": len(remapped),
            "missing": len(missing),
            "unexpected": len(unexpected),
        }

    def load_decoder_pretrained(self, checkpoint_path: str) -> dict:
        """Load V16.3 decoder weights into V18's decoder.

        V18 decoder shares architecture with V16.3; only the latent_dim at
        the entry projection must match. We ensure phoneme_dim + residual_dim
        == V16.3 latent_dim (default 384) in config.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        own_state = self.decoder.state_dict()
        loadable: dict = {}
        for k, v in state.items():
            if k.startswith("decoder."):
                target_key = k[len("decoder."):]  # strip 'decoder.' prefix
                if target_key in own_state and own_state[target_key].shape == v.shape:
                    loadable[target_key] = v
        missing, unexpected = self.decoder.load_state_dict(loadable, strict=False)
        return {
            "attempted": len([k for k in state if k.startswith("decoder.")]),
            "loaded": len(loadable),
            "missing": len(missing),
            "unexpected": len(unexpected),
        }
