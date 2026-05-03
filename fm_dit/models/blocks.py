"""Reusable DiT encoder and decoder blocks."""

from typing import Optional

import torch
import torch.nn as nn

from .attention import SelfAttentionRoPE
from .layers import GatedSkipWithTime, LightweightConvModule, SwiGLU


class EncoderBlock(nn.Module):
    """Encoder block: AdaLN -> self-attention -> conv -> FFN."""

    def __init__(
        self,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        cond_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.0,
        conv_kernel_size: int = 31,
    ):
        super().__init__()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),
        )
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout)
        self.norm_conv = nn.LayerNorm(d_model, elementwise_affine=False)
        self.conv_module = LightweightConvModule(d_model, conv_kernel_size, dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = SwiGLU(d_model, dim_feedforward, dropout)

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mod = self.adaLN_modulation(c)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        h = self.norm1(x) * (1 + gamma1) + beta1
        h = self.self_attn(h, key_padding_mask=key_padding_mask)
        x = x + alpha1 * h

        x = x + self.conv_module(self.norm_conv(x))

        h = self.norm2(x) * (1 + gamma2) + beta2
        h = self.ffn(h)
        return x + alpha2 * h


class DecoderBlock(nn.Module):
    """Decoder block: gated skip -> AdaLN -> self-attention -> conv -> FFN."""

    def __init__(
        self,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        cond_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.0,
        conv_kernel_size: int = 7,
    ):
        super().__init__()

        self.gated_skip = GatedSkipWithTime(d_model, cond_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),
        )
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout)
        self.norm_conv = nn.LayerNorm(d_model, elementwise_affine=False)
        self.conv_module = LightweightConvModule(d_model, conv_kernel_size, dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = SwiGLU(d_model, dim_feedforward, dropout)

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        c: torch.Tensor,
        c_global: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.gated_skip(x, skip, c_global)

        mod = self.adaLN_modulation(c)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        h = self.norm1(x) * (1 + gamma1) + beta1
        h = self.self_attn(h, key_padding_mask=key_padding_mask)
        x = x + alpha1 * h

        x = x + self.conv_module(self.norm_conv(x))

        h = self.norm2(x) * (1 + gamma2) + beta2
        h = self.ffn(h)
        return x + alpha2 * h

