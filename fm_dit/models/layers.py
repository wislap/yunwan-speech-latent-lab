"""Reusable feed-forward, convolution and skip layers for FM-DiT."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward layer."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim_feedforward * 2 / 3)
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        value = self.w2(x)
        return self.dropout(self.w3(gate * value))


class LightweightConvModule(nn.Module):
    """Conformer-style lightweight depthwise convolution module."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.pointwise1 = nn.Linear(d_model, d_model * 2)
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pointwise2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_glu = self.pointwise1(x)
        x_glu, gate = x_glu.chunk(2, dim=-1)
        x_glu = x_glu * torch.sigmoid(gate)
        x_conv = x_glu.transpose(1, 2)
        x_conv = self.depthwise(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.norm(x_conv)
        x_conv = self.pointwise2(x_conv)
        return self.dropout(x_conv)


class GatedSkipWithTime(nn.Module):
    """Time-conditioned gated skip connection."""

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.skip_proj = nn.Linear(d_model, d_model)
        self.time_mod = nn.Linear(cond_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)
        nn.init.zeros_(self.time_mod.weight)
        nn.init.zeros_(self.time_mod.bias)

        self.last_gate_mean = None

    def forward(self, x: torch.Tensor, skip: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(
            self.gate_proj(torch.cat([x, skip], dim=-1))
            + self.time_mod(c).unsqueeze(1)
        )
        self.last_gate_mean = gate.mean().item()
        return self.norm(x + gate * self.skip_proj(skip))


class ConcatSkip(nn.Module):
    """Concat + linear skip connection."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(torch.cat([x, skip], dim=-1)))

