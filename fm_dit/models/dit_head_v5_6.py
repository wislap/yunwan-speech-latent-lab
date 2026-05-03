"""DiT Head v5.6: 纯 DiT (无 Cross-Attention)

核心改动:
1. 去除所有 Cross-Attention
2. 帧级条件: x = input_proj(x_t) + 0.1 * cond_input_proj(cond)
3. Per-Token AdaLN: c = cond + t_emb_seq + cond_pooled
4. Conv 核递减: 编码侧 [15,15,9,5], bottleneck k=9, 解码侧 k=7
5. skip_0: concat+linear (无 gate), skip_1/2/3: gated

架构:
━━━━━━━━━━━━━━━ 编码侧 ━━━━━━━━━━━━━━━
层 0: AdaLN → SA(RoPE) → Conv(k=15) → FFN → skip_0
层 1: AdaLN → SA(RoPE) → Conv(k=15) → FFN → skip_1
层 2: AdaLN → SA(RoPE) → Conv(k=9) → FFN → skip_2
层 3: AdaLN → SA(RoPE) → Conv(k=5) → FFN → skip_3

━━━━━━━━━━━━━━━ Bottleneck ━━━━━━━━━━━
层 4: AdaLN → SA(RoPE) → Conv(k=9) → FFN

━━━━━━━━━━━━━━━ 解码侧 ━━━━━━━━━━━━━━━
层 5: GatedSkipWithTime(skip_3) → AdaLN → SA(RoPE) → Conv(k=7) → FFN
层 6: GatedSkipWithTime(skip_2) → AdaLN → SA(RoPE) → Conv(k=7) → FFN
层 7: GatedSkipWithTime(skip_1) → AdaLN → SA(RoPE) → Conv(k=7) → FFN

━━━━━━━━━━━━━━━ Output Head ━━━━━━━━━━━
ConcatSkip(skip_0) → FinalNorm → ProjOut
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.unsqueeze(-1).float() * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to Q and K"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU FFN"""
    
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
    """Lightweight Convolution Module (Conformer 风格)"""
    
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
    """时间步感知的 Skip Connection
    
    关键洞察：flow matching 不同时间步，skip 的重要性不同
    - t≈1（纯噪声）：skip 的早期对齐信息更重要
    - t≈0（近似干净）：skip 贡献减少，主干精细化为主
    """
    
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.skip_proj = nn.Linear(d_model, d_model)
        self.time_mod = nn.Linear(cond_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # 初始化: gate 偏向小值，skip 贡献初始较小
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)  # sigmoid(-2) ≈ 0.12
        nn.init.zeros_(self.time_mod.weight)
        nn.init.zeros_(self.time_mod.bias)
        
        # 用于监控的 gate 值
        self.last_gate_mean = None
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 当前特征 [B, T, D]
            skip: 编码侧特征 [B, T, D]
            c: 全局条件向量 (含 timestep) [B, D]
        """
        gate = torch.sigmoid(
            self.gate_proj(torch.cat([x, skip], dim=-1))
            + self.time_mod(c).unsqueeze(1)  # [B, 1, D] broadcast
        )
        # 记录 gate 均值用于监控
        self.last_gate_mean = gate.mean().item()
        x = self.norm(x + gate * self.skip_proj(skip))
        return x


class ConcatSkip(nn.Module):
    """Concat + Linear Skip Connection (无 gate)
    
    用于 skip_0，避免 gate 垄断
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 当前特征 [B, T, D]
            skip: 编码侧特征 [B, T, D]
        """
        x = self.norm(self.proj(torch.cat([x, skip], dim=-1)))
        return x


class SelfAttentionRoPE(nn.Module):
    """Self-Attention with RoPE"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        cos, sin = self.rope(x, S)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
        return self.out_proj(out)


class EncoderBlock(nn.Module):
    """编码侧 Block: AdaLN → SA(RoPE) → Conv → FFN (无 CA)"""
    
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
        
        # AdaLN-Per-Token modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),
        )
        
        # Self-Attention
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout)
        
        # Conv
        self.norm_conv = nn.LayerNorm(d_model, elementwise_affine=False)
        self.conv_module = LightweightConvModule(d_model, conv_kernel_size, dropout)
        
        # FFN
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = SwiGLU(d_model, dim_feedforward, dropout)
        
        # Zero-init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(c)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)
        
        # Self-Attention
        h = self.norm1(x) * (1 + gamma1) + beta1
        h = self.self_attn(h)
        x = x + alpha1 * h
        
        # Conv
        h_conv = self.conv_module(self.norm_conv(x))
        x = x + h_conv
        
        # FFN
        h = self.norm2(x) * (1 + gamma2) + beta2
        h = self.ffn(h)
        x = x + alpha2 * h
        
        return x


class DecoderBlock(nn.Module):
    """解码侧 Block: GatedSkip → AdaLN → SA(RoPE) → Conv → FFN (无 CA)"""
    
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
        
        # GatedSkipWithTime
        self.gated_skip = GatedSkipWithTime(d_model, cond_dim)
        
        # AdaLN-Per-Token modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),
        )
        
        # Self-Attention
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout)
        
        # Conv
        self.norm_conv = nn.LayerNorm(d_model, elementwise_affine=False)
        self.conv_module = LightweightConvModule(d_model, conv_kernel_size, dropout)
        
        # FFN
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = SwiGLU(d_model, dim_feedforward, dropout)
        
        # Zero-init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        c: torch.Tensor,
        c_global: torch.Tensor,
    ) -> torch.Tensor:
        # GatedSkipWithTime
        x = self.gated_skip(x, skip, c_global)
        
        mod = self.adaLN_modulation(c)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)
        
        # Self-Attention
        h = self.norm1(x) * (1 + gamma1) + beta1
        h = self.self_attn(h)
        x = x + alpha1 * h
        
        # Conv
        h_conv = self.conv_module(self.norm_conv(x))
        x = x + h_conv
        
        # FFN
        h = self.norm2(x) * (1 + gamma2) + beta2
        h = self.ffn(h)
        x = x + alpha2 * h
        
        return x


class BottleneckBlock(nn.Module):
    """Bottleneck Block: AdaLN → SA(RoPE) → Conv → FFN (无 CA)"""
    
    def __init__(
        self,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        cond_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.0,
        conv_kernel_size: int = 15,
    ):
        super().__init__()
        
        # AdaLN-Per-Token modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),
        )
        
        # Self-Attention
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout)
        
        # Conv
        self.norm_conv = nn.LayerNorm(d_model, elementwise_affine=False)
        self.conv_module = LightweightConvModule(d_model, conv_kernel_size, dropout)
        
        # FFN
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = SwiGLU(d_model, dim_feedforward, dropout)
        
        # Zero-init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(c)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)
        
        # Self-Attention
        h = self.norm1(x) * (1 + gamma1) + beta1
        h = self.self_attn(h)
        x = x + alpha1 * h
        
        # Conv
        h_conv = self.conv_module(self.norm_conv(x))
        x = x + h_conv
        
        # FFN
        h = self.norm2(x) * (1 + gamma2) + beta2
        h = self.ffn(h)
        x = x + alpha2 * h
        
        return x


class DiTHeadV5_6(nn.Module):
    """DiT Flow Head v5.6: 纯 DiT (无 Cross-Attention)
    
    核心改动:
    1. 去除所有 Cross-Attention
    2. 帧级条件直接加到输入
    3. 每层都有 Conv
    4. 保留 U-Net Skip + GatedSkipWithTime
    """
    
    def __init__(
        self,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        cond_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.cond_dim = cond_dim
        
        # Timestep embedding
        self.t_emb = nn.Sequential(
            nn.Linear(d_model, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # 全局池化
        self.cond_pool_proj = nn.Linear(cond_dim, cond_dim)
        
        # 输入投影: x_t + cond
        self.input_proj = nn.Linear(d_model, d_model)
        self.cond_input_proj = nn.Linear(cond_dim, d_model)  # 帧级条件投影
        
        # 编码侧 (层 0-3): Conv 核递减 [15, 15, 9, 5]
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout, conv_kernel_size=15),
            EncoderBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout, conv_kernel_size=15),
            EncoderBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout, conv_kernel_size=9),
            EncoderBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout, conv_kernel_size=5),
        ])
        
        # Bottleneck (层 4): k=9
        self.bottleneck = BottleneckBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout,
                                          conv_kernel_size=9)
        
        # 解码侧 (层 5-7)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout, conv_kernel_size=7),
            DecoderBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout, conv_kernel_size=7),
            DecoderBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout, conv_kernel_size=7),
        ])
        
        # Output Head: skip_0 用 concat+linear (无 gate)
        self.final_concat_skip = ConcatSkip(d_model)
        
        # Output projections
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_proj = nn.Linear(d_model, d_model)
        
        # Zero-init final projection
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        
        # 输入融合: x_t + 0.1 * cond (系数固定 0.1，避免 cond 过于强势)
        x = self.input_proj(x) + 0.1 * self.cond_input_proj(cond)
        
        # Timestep embedding
        t_emb = self.t_emb(timestep_embedding(t, self.d_model))
        t_emb_seq = t_emb.unsqueeze(1).expand(-1, S, -1)  # [B, S, D]
        
        # 全局池化
        cond_pooled = cond.mean(dim=1, keepdim=True).expand(-1, S, -1)
        cond_pooled = self.cond_pool_proj(cond_pooled)
        
        # Per-Token AdaLN 条件: cond + t_emb + cond_pooled
        c = cond + t_emb_seq + cond_pooled
        
        # 全局条件向量 (用于 GatedSkipWithTime)
        c_global = t_emb + cond.mean(dim=1)  # [B, D]
        
        # 编码侧: 存储 skip
        skips = []
        for block in self.encoder_blocks:
            x = block(x, c)
            skips.append(x)  # skip_0, skip_1, skip_2, skip_3
        
        # Bottleneck
        x = self.bottleneck(x, c)
        
        # 解码侧: 融合 skip (逆序)
        # 层 5: skip_3, 层 6: skip_2, 层 7: skip_1
        for i, block in enumerate(self.decoder_blocks):
            skip_idx = 3 - i  # 3, 2, 1
            x = block(x, skips[skip_idx], c, c_global)
        
        # Output Head: skip_0 用 concat+linear (无 gate)
        x = self.final_concat_skip(x, skips[0])
        
        # Final projection
        x = self.final_norm(x)
        x = self.final_proj(x)
        
        return x
    
    def get_gate_stats(self) -> dict:
        """获取所有 gate 的统计信息，用于监控"""
        stats = {}
        for i, block in enumerate(self.decoder_blocks):
            skip_idx = 3 - i
            gate_val = block.gated_skip.last_gate_mean
            if gate_val is not None:
                stats[f'skip_{skip_idx}'] = gate_val
        # skip_0 无 gate (concat+linear)
        stats['skip_0'] = 'concat'
        return stats


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = DiTHeadV5_6(
        d_model=512,
        dim_feedforward=2048,
        cond_dim=512,
        n_heads=8,
    ).to(device)
    
    # 打印层配置
    print("DiT Head v5.6 层配置 (纯 DiT, 无 CA):")
    print("  编码侧:")
    for i, block in enumerate(model.encoder_blocks):
        print(f"    层 {i}: AdaLN → SA(RoPE) → Conv{block.conv_module.kernel_size} → FFN → skip_{i}")
    print("  Bottleneck:")
    print(f"    层 4: AdaLN → SA(RoPE) → Conv{model.bottleneck.conv_module.kernel_size} → FFN")
    print("  解码侧:")
    for i, block in enumerate(model.decoder_blocks):
        skip_idx = 3 - i
        print(f"    层 {5+i}: GatedSkip(skip_{skip_idx}) → AdaLN → SA(RoPE) → Conv{block.conv_module.kernel_size} → FFN")
    print("  Output Head:")
    print("    GatedSkip(skip_0) → FinalNorm → ProjOut")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nDiT Head: {n_params/1e6:.2f}M params")
    
    # Forward test
    B, S, D = 2, 64, 512
    x = torch.randn(B, S, D, device=device)
    t = torch.rand(B, device=device)
    cond = torch.randn(B, S, 512, device=device)
    
    with torch.no_grad():
        out = model(x, t, cond)
    print(f"Input: {x.shape}, Output: {out.shape}")
    
    # Gate stats
    stats = model.get_gate_stats()
    print(f"Gate stats: {stats}")
