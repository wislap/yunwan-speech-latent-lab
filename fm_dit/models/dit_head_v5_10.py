"""DiT Head v5.10: 512 维双分支 + 两次 FFN(half)

核心改动:
1. 保持全维度 512，不分割
2. Bottleneck 用两个 FFN 分别处理:
   - FFN_hf: 小核 Conv (k=3) + FFN
   - FFN_lf: 大核 Conv (k=15) + FFN
3. 两个分支输出相加 (带可学习 gate)

架构:
━━━━━━━━━━━━━━━ 编码侧 ━━━━━━━━━━━━━━━
层 0-3: 同 v5.7

━━━━━━━━━━━━━━━ 双 FFN Bottleneck ━━━━━━━━━━━━━━━
x → SA → 
    ├── Conv(k=3) → FFN_hf (dim_ff=1024)
    └── Conv(k=15) → FFN_lf (dim_ff=1024)
    → gate * hf + (1-gate) * lf

━━━━━━━━━━━━━━━ 解码侧 ━━━━━━━━━━━━━━━
层 5-7: 同 v5.7
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .dit_head_v5_6 import (
    timestep_embedding,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    SwiGLU,
    LightweightConvModule,
    SelfAttentionRoPE,
    EncoderBlock,
    DecoderBlock,
    GatedSkipWithTime,
    ConcatSkip,
)


class DualFFNBottleneck(nn.Module):
    """双 FFN Bottleneck: 高频/低频分别用不同 Conv + FFN"""
    
    def __init__(
        self,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        cond_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.0,
        hf_conv_kernel: int = 3,
        lf_conv_kernel: int = 15,
    ):
        super().__init__()
        
        # AdaLN-Per-Token modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),
        )
        
        # Self-Attention (共享)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout)
        
        # 高频分支: 小核 Conv + FFN
        self.norm_conv_hf = nn.LayerNorm(d_model, elementwise_affine=False)
        self.conv_hf = LightweightConvModule(d_model, hf_conv_kernel, dropout)
        self.norm_ffn_hf = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn_hf = SwiGLU(d_model, dim_feedforward // 2, dropout)
        
        # 低频分支: 大核 Conv + FFN
        self.norm_conv_lf = nn.LayerNorm(d_model, elementwise_affine=False)
        self.conv_lf = LightweightConvModule(d_model, lf_conv_kernel, dropout)
        self.norm_ffn_lf = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn_lf = SwiGLU(d_model, dim_feedforward // 2, dropout)
        
        # 可学习 gate: 控制高频/低频混合比例
        self.gate = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5
        
        # Zero-init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(c)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)
        
        # Self-Attention (共享)
        h = self.norm1(x) * (1 + gamma1) + beta1
        h = self.self_attn(h)
        x = x + alpha1 * h
        
        # 高频分支
        h_hf = self.conv_hf(self.norm_conv_hf(x))
        h_hf = x + h_hf
        h_hf = self.norm_ffn_hf(h_hf) * (1 + gamma2) + beta2
        h_hf = self.ffn_hf(h_hf)
        
        # 低频分支
        h_lf = self.conv_lf(self.norm_conv_lf(x))
        h_lf = x + h_lf
        h_lf = self.norm_ffn_lf(h_lf) * (1 + gamma2) + beta2
        h_lf = self.ffn_lf(h_lf)
        
        # 混合: gate 控制高频/低频比例
        gate = torch.sigmoid(self.gate)
        h_merged = gate * h_hf + (1 - gate) * h_lf
        
        x = x + alpha2 * h_merged
        
        return x


class DiTHeadV5_10(nn.Module):
    """DiT Flow Head v5.10: 512 维双分支 + 两次 FFN(half)
    
    核心改动:
    1. 保持全维度 512
    2. Bottleneck 用两个 FFN 分别处理高频/低频
    3. 可学习 gate 控制混合比例
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
        
        # 输入投影
        self.input_proj = nn.Linear(d_model, d_model)
        self.cond_input_proj = nn.Linear(cond_dim, d_model)
        
        # 编码侧 (层 0-3): Conv 核递减 [15, 15, 9, 5]
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout, conv_kernel_size=15),
            EncoderBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout, conv_kernel_size=15),
            EncoderBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout, conv_kernel_size=9),
            EncoderBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout, conv_kernel_size=5),
        ])
        
        # 双 FFN Bottleneck
        self.bottleneck = DualFFNBottleneck(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            cond_dim=cond_dim,
            n_heads=n_heads,
            dropout=dropout,
            hf_conv_kernel=3,
            lf_conv_kernel=15,
        )
        
        # 解码侧 (层 5-7)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout, conv_kernel_size=7),
            DecoderBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout, conv_kernel_size=7),
            DecoderBlock(d_model, dim_feedforward, cond_dim, n_heads, dropout, conv_kernel_size=7),
        ])
        
        # Output Head
        self.final_concat_skip = ConcatSkip(d_model)
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_proj = nn.Linear(d_model, d_model)
        
        # Zero-init
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
    
    def get_gate_stats(self) -> dict:
        """获取 gate 统计"""
        stats = {}
        # Bottleneck gate
        stats["bn_gate"] = torch.sigmoid(self.bottleneck.gate).item()
        # Decoder gates
        for i, block in enumerate(self.decoder_blocks):
            gate_val = block.gated_skip.last_gate_mean
            if gate_val is not None:
                stats[f"skip_{3-i}"] = gate_val
            else:
                stats[f"skip_{3-i}"] = 0.0
        stats["skip_0"] = "concat"
        return stats
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        
        # 输入融合
        x = self.input_proj(x) + 0.1 * self.cond_input_proj(cond)
        
        # Timestep embedding
        t_emb = self.t_emb(timestep_embedding(t, self.d_model))
        t_emb_seq = t_emb.unsqueeze(1).expand(-1, S, -1)
        
        # 全局池化
        cond_pooled = cond.mean(dim=1, keepdim=True).expand(-1, S, -1)
        cond_pooled = self.cond_pool_proj(cond_pooled)
        
        # Per-Token AdaLN 条件
        c = cond + t_emb_seq + cond_pooled
        c_global = t_emb + cond.mean(dim=1)
        
        # 编码侧: 存储 skip
        skips = []
        for block in self.encoder_blocks:
            x = block(x, c)
            skips.append(x)
        
        # 双 FFN Bottleneck
        x = self.bottleneck(x, c)
        
        # 解码侧: 使用 skip (逆序)
        for i, block in enumerate(self.decoder_blocks):
            skip = skips[-(i+1)]
            x = block(x, skip, c, c_global)
        
        # Output
        x = self.final_concat_skip(x, skips[0])
        x = self.final_norm(x)
        x = self.final_proj(x)
        
        return x
