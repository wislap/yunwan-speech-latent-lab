"""FM-DiT v1 training entry for V16 autoencoder latents.

This is the clean playground3 baseline for representation learning on top of
the V16 time-domain autoencoder latent space. It intentionally starts from the
stable v6.4 DiT recipe, but removes the Mimi/RVQ defaults and exposes the
frame-window controls needed for the higher-rate V16.3 latent stream.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from fm_dit.data.dataset_v3 import get_dataloader_v3
from fm_dit.models.dit_head_v5_10 import DiTHeadV5_10


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding for frame-level condition tokens."""

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds PE max_len {self.pe.size(1)}")
        return x + self.pe[:, : x.size(1)]


def sample_logit_normal(batch_size: int, loc: float = 0.0, scale: float = 1.0, device=None) -> torch.Tensor:
    """采样 Logit-Normal 分布
    
    优点: 更多关注中间困难区域 (t≈0.3-0.7)
    这是高频细节从噪声中"浮现"的关键阶段
    
    分布特性:
    - t<0.1: ~1.3% (vs Beta(0.5,0.5) 的 20.9%)
    - t>0.9: ~1.4%
    - 0.3<t<0.7: ~60.9% (vs Beta(0.5,0.5) 的 26.1%)
    """
    z = torch.randn(batch_size, device=device) * scale + loc
    return torch.sigmoid(z)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lengths_to_valid_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def sample_flow_t(
    batch_size: int,
    mode: str,
    beta_alpha: float,
    beta_beta: float,
    logit_loc: float,
    logit_scale: float,
    eps: float,
    device: torch.device,
) -> torch.Tensor:
    if mode == "uniform":
        t = torch.rand(batch_size, 1, 1, device=device)
    elif mode == "logit_normal":
        t = sample_logit_normal(batch_size, loc=logit_loc, scale=logit_scale, device=device).view(batch_size, 1, 1)
    elif mode == "beta":
        beta_dist = torch.distributions.Beta(beta_alpha, beta_beta)
        t = beta_dist.sample((batch_size, 1, 1)).to(device)
    else:
        raise ValueError(f"unknown t sampling mode: {mode}")
    return t.clamp(eps, 1.0 - eps)


class DropPath(nn.Module):
    """DropPath: 随机跳过整个 block (比 Dropout 更适合 Transformer)"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PreLNTransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer Layer: x = x + DropPath(Attn(LN(x)))"""
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 512,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.drop_path1 = DropPath(drop_path)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop_path2 = DropPath(drop_path)
    
    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-Attention with Pre-LN
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=src_key_padding_mask)
        x = x + self.drop_path1(attn_out)
        
        # FFN with Pre-LN
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x


class PreLNTransformerEncoder(nn.Module):
    """Pre-LN Transformer Encoder: 多层 PreLNTransformerEncoderLayer"""
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 512,
        num_layers: int = 2,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            PreLNTransformerEncoderLayer(d_model, nhead, dim_feedforward, drop_path)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return self.final_norm(x)


class CNNDurationPredictor(nn.Module):
    """CNN Duration Predictor (辅助任务)"""
    
    def __init__(self, d_model: int = 384, hidden_dim: int = 192, n_layers: int = 2):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(n_layers):
            in_dim = d_model if i == 0 else hidden_dim
            self.convs.append(nn.Conv1d(in_dim, hidden_dim, 3, padding=1))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, D] -> duration: [B, S]"""
        x = x.transpose(1, 2)  # [B, D, S]
        
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = x.transpose(1, 2)
            x = norm(x)
            x = x.transpose(1, 2)
            x = F.relu(x)
        
        x = x.transpose(1, 2)  # [B, S, D]
        duration = F.softplus(self.proj(x).squeeze(-1))
        return duration


class FMDiTV16AE(nn.Module):
    """Frame-conditioned DiT wrapper for V16 AE latent flow.

    The head is inherited from the stable v5.10 DiT recipe, while this wrapper
    owns V16-specific conditioning, masking, flow-time sampling and latent
    input/output projection.
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        phoneme_dim: int = 512,
        pitch_dim: int = 128,
        energy_dim: int = 128,
        cond_dim: int = 512,
        latent_dim: int = 512,
        model_dim: int | None = None,
        dit_dim_ff: int = 2048,
        dit_n_heads: int = 8,
        cond_drop_prob: float = 0.1,
        use_duration_predictor: bool = True,
    ):
        super().__init__()
        if model_dim is None:
            model_dim = latent_dim
        
        # Frame-level 条件编码
        self.phoneme_emb = nn.Embedding(vocab_size, phoneme_dim)
        self.pos_enc = SinusoidalPE(phoneme_dim)
        # Dilated Conv Stack (感受野 = 15 帧)
        self.phoneme_conv = nn.Sequential(
            nn.Conv1d(phoneme_dim, phoneme_dim, 3, dilation=1, padding=1),
            nn.GELU(),
            nn.Conv1d(phoneme_dim, phoneme_dim, 3, dilation=2, padding=2),
            nn.GELU(),
            nn.Conv1d(phoneme_dim, phoneme_dim, 3, dilation=4, padding=4),
            nn.GELU(),
        )
        # Pre-LN Transformer (3层, FFN=2048, nhead=8)
        self.phoneme_transformer = PreLNTransformerEncoder(
            d_model=phoneme_dim,
            nhead=8,
            dim_feedforward=2048,
            num_layers=3,
            drop_path=0.1,
        )
        # Prosody Encoder
        prosody_dim = pitch_dim + energy_dim  # 256
        self.pitch_proj = nn.Linear(1, pitch_dim)
        self.energy_proj = nn.Linear(1, energy_dim)
        # Depthwise Separable Conv for temporal modeling
        self.prosody_temporal = nn.Sequential(
            nn.Conv1d(prosody_dim, prosody_dim, 5, padding=2, groups=prosody_dim),
            nn.GELU(),
            nn.Conv1d(prosody_dim, prosody_dim, 1),
            nn.GELU(),
            nn.Conv1d(prosody_dim, prosody_dim, 11, padding=5, groups=prosody_dim),
            nn.GELU(),
            nn.Conv1d(prosody_dim, prosody_dim, 1),
        )
        # Prosody gate: cond = phone + gate * prosody
        self.prosody_gate = nn.Sequential(
            nn.Linear(phoneme_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.prosody_gate[0].weight)
        nn.init.constant_(self.prosody_gate[0].bias, -2.0)
        
        # 融合: 2层 MLP + LayerNorm
        self.cond_proj = nn.Sequential(
            nn.Linear(phoneme_dim + prosody_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_norm = nn.LayerNorm(cond_dim)
        
        # Latent <-> DiT 内部维度投影
        self.latent_in_proj = nn.Identity() if latent_dim == model_dim else nn.Linear(latent_dim, model_dim)
        self.latent_out_proj = nn.Identity() if latent_dim == model_dim else nn.Linear(model_dim, latent_dim)

        # DiT Head v5.10: stable SA + Conv flow head used as the v1 baseline.
        self.dit = DiTHeadV5_10(
            d_model=model_dim,
            dim_feedforward=dit_dim_ff,
            cond_dim=cond_dim,
            n_heads=dit_n_heads,
        )
        
        # Duration Predictor (辅助任务，可选)
        self.use_duration_predictor = use_duration_predictor
        if use_duration_predictor:
            self.phoneme_encoder = nn.Embedding(vocab_size, phoneme_dim)
            self.duration_predictor = CNNDurationPredictor(phoneme_dim, phoneme_dim // 2)
        
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.cond_dim = cond_dim
        self.cond_drop_prob = cond_drop_prob
    
    def encode_cond(self, frame_phoneme_ids, pitch, energy, drop_cond=False):
        """编码条件"""
        if drop_cond:
            B, S = frame_phoneme_ids.shape
            return torch.zeros(B, S, self.cond_dim, device=frame_phoneme_ids.device)
        
        # Phoneme: embedding + PE + Conv1d + Transformer
        phone_emb = self.phoneme_emb(frame_phoneme_ids)
        phone_emb = self.pos_enc(phone_emb)
        phone_emb = self.phoneme_conv(phone_emb.transpose(1, 2)).transpose(1, 2)
        padding_mask = (frame_phoneme_ids == 0)
        phone_emb = self.phoneme_transformer(phone_emb, src_key_padding_mask=padding_mask)
        
        # Prosody (pitch + energy)
        pitch_emb = self.pitch_proj(pitch.unsqueeze(-1))
        energy_emb = self.energy_proj(energy.unsqueeze(-1))
        prosody = torch.cat([pitch_emb, energy_emb], dim=-1)
        
        prosody = self.prosody_temporal(prosody.transpose(1, 2)).transpose(1, 2)
        
        # Prosody gate
        gate = self.prosody_gate(phone_emb)
        
        # 融合
        cond = torch.cat([phone_emb, gate * prosody], dim=-1)
        cond = self.cond_proj(cond)
        cond = self.cond_norm(cond)
        return cond
    
    def forward(
        self,
        latent_target: torch.Tensor,
        frame_phoneme_ids: torch.Tensor,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        phoneme_ids: torch.Tensor = None,
        gt_duration: torch.Tensor = None,
        frame_lengths: torch.Tensor | None = None,
        t_sampling: str = "beta",
        beta_alpha: float = 2.0,
        beta_beta: float = 2.0,
        logit_loc: float = 0.0,
        logit_scale: float = 1.0,
        t_eps: float = 1e-4,
        min_snr_gamma: float = 5.0,
    ):
        """训练 forward"""
        latent_target_model = self.latent_in_proj(latent_target)
        B, S, _ = latent_target_model.shape
        device = latent_target.device
        valid_mask = None
        key_padding_mask = None
        if frame_lengths is not None:
            frame_lengths = frame_lengths.to(device=device).clamp(min=1, max=S)
            valid_mask = lengths_to_valid_mask(frame_lengths, S).unsqueeze(-1)
            key_padding_mask = ~valid_mask.squeeze(-1)
            latent_target_model = latent_target_model * valid_mask.to(dtype=latent_target_model.dtype)
        
        # CFG: 随机 drop 条件
        drop_cond = random.random() < self.cond_drop_prob
        cond = self.encode_cond(frame_phoneme_ids, pitch, energy, drop_cond=drop_cond)
        if valid_mask is not None:
            cond = cond * valid_mask.to(dtype=cond.dtype)
        
        # Flow matching: noise -> latent.
        t = sample_flow_t(
            B, t_sampling, beta_alpha, beta_beta,
            logit_loc, logit_scale, t_eps, device,
        )
        noise = torch.randn_like(latent_target_model)
        if valid_mask is not None:
            noise = noise * valid_mask.to(dtype=noise.dtype)
        
        x_t = (1 - t) * noise + t * latent_target_model
        v_target = latent_target_model - noise
        
        # DiT 预测 velocity (v5.6: 只传 cond，无 context)
        v_pred = self.dit(
            x_t,
            t.squeeze(-1).squeeze(-1),
            cond,
            key_padding_mask=key_padding_mask,
        )
        
        # Min-SNR Weighting
        t_flat = t.squeeze(-1).squeeze(-1)
        snr = t_flat / (1 - t_flat + 1e-8)
        weight = torch.clamp(snr, max=min_snr_gamma) / (snr + 1e-8)
        weight = weight.view(B, 1, 1)
        
        # Weighted MSE loss
        sq_err = (v_pred - v_target) ** 2
        if valid_mask is not None:
            sq_err = sq_err * valid_mask.to(dtype=sq_err.dtype)
            denom = valid_mask.sum(dim=(1, 2)).to(dtype=sq_err.dtype).clamp_min(1.0) * sq_err.shape[-1]
            mse_per_sample = sq_err.sum(dim=(1, 2)) / denom
        else:
            mse_per_sample = sq_err.mean(dim=(1, 2))
        loss_fm = (weight.squeeze() * mse_per_sample).mean()
        
        # Duration Predictor 辅助任务
        loss_dur = torch.tensor(0.0, device=device)
        if self.use_duration_predictor and phoneme_ids is not None and gt_duration is not None and not drop_cond:
            phone_enc = self.phoneme_encoder(phoneme_ids)
            pred_duration = self.duration_predictor(phone_enc)
            
            min_len = min(pred_duration.shape[1], gt_duration.shape[1])
            pred_dur = pred_duration[:, :min_len]
            gt_dur = gt_duration[:, :min_len]
            loss_dur = F.mse_loss(torch.log1p(pred_dur), torch.log1p(gt_dur))
        
        return loss_fm, loss_dur
    
    @torch.no_grad()
    def sample(
        self,
        frame_phoneme_ids: torch.Tensor,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        n_steps: int = 6,
        cfg_scale: float = 2.0,
        latent_mean: float = 0.0,
        latent_std: float = 1.0,
        solver: str = "dpm2",
        frame_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """推理采样"""
        B, S = frame_phoneme_ids.shape
        device = frame_phoneme_ids.device
        
        cond = self.encode_cond(frame_phoneme_ids, pitch, energy)
        uncond = torch.zeros_like(cond)
        key_padding_mask = None
        valid_mask = None
        if frame_lengths is not None:
            frame_lengths = frame_lengths.to(device=device).clamp(min=1, max=S)
            valid_mask = lengths_to_valid_mask(frame_lengths, S).unsqueeze(-1)
            key_padding_mask = ~valid_mask.squeeze(-1)
            cond = cond * valid_mask.to(dtype=cond.dtype)
        
        x = torch.randn(B, S, self.model_dim, device=device)
        if valid_mask is not None:
            x = x * valid_mask.to(dtype=x.dtype)
        dt = 1.0 / n_steps
        
        if solver == "euler":
            for i in range(n_steps):
                t = torch.full((B,), i * dt, device=device)
                v_cond = self.dit(x, t, cond, key_padding_mask=key_padding_mask)
                v_uncond = self.dit(x, t, uncond, key_padding_mask=key_padding_mask)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
                x = x + v * dt
                if valid_mask is not None:
                    x = x * valid_mask.to(dtype=x.dtype)
                
        elif solver == "heun":
            for i in range(n_steps):
                t = torch.full((B,), i * dt, device=device)
                t_next = torch.full((B,), (i + 1) * dt, device=device)
                
                v_cond = self.dit(x, t, cond, key_padding_mask=key_padding_mask)
                v_uncond = self.dit(x, t, uncond, key_padding_mask=key_padding_mask)
                v1 = v_uncond + cfg_scale * (v_cond - v_uncond)
                x_pred = x + v1 * dt
                
                v_cond = self.dit(x_pred, t_next, cond, key_padding_mask=key_padding_mask)
                v_uncond = self.dit(x_pred, t_next, uncond, key_padding_mask=key_padding_mask)
                v2 = v_uncond + cfg_scale * (v_cond - v_uncond)
                
                x = x + 0.5 * (v1 + v2) * dt
                if valid_mask is not None:
                    x = x * valid_mask.to(dtype=x.dtype)
                
        else:  # dpm2
            prev_v = None
            for i in range(n_steps):
                t = torch.full((B,), i * dt, device=device)
                v_cond = self.dit(x, t, cond, key_padding_mask=key_padding_mask)
                v_uncond = self.dit(x, t, uncond, key_padding_mask=key_padding_mask)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
                
                if prev_v is None:
                    x = x + v * dt
                else:
                    x = x + (1.5 * v - 0.5 * prev_v) * dt
                prev_v = v
                if valid_mask is not None:
                    x = x * valid_mask.to(dtype=x.dtype)
        
        x = self.latent_out_proj(x)
        # 处理 latent_std/latent_mean 可能是列表的情况 (per-dim stats)
        if isinstance(latent_std, (list, tuple)):
            latent_std = torch.tensor(latent_std, device=x.device, dtype=x.dtype).view(1, 1, -1)
        if isinstance(latent_mean, (list, tuple)):
            latent_mean = torch.tensor(latent_mean, device=x.device, dtype=x.dtype).view(1, 1, -1)
        x = x * latent_std + latent_mean
        return x
    
    @torch.no_grad()
    def predict_duration(self, phoneme_ids: torch.Tensor) -> torch.Tensor:
        """预测 duration"""
        if not self.use_duration_predictor:
            raise RuntimeError("Duration predictor not enabled")
        phone_enc = self.phoneme_encoder(phoneme_ids)
        return self.duration_predictor(phone_enc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="fm_dit/data/ljspeech_v16_3_256x_dim384/train.json")
    parser.add_argument("--stats-path", type=str, default="fm_dit/data/ljspeech_v16_3_256x_dim384/latent_stats.json")
    parser.add_argument("--latent-dim", type=int, default=384, help="V16 autoencoder latent dimension")
    parser.add_argument("--model-dim", type=int, default=None, help="internal DiT width (default: same as latent-dim)")
    parser.add_argument("--model-version", type=str, default="v1_v16ae")
    parser.add_argument("--output-dir", type=str, default="outputs/fm_dit/v1_v16ae")
    parser.add_argument("--pretrained-cond-encoder", type=str, default=None, help="预训练条件编码器模型路径")
    parser.add_argument("--freeze-cond-encoder", action="store_true", help="冻结条件编码器")
    parser.add_argument("--cond-encoder-lr-mult", type=float, default=1.0, help="条件编码器学习率倍率")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--dit-dim-ff", type=int, default=2048)
    parser.add_argument("--dit-heads", type=int, default=8)
    parser.add_argument("--cond-drop-prob", type=float, default=0.1)
    parser.add_argument("--dur-weight", type=float, default=0.1)
    parser.add_argument("--no-duration-predictor", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=256, help="training crop length in latent frames")
    parser.add_argument("--min-frames", type=int, default=16, help="minimum latent frames to keep a sample")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preload", action="store_true", help="预加载 latent 到内存")
    parser.add_argument("--preload-gpu", action="store_true", help="预加载数据到 GPU 显存")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 继续训练")
    parser.add_argument("--resume-lr-boost", type=float, default=1.0, help="续训时学习率倍数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=1000, help="评估间隔 (steps)")
    parser.add_argument("--eval-frames", type=int, default=256)
    parser.add_argument("--sample-steps", type=int, default=6)
    parser.add_argument("--sample-solver", type=str, default="dpm2", choices=["euler", "heun", "dpm2"])
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--t-sampling", type=str, default="beta", choices=["beta", "uniform", "logit_normal"])
    parser.add_argument("--beta-alpha", type=float, default=2.0)
    parser.add_argument("--beta-beta", type=float, default=2.0)
    parser.add_argument("--logit-loc", type=float, default=0.0)
    parser.add_argument("--logit-scale", type=float, default=1.0)
    parser.add_argument("--t-eps", type=float, default=1e-4)
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    
    # 构建模型
    model = FMDiTV16AE(
        vocab_size=256,
        latent_dim=args.latent_dim,
        model_dim=args.model_dim,
        dit_dim_ff=args.dit_dim_ff,
        dit_n_heads=args.dit_heads,
        cond_drop_prob=args.cond_drop_prob,
        use_duration_predictor=not args.no_duration_predictor,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    dit_params = sum(p.numel() for p in model.dit.parameters())
    print(f"Model: {n_params/1e6:.2f}M params", flush=True)
    print(f"  DiT Head: {dit_params/1e6:.2f}M", flush=True)
    
    start_step = 0
    
    if model.use_duration_predictor:
        dur_params = sum(p.numel() for p in model.duration_predictor.parameters())
        print(f"  Duration Predictor: {dur_params/1e6:.2f}M", flush=True)
    
    # 加载预训练条件编码器
    if args.pretrained_cond_encoder:
        print(f"Loading pretrained condition encoder from {args.pretrained_cond_encoder}...")
        pretrained_ckpt = torch.load(args.pretrained_cond_encoder, map_location=device, weights_only=False)
        pretrained_state = pretrained_ckpt['model']
        
        # 条件编码器相关的 key
        cond_keys = ['phoneme_emb', 'pos_enc', 'phoneme_conv', 'phoneme_transformer',
                     'pitch_proj', 'energy_proj', 'prosody_temporal', 'prosody_gate',
                     'cond_proj', 'cond_norm']
        
        model_state = model.state_dict()
        transferred = 0
        for key in pretrained_state.keys():
            if any(key.startswith(prefix) for prefix in cond_keys):
                if key in model_state and pretrained_state[key].shape == model_state[key].shape:
                    model_state[key] = pretrained_state[key]
                    transferred += 1
        model.load_state_dict(model_state)
        print(f"  Transferred {transferred} condition encoder weights")
    
    # 冻结条件编码器
    if args.freeze_cond_encoder:
        print("Freezing condition encoder...")
        cond_modules = ['phoneme_emb', 'pos_enc', 'phoneme_conv', 'phoneme_transformer',
                        'pitch_proj', 'energy_proj', 'prosody_temporal', 'prosody_gate',
                        'cond_proj', 'cond_norm']
        frozen_params = 0
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in cond_modules):
                param.requires_grad = False
                frozen_params += param.numel()
        print(f"  Frozen {frozen_params/1e6:.2f}M parameters")
    
    # 优化器 - 支持不同学习率
    if args.cond_encoder_lr_mult != 1.0 and not args.freeze_cond_encoder:
        cond_modules = ['phoneme_emb', 'pos_enc', 'phoneme_conv', 'phoneme_transformer',
                        'pitch_proj', 'energy_proj', 'prosody_temporal', 'prosody_gate',
                        'cond_proj', 'cond_norm']
        cond_params = []
        other_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(name.startswith(prefix) for prefix in cond_modules):
                    cond_params.append(param)
                else:
                    other_params.append(param)
        optimizer = AdamW([
            {'params': cond_params, 'lr': args.lr * args.cond_encoder_lr_mult},
            {'params': other_params, 'lr': args.lr},
        ], weight_decay=args.weight_decay)
        print(f"  Condition encoder LR: {args.lr * args.cond_encoder_lr_mult:.2e}")
    else:
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    warmup_steps = min(args.warmup_steps, args.steps // 2)
    def lr_lambda(step):
        if step < warmup_steps:
            return max(1, step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / (args.steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        min_ratio = args.min_lr / args.lr
        return min_ratio + (1 - min_ratio) * cosine_decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 断点续训
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        print(f"  Resumed from {args.resume} (step {start_step})")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print(f"  Restored optimizer state")
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            print(f"  Restored scheduler state (LR={scheduler.get_last_lr()[0]:.2e})")
        if args.resume_lr_boost > 1.0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.resume_lr_boost
            print(f"  LR boosted by {args.resume_lr_boost}x")
    
    # 训练指标追踪
    metrics_history = {
        "steps": [],
        "fm_loss": [],
        "dur_loss": [],
        "correlation": [],
        "latent_mse": [],
        "lr": [],
    }
    
    # 数据
    train_loader = get_dataloader_v3(
        manifest_path=args.data_path,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        num_workers=args.num_workers,
        shuffle=True,
        latent_stats_path=args.stats_path,
        preload_latents=args.preload or args.preload_gpu,
        preload_to_gpu=args.preload_gpu,
    )
    print(f"Train samples: {len(train_loader.dataset)}", flush=True)
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练
    print(f"\n[FM-DiT {args.model_version}] V16 autoencoder latent flow", flush=True)
    print(f"  phoneme_dim=512, pitch/energy_dim=128, cond_dim=512", flush=True)
    print(f"  latent_dim={args.latent_dim} (I/O), model_dim={model.model_dim} (DiT internal)", flush=True)
    print(f"  max_frames={args.max_frames}, min_frames={args.min_frames}", flush=True)
    print(f"  t_sampling={args.t_sampling}, min_snr_gamma={args.min_snr_gamma}", flush=True)
    print(f"  batch_size={args.batch_size}, grad_accum_steps={args.grad_accum_steps}", flush=True)
    print(f"  DiT: SA + Conv (每层), 无 Cross-Attention", flush=True)
    print(f"  条件注入: 输入加法 + Per-Token AdaLN", flush=True)
    print(f"  LR: {args.lr} -> {args.min_lr} (warmup={warmup_steps})", flush=True)
    model.train()
    data_iter = iter(train_loader)
    
    t0 = time.time()
    running_loss_fm = 0.0
    running_loss_dur = 0.0
    running_micro = 0
    amp_enabled = device == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    
    for step in range(start_step, args.steps):
        optimizer.zero_grad()
        for _micro in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            latent = batch["latent"].to(device, non_blocking=True)
            frame_phoneme_ids = batch["frame_phoneme_ids"].to(device, non_blocking=True)
            pitch = batch["pitch"].to(device, non_blocking=True)
            energy = batch["energy"].to(device, non_blocking=True)
            phoneme_ids = batch["phoneme_ids"].to(device, non_blocking=True)
            gt_duration = batch["phoneme_durations"].to(device, non_blocking=True)
            frame_lengths = batch["frame_lengths"].to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                loss_fm, loss_dur = model(
                    latent, frame_phoneme_ids, pitch, energy,
                    phoneme_ids, gt_duration,
                    frame_lengths=frame_lengths,
                    t_sampling=args.t_sampling,
                    beta_alpha=args.beta_alpha,
                    beta_beta=args.beta_beta,
                    logit_loc=args.logit_loc,
                    logit_scale=args.logit_scale,
                    t_eps=args.t_eps,
                    min_snr_gamma=args.min_snr_gamma,
                )
                loss = (loss_fm + args.dur_weight * loss_dur) / args.grad_accum_steps
            
            loss.backward()
            running_loss_fm += loss_fm.item()
            running_loss_dur += loss_dur.item()
            running_micro += 1
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        
        if (step + 1) % args.log_interval == 0:
            avg_fm = running_loss_fm / max(1, running_micro)
            avg_dur = running_loss_dur / max(1, running_micro)
            elapsed = time.time() - t0
            speed = args.log_interval / elapsed
            lr = scheduler.get_last_lr()[0]
            
            # 获取 gate 统计
            gate_stats = model.dit.get_gate_stats()
            gate_str = " ".join([f"g{k}={v:.2f}" if isinstance(v, float) else f"g{k}={v}" for k, v in gate_stats.items()])
            
            print(f"Step {step+1}/{args.steps} | FM: {avg_fm:.4f} | Dur: {avg_dur:.4f} | {gate_str} | LR: {lr:.2e} | {speed:.1f} steps/s", flush=True)
            running_loss_fm = 0.0
            running_loss_dur = 0.0
            running_micro = 0
            t0 = time.time()
        
        if args.save_interval > 0 and (step + 1) % args.save_interval == 0:
            torch.save({
                "model": model.state_dict(),
                "version": args.model_version,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step + 1,
                "args": vars(args),
                "metrics_history": metrics_history,
            }, output_dir / f"step_{step+1}.pt")
        
        # 每 eval_interval steps 评估一次
        if (step + 1) % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                eval_item = train_loader.dataset.items[0]
                T = min(len(eval_item['frame_phoneme_ids']), args.eval_frames)
                eval_phone = torch.tensor([eval_item['frame_phoneme_ids'][:T]], dtype=torch.long, device=device)
                eval_pitch = torch.tensor([eval_item['pitch'][:T]], dtype=torch.float, device=device)
                eval_energy = torch.tensor([eval_item['energy'][:T]], dtype=torch.float, device=device)
                eval_lengths = torch.tensor([T], dtype=torch.long, device=device)
                
                with open(args.stats_path) as f:
                    stats = json.load(f)
                
                # 优先使用 global_mean/global_std (标量)
                latent_mean = stats.get('global_mean', stats.get('mean', 0.0))
                latent_std = stats.get('global_std', stats.get('std', 1.0))
                latent_pred = model.sample(
                    eval_phone, eval_pitch, eval_energy,
                    n_steps=args.sample_steps, cfg_scale=args.cfg_scale,
                    latent_mean=latent_mean, latent_std=latent_std,
                    solver=args.sample_solver,
                    frame_lengths=eval_lengths,
                )
                
                orig_latent = torch.load(eval_item['latent_path'], weights_only=True)
                orig = train_loader.dataset._to_time_channel(orig_latent)
                pred = latent_pred.squeeze(0).cpu()
                min_len = min(orig.shape[0], pred.shape[0])
                orig = orig[:min_len]
                pred = pred[:min_len]
                corr = F.cosine_similarity(
                    orig.flatten().unsqueeze(0),
                    pred.flatten().unsqueeze(0),
                ).item()
                latent_mse = F.mse_loss(pred, orig).item()
                
                metrics_history["steps"].append(step + 1)
                metrics_history["fm_loss"].append(avg_fm if 'avg_fm' in dir() else 0)
                metrics_history["correlation"].append(corr)
                metrics_history["latent_mse"].append(latent_mse)
                metrics_history["lr"].append(scheduler.get_last_lr()[0])
                
                print(f"  [Eval] Step {step+1}: corr={corr:.4f} latent_mse={latent_mse:.6f}", flush=True)
            model.train()
    
    # 保存
    torch.save({
        "model": model.state_dict(),
        "version": args.model_version,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": args.steps,
        "args": vars(args),
        "metrics_history": metrics_history,
    }, output_dir / "fm_dit_v1_v16ae_model.pt")
    print(f"  模型: {output_dir / 'fm_dit_v1_v16ae_model.pt'}", flush=True)
    
    # 测试
    print("\nTesting...", flush=True)
    model.eval()
    
    with open(args.data_path) as f:
        items = json.load(f)
    
    with open(args.stats_path) as f:
        stats = json.load(f)
    
    item = items[0]
    frame_phoneme_ids = torch.tensor([item['frame_phoneme_ids']], dtype=torch.long, device=device)
    pitch = torch.tensor([item['pitch']], dtype=torch.float, device=device)
    energy = torch.tensor([item['energy']], dtype=torch.float, device=device)
    frame_lengths = torch.tensor([len(item['frame_phoneme_ids'])], dtype=torch.long, device=device)
    orig_latent = torch.load(item['latent_path'], weights_only=True)
    
    # 优先使用 global_mean/global_std (标量)
    test_latent_mean = stats.get('global_mean', stats.get('mean', 0.0))
    test_latent_std = stats.get('global_std', stats.get('std', 1.0))
    
    for cfg_scale in [1.0, 1.5, 2.0]:
        latent_pred = model.sample(
            frame_phoneme_ids, pitch, energy,
            n_steps=args.sample_steps, cfg_scale=cfg_scale,
            latent_mean=test_latent_mean, latent_std=test_latent_std,
            solver=args.sample_solver,
            frame_lengths=frame_lengths,
        )
        
        orig = train_loader.dataset._to_time_channel(orig_latent)
        pred = latent_pred.squeeze(0).cpu()
        
        min_len = min(orig.shape[0], pred.shape[0])
        corr = F.cosine_similarity(
            orig[:min_len].flatten().unsqueeze(0),
            pred[:min_len].flatten().unsqueeze(0),
        ).item()
        
        print(f"  CFG={cfg_scale}: corr={corr:.4f}, std={pred.std():.4f}")
    
    # 打印训练指标总结
    if metrics_history["steps"]:
        print("\n" + "="*60)
        print("训练指标总结 (每 1k steps)")
        print("="*60)
        print(f"{'Step':>8} | {'Correlation':>12} | {'LR':>10}")
        print("-"*40)
        for i, step in enumerate(metrics_history["steps"]):
            corr = metrics_history["correlation"][i]
            lr = metrics_history["lr"][i]
            print(f"{step:>8} | {corr:>12.4f} | {lr:>10.2e}")
        
        if len(metrics_history["correlation"]) >= 2:
            corr_trend = metrics_history["correlation"][-1] - metrics_history["correlation"][0]
            trend_str = "↑" if corr_trend > 0 else "↓"
            print("-"*40)
            print(f"Correlation 趋势: {corr_trend:+.4f} {trend_str}")
        print("="*60)


if __name__ == "__main__":
    main()
