"""Multi-Scale STFT + Multi-Scale Waveform Discriminator

两个互补的判别器:

1. MS-STFT Discriminator (频域, 主要)
   - 在复数 STFT 上操作 (real + imag → 2ch)
   - 3 个 STFT 分辨率: (1024, 2048, 512)
   - 2D Conv 在 time-freq 平面上判别
   - 擅长捕捉: 频谱包络误差, 频率凹陷, 谐波失真

2. Multi-Scale Waveform Discriminator (时域, 辅助)
   - 在原始波形 + 下采样版本上操作
   - 3 个尺度: 24kHz, 12kHz, 6kHz
   - 1D grouped Conv 逐步下采样
   - 擅长捕捉: 波形级伪影, 相位不连续, 点击噪声

参考: EnCodec (MS-STFT), SoundStream/HiFi-GAN (MSD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import List, Tuple


# ─── MS-STFT Discriminator ──────────────────────────────────

def _get_2d_padding(kernel_size: Tuple[int, int], dilation: Tuple[int, int] = (1, 1)):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


class STFTDiscriminator(nn.Module):
    """单分辨率 STFT 判别器
    
    输入复数 STFT (real+imag 拼接为 2ch), 用 2D Conv 在 time-freq 平面判别。
    """
    
    def __init__(
        self,
        filters: int = 32,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        kernel_size: Tuple[int, int] = (3, 9),
        stride: Tuple[int, int] = (1, 2),
        dilations: List[int] = [1, 2, 4],
        max_filters: int = 256,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        
        # 输入: 2ch (real + imag)
        self.convs = nn.ModuleList()
        in_ch = 2
        out_ch = filters
        
        # 第一层: 不做 stride
        self.convs.append(nn.Sequential(
            weight_norm(nn.Conv2d(
                in_ch, out_ch, kernel_size=kernel_size,
                padding=_get_2d_padding(kernel_size),
            )),
            nn.LeakyReLU(0.2),
        ))
        in_ch = out_ch
        
        # 中间层: dilation + stride
        for i, dilation in enumerate(dilations):
            out_ch = min(filters * (2 ** (i + 1)), max_filters)
            self.convs.append(nn.Sequential(
                weight_norm(nn.Conv2d(
                    in_ch, out_ch, kernel_size=kernel_size,
                    stride=stride, dilation=(dilation, 1),
                    padding=_get_2d_padding(kernel_size, (dilation, 1)),
                )),
                nn.LeakyReLU(0.2),
            ))
            in_ch = out_ch
        
        # 倒数第二层: k=(3,3)
        out_ch = min(filters * (2 ** (len(dilations) + 1)), max_filters)
        self.convs.append(nn.Sequential(
            weight_norm(nn.Conv2d(
                in_ch, out_ch, kernel_size=(kernel_size[0], kernel_size[0]),
                padding=_get_2d_padding((kernel_size[0], kernel_size[0])),
            )),
            nn.LeakyReLU(0.2),
        ))
        in_ch = out_ch
        
        # 输出层: 1ch
        self.conv_post = weight_norm(nn.Conv2d(
            in_ch, 1, kernel_size=(kernel_size[0], kernel_size[0]),
            padding=_get_2d_padding((kernel_size[0], kernel_size[0])),
        ))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, 1, T] 波形
        Returns:
            logits: [B, 1, T', F'] 判别器输出
            features: 中间层特征列表
        """
        # STFT → complex → real+imag
        x_1d = x.squeeze(1)  # [B, T]
        spec = torch.stft(
            x_1d, self.n_fft, self.hop_length, self.win_length,
            window=self.window, return_complex=True,
        )  # [B, F, T']
        # [B, 2, T', F] — 2ch = real + imag, 转置使 freq 在最后一维
        z = torch.stack([spec.real, spec.imag], dim=1)  # [B, 2, F, T']
        z = z.permute(0, 1, 3, 2)  # [B, 2, T', F]
        
        features = []
        for layer in self.convs:
            z = layer(z)
            features.append(z)
        
        logits = self.conv_post(z)
        return logits, features


class MultiScaleSTFTDiscriminator(nn.Module):
    """多分辨率 STFT 判别器 (3 个分辨率)"""
    
    def __init__(
        self,
        filters: int = 32,
        n_ffts: List[int] = [1024, 2048, 512],
        hop_lengths: List[int] = [256, 512, 128],
        win_lengths: List[int] = [1024, 2048, 512],
        **kwargs,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(
                filters=filters, n_fft=n_ffts[i],
                hop_length=hop_lengths[i], win_length=win_lengths[i],
                **kwargs,
            )
            for i in range(len(n_ffts))
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        logits_list = []
        features_list = []
        for disc in self.discriminators:
            logits, features = disc(x)
            logits_list.append(logits)
            features_list.append(features)
        return logits_list, features_list


# ─── Multi-Scale Waveform Discriminator ─────────────────────

class ScaleDiscriminator(nn.Module):
    """单尺度波形判别器"""
    
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(1, 32, kernel_size=15, padding=7)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(32, 128, kernel_size=41, stride=4, padding=20, groups=4)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(128, 256, kernel_size=41, stride=4, padding=20, groups=16)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(256, 512, kernel_size=41, stride=4, padding=20, groups=64)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(512, 1024, kernel_size=5, padding=2)),
                nn.LeakyReLU(0.2),
            ),
            weight_norm(nn.Conv1d(1024, 1, kernel_size=3, padding=1)),
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return x, features[:-1]


class MultiScaleDiscriminator(nn.Module):
    """多尺度波形判别器 (3 个尺度)"""
    
    def __init__(self, num_scales: int = 3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator() for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        logits_list = []
        features_list = []
        for i, disc in enumerate(self.discriminators):
            logits, features = disc(x)
            logits_list.append(logits)
            features_list.append(features)
            if i < len(self.discriminators) - 1:
                x = self.downsample(x)
        return logits_list, features_list


# ─── Combined Discriminator ─────────────────────────────────

class CombinedDiscriminator(nn.Module):
    """MS-STFT + MSD 组合判别器
    
    返回统一格式的 logits 和 features 列表。
    """
    
    def __init__(
        self,
        use_msstft: bool = True,
        use_msd: bool = True,
        stft_filters: int = 32,
        msd_scales: int = 3,
    ):
        super().__init__()
        self.use_msstft = use_msstft
        self.use_msd = use_msd
        
        if use_msstft:
            self.msstft = MultiScaleSTFTDiscriminator(filters=stft_filters)
        if use_msd:
            self.msd = MultiScaleDiscriminator(num_scales=msd_scales)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        logits_all = []
        features_all = []
        
        if self.use_msstft:
            logits, features = self.msstft(x)
            logits_all.extend(logits)
            features_all.extend(features)
        
        if self.use_msd:
            logits, features = self.msd(x)
            logits_all.extend(logits)
            features_all.extend(features)
        
        return logits_all, features_all


# ─── GAN Losses ─────────────────────────────────────────────

def discriminator_loss(
    real_logits: List[torch.Tensor],
    fake_logits: List[torch.Tensor],
) -> torch.Tensor:
    """Hinge loss for discriminator"""
    loss = 0.0
    for real, fake in zip(real_logits, fake_logits):
        loss += torch.mean(F.relu(1 - real)) + torch.mean(F.relu(1 + fake))
    return loss / len(real_logits)


def generator_loss(fake_logits: List[torch.Tensor]) -> torch.Tensor:
    """Hinge loss for generator"""
    loss = 0.0
    for fake in fake_logits:
        loss += -torch.mean(fake)
    return loss / len(fake_logits)


def feature_matching_loss(
    real_features: List[List[torch.Tensor]],
    fake_features: List[List[torch.Tensor]],
) -> torch.Tensor:
    """Feature matching loss: L1 on discriminator intermediate features"""
    loss = 0.0
    count = 0
    for real_feats, fake_feats in zip(real_features, fake_features):
        for real_f, fake_f in zip(real_feats, fake_feats):
            loss += F.l1_loss(fake_f, real_f.detach())
            count += 1
    return loss / max(count, 1)


