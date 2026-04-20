"""损失函数模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTLoss(nn.Module):
    """单分辨率 STFT 损失
    
    支持:
    - Spectral convergence + log magnitude (原有)
    - Complex loss: real/imag L1 (隐式相位约束)
    - 频带加权: 指定频率范围加权 (如 2-6kHz)
    """
    
    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_size: int = 1024,
        sr: int = 24000,
        complex_weight: float = 0.0,
        band_boost_range: tuple = None,
        band_boost_factor: float = 1.0,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.complex_weight = complex_weight
        self.register_buffer("window", torch.hann_window(win_size))
        
        # 频带加权 mask: [n_freq, 1]
        n_freq = fft_size // 2 + 1
        freq_weight = torch.ones(n_freq, 1)
        if band_boost_range is not None and band_boost_factor > 1.0:
            lo_hz, hi_hz = band_boost_range
            bin_lo = int(lo_hz / (sr / fft_size))
            bin_hi = min(int(hi_hz / (sr / fft_size)), n_freq)
            freq_weight[bin_lo:bin_hi] = band_boost_factor
        self.register_buffer("freq_weight", freq_weight)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, T] 预测音频
            y: [B, 1, T] 目标音频
        """
        # 强制 FP32: torch.stft 在 FP16 下中间值易溢出
        x = x.squeeze(1).float()
        y = y.squeeze(1).float()
        
        x_stft = torch.stft(
            x, self.fft_size, self.hop_size, self.win_size,
            window=self.window, return_complex=True
        )
        y_stft = torch.stft(
            y, self.fft_size, self.hop_size, self.win_size,
            window=self.window, return_complex=True
        )
        
        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)
        
        # 频带加权
        w = self.freq_weight  # [n_freq, 1]
        x_mag_w = x_mag * w
        y_mag_w = y_mag * w
        
        # Spectral convergence loss (加权)
        sc_loss = torch.norm(y_mag_w - x_mag_w, p="fro") / (torch.norm(y_mag_w, p="fro") + 1e-4)
        
        # Log magnitude loss (加权)  [eps=1e-4: FP16 safe, 1e-8 underflows to 0]
        log_loss = F.l1_loss(torch.log(x_mag_w + 1e-4), torch.log(y_mag_w + 1e-4))
        
        loss = sc_loss + log_loss
        
        # Complex loss: real/imag L1 (隐式相位约束)
        if self.complex_weight > 0:
            complex_loss = (
                F.l1_loss(x_stft.real * w, y_stft.real * w)
                + F.l1_loss(x_stft.imag * w, y_stft.imag * w)
            )
            loss = loss + self.complex_weight * complex_loss
        
        return loss


class MultiResolutionSTFTLoss(nn.Module):
    """多分辨率 STFT 损失"""
    
    def __init__(
        self,
        fft_sizes: list = [512, 1024, 2048],
        hop_sizes: list = [128, 256, 512],
        win_sizes: list = [512, 1024, 2048],
        sr: int = 24000,
        complex_weight: float = 0.0,
        band_boost_range: tuple = None,
        band_boost_factor: float = 1.0,
    ):
        super().__init__()
        self.stft_losses = nn.ModuleList([
            STFTLoss(
                fft_size, hop_size, win_size,
                sr=sr,
                complex_weight=complex_weight,
                band_boost_range=band_boost_range,
                band_boost_factor=band_boost_factor,
            )
            for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes)
        ])
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0
        for stft_loss in self.stft_losses:
            loss += stft_loss(x, y)
        return loss / len(self.stft_losses)


class MultiBandSTFTLoss(nn.Module):
    """多频段独立 STFT 损失
    
    将频谱分成多个子带，每个子带独立计算 spectral convergence + log mag loss，
    然后等权求和。避免低频能量主导全局 loss，让中高频获得足够梯度。
    
    默认频段 (针对 decoder Nyquist 频率设计):
      Band 0: 0-500Hz     (基频区)
      Band 1: 500-1500Hz  (Stage 1 Nyquist = 1500Hz)
      Band 2: 1500-3500Hz (Stage 1 imaging 区)
      Band 3: 3500-6000Hz (Stage 2 Nyquist = 6000Hz)
      Band 4: 6000-12000Hz (高频区)
    """
    
    def __init__(
        self,
        fft_size: int = 2048,
        hop_size: int = 512,
        win_size: int = 2048,
        sr: int = 24000,
        band_edges_hz: list = [0, 500, 1500, 3500, 6000, 12000],
        complex_weight: float = 1.0,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.complex_weight = complex_weight
        self.register_buffer("window", torch.hann_window(win_size))
        
        n_freq = fft_size // 2 + 1
        hz_per_bin = sr / fft_size
        
        # 预计算每个 band 的 bin 范围
        self.band_slices = []
        for i in range(len(band_edges_hz) - 1):
            lo_bin = int(band_edges_hz[i] / hz_per_bin)
            hi_bin = min(int(band_edges_hz[i + 1] / hz_per_bin), n_freq)
            self.band_slices.append((lo_bin, hi_bin))
        
        self.n_bands = len(self.band_slices)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 强制 FP32: torch.stft 在 FP16 下中间值易溢出
        x = x.squeeze(1).float()
        y = y.squeeze(1).float()
        
        x_stft = torch.stft(
            x, self.fft_size, self.hop_size, self.win_size,
            window=self.window, return_complex=True
        )
        y_stft = torch.stft(
            y, self.fft_size, self.hop_size, self.win_size,
            window=self.window, return_complex=True
        )
        
        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)
        
        total_loss = 0.0
        for lo, hi in self.band_slices:
            x_band = x_mag[:, lo:hi, :]
            y_band = y_mag[:, lo:hi, :]
            
            # Spectral convergence (per-band, 独立归一化)
            sc = torch.norm(y_band - x_band, p="fro") / (torch.norm(y_band, p="fro") + 1e-4)
            # Log magnitude (per-band)  [eps=1e-4: FP16 safe]
            log_l = F.l1_loss(torch.log(x_band + 1e-4), torch.log(y_band + 1e-4))
            
            band_loss = sc + log_l
            
            # Complex loss per band
            if self.complex_weight > 0:
                x_band_c = x_stft[:, lo:hi, :]
                y_band_c = y_stft[:, lo:hi, :]
                complex_l = (
                    F.l1_loss(x_band_c.real, y_band_c.real)
                    + F.l1_loss(x_band_c.imag, y_band_c.imag)
                )
                band_loss = band_loss + self.complex_weight * complex_l
            
            total_loss = total_loss + band_loss
        
        return total_loss / self.n_bands


class MultiResolutionMultiBandSTFTLoss(nn.Module):
    """多分辨率 × 多频段 STFT 损失
    
    在多个 FFT 分辨率上分别计算 MultiBandSTFTLoss，取平均。
    """
    
    def __init__(
        self,
        fft_sizes: list = [512, 1024, 2048],
        hop_sizes: list = [128, 256, 512],
        win_sizes: list = [512, 1024, 2048],
        sr: int = 24000,
        band_edges_hz: list = [0, 500, 1500, 3500, 6000, 12000],
        complex_weight: float = 1.0,
    ):
        super().__init__()
        self.losses = nn.ModuleList([
            MultiBandSTFTLoss(
                fft_size=f, hop_size=h, win_size=w,
                sr=sr, band_edges_hz=band_edges_hz,
                complex_weight=complex_weight,
            )
            for f, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for l in self.losses:
            loss = loss + l(x, y)
        return loss / len(self.losses)


class HighFreqPenaltyLoss(nn.Module):
    """高频惩罚损失: 惩罚重建音频中多余的高频能量
    
    对 STFT 的高频 bins 施加额外的 L1 惩罚，
    抑制 decoder 产生的高频噪声/电流音。
    """
    
    def __init__(self, fft_size: int = 2048, hop_size: int = 512, 
                 high_freq_ratio: float = 0.3):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.register_buffer("window", torch.hann_window(fft_size))
        self.high_freq_start = int((fft_size // 2 + 1) * (1 - high_freq_ratio))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 强制 FP32
        x = x.squeeze(1).float()
        y = y.squeeze(1).float()
        
        x_stft = torch.stft(
            x, self.fft_size, self.hop_size, self.fft_size,
            window=self.window, return_complex=True
        )
        y_stft = torch.stft(
            y, self.fft_size, self.hop_size, self.fft_size,
            window=self.window, return_complex=True
        )
        
        x_mag = torch.abs(x_stft[:, self.high_freq_start:, :])
        y_mag = torch.abs(y_stft[:, self.high_freq_start:, :])
        
        return F.l1_loss(x_mag, y_mag)


def _set_requires_grad(module: nn.Module, flag: bool):
    """临时开关 module 参数的 requires_grad"""
    for p in module.parameters():
        p.requires_grad = flag


class BranchDecodeLoss(nn.Module):
    """分支独立解码频段 STFT Loss
    
    对 encoder 的 LF/MF/HF 分支 latent 各自 zero-pad 到 128ch,
    独立经过 decoder, 再与 bandpass 滤波后的原始音频计算 STFT loss.
    
    目的: 强制各分支编码对应频段信息, 防止 HF 分支死亡.
    """
    
    def __init__(
        self,
        sr: int = 24000,
        ae_dim: int = 128,
        lf_dim: int = 64,
        mf_dim: int = 32,
        hf_dim: int = 32,
        # 频段边界 (Hz)
        lf_range: tuple = (0, 1500),
        mf_range: tuple = (1500, 6000),
        hf_range: tuple = (6000, 12000),
        # STFT 参数
        fft_size: int = 2048,
        hop_size: int = 512,
    ):
        super().__init__()
        self.sr = sr
        self.ae_dim = ae_dim
        self.lf_dim = lf_dim
        self.mf_dim = mf_dim
        self.hf_dim = hf_dim
        self.lf_slice = (0, lf_dim)
        self.mf_slice = (lf_dim, lf_dim + mf_dim)
        self.hf_slice = (lf_dim + mf_dim, ae_dim)
        
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.register_buffer("window", torch.hann_window(fft_size))
        
        # 预计算频段 bin 范围
        n_bins = fft_size // 2 + 1
        freq_per_bin = sr / fft_size
        
        self.lf_bins = (max(1, int(lf_range[0] / freq_per_bin)), 
                        int(lf_range[1] / freq_per_bin))
        self.mf_bins = (int(mf_range[0] / freq_per_bin),
                        int(mf_range[1] / freq_per_bin))
        self.hf_bins = (int(hf_range[0] / freq_per_bin),
                        min(n_bins, int(hf_range[1] / freq_per_bin)))
        
        self.branch_bins = [self.lf_bins, self.mf_bins, self.hf_bins]
    
    def _band_stft_loss(self, recon: torch.Tensor, target: torch.Tensor, 
                         bin_lo: int, bin_hi: int) -> torch.Tensor:
        """计算指定频段的 STFT magnitude L1 loss"""
        recon = recon.squeeze(1).float()
        target = target.squeeze(1).float()
        
        r_stft = torch.stft(recon, self.fft_size, self.hop_size, self.fft_size,
                            window=self.window, return_complex=True)
        t_stft = torch.stft(target, self.fft_size, self.hop_size, self.fft_size,
                            window=self.window, return_complex=True)
        
        r_mag = torch.abs(r_stft[:, bin_lo:bin_hi, :])
        t_mag = torch.abs(t_stft[:, bin_lo:bin_hi, :])
        
        # Spectral convergence + log magnitude
        sc = torch.norm(t_mag - r_mag, p="fro") / (torch.norm(t_mag, p="fro") + 1e-8)
        log_mag = F.l1_loss(torch.log(r_mag + 1e-7), torch.log(t_mag + 1e-7))
        
        return sc + log_mag
    
    def forward(self, branches: tuple, decoder, audio: torch.Tensor, 
                output_length: int) -> torch.Tensor:
        """
        Args:
            branches: (lf_out, mf_out, hf_out) from encoder, each [B, T, dim]
            decoder: WaveformDecoderV6 instance
            audio: [B, 1, T] original audio
            output_length: target length for decoder
        Returns:
            total branch loss (scalar)
        """
        lf_out, mf_out, hf_out = branches
        B, T_lat, _ = lf_out.shape
        device = lf_out.device
        
        branch_data = [
            (lf_out, self.lf_slice, self.lf_bins),
            (mf_out, self.mf_slice, self.mf_bins),
            (hf_out, self.hf_slice, self.hf_bins),
        ]
        
        total_loss = torch.tensor(0.0, device=device)
        
        for branch_lat, (ch_lo, ch_hi), (bin_lo, bin_hi) in branch_data:
            # Zero-pad 到 128ch
            full_lat = torch.zeros(B, T_lat, self.ae_dim, device=device, dtype=branch_lat.dtype)
            full_lat[:, :, ch_lo:ch_hi] = branch_lat
            
            # Decode — 冻结 decoder 参数, 梯度只流向 encoder 分支
            # 否则 decoder 会学习适应 sparse zero-padded input, 破坏全局重建
            _set_requires_grad(decoder, False)
            with torch.amp.autocast('cuda', enabled=False):
                recon = decoder(full_lat.float(), output_length=output_length)
            _set_requires_grad(decoder, True)
            
            # 频段 STFT loss
            min_len = min(audio.shape[-1], recon.shape[-1])
            loss = self._band_stft_loss(recon[:, :, :min_len], audio[:, :, :min_len],
                                         bin_lo, bin_hi)
            total_loss = total_loss + loss
        
        return total_loss / 3.0  # 平均三个分支


class TailPenaltyLoss(nn.Module):
    """Latent 尾部惩罚: 只惩罚超出 ±k 的值, 不动主体分布
    
    L_tail = mean(relu(|z| - k)^2)
    
    比 KL 轻柔得多: 只打尾巴, 不强迫整体形状
    k=4~6 对应 N(0,1) 的 4~6 sigma
    """
    
    def __init__(self, k: float = 4.0):
        super().__init__()
        self.k = k
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, T, C] latent tensor
        """
        return torch.mean(F.relu(torch.abs(z) - self.k) ** 2)
