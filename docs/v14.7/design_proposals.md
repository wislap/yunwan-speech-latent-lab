# V14.7 设计方案

## 问题诊断

V14.4 SNR 卡在 ~11.4 dB，瓶颈是相位误差：
- Recon mag + Orig phase: 22.3 dB（修复相位 +5.8 dB）
- Recon mag + Recon phase: 16.5 dB（修复幅度仅 +0.8 dB）
- 2kHz 以上相位误差急剧增大，听感为"电流音"
- 根因：encoder 逐级下采样丢失高频相位信息，latent 中相位 R²≈0.5
- 改 loss（V14.7 complex STFT）和改 decoder（V14.8 iSTFT head）都无效，因为信息在 encoder 端就丢了

## 方案 A：双路径 Encoder + AdaLN 交互（推荐）

### 核心思路

Encoder 双路径：时域路径（现有 V14.4）捕捉波形结构，频域路径（新增）捕捉 STFT complex 系数（含完整相位）。两路通过 AdaLN + MLP + Gate 深度交互，最终融合输出 latent。

### 架构

```
输入: wav [B, 1, T]
         │
         ├── 时域路径 (V14.4 encoder blocks)
         │   stem -> block[0] -> block[1] -> block[2] -> block[3] ──→ block[4] -> proj
         │                                                  ↕ AdaLN      ↕ AdaLN
         │                                                  ↕             ↕
         ├── 频域路径 (新增)                                 ↕             ↕
         │   STFT(2048,512) -> 2D Conv compress freq dim   ↕             ↕
         │   -> [B, C, T/512] ─────────────────── F_deep ──→ F_final ───→┘
         │                                                              │
         └── 融合 proj: [B, C_fused, T/512] -> [B, 128*2, T/512] (mean+logvar)
```

### 时间维度对齐

| 层级 | 时域路径 | 频域路径 |
|------|---------|---------|
| block[0] | T/2 (11025Hz) | — (不交互) |
| block[1] | T/8 (2756Hz) | — (不交互) |
| block[2] | T/32 (689Hz) | — (不交互) |
| block[3] | T/128 (172Hz) | STFT(256, 128) → T/128 ✓ 对齐 |
| block[4] | T/512 (43Hz) | STFT(1024, 512) → T/512 ✓ 对齐 |

频域路径在 block[3] 和 block[4] 两个层级注入，使用不同 STFT 分辨率：
- block[3] 层级: n_fft=256, hop=128 → 129 bins, 帧率 172Hz
- block[4] 层级: n_fft=1024, hop=512 → 513 bins, 帧率 43Hz

### DualPathInteraction 模块

```python
class DualPathInteraction(nn.Module):
    """双路径 AdaLN 交互
    
    每路: Norm -> AdaLN(由对方调制) -> MLP -> Gate -> 残差
    不是简单的信息注入，而是互相重塑对方的特征分布。
    """
    def __init__(self, channels):
        # 频域 -> 时域 调制参数 (gamma, beta, gate_scale, gate_bias)
        self.f2t_adaln = nn.Sequential(SiLU(), Conv1d(C, C*4, 1))
        # 时域 -> 频域
        self.t2f_adaln = nn.Sequential(SiLU(), Conv1d(C, C*4, 1))
        # 各自的 MLP
        self.t_mlp = MLP(C -> C*2 -> C)
        self.f_mlp = MLP(C -> C*2 -> C)
        self.t_norm = GroupNorm(1, C)
        self.f_norm = GroupNorm(1, C)

    def forward(self, t, f):
        # 频域调制时域
        γ_t, β_t, s_t, b_t = self.f2t_adaln(f).chunk(4)
        t = t + s_t.sigmoid() * (self.t_mlp(γ_t * self.t_norm(t) + β_t) + b_t)
        
        # 时域调制频域
        γ_f, β_f, s_f, b_f = self.t2f_adaln(t).chunk(4)
        f = f + s_f.sigmoid() * (self.f_mlp(γ_f * self.f_norm(f) + β_f) + b_f)
        
        return t, f
```

### 频域路径 Encoder

```python
class STFTEncoder(nn.Module):
    """频域路径: STFT -> 2D Conv 压缩 freq 维 -> 1D 特征"""
    
    def __init__(self, n_fft, hop, out_channels):
        n_bins = n_fft // 2 + 1
        # 输入: [B, 2, n_bins, T_frames] (real + imag)
        self.conv = nn.Sequential(
            Conv2d(2, 32, (3,3)),      # [B, 32, n_bins, T]
            GELU(),
            Conv2d(32, 64, (3,3), stride=(2,1)),  # freq 下采样
            GELU(),
            Conv2d(64, 128, (3,3), stride=(2,1)),
            GELU(),
            # ... 直到 freq 维度压缩到 1
            # 最终 squeeze freq 维 -> [B, C, T_frames]
        )
```

### Decoder 端对称设计

```
latent z [B, 128, T/512]
    │
    ├── 时域 decoder (V14.4, blocks[0..4] + tail)
    │   block[4] 之后 ↕ AdaLN 交互 ← 频域 decoder
    │   block[3] 之后 ↕ AdaLN 交互 ← 频域 decoder
    │   -> tail -> wav_time
    │
    ├── 频域 decoder (新增)
    │   proj -> expand freq dim -> 2D ConvTranspose
    │   -> [B, 2, n_bins, T/512] -> complex STFT -> iSTFT -> wav_freq
    │
    └── 融合: wav = wav_time + wav_freq (learned blend per frequency band)
```

### 参数量估计

| 模块 | 参数量 |
|------|--------|
| 时域 encoder (V14.4, 冻结或微调) | 56M |
| 频域 encoder (2 个 STFT + 2D Conv) | ~5M |
| DualPathInteraction × 2 | ~2M |
| 时域 decoder (V14.4) | 64M |
| 频域 decoder (iSTFT) | ~5M |
| DualPathInteraction × 2 (decoder) | ~2M |
| 总计 | ~134M (vs V14.4 121M, +13M) |

### 训练策略

1. Phase 1: 冻结时域路径（V14.4 权重），只训练频域路径 + 交互模块
2. Phase 2: 解冻全部，端到端微调，低 lr

### 风险

1. 频域路径可能被时域路径主导（gate 学到 0）→ 缓解：phase 1 冻结时域
2. 两路梯度冲突 → 缓解：AdaLN 的 gate 机制自动调节
3. 训练不稳定 → 缓解：从 V14.4 预训练权重启动

### 预期上限

Oracle 证明修复相位可达 22.3 dB。如果频域路径能把相位 R² 从 0.5 提到 0.8，
预期 SNR 可达 15-18 dB，电流音显著减轻。

---

## 方案 B：频域 AE 蒸馏

训练一个高容量频域 AE (latent_dim=512, 帧率 172Hz)，然后蒸馏到时域 encoder。
问题：频域 AE 在 128 维 43Hz 下本身做不好，teacher 不够强则 student 也不行。
变体：在 decoder 中间特征层对齐而非 latent 层，但复杂度高。

---

## 方案 C：频域注入 Encoder + 逐层相位细化 Decoder（推荐实现）

### 核心思路

结合方案 A 的 encoder 端频域注入和 decoder 端逐层相位预测。
- Encoder: 用 STFT complex 特征通过 AdaLN 注入时域 encoder 深层，让 latent 包含更多相位 hint
- Decoder: 每个上采样层输出后，额外预测该层级的相位修正，逐步从粗到细修正相位
- 最终输出: 时域波形 + 频域相位修正的融合

### 为什么这样设计

1. Encoder 端注入解决"信息来源"问题 — 让 latent 有相位信息可用
2. Decoder 端逐层细化解决"信息利用"问题 — 不指望一步到位预测相位，而是逐层从粗到细
3. 每层的相位预测只需要预测"残差"（和当前波形的相位差），比从零预测容易

### 架构总览

```
=== ENCODER ===

wav [B, 1, T]
  │
  ├─ 时域路径 (V14.4 encoder)
  │  stem -> block[0] (T/2) -> block[1] (T/8) -> block[2] (T/32)
  │                                                    │
  │  block[3] (T/128) ←── AdaLN ←── freq_enc_172Hz ───┤
  │       │                                             │
  │  block[4] (T/512) ←── AdaLN ←── freq_enc_43Hz  ────┘
  │       │
  │  proj -> [B, 256, T/512] -> bottleneck -> latent z [B, 128, T/512]
  │
  └─ 频域特征提取 (新增, 轻量)
     STFT(256, 128)  -> 2D Conv -> [B, C, T/128]  (172Hz, 给 block[3])
     STFT(1024, 512) -> 2D Conv -> [B, C, T/512]  (43Hz, 给 block[4])


=== DECODER ===

latent z [B, 128, T/512]
  │
  proj -> [B, 1024, T/512]
  │
  block[0] (stride=4) -> [B, 1024, T/128]
  │  └─ PhaseRefine: 当前特征 -> 预测 STFT(512, 128) 的 phase delta
  │                   -> iSTFT 得到 phase-corrected 信号片段
  │                   -> 混合回时域特征
  │
  block[1] (stride=4) -> [B, 512, T/32]
  │  └─ PhaseRefine: STFT(128, 32) phase delta
  │
  block[2] (stride=4) -> [B, 512, T/8]
  │
  block[3] (stride=4) -> [B, 256, T/2]
  │
  block[4] (stride=2) -> [B, 128, T]
  │  └─ PhaseRefine: STFT(2048, 512) phase delta (最终修正)
  │
  tail -> wav [B, 1, T]
```

### Encoder 频域注入模块

```python
class STFTFeatureExtractor(nn.Module):
    """从 STFT complex 系数提取特征, 压缩 freq 维到 channel 维"""
    
    def __init__(self, n_fft, hop, out_channels):
        n_bins = n_fft // 2 + 1
        # 输入: [B, 2, n_bins, T_frames] (real, imag)
        self.compress = nn.Sequential(
            nn.Conv2d(2, 32, (3,3), padding=(1,1)),
            nn.GELU(),
            nn.Conv2d(32, 64, (n_bins, 1)),  # 一次性压缩整个 freq 维
            # 输出: [B, 64, 1, T_frames] -> squeeze -> [B, 64, T_frames]
        )
        self.proj = nn.Conv1d(64, out_channels, 1)
    
    def forward(self, wav):
        stft = torch.stft(wav, n_fft, hop, return_complex=True)
        x = torch.stack([stft.real, stft.imag], dim=1)  # [B, 2, F, T]
        x = self.compress(x).squeeze(2)  # [B, 64, T]
        return self.proj(x)  # [B, out_ch, T]
```

### Decoder PhaseRefine 模块

```python
class PhaseRefine(nn.Module):
    """在 decoder 某一层级做相位修正
    
    从当前时域特征预测 STFT phase delta,
    用 iSTFT 合成修正信号, 混合回时域特征。
    """
    
    def __init__(self, in_channels, n_fft, hop):
        n_bins = n_fft // 2 + 1
        # 从时域特征预测 phase delta (cos, sin)
        self.phase_pred = nn.Sequential(
            nn.Conv1d(in_channels, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, n_bins * 2, 1),  # cos_delta, sin_delta
        )
        # 混合门控
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels + 1, in_channels, 1),  # +1 for phase-corrected signal
            nn.Sigmoid(),
        )
    
    def forward(self, feat, wav_so_far=None):
        # feat: [B, C, T_feat] 当前层的时域特征
        # 1. 从特征预测 phase delta
        phase_params = self.phase_pred(feat)  # [B, n_bins*2, T_feat]
        cos_d, sin_d = phase_params.chunk(2, dim=1)
        
        # 2. 对当前波形做 STFT, 保留 mag, 修正 phase
        if wav_so_far is not None:
            stft = torch.stft(wav_so_far, n_fft, hop, return_complex=True)
            mag = stft.abs()
            phase_orig = stft.angle()
            # phase_delta = atan2(sin_d, cos_d)
            phase_delta = torch.atan2(sin_d, cos_d)
            phase_new = phase_orig + phase_delta
            stft_new = mag * torch.exp(1j * phase_new)
            wav_refined = torch.istft(stft_new, n_fft, hop)
            
            # 3. 门控混合
            gate = self.gate(torch.cat([feat, wav_refined.unsqueeze(1)], dim=1))
            feat = feat * gate + (1 - gate) * ...  # 需要维度匹配
        
        return feat
```

### 简化版: 只在最终输出做一次相位修正

上面的逐层方案维度对齐很复杂。更实际的简化版:

```python
class WavVAE_v14_9(nn.Module):
    def forward(self, x):
        # 标准 encode-decode
        z, mean, std = self.encode(x)
        x_coarse = self.decode(z, output_length=x.shape[-1])
        
        # 相位修正 (在最终波形上)
        x_refined = self.phase_refiner(x_coarse, z)
        
        return {"x_hat": x_refined, ...}

class PhaseRefiner(nn.Module):
    """从粗波形 + latent 预测相位修正"""
    
    def __init__(self, latent_dim=128, n_fft=2048, hop=512):
        n_bins = n_fft // 2 + 1
        # 从 latent 预测每帧的 phase correction
        self.phase_net = nn.Sequential(
            nn.Conv1d(latent_dim, 512, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(512, n_bins * 2, 1),  # cos_delta, sin_delta per bin
        )
    
    def forward(self, x_coarse, z):
        # x_coarse: [B, 1, T] 粗波形
        # z: [B, 128, T/512] latent
        
        # 从 latent 预测 phase delta
        phase_params = self.phase_net(z)  # [B, n_bins*2, T/512]
        cos_d, sin_d = phase_params.chunk(2, dim=1)
        phase_delta = torch.atan2(sin_d, cos_d)  # [B, n_bins, T_frames]
        
        # 对粗波形做 STFT
        stft = torch.stft(x_coarse.squeeze(1), n_fft, hop, return_complex=True)
        mag = stft.abs()
        phase = stft.angle()
        
        # 修正相位
        phase_new = phase + phase_delta[:, :, :stft.shape[-1]]
        stft_new = mag * torch.exp(1j * phase_new)
        
        # iSTFT
        x_refined = torch.istft(stft_new, n_fft, hop, length=x_coarse.shape[-1])
        return x_refined.unsqueeze(1)
```

### 推荐实现路径

**先做简化版（只在最终输出修正），验证相位修正是否有效。**
如果有效，再加 encoder 端频域注入。

简化版改动极小:
- 新增 PhaseRefiner 模块 (~3M 参数)
- WavVAE.forward 最后加一步 phase refinement
- 冻结整个 AE，只训练 PhaseRefiner
- Loss: complex STFT loss (直接约束修正后的 real+imag)

### 参数量

| 模块 | 参数 |
|------|------|
| PhaseRefiner (简化版) | ~3M |
| STFTFeatureExtractor × 2 (encoder 注入) | ~2M |
| DualPathInteraction × 2 | ~2M |
| 总新增 | ~7M |

### 训练策略

Phase 1: 冻结 AE，只训练 PhaseRefiner，用 complex STFT loss
Phase 2: 如果 Phase 1 有效，加 encoder 频域注入，解冻全部微调

### 预期

简化版: 如果 phase delta 预测准确度 50%，SNR 从 16.5 → 19 dB
完整版: 如果 encoder 注入提升 latent 相位 R² 到 0.7，SNR 可达 20+ dB
