# V14.6 Changelog

## 目标
突破 V14.4 的 11.36 dB SNR 瓶颈，重点提升 4k-11kHz 高频重建质量。

## 改动

### 1. Multi-band STFT Loss (高频加权)
- `losses_stft.py`: `MultiBandSTFTLoss` 新增 `band_weights` 参数
- 5 个频段: [0-500, 500-1500, 1500-3500, 3500-6000, 6000-11025] Hz
- 权重: [1.0, 1.0, 1.5, 2.5, 3.0] — 高频 band 给 2.5-3x 权重
- `MultiResolutionMultiBandSTFTLoss` 透传 `band_weights`
- SC loss 分母 `.detach()` 保持一致

### 2. Sub-band STFT Discriminator (高频专注)
- `discriminator.py`: 新增 `SubBandSTFTDiscriminator` + `MultiScaleSubBandDiscriminator`
- 只截取 STFT 的 4k-11kHz bins 做 2D Conv 判别
- 和全频段 MS-STFT 判别器互补: 全频段粗粒度 + 窄频段细粒度
- 2 个分辨率: FFT 1024 + 2048

### 3. 训练集成
- `losses.py`: `compute_reconstruction_loss` 新增 `multiband_stft_loss_fn` 参数
- `train.py`: 集成 multi-band STFT loss + sub-band discriminator
  - Sub-band disc 参数和 MS-STFT disc 共享 optimizer
  - 独立的 adv_g_weight 和 fm_weight 控制
  - Checkpoint 保存/恢复 sub-band disc 状态

### 4. 架构回退到 V14.4
- `autoencoder.py`: 移除 V14.5 的 AvgPool skip connections
- 恢复纯 sequential encoder/decoder (DAC-style)

## 配置
- `conf/experiment/v14_6_full.yaml`
- 从 V14.4 best checkpoint 继续训练
- Phase 1 (epoch 0-4): 只用 reconstruction loss + multi-band STFT
- Phase 2 (epoch 5+): 加入 MS-STFT disc + sub-band disc
- Disc warmup: 3 epochs (冻结 AE, 只训练 D)

## 设计思路
V14.4 频段分析显示 4k-11kHz SNR 只有 6-7 dB (vs 低频 25+ dB)。
两个互补策略:
1. Multi-band STFT loss: 让 reconstruction loss 本身更关注高频
2. Sub-band discriminator: 给高频额外的对抗梯度信号
