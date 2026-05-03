# V14 — Oobleck-style Wav-VAE 初始版本

## 架构

- Encoder: Conv1d stem → 3x OobleckEncoderBlock → Conv1d proj
- Decoder: 镜像结构，转置卷积上采样，Tanh 输出
- Bottleneck: softplus std + free-bits KL
- Activation: SnakeBeta (log-scale alpha+beta)
- Normalization: weight_norm，无 LayerNorm
- Shortcut: pixel_unshuffle/shuffle + adaptive_avg_pool1d
- LayerScale init=1e-2

## 配置

- latent_dim=256, strides=[7,7,9], total_stride=441, ~50fps@22050Hz
- encoder_channels=[128, 256, 512, 1024]
- 参数量: ~43M

## 过拟合实验汇总

| 实验 | dim | blocks | steps | SNR | enc_std | 问题 |
|------|-----|--------|-------|-----|---------|------|
| overfit4 | 64 | 3 | 3k | +4.22 | 0.0011 | 基线 |
| overfit5 | 64 | 4 | 3k | -19.63 | 1.06 | 4-block 崩溃 |
| overfit7 | 64 | 4+LS | 3k | +4.06 | 0.0005 | LayerScale 救回 |
| overfit8 | 256 | 3 | 3k | +1.66 | 0.0017 | dim=256 收敛慢 |
| overfit9 | 256 | 3 | 10k | +8.09 | 0.0007 | 最佳，还在涨 |

## 关键发现

1. dim=64 天花板低（有效秩=131，只覆盖 93% 方差）
2. 4-block 不加 LayerScale 会在 step 800 崩溃
3. gnorm 稳定在 ~100，被 clip 掉 99%，优化效率极低
4. softplus > exp(log_var)，后者导致 NaN

## 遗留问题

- 梯度爆炸（STFT loss 分母反传）
- Tanh 压缩动态范围
- Shortcut 的 adaptive_avg_pool 语义不清 + CUDA 警告
