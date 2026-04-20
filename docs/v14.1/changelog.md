# V14.1 变更记录

## 架构改动

| 组件 | V14 | V14.1 | 原因 |
|------|-----|-------|------|
| Decoder 输出 | Tanh | 无激活 | Tanh 在边界梯度消失 |
| Shortcut | adaptive_avg_pool1d | WNConv1d 1x1 | 修复语义错误 + CUDA 警告 |
| STFT loss 分母 | 参与反传 | detach | gnorm 100→10 |
| stft_weight | 1.0 | 0.5 | 配合 detach 调整 |
| grad_clip | 1.0 | 0.5 | 更保守的梯度截断 |
| LR schedule | 固定 1e-4 | warmup 500 步 + cosine, peak 3e-4 | 初期稳定 + 中期加速 |
| Bottleneck | KL (kl_weight=1e-4, free_bits=0.25) | soft range penalty | KL 与低有效秩不匹配 |
| 参数量 | ~43M | ~55M | shortcut 1x1 conv 贡献 ~12M |

## Bottleneck 演进

1. **V14.1a**: KL warmup (0→1e-2 over 2000 steps), free_bits=0.1
2. **V14.1b**: soft range penalty (mean∈[-3,3], std∈[0.1,1.5]), warmup 1000 steps

两者重建质量几乎一致，enc_std 在单条过拟合中都趋近零（正常行为：单样本不需要方差）。

## 收敛加速分析

V14.1 在 1000 步超过 V14 的 10000 步，三个因素：

1. **STFT detach**（最大）: 有效学习率提升 ~10x
2. **去 Tanh**: 大振幅区域梯度畅通
3. **LR warmup**: 初期稳定，中期 lr 比 V14 高 3x
