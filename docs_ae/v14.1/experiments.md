# V14.1 实验记录

## 实验索引

| ID | 类型 | 配置 | SNR@3k | SNR@10k | gnorm | 日志 (远程) |
|----|------|------|--------|---------|-------|-------------|
| E06 | overfit | KL warmup, detach, no Tanh | +10.72 | +16.81 | ~10 | v14_1_overfit_test1.log |
| E07 | overfit | soft range penalty | ~10.7 | TBD | ~10 | v14_1_overfit_softrange.log |

## E06: KL warmup + 架构修复

**配置**
- latent_dim=256, strides=[7,7,9]
- kl_weight=1e-2, free_bits=0.1, kl_warmup=2000
- stft_weight=0.5, grad_clip=0.5, peak_lr=3e-4, warmup=500

**结果**
- SNR: -0.06 → +8.09 (1000步) → +16.81 (10000步)
- gnorm: 12→10 稳定
- enc_std: 0.69 (初始) → 0.008 (收敛后)
- 耗时: 661s

**对比 V14 baseline (E05)**
- SNR +16.81 vs +8.09 (+8.72 dB)
- 到 +8dB 的步数: 900 vs 10000 (11x 加速)

## E07: soft range penalty

**配置**
- 同 E06，但 bottleneck 改为 soft range penalty
- reg_weight=1e-2, margin_mean=3.0, std_range=[0.1, 1.5], warmup=1000

**初步结果 (进行中)**
- SNR 走势与 E06 一致
- enc_std ~0.001，penalty 太弱未起作用 (reg_loss ≈ 0.0001，被重建 loss 淹没)
- 结论: 单条过拟合中 bottleneck 策略对重建质量无影响
