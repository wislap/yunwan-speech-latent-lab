# 实验记录

## 实验索引

| ID | 日期 | 版本 | 类型 | 配置摘要 | 结果 | 日志 |
|----|------|------|------|----------|------|------|
| E01 | 04-20 | V14 | overfit | dim=64, 3block, 3k steps | SNR +4.22dB | v14_overfit4.log |
| E02 | 04-20 | V14 | overfit | dim=64, 4block, 3k steps | SNR -19.63dB (崩溃) | v14_overfit5_4block.log |
| E03 | 04-20 | V14 | overfit | dim=64, 4block+LS, 3k steps | SNR +4.06dB | v14_overfit7_4block_ls1e2_accum4.log |
| E04 | 04-20 | V14 | overfit | dim=256, 3block, 3k steps | SNR +1.66dB | v14_overfit8_dim256.log |
| E05 | 04-20 | V14 | overfit | dim=256, 3block, 10k steps | SNR +8.09dB | v14_overfit9_dim256_10k.log |
| E06 | 04-20 | V14.1a | overfit | KL warmup, detach, no Tanh | SNR +16.81dB | v14_1_overfit_test1.log |
| E07 | 04-20 | V14.1b | overfit | soft range penalty | TBD | v14_1_overfit_softrange.log |

## 详细记录

### E05: V14 baseline (dim=256, 10k steps)
- **目的**: 确认 dim=256 的上限
- **配置**: latent_dim=256, strides=[7,7,9], kl_weight=1e-4, free_bits=0.25, lr=1e-4 固定
- **结果**: SNR +8.09dB, enc_std=0.0007, gnorm~100
- **问题**: 梯度被 clip 掉 99%，后验完全坍缩
- **结论**: 架构能力足够，优化效率极低

### E06: V14.1a KL warmup + 架构修复
- **目的**: 验证 STFT detach + 去 Tanh + LR warmup 的效果
- **配置**: kl_weight=1e-2, free_bits=0.1, kl_warmup=2000, stft_weight=0.5, grad_clip=0.5, peak_lr=3e-4
- **结果**: SNR +16.81dB (+8.72dB vs E05), gnorm~10, enc_std=0.0082
- **关键**: 1000 步就超过 E05 的 10000 步结果
- **问题**: enc_std 仍然很低，后验坍缩未根本解决

### E07: V14.1b soft range penalty
- **目的**: 用 soft range penalty 替代 KL，看能否推高 enc_std
- **配置**: reg_weight=1e-2, margin_mean=3.0, std_range=[0.1, 1.5], warmup=1000
- **结果**: 进行中
- **初步观察**: enc_std ~0.001，penalty 太弱未起作用

## 关键洞察

1. **有效秩限制**: 单条音频有效秩=131，dim=256 中大量维度冗余，VAE 约束与信号结构不匹配
2. **STFT loss 梯度**: spectral convergence 的分母反传是梯度爆炸的根源，detach 后优化效率提升 10x
3. **Tanh 有害**: 压缩动态范围 + 边界梯度消失，对音频重建无益
4. **LongCat 论文验证**: VAE 重建好 ≠ TTS 好，latent dim 大 → DiT 难建模，低帧率对 DiT 友好
