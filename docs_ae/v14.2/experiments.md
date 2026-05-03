# V14.2 实验记录

## E08: V14.2 全数据集训练

**配置**
- latent_dim=128, strides=[2,4,4,4,4]=512, 43.1 Hz
- encoder_channels=[128,256,512,512,1024,1024], 113M params
- 无参数 shortcut (channel averaging / replication)
- L1×10 + L2×5 + STFT×0.5 + Mel×0.1
- batch_size=4, grad_accum=4, effective_batch=16
- lr=3e-4, warmup=500 steps, cosine decay
- grad_clip=1.0, bf16 autocast
- segment_length=51200 (512×100, stride-aligned)
- adv_enable=false
- 数据: LJSpeech 11823 samples, 22050 Hz

**训练进程**
- 每 epoch: 2955 steps, ~539 秒 (~9 分钟)
- 总计跑了 39 epoch (约 5.8 小时)，因服务器定时关机中断
- 零崩溃，零 NaN，零 spike

**结果**

| Epoch | avg_G | Eval SNR | STFT loss | 状态 |
|-------|-------|----------|-----------|------|
| 0 | 2.137 | — | — | |
| 4 | 0.718 | 6.76 dB | 1.037 | new best |
| 9 | 0.571 | 9.88 dB | 0.859 | new best |
| 14 | 0.513 | 10.17 dB | 0.790 | new best |
| 19 | 0.482 | 9.98 dB | 0.744 | |
| 24 | 0.463 | 11.74 dB | 0.714 | new best |
| 29 | 0.451 | 11.32 dB | 0.678 | |
| 34 | 0.445 | 11.96 dB | 0.669 | **best** |
| 38 | 0.438 | — | — | 中断 |

**观察**
1. Loss 持续下降但速度放缓（epoch 30+ 每 epoch 降 ~0.003）
2. SNR 在 epoch 19 和 29 有小幅回落，可能是 eval batch 的随机性
3. Best SNR 11.96 dB at epoch 34，还有上升空间
4. STFT loss 从 1.037 降到 0.669，下降 35%

**对比 V14.1**

| 指标 | V14.1 best | V14.2 best | 提升 |
|------|-----------|-----------|------|
| SNR | 5.16 dB | **11.96 dB** | **+6.8 dB** |
| Epoch | 9 | 34 | |
| STFT | 0.939 | 0.669 | -29% |
| 参数量 | 55M | 113M | +105% |
| 帧率 | 50 Hz | 43.1 Hz | -14% |

## 待做

- [ ] 在新服务器上恢复训练，跑完 50 epoch
- [ ] 对 V14.2 best checkpoint 做 latent space 分析（PCA/t-SNE）
- [ ] 对比 V14.1 和 V14.2 的有效秩变化
- [ ] 听感评估：导出重建音频对比
- [ ] 考虑是否需要判别器（SNR 12+ 后加入可能有帮助）
