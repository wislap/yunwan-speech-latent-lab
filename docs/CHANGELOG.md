# Wav-VAE Changelog

## V14.1 (2026-04-20) — 收敛加速 + 架构修复

### 改动

**架构**
- Decoder 去掉 `nn.Tanh()` 输出激活，避免大振幅梯度消失
- Shortcut 从 `adaptive_avg_pool1d`（无参数，语义错误）改为 `WNConv1d 1x1`（参数化投影）
  - 修复了 CUDA shmem 警告
  - 参数量 43M → 55M（shortcut 贡献 ~12M）

**损失函数**
- STFT spectral convergence 分母 detach：`‖y-x‖/‖y‖.detach()` → gnorm 从 ~100 降到 ~10
- `stft_weight` 默认从 1.0 降到 0.5

**Bottleneck**
- V14.1a: KL warmup（0 → kl_weight 线性升温 2000 步），kl_weight 1e-2，free_bits 0.1
- V14.1b: 替换 KL 为 soft range penalty（mean ∈ [-3,3]，std ∈ [0.1,1.5]），warmup 1000 步

**训练**
- LR warmup 500 步 + cosine decay，peak_lr 3e-4（原 1e-4 固定）
- `grad_clip` 从 1.0 降到 0.5

### 实验结果（单条 LJ001-0001 过拟合，dim=256，10000 步）

| 版本 | SNR@3k | SNR@10k | gnorm | enc_std | 耗时 |
|------|--------|---------|-------|---------|------|
| V14 baseline | +4.22 | +8.09 | ~100 | 0.0007 | 673s |
| V14.1a (KL warmup) | +10.72 | +16.81 | ~10 | 0.0082 | 661s |
| V14.1b (soft range) | ~10.7 | TBD | ~10 | ~0.001 | TBD |

### 关键发现
1. **STFT detach 是最大贡献**：gnorm 100→10，等效学习率提升 ~10x
2. **去 Tanh 解放 decoder**：训练初期 decoder 输出偏大时不再梯度消失
3. **LR warmup 稳定初期**：避免随机参数阶段的大梯度震荡
4. **后验坍缩是结构性的**：无论 KL 还是 soft range，enc_std 都趋近 0。单条音频有效秩 ~131，dim=256 远超信号复杂度，大量维度天然冗余，VAE 约束与信号结构不匹配

### 待定决策
- [ ] Bottleneck 策略：纯 AE + 后处理归一化 vs 继续调 soft range
- [ ] 全数据集训练时机

---

## V14 (2026-04-20) — 初始版本

Oobleck-style Wav-VAE，对齐 LongCat-AudioDiT 架构。

- SnakeBeta activation（log-scale alpha+beta）
- pixel_unshuffle/shuffle shortcuts
- softplus VAE bottleneck + free-bits KL
- weight_norm，无 LayerNorm/BatchNorm
- LayerScale init=1e-2
- strides=[7,7,9], total_stride=441, ~50fps@22050Hz
- encoder_channels=[128,256,512,1024], ~43M params
