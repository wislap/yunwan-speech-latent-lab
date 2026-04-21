# V14.2 变更记录

## 相对 V14.1 的改动

### 架构

| 项目 | V14.1 | V14.2 | 原因 |
|------|-------|-------|------|
| latent_dim | 256 | **128** | V14.1 有效秩仅 41.6，256 维严重过大 |
| strides | [7,7,9]=441 | **[2,4,4,4,4]=512** | 小 stride × 多 block，每步压缩更温和 |
| blocks | 3 | **5** | 更深的层级结构，更大感受野 |
| channels | [128,256,512,1024] | **[128,256,512,512,1024,1024]** | 匹配 5 block |
| params | 55M | **113M** | 接近 LongCat 的 157M |
| fps | 50 Hz | **43.1 Hz** | 更低帧率，对下游 DiT 更友好 |
| max stride | 9 | **4** | 最大单步压缩从 9x 降到 4x |
| Shortcut | 1x1 WNConv (有参数) | **无参数** (channel avg/replication) | 对齐 LongCat 论文 |

### 损失函数

| 项目 | V14.1 | V14.2 | 原因 |
|------|-------|-------|------|
| l1_weight | 1.0 | **10.0** | 让 time-domain L1 主导 |
| l2_weight | — | **5.0** | 新增 L2 loss，惩罚大误差 |
| stft_weight | 0.5 | 0.5 | 不变 |
| mel_weight | 1.0 | **0.1** | mel 之前占 89% total loss，降到 ~15% |
| energy_loss | 有 | **移除** | 是 torchaudio bug 的 workaround，不再需要 |
| grad_clip | 0.5 | **1.0** | gnorm 降低后可以保留更多梯度 |

### 基础设施

- losses.py: 纯 PyTorch mel filterbank，移除 torchaudio 依赖
- train.py: 保存/恢复 scheduler 状态到 checkpoint
- train.py: segment_length 自动对齐 total_stride
- train.py: AudioDataset crop start 对齐 stride

## 收敛加速分析

V14.2 在 epoch 4 就达到 V14.1 epoch 9 的水平：

| 指标 | V14.1 epoch 9 | V14.2 epoch 4 | V14.2 epoch 9 |
|------|--------------|--------------|--------------|
| SNR | 5.16 dB | 6.76 dB | 9.88 dB |
| avg_G | 0.685 | 0.718 | 0.571 |

三个因素：
1. **小 stride 保留更多信息**：max_stride 从 9 降到 4，encoder 每步丢失更少
2. **更多参数**：113M vs 55M，模型容量翻倍
3. **L1+L2 主导 loss**：直接优化波形精度，不被 mel 包络主导
