# V14.4 变更记录

## 目标

修复 2-6 kHz 频率盲区。根因：encoder 的 DownsampleShortcut (pixel_unshuffle) 引入 aliasing 污染 latent。

## 架构改动

| 组件 | V14.2/V14.3 | V14.4 | 原因 |
|------|-------------|-------|------|
| Encoder block | OobleckEncoderBlock (ResUnits + StridedConv + **Shortcut**) | **EncoderBlock** (ResUnits + StridedConv, **无 shortcut**) | 消除 pixel_unshuffle aliasing |
| Decoder block | OobleckDecoderBlock (SubPixelConv + ResUnits + **Shortcut**) | **DecoderBlock** (SubPixelConv + ResUnits, **无 shortcut**) | 对称设计 |
| 设计参考 | LongCat-AudioDiT (有 shortcut) | **DAC** (无 shortcut) | DAC 无盲区 |

其余不变：SnakeBeta, weight_norm, LayerScale, dilations=[1,3,9], sub-pixel decoder

## 参数量

| | V14.2 | V14.3 | V14.4 |
|---|---|---|---|
| Encoder | 56.4M | 56.4M | **56.4M** |
| Decoder | 56.6M | 64.5M | **64.5M** |
| Total | 113M | 121M | **121M** |

Encoder 参数量不变（shortcut 是无参数的），decoder 保持 V14.3 的 sub-pixel 设计。

## 为什么去掉 shortcut 是安全的

1. **DAC 验证**: DAC 用 strides=[2,4,8,8]，无 shortcut，无 LayerScale，训练稳定
2. **我们的 stride 更小**: max_stride=4 (vs DAC 的 8)，梯度传播更容易
3. **LayerScale 保留**: init=0.01 的 LayerScale 已经提供了训练稳定性
4. **Overfit 验证**: 2000 步 SNR +37.76 dB，无 NaN，无崩溃

## Overfit 对比

| 版本 | 步数 | SNR | 有 shortcut |
|------|------|-----|------------|
| V14.1 (dim=256, stride=441) | 10000 | +16.81 dB | ✅ |
| V14.2 (dim=128, stride=512) | 10000 | ~+16 dB (推算) | ✅ |
| **V14.4 (dim=128, stride=512)** | **2000** | **+37.76 dB** | **❌** |

去掉 shortcut 后收敛速度和上限都大幅提升。原因：
- shortcut 主导信号流（std > main path），DilatedResUnit 的非线性学习被压制
- 去掉后 main path 必须独立学习，DilatedResUnit 充分发挥作用

## 全量训练

- 配置: v14_4_full.yaml, 50 epochs, bs=4, grad_accum=4
- 服务器: westd (RTX 5090 32GB)
- 状态: 进行中
- 预计: ~7.5 小时

## 待验证

- [ ] 全量训练 SNR 是否超过 V14.2 的 11.96 dB
- [ ] 2-6 kHz 盲区是否修复（正弦波扫频 + 频段 SNR）
- [ ] 训练稳定性（50 epoch 无崩溃）
- [ ] Latent space 分析（有效秩、维度利用率）
