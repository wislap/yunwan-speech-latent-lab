# V14.3 变更记录

## 目标

修复 V14.2 的 2-6 kHz 频率盲区。初始假设：ConvTranspose1d 的 mirrored aliasing 导致 decoder 无法重建这些频段。

## 架构改动

| 组件 | V14.2 | V14.3 | 原因 |
|------|-------|-------|------|
| Decoder 上采样 | ConvTranspose1d(k=2s, s=s) | Conv1d(k=3) + pixel_shuffle | 消除 ConvTranspose 的 mirrored aliasing |
| Encoder | 不变 | **冻结 V14.2 权重** | 只训练 decoder |
| Shortcut | 保留 | 保留 | 未识别为问题 |
| Decoder params | 57M | 65M | sub-pixel conv channel 膨胀 |

## 训练方式

- **Distillation**: 加载 V14.2 encoder 权重，冻结 encoder+bottleneck
- **Pre-extracted latents**: 27965 segments → packed mmap (5.7GB audio + 0.7GB latent)
- bs=12, grad_accum=1, 12000 samples/epoch, ~4.8 min/epoch

## 结果

| Epoch | avg_G | Eval SNR |
|-------|-------|----------|
| 4 | 0.497 | 10.98 dB |
| 9 | 0.445 | 11.18 dB |
| 14 | 0.431 | **11.61 dB** |

### 频率响应（正弦波扫频，epoch 14）

| 频率 | V14.2 SNR | V14.3 SNR | 变化 |
|------|-----------|-----------|------|
| 200 Hz | 20.57 | 21.42 | +0.9 |
| 500 Hz | 30.66 | 34.93 | +4.3 |
| 2000 Hz | -0.13 | **-0.02** | 无改善 |
| 4000 Hz | -0.45 | **-0.45** | 完全一样 |
| 7000 Hz | 23.10 | 21.85 | -1.3 |

## 关键发现

**Sub-pixel decoder 没有修复 2-6 kHz 盲区。** 频率响应 pattern 和 V14.2 完全一致。

这证明了：
1. 问题不在 decoder 的 ConvTranspose — 换成 pixel_shuffle 后盲区不变
2. 问题在 **encoder** — V14.3 冻结了 V14.2 的 encoder，盲区当然一样
3. 初始假设（decoder aliasing）是错的

## 根因追踪

通过逐层分析 encoder 的 Nyquist 频率链：

```
stem:    22050 Hz, Nyquist = 11025 Hz  ✅
block 0 (s=2): 11025 Hz, Nyquist = 5512 Hz  ✅
block 1 (s=4):  2756 Hz, Nyquist = 1378 Hz  ❌ 2-6kHz 在此被折叠
block 2 (s=4):   689 Hz, Nyquist =  345 Hz
...
```

Block 1 的 stride=4 把有效采样率从 11025 降到 2756 Hz。2000-5512 Hz 的信号超过了新的 Nyquist 1378 Hz，被 aliasing 折叠。

但 DAC 也有同样的 Nyquist 问题（strides=[2,4,8,8]），为什么 DAC 没有盲区？

**因为 DAC 没有 shortcut。** 我们的 DownsampleShortcut 用 pixel_unshuffle 做下采样，这是纯粹的空间折叠，没有任何低通滤波。shortcut 的 aliased 信号和 main path 的 partially-filtered 信号相加后，污染了 latent。

而且 LayerScale init=0.01 让 main path 贡献很小，shortcut 主导了信号流。
