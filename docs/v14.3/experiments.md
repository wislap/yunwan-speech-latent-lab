# V14.3 实验记录

## E09: V14.3 decoder distillation

**配置**
- V14.2 encoder 冻结，新 sub-pixel decoder (Conv k=3 + pixel_shuffle)
- latent_dim=128, strides=[2,4,4,4,4]=512
- Pre-extracted latent dataset (mmap), 12000 samples/epoch
- bs=12, lr=3e-4, 30 epochs

**结果**: Best SNR 11.61 dB (epoch 14)，2-6 kHz 盲区未修复

## E10: 频率响应分析

正弦波扫频确认 V14.3 和 V14.2 的盲区 pattern 完全一致，证明问题在 encoder 而非 decoder。

## E11: Encoder Nyquist 链分析

逐层追踪 2kHz 正弦波在 encoder 中的传播，发现 block 1 (stride=4) 的 Nyquist=1378 Hz 导致 2-6 kHz 信号被折叠。

进一步发现 shortcut 的 pixel_unshuffle 是 aliasing 的主要来源（无低通滤波），而 main path 的 strided conv (kernel=2×stride) 至少有部分 anti-aliasing 效果。
