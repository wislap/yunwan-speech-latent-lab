# V14.7g: Hybrid Decoder — 时域 block[0] + Vocos iSTFT Head

## 核心发现

Decoder 各层的相位信息量（线性探针 R²）：

| 层 | 通道 | 帧率 | 相位 R² |
|----|------|------|---------|
| proj output | 1024 | 43Hz | 1.000 |
| block[0] output | 1024 | 172Hz | 0.958 |
| block[1] output | 512 | 690Hz | 0.242 |
| block[4] output | 128 | 22kHz | 0.038 |

相位信息在 block[1] 之后急剧丢失。block[0] 之后还有 96% 的相位信息。

## 架构

```
Encoder (dim=256, from V14.7e):
  wav → 5x EncoderBlock → proj → z [B, 256, T/512] at 43Hz

Decoder:
  z → proj(256→2048) → block[0](2048→1024, stride=4) → features [B, 1024, T/128] at 172Hz
                                                              ↓
                                                    Vocos Head (ConvNeXt + iSTFT)
                                                              ↓
                                                        wav [B, 1, T]
```

## Vocos Head 设计

输入: [B, 1024, T_feat] at 172Hz
STFT: n_fft=512, hop=128, n_bins=257, 帧率=172Hz (完美对齐)

```
input_proj: Conv1d(1024, 512)
4x ConvNeXt Block (hidden=512, intermediate=1024, kernel=7)
norm: LayerNorm(512)
output_proj: Conv1d(512, 257*2)  → mag_param + phase_param
  mag = softplus(mag_param)
  phase = atan2(sin(p), cos(p))
  stft = mag * (cos(phase) + j*sin(phase))
  wav = iSTFT(stft, n_fft=512, hop=128)
```

## 参数量

| 模块 | 参数 |
|------|------|
| Encoder (dim=256, wider) | 67M |
| Decoder proj + block[0] | ~30M |
| Vocos Head (512d, 4 layers) | ~5M |
| 总计 | ~102M |

## 训练策略

Phase 1: 冻结 encoder + decoder proj + block[0]，只训练 Vocos Head
  - 从 V14.7e checkpoint 加载 encoder + decoder.proj + decoder.blocks[0]
  - Vocos Head 从零训练
  - lr=3e-4, 30 epochs
  - 如果 Vocos Head 能学动（SNR > 5 dB），进入 Phase 2

Phase 2: 解冻全部，端到端微调
  - lr=1e-5 (encoder), lr=1e-4 (Vocos Head)
  - 50 epochs

## 为什么这次能成功

1. Vocos Head 的输入是 block[0] 特征（R²=0.96），不是抽象 latent（R²=0.5）
2. 172Hz 帧率和 STFT(512,128) 完美对齐，不需要上/下采样
3. 1024→514 的映射（2:1 压缩），比 256→1026（1:4 扩展）容易得多
4. block[0] 已经把抽象 latent 展开成具象特征，Vocos 只需要做频域投影
5. softplus + 小初始化防止爆炸

## 风险

1. block[0] 的特征是为时域 decoder 优化的，可能不完全适合频域输出
   → 缓解: Phase 2 端到端微调
2. n_fft=512 的频率分辨率 43Hz/bin，可能不够精细
   → 可以后续试 n_fft=1024, hop=256 (需要 2x 上采样)
3. 4 层 ConvNeXt 可能不够
   → 可以增加到 6-8 层
