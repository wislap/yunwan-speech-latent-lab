# Wav-VAE 项目文档

## 目录结构

```
docs/
├── README.md               # 本文件
├── project_layout.md       # 本地/远程目录说明 + 工作流
├── v14/                    # V14 初始版本
│   └── notes.md
├── v14.1/                  # V14.1 收敛加速 + 架构修复
│   ├── changelog.md
│   ├── experiments.md
│   └── batch_size_scan.md
├── v14.2/                  # V14.2 架构重设计
│   ├── changelog.md
│   └── experiments.md
├── v14.3/                  # V14.3 sub-pixel decoder (失败实验)
│   ├── changelog.md
│   └── experiments.md
├── v14.4/                  # V14.4 去掉 shortcut (DAC-style)
│   └── changelog.md
├── v16.1/                  # V16.1 high-fidelity 128x baseline
│   ├── hf_time.md
│   └── report.md
├── v15/                    # V15 dual-path 实验状态
└── v16.2/                  # V16.2 256x 边界实验与诊断图表
    ├── 256x_time.md
    ├── diagnostics.md
    └── figures/
```

## 版本线

| 版本 | 日期 | 状态 | 核心变化 | Best SNR |
|------|------|------|----------|----------|
| V14 | 04-20 | 完成 | Oobleck-style Wav-VAE 初始实现 | +8.09 dB (overfit) |
| V14.1 | 04-20 | 完成 | STFT detach + 去 Tanh + soft range bottleneck | +5.16 dB (full) |
| V14.2 | 04-21 | 完成 | dim=128, 5-block, 无参数 shortcut, L1+L2 主导 | +11.96 dB (full) |
| V14.3 | 04-21 | 完成 | sub-pixel decoder, 冻结 encoder distillation | +11.61 dB (full) |
| V14.4 | 04-21 | **训练中** | 去掉所有 shortcut (DAC-style) | +37.76 dB (overfit) |

## 关键发现时间线

1. **STFT loss detach** (V14.1) — gnorm 100→10，收敛加速 10x
2. **torchaudio.load 静默失败** (V14.1 full) — 全零训练数据，根因是缺 ffmpeg
3. **dim=256 过大** (V14.1 分析) — 有效秩仅 41.6，dim=128 覆盖 99.5%
4. **大 stride 丢信息** (V14.2) — strides [7,7,9]→[2,4,4,4,4]，SNR 5→12 dB
5. **2-6 kHz 盲区** (V14.2 分析) — encoder block 1 Nyquist=1378 Hz
6. **Shortcut 是 aliasing 根因** (V14.3 分析) — pixel_unshuffle 无低通滤波，污染 latent
7. **去掉 shortcut** (V14.4) — overfit SNR 16→37 dB，收敛速度翻倍
