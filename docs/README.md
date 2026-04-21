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
└── v14.2/                  # V14.2 架构重设计
    ├── changelog.md        # 改动对比 + 收敛分析
    └── experiments.md      # E08 全数据集训练记录
```

## 版本线

| 版本 | 日期 | 状态 | 核心变化 | Best SNR |
|------|------|------|----------|----------|
| V14 | 2026-04-20 | 完成 | Oobleck-style Wav-VAE 初始实现 | +8.09 dB (overfit) |
| V14.1 | 2026-04-20 | 完成 | STFT detach + 去 Tanh + soft range bottleneck | +5.16 dB (full) |
| V14.2 | 2026-04-21 | **进行中** | dim=128, 5-block, 无参数 shortcut, L1+L2 主导 | **+11.96 dB (full)** |
