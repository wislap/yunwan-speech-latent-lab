# Wav-VAE 项目文档

## 目录结构

```
docs/
├── README.md               # 本文件
├── project_layout.md       # 本地/远程目录说明 + 工作流
├── v14/                    # V14 初始版本
│   └── notes.md
└── v14.1/                  # V14.1 收敛加速 + 架构修复
    ├── changelog.md        # 变更记录
    ├── experiments.md      # 实验记录
    └── batch_size_scan.md  # 显存扫描结果
```

## 版本线

| 版本 | 日期 | 状态 | 核心变化 |
|------|------|------|----------|
| V14 | 2026-04-20 | 完成 | Oobleck-style Wav-VAE 初始实现 |
| V14.1 | 2026-04-20 | 进行中 | STFT detach + 去 Tanh + soft range bottleneck |
