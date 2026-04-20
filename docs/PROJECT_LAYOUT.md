# 项目布局

## 本地工作区 (playground3)

```
playground3/
├── autoencoder/                    # 主代码包
│   ├── __init__.py
│   ├── models/
│   │   ├── autoencoder.py          # WavVAE 主模型（encoder/decoder/bottleneck）
│   │   ├── discriminator.py        # MS-STFT + MSD 判别器
│   │   └── discriminator_v14.py    # V14 判别器封装
│   ├── losses.py                   # MultiScaleMelLoss + 聚合函数
│   ├── losses_stft.py              # MultiResolutionSTFTLoss
│   ├── train.py                    # Hydra 训练脚本（全数据集）
│   ├── extract_latents.py          # 潜表示导出工具
│   ├── scripts/
│   │   └── overfit.py              # 单条过拟合测试（快速验证）
│   └── conf/                       # Hydra 配置
│       ├── config.yaml             # 顶层配置（defaults）
│       ├── experiment/
│       │   └── wav_vae.yaml        # 实验超参
│       └── runtime/
│           ├── remote.yaml         # 远程服务器路径
│           └── local.yaml          # 本地路径
├── docs/                           # 文档
│   ├── CHANGELOG.md                # 版本变更记录
│   ├── EXPERIMENTS.md              # 实验记录与分析
│   └── PROJECT_LAYOUT.md           # 本文件
├── .gitignore
└── .kiro/
    └── steering/
        └── project-context.md      # 项目上下文（Kiro steering）
```

## 远程服务器

```
SSH: ssh -p 21466 root@connect.westc.seetacloud.com
GPU: RTX 5090 32GB
Python: /root/miniconda3/bin/python (3.12)

/root/autodl-tmp/project/
├── autoencoder/                    # ← 从本地 rsync 同步
├── data/
│   ├── LJSpeech-1.1/wavs/         # 原始音频
│   └── ljspeech_train_splits.json  # 训练数据索引
├── outputs/
│   ├── v14_overfit*.log            # V14 实验日志
│   ├── v14_1_overfit*.log          # V14.1 实验日志
│   ├── v14_1_overfit/              # 重建音频输出
│   ├── models/                     # 训练 checkpoint
│   └── hydra/                      # Hydra 运行日志
├── conf/                           # 旧项目 (V7-V13) 的 Hydra 配置
├── fm_dit/                         # 下游 DiT TTS 代码
└── fm_dit_code.tar.gz              # LongCat-AudioDiT 参考代码

/root/python_programe/playground2/  # 旧实验代码（V7-V13 时期）
```

## 工作流

1. **本地编辑** → git commit
2. **同步到远程**: `rsync -avz -e "ssh -p 21466" autoencoder/ root@connect.westc.seetacloud.com:/root/autodl-tmp/project/autoencoder/`
3. **远程运行**: `ssh ... 'python -u -m autoencoder.scripts.overfit'` 或 `python -m autoencoder.train`
4. **查看结果**: `ssh ... 'tail outputs/xxx.log'`
