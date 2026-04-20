# 项目布局

## 本地工作区 (playground3)

```
playground3/
├── autoencoder/                    # 主代码包
│   ├── models/
│   │   ├── autoencoder.py          # WavVAE (encoder/decoder/bottleneck)
│   │   ├── discriminator.py        # MS-STFT + MSD 判别器
│   │   └── discriminator_v14.py    # V14 判别器封装
│   ├── losses.py                   # MultiScaleMelLoss + 聚合
│   ├── losses_stft.py              # MultiResolutionSTFTLoss
│   ├── train.py                    # Hydra 全数据集训练
│   ├── extract_latents.py          # 潜表示导出
│   ├── scripts/overfit.py          # 单条过拟合测试
│   └── conf/                       # Hydra 配置
├── docs/                           # 按版本组织的文档
└── .kiro/steering/                 # 项目上下文
```

## 远程服务器

```
SSH: ssh -p 21466 root@connect.westc.seetacloud.com
GPU: RTX 5090 32GB | Python: 3.12 | PyTorch: 2.10

/root/autodl-tmp/project/
├── autoencoder/                    # ← rsync 同步
├── data/
│   ├── LJSpeech-1.1/wavs/
│   └── ljspeech_train_splits.json
├── outputs/                        # 实验日志 + checkpoint + 音频
├── fm_dit/                         # 下游 DiT TTS
└── conf/                           # 旧项目配置 (V7-V13)

/root/python_programe/playground2/  # 旧实验 (V7-V13)
```

## 同步命令

```bash
rsync -avz -e "ssh -p 21466" autoencoder/ root@connect.westc.seetacloud.com:/root/autodl-tmp/project/autoencoder/
```
