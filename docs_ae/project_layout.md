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

### 服务器 A (westc) — 可能已到期
```
SSH: ssh -p 21466 root@connect.westc.seetacloud.com
GPU: RTX 5090 32GB | Python: 3.12 | PyTorch: 2.10
```

### 服务器 B (westd) — 当前活跃
```
SSH: ssh -p 13766 root@connect.westd.seetacloud.com
GPU: RTX 5090 32GB | Python: 3.12 | PyTorch: 2.10
```

两台服务器共享 `/root/autodl-tmp/` 数据盘：
```
/root/autodl-tmp/project/
├── autoencoder/                    # ← rsync 同步
├── data/
│   ├── LJSpeech-1.1/wavs/         # 13100 条 wav
│   └── ljspeech_train_splits.json  # 11823 条训练索引
├── outputs/
│   ├── v14_1_full_train.log
│   ├── v14_2_full_train.log
│   └── models/
│       ├── wav_vae_v14.1_full_remote_main/  # best.pt (SNR 5.16)
│       └── wav_vae_v14.2_full_remote_main/  # best.pt (SNR 11.96)
├── fm_dit/                         # 下游 DiT TTS
└── conf/                           # 旧项目配置 (V7-V13)
```

## 同步命令

```bash
rsync -avz -e "ssh -p 21466" autoencoder/ root@connect.westc.seetacloud.com:/root/autodl-tmp/project/autoencoder/
```
