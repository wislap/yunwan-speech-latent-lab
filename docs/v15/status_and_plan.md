# V15 状态与下一步计划

## 当前状态

### V15 Dual-Path Decoder (正在训练)
- 服务器: `ssh -p 50399 root@connect.westb.seetacloud.com`
  (旧地址 56717 westc 可能也能用)
- 训练: epoch 21/50, SNR=11.79 dB (best)
- 架构: dim=128, V14.4 encoder(冻结) + DualPathDecoder(时域+Vocos blend)
- 结论: **停滞在 11.6-11.8 dB**, 和 V14.4 差不多, blend 没有学到有意义的偏好
- checkpoint: `outputs/models/wav_vae_v15_dual_remote_main/`

### V14.4 Best (基准)
- SNR=11.36 dB, 50 epochs
- checkpoint: `outputs/models/wav_vae_v14.4_full_remote_main/best.pt`
- 问题: 相位误差导致电流音, 2kHz 以上相位 R²<0.5

## 所有实验总结

| 版本 | 架构 | Best SNR | 结论 |
|------|------|----------|------|
| V14.4 | dim=128, 时域 decoder | 11.36 dB | 基准, 相位差 |
| V14.6 | + 判别器 | 11.65→退化 | 判别器崩溃 |
| V14.7 | + complex STFT loss | 11.74 dB | 微提升, 相位改善3% |
| V14.8 | + parallel iSTFT head | 11.63 dB | blend 不动 |
| V14.9 | + PhaseRefiner | 11.98 dB | 微提升, latent 相位信息不够 |
| V14.7d | dim=256, 时域 decoder | 6.5 dB | 爆炸+停滞, decoder 消化不了 |
| V14.7e | dim=256, wider decoder | 6.5 dB | 同上 |
| V14.7f | Vocos decoder (纯) | -0.05 dB | 从抽象 latent 学不动 |
| V14.7g | block[0] + VocosHead | 7.23 dB | 能学动, 还在涨, 但慢 |
| V15 | DualPath (时域+Vocos blend) | 11.79 dB | 停滞, blend 无效 |

## 关键发现

1. **Encoder 深层有完美相位信息 (R²=1.0)**, proj 压缩 1024→128 时丢掉 50%
2. **Decoder block[0] 输出还有 R²=0.96**, block[1] 之后急剧丢失到 0.24
3. **dim=256 latent 能编码 R²=0.97 的相位**, 但时域 decoder 消化不了 (爆炸+停滞)
4. **纯 Vocos decoder 从抽象 latent 学不动**, 但从 block[0] 具象特征能学动
5. **128 维 latent 的信息论极限**: 幅度需要 ~50 维, 相位需要 ~150 维, 128 维不够
6. **所有 dim=128 方案都卡在 11-12 dB**, 不管怎么改 decoder

## 下一步: V16 — 浅网络 + 双路 + dim=256 + Vocos decoder

### 设计理念

- **更浅**: 3 个 block (stride=[8,8,8]=512x) 代替 5 个 block, 减少相位丢失
- **双路 encoder**: 时域路径 + 频域路径 (STFT), AdaLN 交互, 让 latent "频域友好"
- **dim=256**: 足够编码相位 (R²=0.95+)
- **Vocos decoder**: 纯 ConvNeXt + iSTFT, 不做时域上采样
- **更宽**: 3 block 参数少, 可以给更多通道

### 架构

```
=== Encoder (双路) ===

wav [B, 1, T]
  │
  ├── 时域路径
  │   stem(1→128) → block[0](128→512, s=8) → block[1](512→1024, s=8)
  │                                                ↕ AdaLN ← freq_feat
  │   → block[2](1024→2048, s=8) → proj(2048→512) → z [256d, 43Hz]
  │                  ↕ AdaLN ← freq_feat
  │
  └── 频域路径
      STFT(1024, 512) → [B, 2, 513, T/512]
      → Conv2d 压缩 freq 维 → [B, C, T/512]
      → 提供 AdaLN 调制参数给时域路径

=== Decoder (Vocos) ===

z [256d, 43Hz]
  → input_proj(256→1024)
  → 8x ConvNeXt Block (1024d, 全程 43Hz)
  → output_proj(1024→1026)  [513 mag + 513 phase]
  → softplus(mag), atan2(sin(p), cos(p))
  → iSTFT(n_fft=1024, hop=512) → wav
```

### 参数量估计

| 模块 | 参数 |
|------|------|
| 时域 encoder (3 blocks, wider) | ~60M |
| 频域 encoder (2D Conv) | ~3M |
| AdaLN 交互 × 2 | ~2M |
| Bottleneck | ~1M |
| Vocos decoder (8 layers, 1024d) | ~35M |
| 总计 | ~101M |

### 训练策略

- 端到端从头训练 (不需要预训练权重)
- lr=1e-4, warmup=2000 steps (防爆炸)
- Vocos decoder output_proj 用 mag_bias=-5 初始化 (防爆炸)
- Loss: L1 + complex STFT + mel
- 50-100 epochs

### 为什么这次能成功

1. dim=256 提供足够的相位容量 (R²=0.95)
2. 双路 encoder 让 latent 编码方式"频域友好"
3. 更浅的 encoder (3 blocks) 减少相位丢失
4. Vocos decoder 全程 43Hz, 不做上采样, 每个 freq bin 有独立参数
5. 没有时域 decoder 的"消化不了 256 维"问题 (Vocos 是 isotropic)
6. softplus + 小初始化防爆炸

### 风险

1. stride=8 的 block 可能不稳定 (之前只用过 stride=2,4)
2. 双路 encoder 增加训练复杂度
3. 从头训练需要更多 epoch
4. Vocos decoder 之前从抽象 latent 学不动 — 但那是 dim=128, 这次 dim=256 + 频域友好编码

### 需要实现的文件

1. `autoencoder/models/autoencoder.py` — 新增:
   - `STFTFeatureExtractor`: 频域路径, STFT → 2D Conv → 1D features
   - `DualPathAdaLN`: 双路 AdaLN 交互模块
   - `ShallowEncoder`: 3-block encoder with 频域交互
   - 复用现有 `VocosDecoder` (已实现)
   - `WavVAE_V16`: 组合 ShallowEncoder + VocosDecoder

2. `autoencoder/conf/experiment/v16_shallow_dual.yaml` — 配置

3. `autoencoder/train.py` — 支持新模型参数

## 远程服务器信息

- westb: `ssh -p 50399 root@connect.westb.seetacloud.com`
- westc: `ssh -p 56717 root@connect.westc.seetacloud.com` (可能同一台)
- GPU: RTX 5090 32GB
- 数据: `/root/autodl-tmp/project/data/`
- V14.4 checkpoint: `/root/autodl-tmp/project/outputs/models/wav_vae_v14.4_full_remote_main/best.pt`
- V15 训练中: `/root/autodl-tmp/project/outputs/models/wav_vae_v15_dual_remote_main/`
