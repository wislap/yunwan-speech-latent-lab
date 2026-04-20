# Project Context: Wav-VAE Audio Autoencoder

## 项目目标
为 Streaming FM-DiT TTS 系统提供高质量连续潜空间。用 Wav-VAE（基于 LongCat-AudioDiT 架构）替代之前 V13 的 analytic microband bank 方案。

## 架构：Oobleck-style Wav-VAE
- **Encoder**: Conv1d stem → N 个 OobleckEncoderBlock（DilatedResUnit×3 + SnakeBeta activation + strided conv + non-parametric shortcut）→ Conv1d projection
- **Decoder**: 镜像结构，转置卷积上采样
- **VAE Bottleneck**: softplus std 参数化（不是 exp(log_var)），free-bits KL
- **关键技术**: SnakeBeta（log-scale alpha+beta）、weight_norm（无 LayerNorm）、pixel_unshuffle/shuffle shortcuts、LayerScale（init=1e-2）

## 当前最佳配置
- latent_dim=256, strides=[7,7,9], total_stride=441, ~50fps @ 22050Hz
- encoder_channels=[128, 256, 512, 1024]
- 单条过拟合 10000 步: SNR=+8.09 dB，还在上升未饱和
- 参数量: ~43M

## 关键实验发现
1. **dim=64 天花板低**：单条音频有效秩=131，dim=64 只覆盖 93% 方差，SNR 卡在 ~4 dB
2. **dim=256 上限高但收敛慢**：需要 10000+ 步，但 SNR 持续上升
3. **大 stride 需要 LayerScale**：stride=[3,3,7,7] 的 4-block 版本在 step 800 崩溃，加 LayerScale 后稳定
4. **softplus > exp(log_var)**：exp 参数化导致 NaN，softplus+1e-4 天然有界
5. **梯度累积帮助度过过渡期**：SNR 从负转正时最不稳定，大 batch 梯度更平滑
6. **LongCat-AudioDiT 论文发现**：VAE 重建质量好 ≠ TTS 质量好；latent dim 越大 DiT 越难建模；低帧率对 DiT 更友好
7. **V13 失败教训**：LayerNorm 破坏幅度信息；辅助 loss 全部帮倒忙；microband complex → synthesis 链路在高频天然病态

## 远程服务器
- SSH: `ssh -p 21466 root@connect.westc.seetacloud.com`
- GPU: RTX 5090 32GB
- 数据: `/root/autodl-tmp/project/data/LJSpeech-1.1/wavs/`
- 训练数据 JSON: `/root/autodl-tmp/project/data/ljspeech_train_splits.json`

## 代码结构
```
autoencoder/
  models/autoencoder.py    — WavVAE 主模型
  models/discriminator.py  — MS-STFT 判别器
  models/discriminator_v14.py — V14 判别器封装
  losses_stft.py           — MultiResolutionSTFTLoss
  losses.py                — MultiScaleMelLoss + 聚合
  train.py                 — Hydra 训练脚本
  extract_latents.py       — 潜表示导出工具
  scripts/overfit.py       — 单条过拟合测试
  conf/experiment/wav_vae.yaml — Hydra 配置
```

## 参考论文
- LongCat-AudioDiT (Meituan, 2026.03) — Wav-VAE + DiT TTS, MIT 开源
- Oobleck (Harmonai) — VAE codec 架构
- DAC (Descript) — Snake activation, weight_norm
