# V16.1 High-Fidelity Time Baseline

## 目标

V16.1 不追求 512x 高压缩。它的目标是先得到一个足够干净、稳定、可复现的 autoencoder baseline，让后续 DiT latent 表示学习不被 AE 高频噪声污染。

核心判断：

> 如果 128x 低压缩仍然不干净，问题就不只是 512x 信息瓶颈，而要回头查时域 decoder、loss 或数据流程。

## 配置

- 配置文件：`autoencoder/conf/experiment/v16_1_hf_time.yaml`
- 启动：

```bash
python -m autoencoder.train experiment=v16_1_hf_time runtime=remote
```

## 架构

```text
Encoder:
  strides [2, 4, 4, 4] = 128x
  channels [128, 256, 512, 1024, 1024]

Latent:
  256 channels at ~172 Hz

Decoder:
  symmetric pure time-domain decoder
```

## 关键差异

- 不用 Vocos。
- 不用 dual path。
- 不用 PhaseRefiner。
- 不用 adversarial loss。
- 不做 VAE 采样：`sample_latent: false`。
- 首轮不开 multi-band complex STFT。

## 判定标准

- epoch 1-2 的 eval 应明显高于 V16A 同期。
- 如果 epoch 4 仍低于 10 dB，说明纯时域 128x 也没有快速形成可用底座。
- 如果 SNR 快速过 12 dB，还需要听 `eval_wavs`，重点判断高频是否干净。
- 如果高频仍脏，再加 band loss 或转 64x，而不是回到 512x 补丁。
