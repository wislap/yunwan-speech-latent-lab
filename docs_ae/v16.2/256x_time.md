# V16.2 256x Time Boundary Test

## 目标

V16.2 用来判断 256x 是否可以作为比 V16.1 更高压缩的可用 baseline。它不是新方法，而是压缩边界实验：尽量保持 V16.1 的训练 recipe 不变，只把总 stride 从 128x 推到 256x。

## 配置

- 配置文件：`autoencoder/conf/experiment/v16_2_256x_time.yaml`
- 启动：

```bash
python -m autoencoder.train experiment=v16_2_256x_time runtime=remote
```

## 架构

```text
Encoder:
  strides [2, 4, 4, 8] = 256x
  channels [128, 256, 512, 1024, 1024]

Latent:
  256 channels at ~86 Hz

Decoder:
  symmetric pure time-domain decoder
```

## 与 V16.1 的差异

- 总压缩率：128x -> 256x。
- stride：`[2, 4, 4, 4]` -> `[2, 4, 4, 8]`。
- 其他关键变量保持不变：
  - `sample_latent: false`
  - 无 Vocos
  - 无 dual path
  - 无 PhaseRefiner
  - 无 adversarial loss
  - 首轮不开 multi-band complex STFT

## 判定标准

- 如果 epoch 3-5 能明显超过 12 dB，并且听感语义稳定、齿音不脏，256x 值得继续。
- 如果 SNR 能上升但听感出现明显高频毛刺或口齿糊，说明 256x 接近边界，可再试 band loss，但不应直接替代 V16.1。
- 如果 epoch 7 仍显著落后 V16.1，优先把 V16.1 作为 DiT latent baseline，256x 只作为压缩率探索分支。

## 预期

256x 的 latent frame rate 约 86 Hz，理论上比 512x 的约 43 Hz 宽很多，但只有 V16.1 的一半。它可能保留语义，但高频和口齿清晰度会比 128x 更敏感。这个版本的价值在于确认“可用压缩上限”，而不是立刻替代 V16.1。
