# V16A Shallow Time-Domain Baseline

## 目标

V16A 是一个判别实验，不是最终方案。它只验证一个问题：

> 在坚持时域生成路径的前提下，`dim=256 + 更浅的 512x 压缩链路` 是否能突破 V14 系列 11-12 dB 的平台。

这个实验刻意不引入 Vocos、dual-path、PhaseRefiner 或 adversarial loss，避免继续把变量叠在一起。

## 配置

- 配置文件：`autoencoder/conf/experiment/v16a_shallow_time.yaml`
- 启动：

```bash
python -m autoencoder.train experiment=v16a_shallow_time runtime=remote
```

## 架构

```text
Encoder:
  wav
    -> stem 1 -> 128
    -> block stride 8, 128 -> 512
    -> block stride 8, 512 -> 1024
    -> block stride 8, 1024 -> 2048
    -> proj 2048 -> latent mean/std

Latent:
  256 channels, total stride 512

Decoder:
  symmetric time-domain decoder, strides [8, 8, 8]
```

## 判定标准

- 如果 overfit 也困难：优先怀疑 stride=8 block 或 decoder 优化路径。
- 如果 overfit 好、full-set 仍卡住：优先查泛化、正则、数据切分和 augmentation。
- 如果 full-set 稳定超过 12 dB：说明主线成立，后续继续优化时域信息通道。
- 如果 full-set 仍在 11-12 dB：说明瓶颈可能不是 decoder patch，而是 512x 压缩或 latent 目标本身。

## 禁止混入的变量

- 不加 Vocos/iSTFT decoder。
- 不加 dual-path blend。
- 不加 PhaseRefiner。
- 不启用 adversarial loss。
- 不从 V14.4 teacher 初始化。
- 首轮不启用 multi-band complex STFT loss。

V16A 的价值在于干净。如果它给出明确趋势，再决定要不要做 V16B。
