# Yunwan Speech Latent Lab

[English](#english) | [中文](#中文)

## English

Yunwan Speech Latent Lab is an independent research playground for speech
representation learning and latent-space speech generation.

The project explores a simple two-stage architecture:

```text
waveform -> audio autoencoder -> continuous latent -> FM-DiT -> latent -> waveform
```

Instead of relying on a ready-made speech codec or a full TTS framework, this
repository builds a small, controllable stack around a custom waveform
autoencoder and a flow-matching DiT model trained on its latent space.

## Overview

The project has two main parts:

- `autoencoder/`: trains a waveform-level Wav-VAE / audio autoencoder.
- `fm_dit/`: trains a flow-matching DiT model over autoencoder latents.

The current baseline uses the V16.3 autoencoder:

- sample rate: `22050 Hz`
- total compression stride: `256x`
- latent dimension: `384`
- latent frame rate: `22050 / 256 = 86.13 fps`

The autoencoder is treated as the stable acoustic representation layer. The
current research focus is on generating or predicting those latents with
FM-DiT, then decoding them back to waveform audio.

## Architecture

### 1. Audio Autoencoder

The first stage is a waveform autoencoder:

```text
mono waveform -> encoder -> latent -> decoder -> reconstructed waveform
```

The current autoencoder line is based on 1D convolutional audio modeling. It
uses:

- 1D convolutional encoder and decoder blocks
- dilated residual units
- SnakeBeta activations
- sub-pixel upsampling in the decoder
- L1 waveform reconstruction loss
- multi-resolution STFT loss
- multi-scale mel loss

The goal is to learn a continuous latent space that preserves enough acoustic
detail for high-quality reconstruction while remaining easier to model than raw
waveform samples.

The current recommended autoencoder configuration is:

```text
autoencoder/conf/experiment/v16_3_256x_dim384_time.yaml
```

Example training command:

```bash
python -m autoencoder.train experiment=v16_3_256x_dim384_time runtime=remote
```

### 2. FM-DiT Latent Generator

The second stage models the autoencoder latent space instead of waveform audio.

```text
text / phoneme / pitch / energy -> FM-DiT -> AE latent -> AE decoder -> waveform
```

The FM-DiT branch uses flow matching over normalized V16.3 autoencoder latents.
Its inputs can include phoneme ids, phoneme durations, frame-level phoneme ids,
pitch, energy, and text metadata.

Main entry point:

```bash
python -m fm_dit.train_v1_v16ae
```

Small smoke-run example:

```bash
python -m fm_dit.train_v1_v16ae \
  --data-path fm_dit/data/ljspeech_v16_3_256x_dim384/train.json \
  --stats-path fm_dit/data/ljspeech_v16_3_256x_dim384/latent_stats.json \
  --latent-dim 384 \
  --model-dim 512 \
  --max-frames 256 \
  --max-samples 200 \
  --batch-size 64 \
  --t-sampling beta \
  --min-snr-gamma 5.0 \
  --steps 2000
```

## Datasets

This repository uses two main data sources.

### LJSpeech

LJSpeech is used for autoencoder reconstruction experiments and baseline latent
export.

Typical uses:

- waveform autoencoder training
- reconstruction quality evaluation
- latent extraction
- FM-DiT baseline manifest construction

Audio is normalized into the project training setup, typically using
`22050 Hz` mono waveform input.

### CosyVoice3 Synthetic Speech

The project also uses CosyVoice3-generated speech data for multi-speaker and
Chinese TTS-style experiments.

Typical uses:

- multi-speaker synthetic speech generation
- same-text / cross-speaker paired data
- pinyin and phoneme duration metadata
- forced-alignment preparation
- V17/V18 latent structure probes
- conditional latent generation experiments

Data preparation utilities live in:

```text
tools/tts/
```

These scripts handle tasks such as manifest filtering, duration filtering,
pinyin label generation, phoneme alignment scaffolding, forced-alignment corpus
export, prompt speaker selection, and CosyVoice3 batch synthesis.

## Repository Layout

```text
.
├── autoencoder/      # Wav-VAE / waveform autoencoder code
├── fm_dit/           # FM-DiT training over V16.3 autoencoder latents
├── tools/tts/        # TTS data preparation and manifest tooling
├── scripts/          # experiment launch scripts
├── docs_ae/          # autoencoder experiment notes and version history
└── outputs/          # local experiment outputs, checkpoints, and logs
```

## Manifest Format

FM-DiT training expects a JSON manifest where each sample usually contains:

```json
{
  "id": "sample-id",
  "latent_path": "path/to/latent.pt",
  "phoneme_ids": [1, 2, 3],
  "phoneme_durations": [5, 7, 4],
  "frame_phoneme_ids": [1, 1, 1, 1],
  "pitch": [0.0, 0.0, 0.0],
  "energy": [0.0, 0.0, 0.0],
  "duration": 128,
  "text": "optional text",
  "audio_path": "optional/path.wav"
}
```

Latent tensors may be stored as either `[C, T]` or `[T, C]`. The dataset loader
normalizes latents with `latent_stats.json`, preferring per-channel mean and
standard deviation when available.

## Experiment Notes

Experiment history is documented in:

```text
docs_ae/
```

Important lines:

- **V14.x**: early Wav-VAE architecture exploration.
- **V14.4**: removed shortcut paths that caused aliasing.
- **V16.3**: current stable autoencoder baseline.
- **V17**: frozen-encoder probes and teacher-distillation experiments.
- **V18**: generation-aware autoencoder pressure test, now archived.

The main conclusion from V18 is that the V16.3 bottleneck is strong for
reconstruction-only training, but not robust when extra CTC / phoneme objectives
are added directly to the autoencoder. The current direction is therefore to
keep V16.3 fixed and solve the generation problem in `fm_dit/`.

## Current Direction

The project is currently focused on FM-DiT over V16.3 latents:

1. Train or load the V16.3 autoencoder.
2. Export continuous latents from speech audio.
3. Build manifests with phoneme, pitch, energy, and duration metadata.
4. Train FM-DiT to model the normalized latent space.
5. Decode generated latents through the autoencoder decoder.
6. Evaluate both latent metrics and decoded waveform quality.

Useful evaluation signals include latent MSE/cosine, PCA subspace residuals,
decoded-audio SNR, multi-band SNR, frequency response, phase consistency, and
fixed validation samples.

## Status

This repository is an active research workspace. The architecture is intentionally
simple and modular so that representation-learning ideas can be tested quickly:

```text
custom audio autoencoder + latent-space FM-DiT
```

It is not intended to be a production-ready TTS system yet.

## 中文

Yunwan Speech Latent Lab 是一个面向语音表示学习和 latent-space 语音生成的
独立研究项目。

项目探索的是一个简单、可控的两阶段架构：

```text
waveform -> audio autoencoder -> continuous latent -> FM-DiT -> latent -> waveform
```

它没有直接复刻现成的语音 codec 或完整 TTS 框架，而是围绕自定义 waveform
autoencoder 和基于 latent space 的 flow-matching DiT 模型，搭建了一套轻量
的实验架构。

## 项目概览

项目主要包含两个部分：

- `autoencoder/`：训练 waveform-level Wav-VAE / audio autoencoder。
- `fm_dit/`：在 autoencoder latent 上训练 flow-matching DiT 模型。

当前 baseline 使用 V16.3 autoencoder：

- sample rate：`22050 Hz`
- 总压缩 stride：`256x`
- latent dimension：`384`
- latent frame rate：`22050 / 256 = 86.13 fps`

Autoencoder 被视为稳定的声学表示层。当前研究重点是在 FM-DiT 中生成或预测
这些 latent，再通过 autoencoder decoder 将 latent 还原为 waveform audio。

## 架构设计

### 1. Audio Autoencoder

第一阶段是 waveform autoencoder：

```text
mono waveform -> encoder -> latent -> decoder -> reconstructed waveform
```

当前 autoencoder 线基于 1D convolutional audio modeling，主要使用：

- 1D convolutional encoder / decoder blocks
- dilated residual units
- SnakeBeta activation
- decoder 中的 sub-pixel upsampling
- L1 waveform reconstruction loss
- multi-resolution STFT loss
- multi-scale mel loss

目标是学习一个连续 latent space：它要保留足够的声学细节用于高质量重建，
同时比 raw waveform 更容易被下游生成模型建模。

当前推荐的 autoencoder 配置：

```text
autoencoder/conf/experiment/v16_3_256x_dim384_time.yaml
```

训练命令示例：

```bash
python -m autoencoder.train experiment=v16_3_256x_dim384_time runtime=remote
```

### 2. FM-DiT Latent Generator

第二阶段建模的是 autoencoder latent space，而不是直接建模 waveform。

```text
text / phoneme / pitch / energy -> FM-DiT -> AE latent -> AE decoder -> waveform
```

FM-DiT 分支在归一化后的 V16.3 autoencoder latent 上使用 flow matching。
输入条件可以包括 phoneme ids、phoneme durations、frame-level phoneme ids、
pitch、energy 和文本 metadata。

主要入口：

```bash
python -m fm_dit.train_v1_v16ae
```

小规模 smoke run 示例：

```bash
python -m fm_dit.train_v1_v16ae \
  --data-path fm_dit/data/ljspeech_v16_3_256x_dim384/train.json \
  --stats-path fm_dit/data/ljspeech_v16_3_256x_dim384/latent_stats.json \
  --latent-dim 384 \
  --model-dim 512 \
  --max-frames 256 \
  --max-samples 200 \
  --batch-size 64 \
  --t-sampling beta \
  --min-snr-gamma 5.0 \
  --steps 2000
```

## 数据集

项目主要使用两类数据来源。

### LJSpeech

LJSpeech 主要用于 autoencoder 重建实验和 baseline latent 导出。

典型用途：

- waveform autoencoder training
- reconstruction quality evaluation
- latent extraction
- FM-DiT baseline manifest construction

音频通常会被处理为 `22050 Hz` 单声道 waveform 输入。

### CosyVoice3 Synthetic Speech

项目也使用 CosyVoice3 生成的合成语音数据，用于多说话人和中文 TTS 风格实验。

典型用途：

- multi-speaker synthetic speech generation
- same-text / cross-speaker paired data
- pinyin and phoneme duration metadata
- forced-alignment preparation
- V17/V18 latent structure probes
- conditional latent generation experiments

数据处理工具位于：

```text
tools/tts/
```

这些脚本负责 manifest 过滤、duration 过滤、拼音标签生成、phoneme alignment
scaffold、forced-alignment corpus 导出、prompt speaker 选择和 CosyVoice3
批量合成等任务。

## 仓库结构

```text
.
├── autoencoder/      # Wav-VAE / waveform autoencoder code
├── fm_dit/           # FM-DiT training over V16.3 autoencoder latents
├── tools/tts/        # TTS data preparation and manifest tooling
├── scripts/          # experiment launch scripts
├── docs_ae/          # autoencoder experiment notes and version history
└── outputs/          # local experiment outputs, checkpoints, and logs
```

## Manifest 格式

FM-DiT 训练使用 JSON manifest。每个样本通常包含：

```json
{
  "id": "sample-id",
  "latent_path": "path/to/latent.pt",
  "phoneme_ids": [1, 2, 3],
  "phoneme_durations": [5, 7, 4],
  "frame_phoneme_ids": [1, 1, 1, 1],
  "pitch": [0.0, 0.0, 0.0],
  "energy": [0.0, 0.0, 0.0],
  "duration": 128,
  "text": "optional text",
  "audio_path": "optional/path.wav"
}
```

Latent tensor 可以保存为 `[C, T]` 或 `[T, C]`。Dataset loader 会使用
`latent_stats.json` 对 latent 进行归一化，并优先使用 per-channel mean 和
standard deviation。

## 实验记录

实验历史记录在：

```text
docs_ae/
```

重要版本：

- **V14.x**：早期 Wav-VAE 架构探索。
- **V14.4**：去掉导致 aliasing 的 shortcut path。
- **V16.3**：当前稳定 autoencoder baseline。
- **V17**：frozen-encoder probes 和 teacher-distillation 实验。
- **V18**：generation-aware autoencoder 压力测试，现已归档。

V18 的主要结论是：V16.3 bottleneck 对 reconstruction-only 训练很有效，
但直接在 autoencoder 上加入 CTC / phoneme 等额外目标时并不稳定。因此当前
方向是固定 V16.3，并在 `fm_dit/` 中解决 latent generation 问题。

## 当前方向

项目当前聚焦于 V16.3 latent 上的 FM-DiT：

1. 训练或加载 V16.3 autoencoder。
2. 从语音音频中导出连续 latent。
3. 构造带有 phoneme、pitch、energy 和 duration metadata 的 manifest。
4. 训练 FM-DiT 建模归一化后的 latent space。
5. 通过 autoencoder decoder 解码生成的 latent。
6. 同时评估 latent 指标和 decoded waveform 质量。

有用的评估信号包括 latent MSE/cosine、PCA subspace residual、decoded-audio
SNR、multi-band SNR、frequency response、phase consistency 和固定验证样本。

## 状态

这是一个活跃的研究工作区。项目架构有意保持简单和模块化，方便快速验证语音
表示学习想法：

```text
custom audio autoencoder + latent-space FM-DiT
```

它目前还不是生产级 TTS 系统。
