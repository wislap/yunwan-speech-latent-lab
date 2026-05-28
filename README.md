# Yunwan Speech Latent Lab

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

## Technical Stack

The codebase is primarily written in Python and PyTorch.

Main technologies:

- **Python** for experiments, training scripts, and data tools.
- **PyTorch** for neural network modules, training loops, losses, and tensor
  computation.
- **torchaudio** and **soundfile** for waveform loading, resampling, and audio
  processing.
- **Hydra** and **OmegaConf** for autoencoder experiment configuration.
- **NumPy** for data processing and analysis utilities.
- **Matplotlib / Librosa** for diagnostics, plots, and spectral analysis.
- **CosyVoice3-related tooling** for synthetic multi-speaker Chinese speech
  data construction.

This is a research repository rather than a packaged library. Some analysis
scripts may require additional local dependencies depending on the experiment.

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
