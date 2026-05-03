# FM-DiT on V16 Autoencoder Latents

This folder is the clean representation-learning branch for playground3.

The autoencoder code in `../autoencoder` learns a high-quality waveform latent
space. FM-DiT starts from that latent space and studies flow matching / DiT
dynamics over V16 AE latents, instead of inheriting the Mimi/RVQ assumptions
from the older playground2 project.

## Current Baseline

- AE target: V16.3, 256x stride, 384 channels.
- Latent frame rate: `22050 / 256 = 86.1328125 fps`.
- First DiT entry point: `python -m fm_dit.train_v1_v16ae`.
- Expected data root: `fm_dit/data/ljspeech_v16_3_256x_dim384/`.

The first smoke baseline should use short windows before scaling sequence
length:

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

## Manifest Contract

`dataset_v3.py` expects each item to contain:

- `id`
- `latent_path`
- `phoneme_ids`
- `phoneme_durations`
- `frame_phoneme_ids`
- `pitch`
- `energy`
- `duration`
- optional `text` and `audio_path`

Latents should be saved as `[C, T]` or `[T, C]` tensors. The dataset normalizes
with per-channel `mean`/`std` when available, falling back to scalar
`global_mean`/`global_std`.

The dataloader returns `frame_lengths`; training uses it to mask padded frames
in both self-attention and flow-matching loss. This is important for V16 AE
latents because sequence lengths vary substantially after random cropping and
padding.

## Model Layout

- `models/embeddings.py`: timestep and RoPE utilities.
- `models/attention.py`: RoPE self-attention with padding masks.
- `models/layers.py`: SwiGLU, lightweight conv and skip layers.
- `models/blocks.py`: reusable DiT encoder/decoder blocks.
- `models/dit_head_v5_10.py`: the current v1 flow head assembled from those
  components.

## Research Notes

Do not judge this branch with latent cosine alone. The useful evaluation stack
is:

- normalized and raw latent MSE/cosine,
- PCA-subspace residual analysis using the V16.3 p95 directions,
- AE-decoded audio SNR, multi-band SNR, frequency response and phase coherence,
- fixed validation cases with local time-error curves.
