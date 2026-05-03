# FM-DiT V1: V16 AE Latent Baseline

## Goal

Establish a stable, fast-converging DiT baseline in the V16.3 autoencoder
latent space so the next experiments can focus on representation learning
rather than rebuilding the codec.

## Starting Point

- Autoencoder: V16.3, 256x stride, 384-dimensional deterministic latent.
- Current AE evidence:
  - high-quality reconstruction,
  - frame-level effective rank uses the wider 384d space,
  - generalization stress tests do not show boundary fragility,
  - latent perturbations around the 0.005-0.01 scale remain usable.

## Key Differences From The Older FM-DiT Project

- No Mimi/RVQ default data path.
- No 12.5fps assumption.
- Training window is explicit via `--max-frames`.
- Dataset normalization supports per-channel latent stats.
- `frame_lengths` masks padded frames in attention, global condition pooling,
  sampling and FM loss.
- Evaluation should decode through the V16 AE decoder, not stop at latent
  cosine correlation.

## First Smoke

Use V16.3 exported LJSpeech latents with:

- `latent_dim=384`
- `model_dim=512`
- `max_frames=256`
- small step count first, then scale to `512` frames if memory allows.
