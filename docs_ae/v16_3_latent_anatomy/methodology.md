# V16.3 Latent Anatomy — Methodology

This document pins down every configurable parameter so that all
results under `docs_ae/v16_3_latent_anatomy/` are reproducible.

## Checkpoints

- V16.3:
  `outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt`
  (epoch 13, val SNR 20.73 dB, dim=384, strides=[2,4,4,8])
- V18:
  `outputs/models/wav_vae_v18_base_remote_main/best.pt`
  (epoch 37, val SNR 15.70 dB, dim=384 concat, strides=[2,4,4,8])

## Datasets

- **LJSpeech val subset**:
  - JSON: `data/ljspeech_train_splits.json`, `val` split
  - Sample count: 128 (fixed seed 20260510)
  - Segment length: 65536 samples (V16.3 training segment length)
  - Sampling rate: 22050

- **CosyVoice3 4-spk crossed val**:
  - Manifest: `CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl`
  - Sample count: 112
  - Duration: ~8-13 s each
  - Phoneme alignment: `pinyin_durations` field in manifest

## Label pipeline

Implemented in `autoencoder/scripts/anatomy/labels.py`.

### Phoneme frame labels

- CosyVoice: from `pinyin_durations` field, mapped to AE frame rate
  using integer frame index rounding.
- LJSpeech: not used in label-requiring blocks. Phoneme-label-requiring
  blocks (B2 phoneme probe, B3 phoneme subspace) run only on CosyVoice.

### F0

- `pyin` with `fmin=50, fmax=500, sr=22050, frame_length=1024,
  hop_length=256`
- Interpolated linearly to AE frame rate (one AE frame per 256 audio
  samples).
- Voiced / unvoiced flag from `pyin` confidence.

### Energy

- `log(1 + RMS)` per AE frame. RMS computed over 256-sample frame
  aligned with AE stride.

### Spectral centroid

- `librosa.feature.spectral_centroid` with `n_fft=1024, hop_length=256`.
- Converted to kHz.

### Cached outputs

All per-sample label arrays are cached to
`docs_ae/v16_3_latent_anatomy/data/labels/{dataset}/{sample_id}.npz`
with keys `phoneme`, `speaker`, `f0`, `voiced`, `energy`, `centroid`.

## Seeds

- Global numpy seed: 20260510
- Global torch seed: 20260510
- CUDA deterministic: True

## Statistical configuration

- Bootstrap resamples: 5 (quick) or 100 (where feasible)
- Shuffle-null resamples in B3: 1000
- CV folds in B4: 5, split by utterance (not by frame)
- Confidence level: 95%

## Compute budget

Single RTX 5090 32 GB. Expected wall-clock per block:

- B1: ~20 min (encode 128 samples + SVD + TwoNN)
- B2: ~40 min (PCA + 32×6 probe × 5-fold CV × 2 datasets)
- B3: ~2 h (subspace SVD + 1000 shuffle null + both datasets)
- B4: ~1 h (linear + MLP × 5 labels × 5-fold × 2 datasets)
- B5: ~2 h (50 directions × 5 alpha × 8 samples, with decoder forward)
- B6: ~20 min (compilation, reruns of B2/B3/B4 on V18 already covered)

Total compute: ~7-8 h.

## Run commands

Commands assume the remote working directory is
`/root/autodl-tmp/project`. Prepend `ssh -p 14025
root@connect.westc.seetacloud.com` if invoking from the local machine.

### B1 — global and local spectrum (LJSpeech)

```
python -m autoencoder.scripts.anatomy.spectrum \
  --checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
  --dataset-name ljspeech \
  --manifest data/ljspeech_train_splits.json \
  --n-samples 128 \
  --segment-samples 65536 \
  --crop-mode center \
  --twonn-k 2 \
  --twonn-subsample 4000 \
  --twonn-bootstrap 20 \
  --max-lag 16 \
  --out docs_ae/v16_3_latent_anatomy/data/b1_v16_3_ljspeech \
  --verbose
```

### B1 — global and local spectrum (CosyVoice)

Uses the CosyVoice 4-speaker crossed val manifest. Provides a comparator
to the LJSpeech baseline that will be more directly comparable to B3's
CosyVoice-only speaker analysis.

```
python -m autoencoder.scripts.anatomy.spectrum \
  --checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
  --dataset-name cosyvoice_4spk_val \
  --manifest CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl \
  --path-root CosyVoice_main \
  --path-root . \
  --n-samples 112 \
  --segment-samples 131072 \
  --crop-mode center \
  --twonn-k 2 \
  --twonn-subsample 4000 \
  --twonn-bootstrap 20 \
  --max-lag 16 \
  --out docs_ae/v16_3_latent_anatomy/data/b1_v16_3_cosyvoice \
  --verbose
```

Later blocks' run commands will be added here as each module is
implemented.

### B2 — PCA directions vs semantic labels (CosyVoice)

```
python -m autoencoder.scripts.anatomy.pca_semantics \
  --checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
  --dataset-name cosyvoice_4spk_val \
  --manifest CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl \
  --path-root CosyVoice_main \
  --path-root . \
  --n-samples 112 \
  --segment-samples 131072 \
  --k-directions 32 \
  --n-folds 5 \
  --out docs_ae/v16_3_latent_anatomy/data/b2_v16_3_cosyvoice \
  --verbose
```

Optional diagnostic for distinguishing "no info" from "distributed
info":

```
python -m autoencoder.scripts.anatomy._b2_diagnostic \
  --checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
  --manifest CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl \
  --path-root CosyVoice_main \
  --path-root . \
  --n-samples 112 \
  --segment-samples 131072 \
  --k-directions 32 \
  --out docs_ae/v16_3_latent_anatomy/data/b2_v16_3_cosyvoice
```

### B2.5 — Sanity checks

#### S1: TwoNN calibration and N-scaling

```
python -m autoencoder.scripts.anatomy.b2_5_twonn_scaling \
  --checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
  --manifest data/ljspeech_train_splits.json \
  --n-samples 128 --segment-samples 65536 \
  --sweep-n 500,1000,2000,4000,8000,16000 \
  --n-repeats 5 \
  --calibration-dims 10,20,40,80,160 \
  --out docs_ae/v16_3_latent_anatomy/data/b2_5 \
  --verbose
```

#### S2: Full-384-dim linear probe (real labels)

```
python -m autoencoder.scripts.anatomy.b2_5_full_linear \
  --checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
  --manifest CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl \
  --path-root CosyVoice_main --path-root . \
  --n-samples 112 --segment-samples 131072 \
  --ridge-alpha 10.0 --logistic-alpha 1.0 --logistic-iter 400 \
  --b2-json docs_ae/v16_3_latent_anatomy/data/b2_v16_3_cosyvoice/b2_summary.json \
  --out docs_ae/v16_3_latent_anatomy/data/b2_5 \
  --verbose
```

#### S3: Shuffle null (same script with `--shuffle-labels`)

```
python -m autoencoder.scripts.anatomy.b2_5_full_linear \
  --checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
  --manifest CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl \
  --path-root CosyVoice_main --path-root . \
  --n-samples 112 --segment-samples 131072 \
  --ridge-alpha 10.0 --logistic-alpha 1.0 --logistic-iter 400 \
  --shuffle-labels \
  --out docs_ae/v16_3_latent_anatomy/data/b2_5 \
  --verbose
```

### B3 — Supervised subspaces and principal angles

```
python -m autoencoder.scripts.anatomy.subspace \
  --checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
  --dataset-name cosyvoice_4spk_val \
  --manifest CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl \
  --path-root CosyVoice_main --path-root . \
  --n-samples 112 --segment-samples 131072 \
  --phoneme-min-count 100 \
  --variance-target 0.90 \
  --n-shuffles 1000 \
  --out docs_ae/v16_3_latent_anatomy/data/b3_v16_3_cosyvoice \
  --verbose
```
