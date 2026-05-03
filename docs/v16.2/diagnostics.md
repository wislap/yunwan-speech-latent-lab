# V16.2 Diagnostics

Date: 2026-05-03

## GPU Status

V16.2 training is the only GPU-heavy process.

- GPU: NVIDIA GeForce RTX 5090
- Memory: about 8.7 GiB used / 32.6 GiB total
- Free memory: about 24 GiB
- Utilization during training: about 60-80%

This is enough memory for lightweight checkpoint/eval work, but diagnostics should prefer CPU-only audio analysis while training is active. Avoid launching another full training run or GPU ASR job during the current V16.2 run.

## Diagnostic Script

Added:

- `autoencoder/scripts/diagnose_recon_wavs.py`
- `autoencoder/scripts/plot_recon_diagnostics.py`

The script compares exported `eval_wavs` pairs only. It does not load the model and does not use GPU.

Metrics:

- full-band time-domain SNR
- log spectral distance
- band SNR:
  - 0-500 Hz
  - 500-1500 Hz
  - 1500-3500 Hz
  - 3500-6000 Hz
  - 6000-11025 Hz
- high-frequency error ratio for 6000-11025 Hz
- high-frequency gap: average low/mid band SNR minus 6000-11025 Hz SNR

## Single-Sample Eval-Wav Diagnostics

These numbers are from the exported `original_0.wav` / `recon_0.wav` pairs, so they are listening-sample diagnostics, not the averaged validation metric.

### V16.2 256x

| epoch | full SNR | LSD | 0-500 | 500-1500 | 1500-3500 | 3500-6000 | 6000-11025 | HF err ratio | HF gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.09 | 12.11 | 16.13 | 11.70 | 0.67 | -0.55 | -1.09 | 1.2854 | 10.59 |
| 3 | 14.57 | 10.66 | 23.24 | 21.88 | 0.25 | -0.65 | 1.01 | 0.7924 | 14.11 |
| 5 | 14.06 | 9.06 | 19.69 | 19.39 | 13.46 | 0.22 | 6.76 | 0.2108 | 10.76 |
| 7 | 17.23 | 8.58 | 19.32 | 18.99 | 14.07 | 4.91 | 6.04 | 0.2490 | 11.42 |
| 9 | 21.53 | 7.65 | 30.35 | 19.33 | 9.67 | 6.24 | 0.20 | 0.9557 | 19.59 |
| 11 | 21.46 | 7.49 | 24.10 | 24.60 | 16.50 | 11.81 | 7.10 | 0.1949 | 14.63 |
| 13 | 22.00 | 7.16 | 27.44 | 28.02 | 23.92 | 11.62 | 9.13 | 0.1221 | 17.32 |

### V16.1 128x

| epoch | full SNR | LSD | 0-500 | 500-1500 | 1500-3500 | 3500-6000 | 6000-11025 | HF err ratio | HF gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.97 | 11.87 | 17.51 | 16.66 | -0.66 | -0.50 | -1.40 | 1.3804 | 12.57 |
| 3 | 16.76 | 9.40 | 23.21 | 22.78 | 3.60 | 1.16 | 2.21 | 0.6009 | 14.32 |
| 5 | 18.03 | 8.88 | 24.29 | 16.14 | 6.72 | 4.75 | 5.15 | 0.3054 | 10.57 |
| 7 | 24.25 | 7.25 | 32.51 | 24.16 | 16.59 | 10.32 | 3.92 | 0.4060 | 20.50 |
| 9 | 19.71 | 7.28 | 20.23 | 21.76 | 13.84 | 13.70 | 5.71 | 0.2685 | 12.90 |
| 11 | 25.93 | 6.33 | 25.33 | 28.20 | 22.61 | 16.52 | 12.90 | 0.0513 | 12.48 |
| 13 | 23.21 | 6.08 | 24.44 | 22.11 | 20.07 | 18.82 | 14.19 | 0.0381 | 8.02 |

## Interpretation

V16.2 is not collapsing semantically. By epoch 13, the single exported sample has full-band SNR around 22 dB, which is close to the usable range.

The main remaining gap is high frequency:

- V16.2 epoch 13 reaches 9.13 dB in 6-11 kHz.
- V16.1 epoch 13 reaches 14.19 dB in 6-11 kHz.
- V16.2 epoch 13 high-frequency error ratio is 0.1221.
- V16.1 epoch 13 high-frequency error ratio is 0.0381.

So V16.2's high-frequency error energy is still about 3.2x V16.1 on this sample. This is consistent with possible articulation blur, consonant softness, or dirty sibilance.

## Current Direction

Keep V16.2 running through at least epoch 17 or 19. If full-band SNR keeps improving but 6-11 kHz remains far behind V16.1, the next 256x experiment should add high-frequency-focused reconstruction pressure before trying GAN or decoder complexity:

1. enable multi-band STFT loss
2. increase high-band weights moderately
3. keep deterministic latent and pure time-domain decoder
4. do not add adversarial loss yet

V16.1 remains the high-fidelity anchor. V16.2 is now a real compression candidate, but its acceptance depends on high-frequency/listening quality rather than semantic preservation alone.

## Report Figures

Generated under:

- remote working output: `/root/autodl-tmp/project/outputs/diagnostics/v16_recon_report/`
- report figures: `docs/v16.2/figures/`

Files:

- `epoch_curves.png`: full-band SNR, high-frequency SNR, high-frequency phase MAE, high-frequency phase coherence over epochs.
- `latest_band_snr.png`: latest epoch multi-band SNR comparison.
- `latest_frequency_response_bias.png`: latest epoch frequency response bias, measured as mean recon/original magnitude ratio in dB.
- `latest_phase_error.png`: latest epoch weighted phase MAE by frequency band.
- `latest_phase_coherence.png`: latest epoch phase coherence by frequency band.
- `V16.1_band_snr_heatmap.png`
- `V16.2_band_snr_heatmap.png`
- `V16.1_phase_coherence_heatmap.png`
- `V16.2_phase_coherence_heatmap.png`
- `recon_diagnostics_detail.csv`: detailed numeric table.

Local links:

- [epoch_curves.png](figures/epoch_curves.png)
- [latest_band_snr.png](figures/latest_band_snr.png)
- [latest_frequency_response_bias.png](figures/latest_frequency_response_bias.png)
- [latest_phase_error.png](figures/latest_phase_error.png)
- [latest_phase_coherence.png](figures/latest_phase_coherence.png)
- [recon_diagnostics_detail.csv](figures/recon_diagnostics_detail.csv)

## Fine-Band Latest-Epoch Snapshot

Latest exported sample:

- V16.1: epoch 13
- V16.2: epoch 13

| band Hz | V16.1 SNR | V16.2 SNR | V16.1 mag bias | V16.2 mag bias | V16.1 phase MAE | V16.2 phase MAE | V16.1 coh | V16.2 coh |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-250 | 20.89 | 20.75 | 2.16 | 3.32 | 0.4 | 1.1 | 1.000 | 0.999 |
| 250-500 | 27.71 | 31.73 | 0.61 | 0.27 | 0.3 | 0.4 | 1.000 | 1.000 |
| 500-1000 | 22.97 | 27.52 | 0.96 | 0.31 | 0.9 | 0.9 | 0.999 | 1.000 |
| 1000-1500 | 22.05 | 25.99 | 0.86 | 0.65 | 0.6 | 1.3 | 1.000 | 0.999 |
| 1500-2500 | 19.34 | 24.15 | 0.47 | 0.08 | 1.1 | 1.2 | 0.998 | 0.999 |
| 2500-3500 | 20.41 | 16.36 | 0.11 | 0.32 | 1.1 | 2.7 | 0.999 | 0.994 |
| 3500-5000 | 20.47 | 16.81 | -0.33 | 0.24 | 1.1 | 3.2 | 0.999 | 0.994 |
| 5000-7000 | 16.58 | 12.09 | -0.98 | -0.14 | 2.3 | 4.3 | 0.996 | 0.981 |
| 7000-9000 | 18.11 | 11.36 | -0.76 | -0.12 | 2.2 | 3.7 | 0.997 | 0.972 |
| 9000-11025 | 0.61 | 1.16 | -1.24 | -0.40 | 40.9 | 36.0 | 0.606 | 0.661 |

Important reading:

- V16.2 is very strong below 2.5 kHz on this sample.
- The meaningful gap starts around 2.5-3.5 kHz and becomes obvious in 5-9 kHz.
- 9-11 kHz has weak phase coherence for both V16.1 and V16.2, so this top band should not be over-interpreted alone.
- The best diagnostic band for articulation and sibilance is currently 3.5-9 kHz.
- Frequency response bias is not the main problem. V16.2's magnitude ratio is close to 0 dB above 2.5 kHz, but SNR and phase metrics are worse. That suggests residual texture/noise/phase-detail error rather than a simple high-frequency rolloff.

## Effective Rank

Output:

- [effective_rank_summary.csv](figures/effective_rank/effective_rank_summary.csv)
- [effective_rank_summary.png](figures/effective_rank/effective_rank_summary.png)

Setup:

- 64 LJSpeech samples
- first 65,536 samples per item
- deterministic encoder output
- latent_dim 256
- frame-level PCA over individual latent frames
- utterance-mean PCA over `z.mean(time)`, matching older V14-style analysis

| run | stride | ckpt epoch | ckpt SNR | frame effective rank | frame 99% dims | mean effective rank | mean 99% dims | active dims |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V16.1 | 128x | 11 | 23.21 | 125.3 | 217 | 26.3 | 53 | 256/256 |
| V16.2 | 256x | 13 | 19.93 | 144.3 | 229 | 38.0 | 57 | 256/256 |

Interpretation:

- V16.2 does not show latent collapse.
- V16.2 uses at least as many dimensions as V16.1, and by entropy rank it uses more.
- All 256 dims are active under frame-level std > 0.1 for both runs.
- Therefore the current 256x quality gap is not caused by low effective rank or dead latent dimensions.
- The bottleneck is more likely temporal compression/detail allocation: V16.2 has fewer latent frames and must encode more articulation and high-frequency phase detail per frame.
