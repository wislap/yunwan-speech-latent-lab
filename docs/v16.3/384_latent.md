# V16.3 256x Dim384 Capacity Test

## 目标

V16.3 测试一个问题：V16.2 的 256x 高频/口齿差距，是否主要来自 latent channel capacity 不够。

V16.2 的有效秩显示 latent 没有塌缩，但 256 dim 已经被充分使用：

- frame effective rank: 144.3
- frame 99% dims: 229 / 256
- active dims: 256 / 256

所以 V16.3 只扩大 latent width，不引入新 loss 或新 decoder。

## 配置

- 配置文件：`autoencoder/conf/experiment/v16_3_256x_dim384_time.yaml`
- 启动：

```bash
python -m autoencoder.train experiment=v16_3_256x_dim384_time runtime=remote
```

## 与 V16.2 的差异

- `latent_dim: 256` -> `latent_dim: 384`

保持不变：

- stride `[2, 4, 4, 8] = 256x`
- `sample_latent: false`
- pure time-domain decoder
- no GAN
- no Vocos
- no PhaseRefiner
- no multi-band STFT in the first capacity test

## 判定标准

V16.3 成立需要同时看到：

- SNR 比 V16.2 同 epoch 更快或更高
- 3.5-9 kHz band SNR 改善
- phase coherence/phase MAE 改善
- effective rank 不再贴近 384 上限
- 听感口齿比 V16.2 更清楚

如果 dim384 明显改善，说明 256x 的主瓶颈是 channel capacity。  
如果 dim384 改善有限，优先考虑 192x 或高频 band loss，而不是继续堆复杂 decoder。

## 启动记录

Date: 2026-05-03

- Remote log: `/root/autodl-tmp/project/v16_3_launch.log`
- Remote output dir: `/root/autodl-tmp/project/outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main`
- Remote train process: `PID 11486`
- Smoke:
  - params: 127,472,641
  - encoder: 57,992,064
  - decoder: 69,480,577
  - input 8192 samples -> latent `[1, 384, 32]`
  - output length aligned
  - deterministic encode confirmed
- First train log line:
  - epoch 0 step 50, `G=2.8256`

Note: V16.2 is still running in parallel, so wall-clock speed may be slower than V16.2's original single-run speed.

## 暂停与测试

Date: 2026-05-03

Training was paused after epoch 16. The best checkpoint is epoch 13:

| epoch | SNR | STFT |
|---:|---:|---:|
| 1 | 7.19 | 1.4264 |
| 3 | 10.24 | 1.1583 |
| 5 | 13.42 | 0.9686 |
| 7 | 16.90 | 0.8628 |
| 9 | 18.25 | 0.7563 |
| 11 | 18.50 | 0.7478 |
| 13 | 20.73 | 0.6351 |
| 15 | 20.47 | 0.6412 |

Generated diagnostics:

- [recon_report/epoch_curves.png](figures/recon_report/epoch_curves.png)
- [recon_report/latest_band_snr.png](figures/recon_report/latest_band_snr.png)
- [recon_report/latest_frequency_response_bias.png](figures/recon_report/latest_frequency_response_bias.png)
- [recon_report/latest_phase_error.png](figures/recon_report/latest_phase_error.png)
- [recon_report/latest_phase_coherence.png](figures/recon_report/latest_phase_coherence.png)
- [recon_report/recon_diagnostics_detail.csv](figures/recon_report/recon_diagnostics_detail.csv)
- [effective_rank/effective_rank_summary.csv](figures/effective_rank/effective_rank_summary.csv)
- [effective_rank/effective_rank_summary.png](figures/effective_rank/effective_rank_summary.png)

### Effective Rank

| run | latent dim | stride | frame effective rank | frame 99% dims | mean effective rank | mean 99% dims |
|---|---:|---:|---:|---:|---:|---:|
| V16.1 | 256 | 128x | 125.3 | 217 | 26.3 | 53 |
| V16.2 | 256 | 256x | 144.3 | 229 | 38.0 | 57 |
| V16.3 | 384 | 256x | 175.1 | 313 | 35.5 | 57 |

Interpretation:

- dim384 is being used; the extra channels are not dead.
- V16.3 frame-level rank rises from 144.3 to 175.1.
- 99% variance needs 313 dims, so 384d gives real headroom but is still meaningfully occupied.

### Band/Phase Result

At epoch 13, V16.3 improves full-band SNR over V16.2:

| run | full SNR | LSD |
|---|---:|---:|
| V16.2 epoch 13 | 22.00 | 7.16 |
| V16.3 epoch 13 | 22.98 | 7.19 |

High-frequency comparison at epoch 13:

| band Hz | V16.2 SNR | V16.3 SNR | V16.2 phase MAE | V16.3 phase MAE |
|---:|---:|---:|---:|---:|
| 2500-3500 | 16.36 | 15.93 | 2.7 | 2.7 |
| 3500-5000 | 16.81 | 15.93 | 3.2 | 3.3 |
| 5000-7000 | 12.09 | 13.70 | 4.3 | 3.4 |
| 7000-9000 | 11.36 | 11.92 | 3.7 | 3.4 |
| 9000-11025 | 1.16 | 1.68 | 36.0 | 31.1 |

Latest exported epoch 15 shows a stronger high-frequency benefit:

| band Hz | V16.2 SNR | V16.3 SNR | V16.2 phase MAE | V16.3 phase MAE |
|---:|---:|---:|---:|---:|
| 2500-3500 | 14.46 | 16.75 | 2.9 | 2.8 |
| 3500-5000 | 13.09 | 16.80 | 3.2 | 2.8 |
| 5000-7000 | 9.19 | 14.69 | 6.9 | 2.8 |
| 7000-9000 | 6.01 | 11.38 | 10.1 | 3.4 |
| 9000-11025 | -0.67 | 1.78 | 57.4 | 30.0 |

Conclusion:

dim384 gives a real capacity gain. It improves V16.2 by about +0.8 dB best checkpoint SNR and noticeably improves 5-9 kHz band SNR/phase stability. The remaining gap to V16.1 means 256x still has a temporal-detail bottleneck, but channel capacity is definitely part of the problem.

## Generalization Stress Test

Output:

- [generalization/per_case_metrics.csv](figures/generalization/per_case_metrics.csv)
- [generalization/feature_correlations.csv](figures/generalization/feature_correlations.csv)
- [generalization/v16_3_minus_v16_2_delta.csv](figures/generalization/v16_3_minus_v16_2_delta.csv)
- [generalization/snr_boxplot.png](figures/generalization/snr_boxplot.png)
- [generalization/time_local_snr_curves.png](figures/generalization/time_local_snr_curves.png)
- [generalization/boundary_local_snr_curves.png](figures/generalization/boundary_local_snr_curves.png)
- [generalization/feature_snr_correlation.png](figures/generalization/feature_snr_correlation.png)

Setup:

- 24 random full-length LJSpeech samples
- 24 variants with a short random inserted silence span
- 12 concatenated full-length sample pairs with a short silence seam
- 60 cases per model, 180 reconstructions total
- no audio is saved under docs

### SNR By Case Type

| run | full mean | full min | silence mean | silence min | concat mean | concat min |
|---|---:|---:|---:|---:|---:|---:|
| V16.1 | 23.35 | 19.93 | 23.27 | 19.94 | 23.60 | 22.22 |
| V16.2 | 19.63 | 15.33 | 19.58 | 15.30 | 20.02 | 18.41 |
| V16.3 | 20.55 | 16.75 | 20.53 | 16.67 | 21.03 | 19.31 |

### Boundary Sensitivity

Boundary error ratio is the local seam/silence-boundary error power divided by global error power.

| run | mean | median | max |
|---|---:|---:|---:|
| V16.1 | 0.53 | 0.34 | 1.64 |
| V16.2 | 0.35 | 0.23 | 1.26 |
| V16.3 | 0.34 | 0.22 | 1.18 |

Interpretation:

- Inserted silence does not create a new failure mode. Full and silence scores are nearly identical.
- Concatenation does not collapse reconstruction; concat is slightly easier on average because the selected pairs have favorable energy/spectral distribution.
- V16.3 slightly improves boundary robustness over V16.2.

### Sample Association

The strongest correlation with SNR is sample high-frequency content:

| run | spectral centroid r | 3.5-9k energy ratio r | RMS r |
|---|---:|---:|---:|
| V16.1 | -0.824 | -0.811 | 0.434 |
| V16.2 | -0.745 | -0.747 | 0.418 |
| V16.3 | -0.731 | -0.747 | 0.430 |

Interpretation:

- Hard samples are high-centroid/high-frequency samples.
- This confirms that the remaining failure mode is tied to high-frequency content, not silence insertion or concatenation boundaries.
- V16.3 improves V16.2 by about +0.9 to +1.0 dB across full/silence/concat cases.
- V16.3 gains are larger on higher-centroid samples: delta SNR vs spectral centroid has r=0.423; delta SNR vs 3.5-9k energy ratio has r=0.311.

## Latent PCA Directions For Noise Robustness

Output:

- [latent_pca_directions/v16_3/metadata.json](figures/latent_pca_directions/v16_3/metadata.json)
- [latent_pca_directions/v16_3/frame_p95_pca_directions.npz](figures/latent_pca_directions/v16_3/frame_p95_pca_directions.npz)
- [latent_pca_directions/v16_3/frame_p95_pca_component_summary.csv](figures/latent_pca_directions/v16_3/frame_p95_pca_component_summary.csv)
- [latent_pca_directions/v16_3/frame_p95_pca_directions_long.csv](figures/latent_pca_directions/v16_3/frame_p95_pca_directions_long.csv)
- [latent_pca_directions/v16_3/latent_mean.csv](figures/latent_pca_directions/v16_3/latent_mean.csv)
- [latent_pca_directions/v16_3/frame_pca_cumulative_variance.png](figures/latent_pca_directions/v16_3/frame_pca_cumulative_variance.png)

Setup:

- checkpoint: V16.3 best, epoch 13
- latent dim: 384
- total stride: 256
- samples: 128
- frames: 32,048
- segment length: 65,536 samples
- PCA view: frame-level latent vectors
- target cumulative explained variance: 95%

Result:

- p95 component count: 234
- each component is a unit vector in 384-dimensional latent channel space
- sign convention: largest absolute loading is positive

The NPZ contains:

```text
components                  [234, 384] float32 unit directions
mean                        [384] latent mean vector
explained_variance_ratio    [234]
cumulative_variance         [234]
eigenvalues                 [234]
pca_score_std               [234]
k_p95                       scalar, 234
```

For robustness tests, perturb a latent tensor `z [B, C, T]` along direction `u_i`:

```python
z_noisy = z + alpha * pca_score_std[i] * u_i[None, :, None] * noise_schedule
```

Recommended first sweep:

- component groups: top 1-8, 9-32, 33-96, 97-234
- alpha: 0.05, 0.1, 0.2, 0.5, 1.0
- schedules:
  - all frames
  - random 20% frames
  - high-energy frames only
  - silence frames only

Measure:

- reconstruction SNR drop
- 3.5-9 kHz SNR drop
- phase coherence drop
- ASR/WER if available
- whether high-rank directions cause localized high-frequency artifacts

## PCA-Shaped Latent Gaussian Noise Test

Output:

- [pca_noise_robustness/noise_summary.csv](figures/pca_noise_robustness/noise_summary.csv)
- [pca_noise_robustness/per_sample_noise_metrics.csv](figures/pca_noise_robustness/per_sample_noise_metrics.csv)
- [pca_noise_robustness/noise_robustness_curves.png](figures/pca_noise_robustness/noise_robustness_curves.png)
- [pca_noise_robustness/metadata.json](figures/pca_noise_robustness/metadata.json)

Noise definition:

```text
eps_i(t) ~ N(0, variance_scale * eigenvalue_i)
noise(t) = sum_i eps_i(t) * u_i
z_noisy = z + noise
```

Compared against channel-isotropic noise with similar relative latent RMS.

Setup:

- model: V16.3 best checkpoint
- samples: 24
- segment length: 65,536
- PCA directions: p95, 234 components
- variance scale: 0, 0.005, 0.01, 0.02, 0.05, 0.1
- modes: `pca`, `channel_iso`

Summary:

| mode | variance scale | latent noise RMS / latent RMS | SNR drop | 3.5-9k SNR drop | 3.5-9k coherence drop |
|---|---:|---:|---:|---:|---:|
| channel_iso | 0.005 | 0.071 | 0.26 | 0.11 | 0.0004 |
| channel_iso | 0.010 | 0.100 | 0.49 | 0.21 | 0.0007 |
| channel_iso | 0.020 | 0.141 | 0.93 | 0.42 | 0.0014 |
| channel_iso | 0.050 | 0.223 | 2.02 | 0.95 | 0.0035 |
| channel_iso | 0.100 | 0.315 | 3.38 | 1.73 | 0.0071 |
| pca | 0.005 | 0.072 | 1.18 | 0.18 | 0.0007 |
| pca | 0.010 | 0.101 | 2.11 | 0.35 | 0.0012 |
| pca | 0.020 | 0.143 | 3.48 | 0.68 | 0.0025 |
| pca | 0.050 | 0.227 | 6.06 | 1.51 | 0.0062 |
| pca | 0.100 | 0.320 | 8.52 | 2.61 | 0.0122 |

Interpretation:

- PCA-shaped noise is much more damaging than channel-isotropic noise at similar relative latent RMS.
- This means decoder sensitivity is aligned with the real high-variance latent manifold, not with arbitrary channel directions.
- Very small PCA-shaped variance already matters: `variance_scale=0.005` gives about 1.18 dB SNR drop.
- High-frequency degradation is smoother than full-band degradation but still monotonic.
- A practical DiT target should probably keep natural latent prediction noise below the `0.005-0.01` variance-scale range if we want minimal AE-decoder quality loss.

## Monte Carlo PCA-Shaped Noise Test

Output:

- [pca_noise_mc/mc_summary.csv](figures/pca_noise_mc/mc_summary.csv)
- [pca_noise_mc/mc_per_trial_metrics.csv](figures/pca_noise_mc/mc_per_trial_metrics.csv)
- [pca_noise_mc/mc_per_sample_sensitivity.csv](figures/pca_noise_mc/mc_per_sample_sensitivity.csv)
- [pca_noise_mc/mc_mean_p05_p95_curves.png](figures/pca_noise_mc/mc_mean_p05_p95_curves.png)
- [pca_noise_mc/snr_drop_violin.png](figures/pca_noise_mc/snr_drop_violin.png)
- [pca_noise_mc/pca_per_sample_snr_drop_heatmap.png](figures/pca_noise_mc/pca_per_sample_snr_drop_heatmap.png)
- [pca_noise_mc/channel_iso_per_sample_snr_drop_heatmap.png](figures/pca_noise_mc/channel_iso_per_sample_snr_drop_heatmap.png)

Setup:

- samples: 12
- repeats per nonzero noise scale: 16
- total trials: 2,328
- variance scales: 0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1
- modes: `pca`, `channel_iso`

### Full-Band SNR Drop

| mode | variance scale | mean | p05 | p95 |
|---|---:|---:|---:|---:|
| channel_iso | 0.002 | 0.10 | 0.08 | 0.14 |
| channel_iso | 0.005 | 0.26 | 0.19 | 0.35 |
| channel_iso | 0.010 | 0.50 | 0.38 | 0.65 |
| channel_iso | 0.020 | 0.95 | 0.73 | 1.22 |
| channel_iso | 0.050 | 2.06 | 1.67 | 2.59 |
| channel_iso | 0.100 | 3.47 | 2.89 | 4.21 |
| pca | 0.002 | 0.54 | 0.29 | 0.79 |
| pca | 0.005 | 1.22 | 0.68 | 1.73 |
| pca | 0.010 | 2.16 | 1.29 | 2.99 |
| pca | 0.020 | 3.59 | 2.27 | 4.75 |
| pca | 0.050 | 6.26 | 4.40 | 7.84 |
| pca | 0.100 | 8.77 | 6.54 | 10.50 |

### 3.5-9 kHz SNR Drop

| mode | variance scale | mean | p05 | p95 |
|---|---:|---:|---:|---:|
| channel_iso | 0.002 | 0.04 | 0.01 | 0.07 |
| channel_iso | 0.005 | 0.10 | 0.03 | 0.15 |
| channel_iso | 0.010 | 0.19 | 0.04 | 0.30 |
| channel_iso | 0.020 | 0.37 | 0.08 | 0.55 |
| channel_iso | 0.050 | 0.87 | 0.20 | 1.27 |
| channel_iso | 0.100 | 1.58 | 0.43 | 2.24 |
| pca | 0.002 | 0.08 | 0.02 | 0.13 |
| pca | 0.005 | 0.21 | 0.05 | 0.31 |
| pca | 0.010 | 0.40 | 0.11 | 0.58 |
| pca | 0.020 | 0.77 | 0.19 | 1.10 |
| pca | 0.050 | 1.71 | 0.47 | 2.30 |
| pca | 0.100 | 2.92 | 0.87 | 3.85 |

Interpretation:

- The Monte Carlo result confirms the single-draw test.
- PCA-shaped noise has much higher full-band impact than channel-isotropic noise at the same variance scale.
- At `variance_scale=0.005`, PCA-shaped noise has mean full-band drop 1.22 dB and p95 drop 1.73 dB.
- At `variance_scale=0.01`, p95 full-band drop is already about 2.99 dB.
- 3.5-9 kHz degradation is smaller than full-band degradation but monotonic and clearly separated between PCA-shaped and isotropic noise.
- A realistic DiT latent predictor should probably target PCA-shaped residual variance below 0.005 if the goal is near-transparent AE decoding.
