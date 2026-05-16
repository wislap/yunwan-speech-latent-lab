# V16.3 Latent Anatomy — Master Results

This is the canonical place to cite anatomy findings elsewhere in the
project. Every row is a named metric with its 95% CI (if applicable),
dataset, and block of origin.

Entries are filled in as blocks complete. Blocks not yet run show "—".

## Block B1 — Global and local spectrum

| Metric | Dataset | Value | 95% CI |
|--------|---------|-------|--------|
| `B1.rank_entropy` | LJSpeech | 181.4 | [177.6, 183.2] |
| `B1.rank_95` | LJSpeech | 236 | [234, 236] |
| `B1.rank_99` | LJSpeech | 315 | — |
| `B1.local_ID_TwoNN_raw` | LJSpeech | 26.4 | [23.5, 27.6] (N≥8000, numerically-stable rerun) |
| ~~`B1.local_ID_median` (old)~~ | ~~LJSpeech~~ | ~~35.1~~ | withdrawn (B2.5-S1 bug fix) |
| `B1.local_ID_calibrated` | LJSpeech | 60-100 | soft range, see b2_5_sanity.md |
| `B1.local_global_ratio_calibrated` | LJSpeech | 0.3-0.5 | soft |
| `B1.effective_frames_per_sample` | LJSpeech | 216.9 | [174.5, 256.0] |
| `B1.dead_dims_frac` | LJSpeech | 0 / 384 | — |
| `B1.mean_rank_95` | LJSpeech | 69 | biased (see b1_spectrum.md) |

## Block B2.5 — Sanity checks (TwoNN calibration, full-384 linear probe, shuffle null)

### S1: TwoNN calibration

| True d (synthetic) | TwoNN | L-B (k=20) |
|--------------------|-------|------------|
| 10 | 6.4 | 9.3 |
| 20 | 11.6 | 15.8 |
| 40 | 19.2 | 25.5 |
| 80 | 29.6 | 38.9 |
| 160 | 44.6 | 58.2 |

Both estimators under-count at d > 40. TwoNN factor 0.28-0.64, L-B
factor 0.36-0.93.

### S2/S3: Full-384 linear probe vs shuffle null

| Label | Real | Shuffle null | Real > null? |
|-------|------|--------------|--------------|
| `B2_5.linear_phoneme_acc` | 0.068 | 0.075 | **no** |
| `B2_5.linear_phoneme_top5` | 0.340 | 0.330 | **no** |
| `B2_5.linear_speaker_acc` | 0.618 (chance 0.25) | 0.250 | yes |
| `B2_5.linear_f0_r2` | 0.199 | -0.011 | yes |
| `B2_5.linear_energy_r2` | 0.809 | -0.007 | yes |
| `B2_5.linear_centroid_r2` | 0.515 | -0.009 | yes |
| `B2_5.linear_voiced_r2` | 0.189 | -0.009 | yes |

**Headline findings**:

- Speaker, energy, centroid linearly readable from full 384-dim latent.
- **Phoneme not linearly readable** (indistinguishable from shuffled).
- Distributed encoding confirmed: best-single-PC R² < 0.1 for every
  label, but full-384 R² up to 0.81. Jump factor ≥ 13× for energy.

## Block B2 — PCA directions vs semantic labels

| Metric | Dataset | Value | Note |
|--------|---------|-------|------|
| `B2.K_phoneme` (of 32) | CosyVoice | 0 | All unassigned |
| `B2.K_speaker` (of 32) | CosyVoice | 0 | All unassigned |
| `B2.K_prosody` (of 32) | CosyVoice | 0 | All unassigned |
| `B2.K_unassigned` (of 32) | CosyVoice | 32 | thr=0.30 |
| `B2.best_phoneme_eta2_cv` | CosyVoice | 0.003 | 5-fold |
| `B2.best_speaker_eta2_cv` | CosyVoice | 0.013 | 5-fold |
| `B2.best_f0_r2_cv` | CosyVoice | ≤ 0 | voiced frames |
| `B2.best_energy_r2_cv` | CosyVoice | 0.061 | 5-fold |
| `B2.best_centroid_r2_cv` | CosyVoice | 0.021 | 5-fold |
| `B2.multivariate_energy_r2_insample` | CosyVoice | 0.355 | all 32 PCs jointly |
| `B2.multivariate_f0_r2_insample` | CosyVoice | 0.013 | all 32 PCs jointly |
| `B2.multivariate_centroid_r2_insample` | CosyVoice | 0.099 | all 32 PCs jointly |

## Block B3 — Supervised subspaces and principal angles

| Metric | Dataset | Value | 95% CI |
|--------|---------|-------|--------|
| `B3.phoneme_subspace_dim` | CosyVoice | 30 | variance target 0.90 |
| `B3.phoneme_variance_fraction_on_frames` | CosyVoice | 0.306 | — |
| `B3.phoneme_within_class_frac` | CosyVoice | 0.998 | — |
| `B3.phoneme_between_class_frac` | CosyVoice | 0.002 | — |
| `B3.speaker_subspace_dim` | CosyVoice | 2 | 4 speakers |
| `B3.speaker_variance_fraction_on_frames` | CosyVoice | 0.012 | — |
| `B3.speaker_variance_fraction_on_utt_means` | CosyVoice | 0.441 | — |
| `B3.principal_angle_mean_deg` | CosyVoice | **30.18** | observed |
| `B3.principal_angle_min_deg` | CosyVoice | 25.93 | observed |
| `B3.shuffle_null_mean_angle_deg` | CosyVoice | 64.11 | [59.79, 68.47] |
| `B3.shuffle_p_mean` | CosyVoice | < 0.001 | 0 of 1000 |

**Headline**: phoneme and speaker subspaces overlap heavily (observed
30° principal angle vs null 64°, p < 0.001). V16.3 latent is
factor-mixed, not axis-aligned.

## Block B4 — Frame-level linear vs MLP probes

| Metric | Dataset | Linear | MLP | Gap |
|--------|---------|--------|-----|-----|
| phoneme accuracy | CosyVoice | — | — | — |
| speaker accuracy | CosyVoice | — | — | — |
| F0 bin accuracy | CosyVoice | — | — | — |
| energy bin accuracy | CosyVoice | — | — | — |
| voicedness accuracy | CosyVoice | — | — | — |

## Block B5 — Latent curvature

| Direction family | Median κ | IQR | Notes |
|------------------|----------|-----|-------|
| `U_phoneme` | — | — | — |
| `U_speaker` | — | — | — |
| `U_residual` | — | — | — |
| top-PCA 1-8 | — | — | — |

## Block B6 — V16.3 vs V18

Filled in as a complete delta table once all earlier blocks are
reproduced on V18. See `b6_v16_vs_v18.md` for the detailed side-by-side.

## Interpretation (to be written)

Once B3 completes, this section records the top-level claims that
other docs can cite. For example:

- "V16.3 phoneme and speaker subspaces are (aligned / mixed / orthogonal)
  by principal-angle test, p=X vs shuffle null."
- "V18 supervision collapse reduces effective rank from R_v16 to R_v18
  while preserving the phoneme subspace variance fraction within δ."
