# B1 — Global and local spectrum (V16.3 + LJSpeech val)

Date: 2026-05-10

> **Erratum (2026-05-10, after B2.5-S1)**: the originally reported
> `local_ID = 35.1` came from a pairwise-distance routine with a
> numerical issue at large N. After switching to a dot-product-based
> distance routine and checking TwoNN's finite-sample bias against a
> synthetic Gaussian calibration, the correct values are:
>
> - TwoNN at N≥8000: **26.4**, CI [23.5, 27.6]
> - Levina-Bickel MLE (k=20) at N=4000: 73 (noisy)
> - Calibration shows both estimators under-count true d at d>40.
>   Calibrated estimate: **local ID ≈ 60-100**, not 35.
>
> The **qualitative direction is preserved** — local ID < global rank
> — but the factor is about 2-3×, not 5×. The interpretation section
> below is updated inline. See `b2_5_sanity.md` for details.

## Setup

- Checkpoint: `outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt`
  (epoch 13, best val SNR 20.73 dB, latent_dim=384, stride=256)
- Dataset: LJSpeech, `data/ljspeech_train_splits.json`, 128 samples
  selected with seed 20260510
- Segment length: 65 536 samples (= 256 frames per utterance at
  stride=256)
- Total frames analysed: 32 768
- All 384 latent channels are active (std > 0.1), no dead dims
- Bootstrap: 20 resamples over utterances, seed 20260510

Artifacts:

- `data/b1_v16_3_ljspeech/b1_summary.json`
- `data/b1_v16_3_ljspeech/b1_spectrum.npz`
- `data/b1_v16_3_ljspeech/figures/b1_cumulative_variance.png`
- `data/b1_v16_3_ljspeech/figures/b1_dim_std_histogram.png`
- `data/b1_v16_3_ljspeech/figures/b1_local_id.png`
- `data/b1_v16_3_ljspeech/figures/b1_temporal_autocorr.png`

## Key numbers

### Frame-level spectrum

| Metric | Value | 95% CI (boot) |
|--------|-------|---------------|
| entropy rank | 181.4 | [177.6, 183.2] |
| participation rank | 106.2 | — |
| rank_90 | 189 | — |
| rank_95 | **236** | [234.0, 236.0] |
| rank_99 | 315 | — |

Matches prior V16.3 PCA results (234 components for 95% variance). Our
expectation range in `plan.md` was rank_95 ∈ [200, 280]. Confirmed.

### Utterance-mean spectrum

| Metric | Value | Note |
|--------|-------|------|
| entropy rank | 52.7 | — |
| rank_90 | 54 | — |
| rank_95 | 69 | — |
| rank_99 | 94 | — |

**Caveat on bootstrap CI**: bootstrap with replacement on 128
utterances yields ~81 unique utterances per resample, which caps the
bootstrap SVD rank and makes the reported CI [47.5, 53.0] an
under-estimate. The point estimate 69 on the full 128 is the correct
figure to cite. For B1 we report this as "rank_95 = 69 (bootstrap CI
biased down)". For future blocks where we need mean-level bootstrap
CI, we will use a larger utterance set or a without-replacement
subsampling scheme.

### Per-dimension activity

| Statistic | Frame-level | Utterance-mean |
|-----------|-------------|----------------|
| std mean | 0.163 | 0.012 |
| std min | 0.124 | — |
| std max | 0.267 | — |
| dead dims (std < 0.01) | 0 | — |
| active dims (std > 0.1) | 384 | — |

All 384 latent channels carry non-trivial variance frame-to-frame. The
utterance-mean std being 14× smaller than the frame-level std is
expected and confirms that most channel variation is within-utterance,
not between-utterance.

### Local intrinsic dimension (TwoNN, k=2)

| Metric | Value | 95% CI |
|--------|-------|--------|
| local_ID_median (numerically stable rerun, N=16000) | **26.4** | [23.5, 27.6] |
| local_ID_median (original buggy run, N=4000) | ~~35.1~~ | withdrawn |
| calibrated true local ID | ≈ 60-100 | (from B2.5-S1) |
| local / global ratio (calibrated) | ≈ 0.3-0.5 | (from B2.5-S1) |

Configuration: 4 000-16 000 frames subsampled per bootstrap run, 5
bootstrap runs.

See B2.5-S1 for the calibration curve and bug description. **The
revised figure is used for citation. The original 35.1 is retained
only so the erratum trail is traceable.**

Even under the more conservative calibrated estimate, local ID
(60-100) is below the global rank (236), so the qualitative claim
"local < global" is preserved. The earlier framing of a sharp 5×
gap is withdrawn.

### Temporal autocorrelation

| Lag (frames) | mean autocorrelation |
|--------------|----------------------|
| 0 | 1.000 |
| 1 | 0.089 |
| 2 | 0.027 |
| 3 | 0.017 |
| 4 | 0.006 |
| 8 | -0.003 |
| 16 | -0.005 |

Effective frame count per 256-frame segment: **217** (CI [174.5, 256]).

Temporal correlation is surprisingly weak — lag-1 autocorrelation only
0.09. V16.3 latent has largely decorrelated neighboring frames at the
86 Hz latent frame rate. Effective frames of ~217 out of 256 nominal
≈ 0.85 × nominal suggests little redundancy along time.

## Interpretation

1. **Global rank is genuine.** 236 dims carry 95% of frame variance and
   the bootstrap CI is tight. The "234 PCA components at 95%" figure
   from the original V16.3 report reproduces.
2. **Local manifold is smaller than global rank.** After calibration
   against known-d Gaussian baselines (B2.5-S1), the true local
   intrinsic dimension is ≈ 60-100, not 35.1. Still less than 236,
   but not by the original 5× factor. Around 2-3×.
3. **The implication for "collapse"**: V18's rank=10 is a **real**
   reduction relative to V16.3's calibrated local ID of 60-100. The
   sharp rhetorical framing from the first B1 draft ("V18 rank=10 is
   close to V16.3 local ID") is withdrawn. V18 still lost substantial
   local capacity.
4. **Low temporal autocorrelation** means that the encoder is not
   encoding slowly-varying prosody through simple smoothing. Whatever
   prosody information is in the latent, it is already in a
   frame-local format.

These interpretations are consistent with V16.3 being a high-quality
reconstruction AE: local richness per frame is high enough to carry
acoustic detail (local ID in the 60-100 range), and global richness
beyond that is available for slow factor variation across frames.

## Decisions for later blocks

- B2 and B3 should report both global-rank and local-ID-scaled numbers
  when comparing subspace sizes. A phoneme subspace of 20 dims is
  "local-dimensional" (~57% of local ID), not "small" (~8% of global
  rank).
- B6 comparison with V18 should measure local ID on V18 too. If V18
  local ID is 10 (matching the global rank=10), the entire latent has
  become locally low-dim. If V18 local ID is 25-30, V18 has preserved
  local structure but lost the slow-direction capacity.

## Status

**B1 complete on V16.3 + LJSpeech.** The CosyVoice-side run of B1 is
not strictly required for later blocks (speaker-independent rank/ID
transfer across distributions is not the main question), but we may
still run it as a quick cross-check.

Citable handles for downstream docs:

- `B1.rank_95 = 236 (CI 234-236)` on LJSpeech
- `B1.entropy_rank = 181.4 (CI 177.6-183.2)` on LJSpeech
- `B1.local_ID_TwoNN_raw = 26.4 (CI 23.5-27.6)` on LJSpeech (numerically stable version)
- `B1.local_ID_calibrated ≈ 60-100` (soft range from B2.5-S1 calibration)
- `B1.effective_frames_per_sample = 217` on LJSpeech (out of 256 nominal)
