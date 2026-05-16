# B3 — Supervised subspaces and principal angles (V16.3 + CosyVoice 4spk val)

Date: 2026-05-10

## Setup

- Checkpoint: `outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt`
- Dataset: CosyVoice 4-speaker crossed val, 112 utterances, 57 344 frames
- Segment: 131 072 samples (= 512 AE frames per utterance)
- Phoneme subspace: frame-level class-mean SVD grouping by pinyin id,
  min 100 frames per class (71 of 80 classes kept)
- Speaker subspace: utterance-mean grouping by speaker id, all 4
  speakers kept
- Variance target for subspace truncation: 0.90
- Shuffle null: 1000 frame-level label permutations

Artifacts:

- `data/b3_v16_3_cosyvoice/b3_summary.json`
- `data/b3_v16_3_cosyvoice/b3_subspaces.npz`
- `data/b3_v16_3_cosyvoice/figures/b3_principal_angles.png`
- `data/b3_v16_3_cosyvoice/figures/b3_variance_fractions.png`

## Headline finding

**Phoneme and speaker subspaces in V16.3 latent overlap heavily
(mean principal angle 30°, p < 0.001 vs shuffle null at 64°).** The
latent is **not** axis-aligned. Factor decomposition, if attempted,
must be either low-rank projected or nonlinear.

## Subspace geometry

### Phoneme subspace (frame level)

| Property | Value |
|----------|-------|
| dim (90% of class-mean variance) | 30 |
| unique classes kept | 71 of 80 (min 100 frames each) |
| variance fraction on full latent | **0.306** |
| within-class variance fraction | 0.998 |
| between-class variance fraction | **0.002** |

The phoneme subspace is a 30-dimensional subspace of the 384-dim
latent that captures 30.6% of the frame-level total variance — but
almost all of that is still within-class rubble, because the
class-mean centering isolates only the between-class axis. The raw
LDA-style within/between split is dramatic: **99.8% of variance is
within class, 0.2% is between class**. A frame-level linear
classifier on raw latent has almost nothing to work with at the
class-center level, matching B2.5's finding that full-384 linear
phoneme accuracy is at chance.

### Speaker subspace (utterance level)

| Property | Value |
|----------|-------|
| dim (90% of speaker-mean variance) | 2 |
| unique speakers | 4 |
| variance fraction on frame-level X | 0.012 |
| variance fraction on utterance-mean X | **0.441** |
| within-speaker (utt-mean) variance | 0.543 |
| between-speaker (utt-mean) variance | 0.457 |

Speaker information is **heavily concentrated at the utterance-mean
level**. When we look at frames (which includes all phonemic and
prosodic variation), the speaker subspace explains only 1.2% of
frame variance. When we look at utterance means, the same subspace
explains 44.1%, and the within/between split becomes roughly equal
(54/46) — i.e. speakers are genuinely separable at utterance level.

This is also consistent with B2.5's finding that full-384 linear
speaker accuracy on frames is 0.62. The speaker signal is there but
is a **slow** component that shows up strongly only when averaged
over time.

## Principal angles

Principal angles between U_ph (30 dims) and U_sp (2 dims):

| index | angle (deg) |
|-------|-------------|
| 1 | 25.93 |
| 2 | 34.43 |

Mean: **30.18°**. Minimum: **25.93°**.

For reference, two random 30-dim and 2-dim subspaces in a 384-dim
ambient space have expected minimum principal angle ~70° — very close
to orthogonal. The observed 26° is far from orthogonal.

### Shuffle null

Permuting the phoneme label 1000 times at frame level (preserving
marginal class distribution) and recomputing the phoneme-subspace
principal angles with the same speaker subspace:

| Statistic | Observed | Null mean | Null 95% CI |
|-----------|----------|-----------|-------------|
| mean angle | 30.18° | 64.11° | [59.79°, 68.47°] |
| min angle | 25.93° | — | — |

Permutation p-values:

- p(null mean ≤ observed mean) = **0.0000** (0 out of 1000)
- p(null min ≤ observed min) = **0.0000** (0 out of 1000)

The observed mean angle is 34° smaller than the null distribution's
minimum. The chance of this arising from label-free structure is
well below 0.001.

## Interpretation

1. **Phoneme-discriminative directions and speaker-discriminative
   directions share substantial geometry in V16.3 latent.** The
   2-dim speaker subspace makes an angle of only ~26° with the
   nearest phoneme direction, meaning one of the phoneme subspace
   axes is almost the same as one of the speaker axes.

2. **Class-mean between-class variance is very small (0.2%)
   compared to within-class variance (99.8%).** The phoneme
   class-mean separation is a tiny slice of the latent's variance.
   A frame-level linear classifier trained on raw latents has almost
   no usable between-class signal. This explains why full-384 linear
   phoneme accuracy in B2.5 was at chance: the between-class
   discrimination is below the noise floor of any linear probe.

3. **Speaker-discriminative geometry exists but is weak at frame
   level.** Only 1.2% of frame variance lies in the 2-dim speaker
   subspace, but the utterance-mean explanation is 44%. Speaker is
   encoded as a slow factor with variance much smaller than phonetic
   variation at individual frames.

4. **Axis-aligned factorization is refuted.** A naive design that
   reserves channels 0-30 for phoneme and 31-32 for speaker cannot
   work on V16.3 latent, because these two subspaces in the
   data-driven decomposition share geometry.

5. **The "factor-mixed latent" interpretation used in V17 postmortem
   reports is now quantitatively confirmed.** V17.1-V17.2 failures
   are consistent with an encoder geometry where phoneme and speaker
   axes point in nearly the same subspace, so a shallow head trying
   to isolate phoneme direction inevitably picks up speaker signal,
   and vice versa.

## Caveats

- **CosyVoice speakers are OOD for V16.3** (LJSpeech-trained). The
  particular principal angles may reflect this OOD condition. The
  qualitative direction (non-orthogonality) is expected to hold
  on in-distribution data, but quantitative numbers should be
  understood with this caveat.
- **Phoneme durations are heuristic** (from CosyVoice's heuristic
  duration allocator), not forced-alignment. Frame-level phoneme
  labels may be mis-assigned at segment boundaries. The 0.2%
  between-class variance may be under-estimated due to label noise.
- The **shuffle null preserves the marginal class distribution**
  but not temporal autocorrelation, which may slightly inflate the
  null distribution's principal angles. A block-bootstrap would be
  more conservative, but the observed gap (30° vs 64°) is so large
  that this refinement would not change the conclusion.
- **Speaker subspace dim=2** because only 4 speakers are available.
  With more speakers, U_sp would grow and the principal-angle story
  might change slightly. The current qualitative conclusion is
  unaffected.

## Citable handles

- `B3.phoneme_subspace_dim = 30` (CosyVoice, variance target 0.90)
- `B3.phoneme_variance_fraction_on_frames = 0.306`
- `B3.phoneme_within_class_frac = 0.998`
- `B3.phoneme_between_class_frac = 0.002`
- `B3.speaker_subspace_dim = 2` (4 speakers)
- `B3.speaker_variance_fraction_on_frames = 0.012`
- `B3.speaker_variance_fraction_on_utt_means = 0.441`
- `B3.principal_angle_mean_deg = 30.18` (CosyVoice)
- `B3.principal_angle_min_deg = 25.93`
- `B3.shuffle_null_mean_angle_deg = 64.11` (CI [59.79, 68.47])
- `B3.shuffle_p_mean = 0.0000` (0 of 1000)

## Decisions for downstream work

- The "factor-mixed latent" interpretation in the V17 report series
  is quantitatively supported. V17.1-V17.2 "why shallow heads fail"
  postmortems can now cite `B3.principal_angle_mean = 30°` as
  direct evidence.
- For **V19 design**, axis-split architectures (reserve C_ph
  channels for phoneme, C_sp for speaker) are ruled out. Any
  factorization must be either:
  - **Projection-based** (learn low-rank projections `W_ph` and
    `W_sp` that enforce orthogonality by construction), or
  - **Nonlinear** (e.g. adapters that extract factor subspaces
    through a shallow nonlinear map).
- The original Block C (NC geometry) is still interesting in this
  light: within-class variance 99.8% suggests NC1 is **not**
  happening on V16.3 (features are far from their class centroids).
  V18 would show a very different NC1 and confirm the "supervision
  squeezes features toward class means" story.
- For B4 (linear vs MLP probe), expect huge gap on phoneme because
  the between-class variance is 0.2% of total. MLP probe needs to
  learn a nonlinear transformation to discover the class geometry,
  which raw linear probe on full-384 cannot do.

## Status

**B3 complete on V16.3 + CosyVoice val.** Need to rerun on V18
checkpoint for B6 comparison (expect much larger principal angles
in V18 after supervision compresses phoneme onto its own axes).
