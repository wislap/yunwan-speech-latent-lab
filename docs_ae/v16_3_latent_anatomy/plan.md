# V16.3 Latent Anatomy — Plan

Date: 2026-05-10

## Purpose

Produce a set of internally-citable, statistically grounded measurements
about the geometry of the V16.3 reconstruction-only autoencoder latent
space, so that downstream design decisions (V19 architecture, collapse
mechanism claims, probe design) can rest on evidence rather than
intuition.

This is not a paper-grade theoretical study. It is a probe suite whose
results are meant to be reused ("we showed in B3 that ..."), not
re-argued.

## Design principles

- **Citable conclusions**: every block ends with a small number of named
  numerical results, each with a confidence interval where applicable.
- **Reproducibility first**: fixed seeds, fixed segment positions, fixed
  sample counts, fixed label pipeline. All configuration lives in
  `methodology.md` so that any result can be re-run and return the same
  numbers.
- **Defensible method**: every metric has a one-sentence justification.
  No technique is used that we cannot explain on the spot.
- **Falsifiable**: every block has pre-declared thresholds that
  distinguish "evidence supports X" from "evidence refutes X".
- **Not paper-theory**: we deliberately avoid Hessian eigenspectra on
  full parameters, strict neural-collapse claims on CTC, and other
  techniques whose correct handling exceeds the scope of this probe
  suite.

## Datasets

- **LJSpeech val subset**: 128 random full-length samples, fixed seed.
  V16.3's training distribution. Used for B1 baseline geometry, B2 PCA
  semantics, B3 phoneme subspace, B4 frame-level readability, B5
  curvature.
- **CosyVoice3 4-speaker crossed val**: 112 items, 28 text groups,
  existing phoneme alignments. Used for B2 (speaker label), B3 (speaker
  subspace), B4 (speaker probe), B6 (V16.3 vs V18 comparison on the
  same data the V18 run used).

Why two sets: V16.3 is LJSpeech-trained and single-speaker. Speaker
geometry cannot be measured on LJSpeech. CosyVoice introduces
distribution shift, which is called out explicitly and treated as a
limitation rather than hidden.

## Labels

All labels are computed once and cached.

- `phoneme`: pinyin+tone frame labels from CosyVoice manifest
  (`pinyin_durations`) where available; otherwise per-frame silence vs.
  voiced flag from a forced-alignment-free estimator.
- `speaker`: speaker id from CosyVoice manifest.
- `F0`: `pyin` frame-level, 10 ms hop mapped to AE frame rate by linear
  interpolation.
- `energy`: log RMS per AE frame.
- `centroid`: spectral centroid per AE frame using a 1024-sample FFT
  centered on the AE frame.
- `voicedness`: binary derived from `pyin` probability threshold.

Implementation in `autoencoder/scripts/anatomy/labels.py`.

## Block overview

| Block | Topic | V16.3 | V18 | Duration |
|-------|-------|:-----:|:---:|----------|
| B1 | Global + local spectrum | Yes | — | 1 day |
| B2 | PCA directions vs semantic labels | Yes | Yes | 1 day |
| B3 | Supervised subspaces + principal angles + shuffle null | Yes | Yes | 1.5 days |
| B4 | Frame-level linear / MLP probes | Yes | Yes | 0.5 days |
| B5 | Latent curvature along semantic directions | Yes | — | 1 day |
| B6 | V16.3 vs V18 comparison table | — | — | 0.5 days |

Total: 5.5 days of compute and analysis.

---

## Block B1 — Global and local spectrum

### Objective

Establish baseline global and local dimensionality of V16.3 latent.

### Procedure

1. Encode 128 LJSpeech samples, 65536 samples each, to frame-level
   latents. Collect all frames into one matrix `Z ∈ R^{N × 384}`.
2. Center and compute SVD. Report:
   - full singular value spectrum `σ_i`
   - cumulative variance curve
   - `rank_90`, `rank_95`, `rank_99`
   - entropy rank `exp(-Σ p_i log p_i)` where `p_i = σ_i² / Σ σ_j²`
   - participation ratio `(Σ σ_i²)² / Σ σ_i⁴`
3. Per-dimension activity: std, kurtosis, fraction of dead dims
   (std < 0.01).
4. Local intrinsic dimension via TwoNN (Facco et al. 2017), k=10 nearest
   neighbors. Report median, IQR, per-class bootstrap 5 times.
5. Temporal autocorrelation: for each latent channel, autocorrelation at
   lags 1, 2, 4, 8, 16 frames. Effective frame count defined as `T /
   (1 + 2 Σ_{k=1}^{T-1} (1 - k/T) ρ_k)`.

### Falsifiable predictions

- `rank_95` is expected in [200, 280] based on V16.3 prior PCA work.
- `local_ID / global_rank` ratio: if ≈ 1, latent uses nearly all of its
  global dimensions at every frame. If << 1 (say < 0.3), most global
  dimensions encode slow / speaker / prosody factors that vary across
  frames but not within a single frame.
- Effective frame count is expected to be much smaller than nominal
  frame count, indicating temporal smoothness.

### Citable outputs

- `B1.rank_entropy`, `B1.rank_95`, `B1.local_ID_median`, `B1.local_ID_iqr`,
  `B1.effective_frames_per_sample`.

---

## Block B2 — PCA directions vs semantic labels

### Objective

For each of the top-32 PCA directions of V16.3 latent, determine how
much it linearly encodes each known semantic factor.

### Procedure

1. Compute top-32 PCA directions `u_1 ... u_32` on the LJSpeech frame
   matrix from B1. Also on the CosyVoice frame matrix (separately; PCA
   directions can differ).
2. For each direction `u_i`, per-frame projection `α_i(t) = u_i^T Z(t)`.
3. For each label `y ∈ {phoneme_bag, speaker_id, F0, log_energy, centroid,
   voicedness}`:
   - Linear probe: 5-fold CV `R²` predicting `y` from `α_i(t)`
     (classification uses accuracy; continuous uses R²)
   - Report mean R² / accuracy and 95% CI via Fisher z-transform.
4. Assign each direction a primary label via `argmax_y R²` if `max R² >
   0.3`; otherwise label "unassigned".
5. Produce a 32 × 6 heatmap of R² / accuracy.

### Falsifiable predictions

- Fraction of directions labelled phoneme / speaker / prosody /
  unassigned under the R² > 0.3 threshold. Our baseline expectation
  before running: roughly 30% phoneme, 10% speaker (only on CosyVoice),
  20% prosody-related, 40% unassigned.
- "unassigned" > 60% would mean most variance is in nonlinear /
  unmeasured factors and linear subspace analysis has a low ceiling.

### Citable outputs

- `B2.K_phoneme`, `B2.K_speaker`, `B2.K_prosody`, `B2.K_unassigned`
- `B2.max_R2_by_label` (single number per label indicating best
  direction's R²)

---

## Block B3 — Supervised subspaces and principal angles

### Objective

Find the phoneme-aligned subspace and speaker-aligned subspace of
V16.3 latent, then quantify how much they overlap.

### Procedure

1. **Phoneme subspace** (CosyVoice):
   - Group frames by pinyin class (collapse silence, drop classes with
     fewer than 50 frames).
   - Compute class mean vectors `μ_k`.
   - Stack and center: `M = [μ_1 - μ̄, ..., μ_K - μ̄]`.
   - SVD on `M`. Phoneme subspace `U_ph` = left singular vectors of
     cumulative 90% variance, with dim `d_ph`.
2. **Speaker subspace** (CosyVoice):
   - Utterance-mean vectors, grouped by speaker id.
   - Same SVD procedure; `U_sp` with dim `d_sp`.
3. **Principal angles**:
   - Compute principal angles between `U_ph` and `U_sp` via `svd(U_ph^T
     U_sp)`. Report full angle spectrum.
   - Summary statistics: mean angle, number of angles below 30°, number
     above 60°.
4. **Shuffle null**:
   - Permute phoneme labels across frames (preserving marginal
     distribution), recompute `U_ph_null`.
   - Compute principal angles `U_ph_null` vs `U_sp`.
   - Repeat 1000 times, build null distribution of mean principal angle.
   - p-value = fraction of null runs with mean angle lower than
     observed.
5. **Variance fractions**:
   - `v_ph = ||P_{U_ph} Z||² / ||Z||²` averaged over frames
   - `v_sp` on utterance means
   - Report both.

### Falsifiable predictions

- If V16.3 latent has axis-aligned factorization: observed mean
  principal angle ≫ 30°, p << 0.01.
- If latent is factor-mixed: mean angle close to shuffle null's median,
  p > 0.1.
- If factor-mixed, the interpretation used throughout the rest of this
  work is supported: supervision pushes shared directions, which is why
  adding a supervision loss compresses shared dimensions.

### Citable outputs

- `B3.d_phoneme`, `B3.d_speaker`
- `B3.mean_principal_angle_deg`, `B3.n_angles_below_30deg`
- `B3.shuffle_null_mean_angle_deg`, `B3.shuffle_null_p`
- `B3.v_phoneme`, `B3.v_speaker`

### Scope note

Prosody and high-frequency subspaces are not constructed here.
F0/energy/centroid are scalar and do not define a natural subspace of
sufficient dimension to be meaningful under principal-angle analysis.
Their information content is covered by B2 (per-direction R²) and B4
(frame-level readability).

---

## Block B4 — Frame-level linear and MLP probes

### Objective

Measure how much information lives in V16.3 latent about each semantic
factor, and how much of it is linearly readable.

### Procedure

1. Freeze encoder. Extract frame-level latents with aligned labels for
   all 112 CosyVoice val frames.
2. For each label `y`:
   - **Linear probe**: logistic regression (classification) or ridge
     regression (continuous), 5-fold CV on utterance-split folds (no
     frame leakage), report accuracy / R².
   - **MLP probe**: 2-layer MLP with hidden 128, same CV scheme.
3. For each label compute gap `Δ = MLP_score - Linear_score`.
4. Bootstrap resample 5 times over the sample axis for each number.

### Falsifiable predictions

- If linear probe accuracy on phoneme is high (> 0.6) while previous
  utterance-mean probes failed: the V17.2 failure is a pooling
  problem, not an encoding problem.
- If linear is much lower than MLP (Δ > 0.15): information is
  distributed nonlinearly, shallow heads are insufficient.

### Citable outputs

- `B4.linear_phoneme`, `B4.mlp_phoneme`, `B4.gap_phoneme`
- Same for speaker, F0 bin, energy bin, voicedness.

---

## Block B5 — Latent curvature along semantic directions

### Objective

For each semantic subspace identified in B3, determine how much
reconstruction loss increases per unit perturbation along that
subspace. Flat directions are the ones supervision will preferentially
push.

### Procedure

1. Fix V16.3 decoder. Take a held-out set of 8 samples.
2. For each sample, encode to `Z ∈ R^{384 × T}`.
3. For each test direction family `D ∈ {U_ph, U_sp, U_residual,
   top_PCA_1-8}`:
   - Pick 10 random unit vectors in that subspace (residual subspace is
     orthogonal complement of `U_ph ∪ U_sp` within the top-234 PCA
     basis).
   - For each `α ∈ {0, 0.1, 0.2, 0.5, 1.0}`, compute `L_recon(Z + α d)`
     and extract decoder output `x_hat`, measure SNR drop.
   - Fit quadratic `ΔL ≈ κ α²` to get curvature `κ` per direction.
4. Aggregate by direction family: median κ, IQR, 5-bootstrap CI.

### Falsifiable predictions

- If κ_phoneme < κ_residual significantly: phoneme subspace is flat
  under recon loss, supporting the "supervision exploits flat
  directions" mechanism.
- If κ_phoneme ≈ κ_speaker ≈ κ_residual: recon loss is isotropic on the
  latent, supervision collapse needs a different explanation.
- Top-PCA directions should be more curved than random or residual
  directions (since PCA picks high-variance directions, which
  correspond to decoder-active directions).

### Citable outputs

- `B5.kappa_phoneme`, `B5.kappa_speaker`, `B5.kappa_residual`
- `B5.kappa_top_PCA_1_to_8_median`

### Scope note

We do not compute parameter-space Hessian eigenvalues. Latent-space
curvature is measured via finite differences with the decoder frozen,
which is numerically stable and conceptually clean. Any claim about
loss landscape geometry is restricted to the latent input surface of
the decoder.

---

## Block B6 — V16.3 vs V18 comparison

### Objective

Compile a side-by-side table of B1, B2, B3, B4 results for V16.3 and
V18, on the same CosyVoice evaluation set where possible.

### Procedure

1. Rerun B2, B3, B4 on V18 best.pt with identical data, label pipeline,
   and seeds.
2. Produce a single comparison table `docs_ae/v16_3_latent_anatomy/b6_v16_vs_v18.md`.
3. Highlight the three most diagnostic differences (expected: effective
   rank, principal angle between phoneme/speaker subspaces, and
   per-direction R² distribution).

### Citable outputs

- A complete delta table of every `B1.*`, `B2.*`, `B3.*`, `B4.*` metric.

---

## Execution order and decision gates

1. **Day 1**: B1 on V16.3. Decision: if local ID is much smaller than
   global rank, revise framing in subsequent blocks.
2. **Day 2**: B2 on V16.3 + CosyVoice. Decision: if unassigned fraction
   > 60%, reduce scope of B3 (expect subspace variance fractions to be
   small) and plan to add additional labels (MFCC, HuBERT feature
   regression) before going further.
3. **Days 3-4**: B3 on both datasets with shuffle null. Decision: the
   p-value of the phoneme/speaker principal angle test is the main
   gate. Below 0.05 means the "factor-mixed" interpretation needs
   refinement. Above 0.5 means V16.3 latent is substantially
   factor-mixed, and our preferred downstream story is supported.
4. **Day 4**: B4 on both datasets. Decision: Δ (MLP − linear) scale
   informs whether V19 needs shallow or deep heads on the supervised
   stream.
5. **Day 5**: B5 on V16.3. Decision: curvature ordering determines
   whether "supervision pushes flat direction" is a valid internal
   claim.
6. **Day 5-6**: B6 compilation.

## Output layout

```
docs_ae/v16_3_latent_anatomy/
  plan.md                          (this file)
  methodology.md                   (detailed params + run commands)
  results.md                       (master citable table)
  b1_spectrum.md                   (plots + detailed readings)
  b2_pca_semantics.md
  b3_subspaces.md
  b4_readability.md
  b5_curvature.md
  b6_v16_vs_v18.md
  figures/                          (all PNGs here)
  data/                             (cached label arrays, PCA bases)
```

## Code layout

```
autoencoder/scripts/anatomy/
  __init__.py
  labels.py              — F0/energy/centroid/voicedness/phoneme label pipeline
  dataset.py             — encode audio → cached (latent, labels) batches
  spectrum.py            — B1
  pca_semantics.py       — B2
  subspace.py            — B3 + shuffle null
  readability.py         — B4
  curvature.py           — B5
  compare.py             — B6 aggregator
scripts/run_anatomy.sh   — driver for all blocks
```

## Explicit limitations

- Speaker subspace measured on CosyVoice introduces out-of-distribution
  for V16.3, which is LJSpeech-trained. We call this out in every
  speaker-related finding.
- Shuffle null permutes labels but preserves marginal distribution only;
  temporal autocorrelation of labels is not preserved, which may
  overestimate the observed-vs-null gap. If the shuffle p-value is
  borderline, we will add a block-bootstrap alternative.
- Curvature in B5 is around the encoder output `Z = encoder(x)`. It does
  not measure curvature of the full `recon(encoder(x))` pipeline with
  respect to the encoder parameters.
- B3 subspaces are defined via label-mean SVD, which is an LDA-style
  decomposition and implicitly assumes within-class isotropy. Departures
  from this assumption are a methodology risk, not corrected here.

## Not included

- Neural-collapse NC1-NC4 measurements under the strict Papyan 2020
  definition. Our CTC + absorbing-blank + AE-without-classifier setting
  is not covered by the original NC theory. A principled extension
  is out of scope.
- Parameter-space Hessian spectra.
- Any training from scratch. All probes are on existing checkpoints.
- Any test set beyond LJSpeech val and CosyVoice crossed val.
