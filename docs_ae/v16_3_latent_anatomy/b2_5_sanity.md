# B2.5 — Probe saturation and baseline sanity (V16.3)

Date: 2026-05-10

## Motivation

B1 reported `local_ID_median = 35.1` from TwoNN, and B2 reported that
no top-32 PCA direction reaches R² > 0.1 on any label. Before B3,
we need to know:

- Is the `local_ID = 35` a real manifold dimension, or a TwoNN
  under-count?
- Is "no information along top-32 PCA" the same as "no information"?
  Could it be all in the tail PCs?
- Are the near-zero CV numbers real, or a methodological artifact
  (fold leakage, label bug)?

Three sanity subtests: S1 (TwoNN N-scaling + calibration), S2
(full-384 linear probe), S3 (label shuffle null).

## B2.5-S1 — TwoNN calibration and N-scaling

### S1.1 — Synthetic Gaussian calibration

For known d, sample `N(0, I_d)` and measure both estimators at
N=4000:

| True d | TwoNN | TwoNN / d | Levina-Bickel (k=20) | L-B / d |
|--------|-------|-----------|----------------------|---------|
| 10 | 6.43 | 0.64 | 9.31 | 0.93 |
| 20 | 11.55 | 0.58 | 15.80 | 0.79 |
| 40 | 19.22 | 0.48 | 25.47 | 0.64 |
| 80 | 29.55 | 0.37 | 38.85 | 0.49 |
| 160 | 44.59 | 0.28 | 58.17 | 0.36 |

**Both estimators systematically underestimate true d.** The
underestimation factor worsens as d grows. TwoNN at d=80 reports 30
(0.37× truth). Levina-Bickel at d=80 reports 39 (0.49× truth). Both
estimators at d=160 are off by a factor 2.7× / 2.75×.

This is an expected property of finite-sample ID estimators on
high-dimensional Gaussians: far from the manifold boundary, the
neighborhood structure is Euclidean, but the estimator requires enough
points within each "ID-many" nearest neighbors to trust the ratio.
N=4000 is too few for d=80 or higher.

### S1.2 — V16.3 latent N-scaling

V16.3 LJSpeech 32 768 frames (128 samples × 256 frames each), drawn
by bootstrap for 5 reps per N:

| N | TwoNN (median) | TwoNN (p05-p95) | L-B k=20 |
|---|----------------|------------------|----------|
| 500 | 11.60 | 7.98 – 16.69 | 19.83 |
| 1000 | 39.70 | 17.98 – 43.25 | 29.93 |
| 2000 | 37.18 | 18.68 – 38.27 | 14.08 |
| 4000 | 26.44 | 21.40 – 36.61 | 73.34 |
| 8000 | 25.43 | 23.47 – 31.40 | 74.50 |
| 16000 | 26.44 | 23.47 – 27.64 | 21.15 |

TwoNN plateaus around 25-27 for N ≥ 4000. Levina-Bickel is erratic
(14-75 range across N), the k=20 variant is numerically fragile on
real data.

### S1.3 — Numerical-stability bug discovered and fixed

The first run of B1 and B2.5 used a pairwise-distance routine based
on `diff[:, None, :] - x[None, :, :]` which is mathematically
equivalent to the `sqrt(|a|² + |b|² - 2 a·b)` formulation we now
use, but ran out of GPU-ish chunks on large N and produced slightly
different numbers. The **rewritten version in
`autoencoder/scripts/anatomy/spectrum.py` and `b2_5_twonn_scaling.py`
is the authoritative one**.

After the fix, on the **same data** (V16.3 LJSpeech), TwoNN at
N=4000 gives ~26 rather than the originally reported 35.

### S1.4 — Interpretation

Applying the synthetic calibration curve back to our V16.3
observations (TwoNN ≈ 26 in the plateau region):

| TwoNN observed | True d estimate from calibration |
|----------------|----------------------------------|
| TwoNN(d=40) = 19.22 | V16.3 TwoNN=26 implies true d > 40 |
| TwoNN(d=80) = 29.55 | V16.3 TwoNN=26 implies true d ≈ 60-70 |

So **V16.3 local intrinsic dimension is in the range 60-100, not 35**.
A more conservative bound: the local ID is **definitely above 40** and
**not larger than 160** (since TwoNN at 160 gives 44.6, higher than
our 26).

### S1.5 — Revised story for the paper

- B1's original `local_ID = 35` is incorrect. The correct TwoNN value
  is 26, which corresponds to a true local dimension of roughly 60-100
  after calibration.
- The `local / global ratio` is therefore closer to 0.3-0.5, not
  0.19. This is still less than 1, so there is still meaningful
  evidence that "global rank > local rank", but the factor is ~2-3×,
  not 5×.
- **The "collapse is smaller than it looks" claim weakens**. V18's
  rank=10 is likely still a serious reduction relative to V16.3's
  local ID of 60-100. We should retract the rhetorical framing.

### S1.6 — Citable handles (revised)

- `B1.local_ID_TwoNN_raw = 26.44` at N=16000 (5 reps bootstrap, CI
  [23.5, 27.6])
- `B1.local_ID_calibration_note = "TwoNN undercounts by 0.28-0.64
  factor between d=160 and d=10. True local ID is in [40, 160]."`
- `B1.local_ID_calibrated_estimate ≈ 60-100` (soft)

## B2.5-S2 — Full 384-dim linear probe

5-fold utterance-split CV on V16.3 + CosyVoice 4-spk val, 112
utterances, 57 344 frames. Ridge α = 10.0 for continuous targets,
L2-regularized multi-class logistic regression for categorical.

| Label | Best single PC (B2) | Full 384 (B2.5) | Shuffle null (S3) | Real ≫ null? |
|-------|---------------------|-----------------|-------------------|--------------|
| phoneme | η²=0.003 | acc=0.068, top5=0.340, chance=0.086 | acc=0.075, top5=0.330 | **NO** |
| speaker | η²=0.013 | acc=0.618, chance=0.250 | acc=0.250 | YES |
| f0 | r²≤0 | R²=0.199 ± 0.040 | R²=-0.011 | YES |
| energy | r²=0.061 | **R²=0.809 ± 0.034** | R²=-0.007 | YES |
| centroid | r²=0.021 | R²=0.515 ± 0.043 | R²=-0.009 | YES |
| voiced | r²=0.012 | R²=0.189 ± 0.035 | R²=-0.009 | YES |

### Interpretation

1. **Energy (R²=0.81) and centroid (R²=0.52) are linearly present in
   V16.3 latent, but spread across many dimensions**. The best single
   PC alone could only reach R²=0.06 for energy. The factor 13× jump
   between best-PC and full-384 is the clearest evidence of
   non-axis-aligned, distributed encoding.

2. **Speaker (acc=0.62, vs chance 0.25) is also linearly readable
   from the full latent**. This is notable because V16.3 was trained
   only on LJSpeech (single speaker). The CosyVoice 4 speakers are
   strictly out of distribution. Still, a linear readout separates
   them above chance, which means V16.3 encodes **some** acoustic
   feature that happens to distinguish these 4 out-of-distribution
   voices.

3. **Phoneme is NOT linearly readable**. acc=0.068 is below chance
   baseline 0.086 (from majority class frequency). Even the top-5
   accuracy of 0.34 is indistinguishable from the shuffled null of
   0.33. **Phoneme information, if present at all, is encoded
   nonlinearly in V16.3**.

4. **F0 (R²=0.20) and voicedness (R²=0.19) are weakly present**.
   Still clearly above shuffle null (-0.01) so non-zero, but small.
   Part of this weakness may be pitch-tracker noise (our F0 label
   uses a simple autocorrelation, not a production pitch tracker).

### Full-384 vs top-32 revelation

The B2 conclusion "no top-32 PCA direction encodes any semantic
factor" has changed meaning. It is still technically true, but now
the full story is:

- Linear information is present (for all labels except phoneme)
- It is **distributed** across dimensions, not axis-aligned
- The top-32 PCs, which cover 55.8% of variance, contain **only a
  fraction** of the linearly-readable information. The remaining
  52 PCs (covering the bottom 44.2% of variance) contain a lot of
  linearly-readable energy / centroid / speaker signal.

This inverts what one would naturally assume. Usually high-variance
PCs carry "big" factors like speaker timbre and phoneme identity.
Here, the reconstruction-only V16.3 has distributed speaker info
across the entire 384-dim cloud without concentrating it in any
small subset.

## B2.5-S3 — Label shuffle null

Same configuration as S2 but labels randomly permuted frame-by-frame
before fold construction. Results above show all shuffled R²/accuracy
values within ±0.01 of chance, confirming that the fold splitter is
clean and the real-data values above are real signal, not artifacts.

Noteworthy: the shuffled phoneme top-5 accuracy of 0.33 matches real
0.34 within noise. **This confirms that V16.3 latent has no
linearly readable frame-level phoneme content beyond what a uniform
shuffled model would give.** The marginal class frequency dominates.

## Revised citable handles

### Linear probe ceiling (B2.5-S2)

- `B2_5.linear_phoneme_acc = 0.068` (≈ shuffled, no linear signal)
- `B2_5.linear_phoneme_top5 = 0.340` (≈ shuffled)
- `B2_5.linear_speaker_acc = 0.618` (chance 0.250, shuffled 0.250)
- `B2_5.linear_f0_r2 = 0.199` (shuffled -0.011)
- `B2_5.linear_energy_r2 = 0.809` (shuffled -0.007)
- `B2_5.linear_centroid_r2 = 0.515` (shuffled -0.009)
- `B2_5.linear_voiced_r2 = 0.189` (shuffled -0.009)

### Revised B1 (TwoNN)

- `B1.local_ID_TwoNN_raw = 26.4 (CI 23.5-27.6)` at N≥8000
- `B1.local_ID_calibrated = 60-100` (soft range from calibration)

## Decision-level implications

### What stands

- **B2's "no PC is axis-aligned with any label" stays correct**. The
  PC basis is not aligned with any semantic factor, for all labels.
- **"Distributed encoding" claim now has a quantitative anchor**:
  energy goes from R²=0.06 (best PC) to R²=0.81 (all 384 PCs).

### What needs revising

- **B1's "5× collapse story"** (local ratio 0.19) must be withdrawn.
  The ratio is closer to 0.3-0.5. We keep the qualitative observation
  "local < global" but drop the sharp quantitative factor.
- **"V16.3 encodes phoneme in latent"** must be qualified. Linearly,
  no. Nonlinearly — we have not measured yet.

### What becomes clearer for B3 and paper

1. **B3 phoneme subspace** — if phoneme is linearly invisible in the
   full latent, then the B3 phoneme subspace constructed from class
   means will **also** be a small subspace with low variance fraction.
   We should adjust B3 expectations: phoneme subspace may cover only
   ~5-15% of total variance.
2. **B4 MLP-vs-linear gap will be huge** — for phoneme, maybe 0.07
   linear vs 0.5+ MLP. This will be the clearest evidence of
   "phoneme is nonlinearly encoded."
3. **Paper headline line**: "Reconstruction-only V16.3 latent
   encodes speaker and energy linearly but phoneme nonlinearly; no
   factor is axis-aligned with PCA directions."
4. **This reframes supervision collapse**: CTC supervision tries to
   make phoneme **linearly** separable. The encoder, to satisfy CTC,
   must re-route phoneme information to a linearly readable form —
   which reshapes latent geometry substantially. This is a stronger
   mechanism than "flat direction shortcut": the encoder is being
   asked to change **the geometric encoding of phoneme from
   nonlinear to linear**, forcing a large geometry change that must
   come from somewhere.

## Status

- S1 complete, 1 hour compute
- S2 complete, 5 min compute
- S3 complete (via `--shuffle-labels` path), 5 min compute
- **B1 report requires an erratum**: `local_ID` figure must be
  updated from 35.1 to 26.4 (TwoNN raw) / 60-100 (calibrated). The
  B1 interpretation paragraph that talks about "0.19 ratio" needs
  rewriting.

## Next

- Write erratum on `b1_spectrum.md`
- Update `results.md` master table with corrected numbers
- Proceed to B3 with adjusted expectations
