# B2 — PCA directions vs semantic labels (V16.3 + CosyVoice 4spk val)

Date: 2026-05-10

## Setup

- Checkpoint: `outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt`
- Dataset: CosyVoice 4-speaker crossed val, full 112 utterances
- Segment length: 131 072 samples (~5.9 s at 22 050 Hz, = 512 AE frames)
- Total frames analysed: 57 344
- Top-32 PCA directions, covering **55.8 % cumulative variance**
- 5-fold CV, splits by utterance (no frame leakage)
- Labels: phoneme (pinyin, 80 unique classes, all frames labelled),
  speaker (4 classes), F0 (voiced frames only), energy, centroid,
  voicedness

Artifacts:

- `data/b2_v16_3_cosyvoice/b2_heatmap.npz`
- `data/b2_v16_3_cosyvoice/b2_summary.json`
- `data/b2_v16_3_cosyvoice/b2_diagnostic.json` (raw no-CV sanity
  check)
- `data/b2_v16_3_cosyvoice/figures/b2_heatmap.png`
- `data/b2_v16_3_cosyvoice/figures/b2_bar_primary.png`

## Headline finding

**None of the top-32 PCA directions of V16.3 latent linearly encodes
any of the six semantic factors above threshold (R² / η² < 0.30).**

All 32 directions are classified `unassigned`. The best scores per
label, under utterance-split 5-fold CV:

| Label | Best CV score | Metric |
|-------|---------------|--------|
| phoneme | 0.003 | η² (correlation ratio) |
| speaker | 0.013 | η² |
| f0 | ≤ 0 | ridge R² (voiced frames only) |
| energy | 0.061 | ridge R² |
| centroid | 0.021 | ridge R² |
| voiced | 0.012 | ridge R² |

Nearly all CV R² values are slightly **negative**, which means the
single-direction linear model does worse than predicting the mean on
held-out utterances. This rules out the possibility that the zero
scores are numerical noise.

## Is this "no information" or "distributed information"?

To separate "latent doesn't encode this factor at all" from "latent
encodes this factor but spread across many directions", we ran a
diagnostic with two checks, on the same 57 344 frames:

### (a) Raw in-sample correlation, per direction

No cross-validation, just `corr(α_i, y)²` or full-data η² per
direction. Results:

| Label | Best raw score across 32 PCs | Top-3 |
|-------|------------------------------|-------|
| phoneme η² | 0.0033 | 0.0033, 0.0030, 0.0028 |
| speaker η² | 0.0133 | 0.0133, 0.0118, 0.0102 |
| f0 r² (voiced) | 0.0043 | 0.0043, 0.0031, 0.0016 |
| energy r² | 0.0838 | 0.0838, 0.0614, 0.0481 |
| centroid r² | 0.0227 | 0.0227, 0.0200, 0.0142 |
| voiced r² | 0.0129 | 0.0129, 0.0109, 0.0084 |

Even in-sample, no single PCA direction captures more than 8 % of
variance of any label.

### (b) Multivariate in-sample R² (all 32 PCs jointly as features)

| Label | All-PC multivariate R² |
|-------|------------------------|
| f0 | 0.013 |
| energy | **0.355** |
| centroid | 0.099 |
| voiced | 0.054 |

The energy label jumps from 0.084 (best single PC) to 0.355 (all 32
jointly). This is a clear signature of **distributed encoding**: the
label information is there, but spread across many PCs rather than
aligned with any of them.

Centroid and voicedness also benefit from multivariate pooling, but
the absolute R² is still below 0.1. These probably need more PCs or
a nonlinear probe (B4).

F0 does not benefit from multivariate pooling. This may mean either
F0 is encoded very nonlinearly, or that our autocorrelation-based F0
label has too much noise. B4 will give a more reliable answer using
a nonlinear probe with quantile-binned F0.

## Interpretation

1. **V16.3 latent is not axis-aligned.** Top-32 unsupervised PCA
   directions are not individual "phoneme", "speaker", or "prosody"
   directions. The encoder does not spontaneously separate semantic
   factors onto coordinate axes.
2. **Semantic information is present but distributed.** Energy's
   multivariate jump from 0.08 to 0.36 demonstrates that the needed
   information is in the latent, just not concentrated.
3. **The direct implication for downstream design**: a head that
   mean-pools and reads a single direction from the latent is doomed
   to fail (this matches V17.2's observation). Supervised subspace
   decomposition (B3) is the correct tool to isolate per-factor
   structure from this factor-mixed latent.
4. **The implication for the "supervision collapse" story**: when
   CTC / teacher-distill / contrastive loss is added, the encoder
   finds it much easier to compress along the PCA directions we see
   here, because **those directions are not needed by any particular
   semantic factor**. Any reweighting that makes one direction more
   important for a supervision target is unopposed by a
   per-direction recon signal — recon gradient is diffuse across the
   exact same directions.

## Caveats and limitations

- **F0 label quality**: the autocorrelation-based F0 estimator is not
  pitch-tracker grade. Low F0 R² may partly be label noise, not model
  uninformedness. B4 will use a binned-F0 classifier to reduce
  sensitivity to pitch-tracker noise.
- **Linear only**: B2 probes linear readability along unsupervised PC
  directions. It is not saying "the latent has no phoneme information";
  B4 answers that question with nonlinear probes.
- **CosyVoice distribution shift**: V16.3 is LJSpeech-trained. The
  speaker factor (4 CosyVoice speakers) is entirely OOD for this
  encoder. The near-zero speaker η² may partly reflect this, not only
  the lack of axis alignment. We note this and will cross-check in B3.
- **Threshold 0.30 is somewhat arbitrary**. Lowering to 0.15 changes
  nothing: the best CV score is 0.061 on energy, so no direction
  passes 0.15 either.

## Decisions for later blocks

- B3 (supervised subspaces) must do the heavy lifting. PCA directions
  do not give us factor axes. Class-mean SVD (phoneme) and per-speaker
  SVD will give us the proper subspaces.
- B4 (MLP probe) is now more important than initially planned. If the
  linear-vs-MLP gap is large, this quantifies the "distributed /
  nonlinear" nature of V16.3 encoding.
- The shuffle null in B3 becomes more interesting: since random
  directions capture very little label variance, the subspace-level
  signal must be measured carefully against this tiny baseline.

## Citable handles

- `B2.K_phoneme = 0 / 32` (no PCA direction is phoneme-primary)
- `B2.K_speaker = 0 / 32`
- `B2.K_unassigned = 32 / 32`
- `B2.best_phoneme_eta2_cv = 0.003`
- `B2.best_energy_r2_cv = 0.061`
- `B2.multivariate_energy_r2_insample = 0.355` (32 PCs jointly)
- `B2.multivariate_f0_r2_insample = 0.013`

## Status

**B2 complete on V16.3 + CosyVoice val.** No cross-check on LJSpeech
(no phoneme or speaker labels there; only F0 / energy / centroid
would be available, and those already showed the distributed pattern
on CosyVoice).
