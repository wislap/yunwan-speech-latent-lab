# V18 Training Log

Training run: `outputs/models/wav_vae_v18_base_remote_main` (server 14025).

## Setup

- Config: `v18_base.yaml`
- Init: V17.3 Conformer teacher (134/230 phoneme keys) + V16.3 decoder (147/147 keys)
- `lambda_phoneme=0.3`, recon weights l1=10.0/stft=0.5/mel=0.1
- batch_size=1, grad_accum=16, segment=65536, lr=1e-4, 40 epochs
- ~250s per epoch -> ~2.8 hours for full run

## Stage 1 warmup outcome

Attempted first; failed as expected:
- Froze residual+decoder, trained only phoneme+CTC head with l1=0.1
- Reconstruction exploded after ~900 steps because the decoder was fed
  `concat(phoneme_learning, residual_random)` which is never in its
  training distribution
- Stage 1 warmup is not useful for this architecture; skipped in favor of
  direct Stage 2 joint training

## Stage 2 joint training (active)

### Reconstruction SNR per eval

| Epoch | Val SNR (dB) | CTC train | CTC non-blank frac |
|-------|--------------|-----------|---------------------|
| 1 | 5.56 | ~27 | 1.00 |
| 3 | 8.48 | ~9 | 1.00 |
| 5 | 10.31 | ~6 | 1.00 |
| 7 | 11.42 | ~5 | 1.00 |
| 9 | 10.98 | ~4.5 | 0.00-0.10 |
| 11 | 11.25 | ~4.0 | 0.10 |
| 13 | 12.93 | ~3.3 | 0.13 |
| 15 | 6.88 (outlier) | ~3.2 | 0.14 |
| 17 | 13.50 | ~2.9 | 0.14-0.16 |

### Observations

1. Reconstruction trajectory is healthy: +1.5 dB every 4 epochs with
   monotone improvement on aggregate, punctuated by noisy dips.
2. CTC loss decreases from 27 to ~3 over 17 epochs.
3. CTC exited the all-blank regime around epoch 9-10. Non-blank fraction
   now stabilizes near 14-16%.
4. The all-blank -> non-blank transition caused a small SNR dip
   (epoch 9 -> 10.98), followed by recovery.
5. Epoch 15 eval SNR spike (6.88) appears to be eval-set variance; the
   next epoch immediately recovered to 13.50 and training loss did not
   regress.
6. No exploding losses, no OOM, training speed stable at ~250s/epoch.

### Projected outcome at epoch 40

Extrapolating:
- Val SNR: ~17-19 dB (close to 18 dB acceptance gate)
- CTC loss: ~1.5-2.0
- Non-blank fraction: ~25-35%

### Next steps after training completes

- Run `v18_ae_diagnostics.py` on the best checkpoint
- Evaluate all six acceptance-gate criteria
- If pass: export latents, proceed to FM-DiT training
- If fail on criterion 6 (FM-DiT t=0): schedule V18.1 (add speaker
  encoder)
- If fail on criteria 1-5: analyze which specific metric failed and
  decide whether to iterate or pivot


## Stage 2 final result (after 40 epochs)

Training completed in ~2.8 hours.

### Best checkpoint metrics (epoch 37, val SNR 15.70 dB)

| Metric | Value | Target | Pass |
|--------|-------|--------|------|
| Recon SNR (mean) | 16.02 dB | >= 18 dB | FAIL |
| Recon STFT (mean) | 0.7522 | (V16.3 ~0.60) | - |
| Latent rank95 on z | 10 | >= 200 | FAIL (severe) |
| rank95 on z_phoneme | 8 | - | - |
| rank95 on z_residual | 37 | - | - |
| CTC non-blank fraction | 0.047 | >= 0.30 | FAIL |
| z_phoneme cross-speaker retrieval top1 | 0.259 | - | partial |
| z_phoneme speaker probe acc | 0.750 | <= 0.60 | FAIL |
| z_residual speaker probe acc | 0.991 | >= 0.90 | PASS |
| Residual zero SNR drop | 16.04 dB | > 10 dB | PASS |

Gate: **FAIL** (2/6 pass).

### Key interpretation

1. **Latent collapse**: rank95=10 means the full 384d latent is severely
   compressed. z_phoneme uses 8 of 128 dims, z_residual uses 37 of 256.
   Likely root cause: V16.3 range-penalty bottleneck applied on the
   concatenated latent compresses it to satisfy mean/std margins, and
   under CTC gradient pressure the encoder learns a low-rank shortcut.

2. **CTC under-learning at eval time**: training-time non-blank fraction
   was 14-19%, but val set averages only 4.7%. Suggests CTC head learned
   per-sample features that don't generalize.

3. **Phoneme encoder is speaker-leaking**: probe reads speaker at 75%
   from z_phoneme (target <= 60%). CTC supervision was too weak to
   enforce speaker invariance in phoneme stream.

4. **Residual encoder works as designed**: z_residual drop triggers 16
   dB SNR loss, proving residual carries real info; speaker probe 99%
   confirms speaker info lives primarily in residual.

### Diagnosis

The architecture split (phoneme + residual with decoder concat) is
**valid in principle** — the residual zero-drop and residual speaker probe
show the decomposition works. The failure modes are:

- Bottleneck range penalty compresses latent rank under CTC gradient
  influence
- `lambda_phoneme=0.3` is too weak to drive phoneme-only content in
  z_phoneme
- CTC loss alone cannot produce speaker invariance without additional
  gradient reversal or MI constraints

### Status: archived (2026-05-16)

V18 stops here. The run is reframed as a stress-test of V16.3 bottleneck +
loss recipe under additional supervision, not as a product line. See
[lessons.md](lessons.md) for the consolidated takeaways. Next work moves to
the FM-DiT side (`../../fm_dit/`).

### Ablations considered but not run

The following V18.0.1 ablations were planned but skipped after the gate
result. Marginal information value is low relative to the cost; the V18
result already supports the conclusions in lessons.md.

1. **lambda_phoneme = 0.0**: pure recon under V18 architecture. Is SNR
   ~ V16.3's 20.7 dB? If yes, the architecture is fine and CTC is the
   problem. If no, the residual encoder's reduced channels are the
   bottleneck.

2. **reg_weight = 0**: drop bottleneck range penalty entirely. Does
   latent rank95 recover to > 100?

3. **lambda_phoneme = 0.1 + CTC warmup**: 2000-step warmup where CTC is
   disabled, then ramp. Reduces early-epoch phoneme disruption.

4. **Adversarial speaker removal on z_phoneme**: gradient reversal layer
   + speaker classifier. Forces speaker invariance explicitly.

These are small incremental experiments (~1-2 days each). The V18 core
hypothesis is not yet falsified; we've learned that the architecture
decomposes correctly but the loss recipe needs tuning.
