# V17.1 Structure/Reconstruction Probe

This experiment asks one narrow question:

> Does the V17.1 cross-factor structure constraint improve latent structure while
> preserving or improving reconstruction, instead of creating a persistent
> gradient conflict with reconstruction?

It is not a scale run. It is a controlled probe before spending more GPU time.

## Hypothesis

V17.1 should keep the mature V16.3 autoencoder behavior as a reconstruction
guardrail, while adding shallow cross-sample factor heads that make the latent
space more structured:

- same text, different speaker: phoneme/content projection should align.
- same text, different speaker: speaker projection should move apart.
- locally similar phoneme texts: phoneme projection should preserve soft local
  geometry rather than becoming a flat ID classifier.
- reconstruction should not collapse, because the decoder still receives the
  full latent and reconstruction loss remains the main objective.

The expected healthy pattern is:

- early steps: mild recon/structure friction is acceptable.
- middle steps: structure metrics improve while SNR/STFT recover toward the
  baseline.
- later steps: recon stays near or above baseline and structure metrics do not
  regress.

## Fixed Inputs

Use one fixed initialization and one fixed dataset split for every run.

Initialization:

```bash
/root/autodl-tmp/project/outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt
```

Training manifest:

```bash
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_2spk_calib_qc_8_13p5_train_phalign_neighbors.jsonl
```

Validation manifest:

```bash
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_2spk_calib_qc_8_13p5_val_phalign_neighbors.jsonl
```

Optional structured geometry manifest:

```bash
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_align_v0_2spk_calib_train_phalign_neighbors.jsonl
```

For the first probe, keep the broad 4h split as the training distribution and
use the align set for diagnostics. Do not upweight align reconstruction yet,
otherwise it becomes harder to separate data-shape effects from loss effects.

## Run Matrix

Every run uses the same V16.3 backbone config and differs only in structure
pressure.

| Run | V17.1 setting | Purpose |
| --- | --- | --- |
| `baseline_recon` | `v17_constraints_enable=false` | Reconstruction reference. |
| `cross_w01` | `v17_cross_factor_weight=0.1` | Low-pressure structure. |
| `cross_w04` | `v17_cross_factor_weight=0.4` | Current main candidate. |
| `cross_w08` | `v17_cross_factor_weight=0.8` | Conflict stress test. |
| `cross_w04_detach` | `weight=0.4`, `v17_cross_factor_detach_latent=true` | Detect whether heads learn alone without changing the AE latent. |

The detach run is important. If detach improves only the constraint head metrics
but not latent/factor diagnostics, the head is absorbing the task. A useful V17.1
run should beat detach on latent structure while staying close to baseline on
reconstruction.

## Metrics

### Reconstruction Guardrail

Measure on the same validation set:

- SNR
- MR-STFT loss
- mel loss
- generated eval wav listening check
- optional ASR CER/WER after the first successful probe
- optional F0/envelope diagnostics for a second pass

A reasonable first-pass pass condition:

```text
SNR drop <= 0.5 dB vs baseline
STFT/mel not materially worse
no audible jumps/collapse in eval_wavs
```

V17.1 does not need to win reconstruction immediately, but it must not buy
structure by destroying the AE.

### Structure

Run cross-factor diagnostics on both the broad validation set and the align set.

Important fields:

- `phoneme_text_cross_speaker_retrieval.top1`
- `phoneme_pair_cos.same_text_diff_speaker_cos_mean`
- `speaker_pair_cos.same_text_diff_speaker_cos_mean`
- `speaker_speaker_centroid_acc`
- `phoneme_rank.rank95`
- `speaker_rank.rank95`
- `latent_mean_rank.rank95`

Healthy direction:

```text
phoneme same-text cross-speaker similarity goes up
speaker same-text cross-speaker similarity goes down
speaker centroid accuracy stays high
rank95 does not collapse
```

### Gradient Conflict

For each checkpoint, compute encoder/component gradient cosine between:

```text
reconstruction loss
cross_factor loss
```

Important fields:

- component-level `cosine_mean`
- `cross_factor_weighted_over_recon_mean`
- bottleneck/late encoder cosine
- minimum cosine across pairs

Interpretation:

```text
cosine < -0.2 for many components: strong conflict
-0.1 <= cosine <= 0.2: manageable friction
cosine > 0 later in training: structure is becoming compatible with recon
weighted_over_recon >> 1: structure can dominate and damage recon
weighted_over_recon << 0.01: structure is probably too weak
```

## Execution

From the project root on the training server:

```bash
bash scripts/run_v17_1_probe_matrix.sh
```

Run a subset:

```bash
RUNS=cross_w01,cross_w04 bash scripts/run_v17_1_probe_matrix.sh
```

Run diagnostics for a finished checkpoint:

```bash
/root/miniconda3/bin/python -m autoencoder.scripts.v17_gradient_alignment \
  --constraint cross_factor \
  --checkpoint /root/autodl-tmp/project/outputs/models/v17_1_probe_cross_w04/latest.pt \
  --manifest /root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_align_v0_2spk_calib_train_phalign_neighbors.jsonl \
  --path-root /root/autodl-tmp/project/CosyVoice_main \
  --out /root/autodl-tmp/project/outputs/diagnostics/v17_1_probe_cross_w04_grad_align
```

Cross-factor latent diagnostics:

```bash
/root/miniconda3/bin/python -m autoencoder.scripts.v17_cross_factor_diagnostics \
  --checkpoint /root/autodl-tmp/project/outputs/models/v17_1_probe_cross_w04/latest.pt \
  --manifest /root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_align_v0_2spk_calib_train_phalign_neighbors.jsonl \
  --path-root /root/autodl-tmp/project/CosyVoice_main \
  --out /root/autodl-tmp/project/outputs/diagnostics/v17_1_probe_cross_w04_factor_align
```

## Decision Rule

Promote `cross_w04` or `cross_w01` to a larger run only if it satisfies all of:

- reconstruction is close to baseline.
- structure diagnostics improve over baseline/detach.
- gradient cosine is not persistently strongly negative.
- effective rank does not collapse.

If `cross_w08` improves structure but hurts SNR/STFT and has negative gradient
cosine, it is a useful upper bound, not a candidate setting.

If detach performs similarly to non-detach, the current heads are learning too
much internally and the latent itself is not being shaped enough.
