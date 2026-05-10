# V18 Design: Generation-Aware Autoencoder for Flow-Matching TTS

Date: 2026-05-10

## Context

V16.3 is our current stable AE baseline (20.7 dB SNR, 384d latent, 256x stride).
FM-DiT on V16.3 latent is functional but suboptimal: teacher-forced t=0 SNR is
near zero (-0.44 dB), indicating weak endpoint coupling between condition and
latent. V17.1-V17.4 explored post-hoc structural constraints on the frozen
V16.3 encoder and consistently hit a gradient-conflict wall: recon and
structure losses compete on shared encoder parameters.

V18 takes a different stance. Instead of post-hoc shaping a recon-optimal
encoder, V18 redesigns the encoder side of the AE so that a fraction of the
latent is supervised semantic content, while another fraction is a free
residual that absorbs everything else. This is the minimum architectural
change that implements the "generation-aware supervision" idea.

## Core Hypothesis

> An AE whose latent is partially supervised by a frame-level phonetic
> classifier produces a downstream-friendlier latent for FM-DiT than an AE
> trained purely on reconstruction, without sacrificing reconstruction quality.

"Downstream-friendlier" is measured directly:

- FM-DiT teacher-forced t=0 SNR (primary)
- FM-DiT pure sampling std/corr (secondary)
- Decoded audio WER / speaker similarity (end-to-end)

## Architecture

### Top-Level Structure

```
audio ─┬─→ phoneme_encoder (supervised) ────→ z_phoneme  [128d, T_frames]
       │
       └─→ residual_encoder (unsupervised) ──→ z_residual [256d, T_frames]

z = concat([z_phoneme, z_residual], dim=1)  # [384d, T_frames]
z → decoder (V16.3-derived) → audio_hat
```

Rationale for concat at the full-latent level:

- Keeps total dim = 384 to preserve V16.3 decoder compatibility
- Gives `z_phoneme` and `z_residual` independent channels, avoiding gradient
  cross-talk during training
- Does not commit to harder disentanglement than necessary for the minimal
  hypothesis
- Downstream FM-DiT sees a 384d latent that is decomposable at inference time

### Phoneme Encoder

Backbone: LogMel + Conformer, derived from the V17.3 Conformer content
teacher which already achieves cross-speaker text retrieval of ~0.47 on the
4-speaker CosyVoice3 crossed validation set.

Outputs:

- Frame-level embeddings at AE frame rate (256x stride, ~86 Hz at 22050 Hz)
- Dimension: 128

Heads (training only):

- Frame CTC over pinyin+tone vocabulary (strong supervision)
- K-way soft cluster prediction over a learned codebook (weak-discrete prior,
  optional)

Stride matching:

- Conformer operates at log-mel frame rate (256 hop), then 2x conv subsampling
  matches final latent frame rate
- The encoder must be able to receive a full audio input and output frame
  embeddings aligned with V16.3 latent frames

Initialization:

- Bootstrap from `outputs/models/v17_3_conformer_teacher_4spk_qc_1000/best.pt`
- Only the backbone weights transfer; heads are re-initialized for V18 targets

### Residual Encoder

Backbone: V16.3-derived WavVAEEncoder, but narrower.

- Reduced channels: `[96, 192, 384, 768, 768]` (V16.3 uses `[128, 256, 512,
  1024, 1024]`)
- Same strides `[2, 4, 4, 8]` and dilations `[1, 3, 9]`
- Output dim: 256
- No bottleneck regularization at first (mirror V16.3 `sample_latent=false`)

Rationale for narrower:

- Residual carries less structured information than the full V16.3 latent
- Smaller encoder reduces gradient noise and OOM risk
- Keeps total trainable AE parameters comparable to V16.3

Initialization:

- From scratch for the architecture shape difference
- Alternative: load V16.3 encoder weights for the first 4 blocks with channel
  truncation; test both

### Decoder

Backbone: V16.3 decoder, minor input adaptation.

- Input: `[B, 384, T_frames]` from concat, matches V16.3
- Everything downstream of the bottleneck identical to V16.3

Initialization:

- Load V16.3 decoder weights directly (same architecture)

### Bottleneck

Keep V16.3-style soft range penalty bottleneck on the concatenated latent:

- `sample_latent=false`
- `reg_weight=1e-4, margin_mean=6.0, std_min=0.05, std_max=3.0`
- `reg_warmup_steps=5000`

This keeps the latent scale-compatible with V16.3 for downstream FM-DiT
comparison.

## Training Losses

```
L_total = l1_time_weight * L1(x_hat, x)
        + stft_weight * MRSTFT(x_hat, x)
        + mel_weight * MultiScaleMel(x_hat, x)
        + reg_weight * bottleneck_range_penalty
        + lambda_phoneme * CTC(phoneme_logits, pinyin_targets)
        + lambda_cluster * CE(cluster_logits, cluster_targets)  # optional
```

Weights:

- `l1=10.0, stft=0.5, mel=0.1` (match V16.3 baseline)
- `lambda_phoneme=0.3` (target: phoneme loss contributes ~10-20% of total)
- `lambda_cluster=0.1` if used, otherwise 0

The phoneme loss only touches `z_phoneme` (CTC head reads from
`z_phoneme` only, not from decoder output). Residual encoder sees no
supervision beyond reconstruction gradient.

## Training Schedule

### Stage 1: Phoneme Encoder Warmup (optional, ~2000 steps)

- Only phoneme_encoder and CTC head are trainable
- Decoder and residual encoder frozen
- Loss: CTC only
- Purpose: align phoneme encoder output to V16.3 frame rate without having
  to fight recon gradient from the start
- Skip if warmstarting from V17.3 teacher already aligns the frame rate

### Stage 2: Joint AE Training (~40 epochs)

- All components trainable
- Full loss active
- `lr=1e-4`, `lr_warmup_steps=1000`, cosine decay
- `batch_size=1, grad_accum_steps=16, segment_length=65536` (match V16.3
  recipe that trained stably)
- `use_checkpoint=true` on encoders for memory

### Stage 3: Optional Fine-tune (if needed)

- Unfreeze anything previously frozen
- Reduce `lambda_phoneme` to 0.1
- Small lr (`1e-5`)
- Only if Stage 2 produces SNR < V16.3 by more than 2 dB

## Downstream Integration

### FM-DiT Training on V18 Latent

The latent is 384d continuous, same as V16.3. FM-DiT can operate directly,
but we introduce one new option:

- **Latent split condition**: at every FM-DiT layer, z_phoneme (the first 128
  channels) can be provided as strong conditional input, either via FiLM or
  via attention-to-phoneme-token-sequence
- Residual (the last 256 channels) is what FM-DiT generates

Concretely:

```
t=0: x_t = noise (384d)
condition: text-derived phoneme embeddings + speaker/prosody refs
FM-DiT predicts velocity on 384d
  but the phoneme channels are anchored to z_phoneme_predicted (from condition)
  residual channels are free to flow

Equivalent formulation:
  FM-DiT predicts a 256d residual velocity conditioned on phoneme
  then concat(z_phoneme_from_text, z_residual_generated) is the latent
```

### Text -> Phoneme Embedding Predictor

To close the loop at inference time, we need: `text -> z_phoneme`.

Since V18 trained phoneme_encoder with frame-level CTC, the phoneme encoder's
output distribution is known. A small `text -> phoneme_embedding` predictor
can be trained separately:

- Input: phoneme sequence (from G2P) + duration
- Output: frame-level embeddings aligned to V18 phoneme encoder output
- Loss: MSE to phoneme_encoder(audio) on paired data

This predictor is effectively an inverse of the phoneme encoder at the
embedding level.

## Success Criteria

### V18 AE Success (Stage 2 complete)

- SNR on CosyVoice3 val: within 2 dB of V16.3 baseline (>= 18 dB)
- Phoneme frame CTC loss on val: < 1.5 (equivalent to ~70% frame accuracy)
- Latent effective rank on val: >= 200 (not collapsed)

### V18 Downstream Success (FM-DiT on V18 latent)

- Teacher-forced t=0 SNR: **> 5 dB** (vs V16.3's -0.44 dB)
- Pure sampling corr: > 0.3 (vs V16.3's 0.04)
- End-to-end ASR WER on decoded audio: < V16.3 + FM-DiT baseline

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Recon SNR drops too much | Reduce `lambda_phoneme`; add Stage 3 fine-tune |
| Phoneme CTC won't converge | Bootstrap from V17.3 teacher; inspect alignment |
| Residual steals phoneme info | Optional: information bottleneck on residual (small KL) |
| Phoneme head learns without shaping z_phoneme | Head must be linear or shallow (1 layer), not deep |
| FM-DiT still fails endpoint | Indicates phoneme channels alone don't solve; add speaker encoder in V18.1 |
| OOM with two encoders | Residual encoder is narrower; checkpoint on both; reduce segment if needed |
| Joint training instability | Keep V16.3 stability tricks: bottleneck warmup, grad_clip=1.0, lr_warmup |

## What V18 Explicitly Does Not Include

Deferred to V18.x variants if V18 core succeeds:

- Speaker encoder as a separate stream (V18.1)
- Prosody encoder (V18.2)
- ASR-TTS cyclic training (V18.3 — the phoneme encoder can be jointly used
  for ASR inference, but V18 trains only for TTS use)
- Flow-matching-loss during AE training (V18.4)
- RAE-style frozen semantic encoder (V18.5 — if V18 joint training fails)
- FiLM or attention-based fusion (V18 uses concat only)
- Low-rank projections and MI overlap control
- Information bottlenecks beyond V16.3 standard range penalty

These are all reasonable extensions but V18 core must first prove that a
single supervised phonetic stream on top of a residual stream improves
downstream FM-DiT. Every deferred feature is separately testable later.

## Implementation Plan

### Files to Add

- `autoencoder/models/v18_encoder.py`: `V18PhonemeEncoder` (Conformer-based),
  `V18ResidualEncoder` (narrow WavVAEEncoder), `V18Autoencoder` (composite)
- `autoencoder/conf/experiment/v18_base.yaml`: joint training config
- `autoencoder/conf/experiment/v18_phoneme_warmup.yaml`: Stage 1 config
- `scripts/run_v18_train.sh`: launch script on remote

### Files to Modify

- `autoencoder/train.py`: detect `experiment.train.model_type == 'v18'` and
  instantiate composite model; carry phoneme label from manifest into the
  loss

### Files to Reuse

- `autoencoder/scripts/v17_3_train_conformer_teacher.py`: source for phoneme
  encoder architecture
- `autoencoder/models/autoencoder.py`: WavVAE encoder/decoder building blocks
- CosyVoice3 crossed_zh_4spk manifests with pinyin durations (already
  present)

### Rough Timing

- Day 1-2: implement V18Autoencoder and phoneme warmup
- Day 3-4: Stage 1 phoneme warmup + validate frame-level CTC converges
- Day 5-7: Stage 2 joint training
- Day 8-9: diagnostics (SNR, rank, phoneme accuracy, latent PCA)
- Day 10-12: FM-DiT training on V18 latent
- Day 13-14: end-to-end evaluation vs V16.3 baseline

Total: ~2 weeks to a conclusive result.

## Evaluation Protocol

All metrics on the 4-speaker CosyVoice3 crossed validation set
(`crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl`,
112 items, 28 text groups).

Required artifacts for the V18 report:

- V18 AE training log and best checkpoint
- Reconstruction SNR/STFT/PESQ on val
- Latent PCA effective rank summary
- Phoneme CTC frame accuracy on val
- Cross-speaker text retrieval top1 on z_phoneme (should be high, as a
  sanity check that z_phoneme holds phonetic content)
- FM-DiT training log and best checkpoint on V18 latent
- Teacher-forced t sweep JSON on val
- Sampled audio WER on val (Whisper or FunASR)
- Side-by-side comparison table vs V16.3 + FM-DiT baseline
