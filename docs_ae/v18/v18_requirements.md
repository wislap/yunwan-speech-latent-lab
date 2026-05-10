# V18 Requirements

Date: 2026-05-10

These requirements are derived from `v18_design.md`. They state the
functional and quantitative targets that V18 must satisfy, independent of
implementation choices.

## Context

- Baseline: V16.3 pure-reconstruction WavVAE (384d latent, 256x stride,
  SNR 20.7 dB on val).
- Downstream: FM-DiT on V16.3 latent works but has weak endpoint coupling
  (teacher-forced t=0 SNR ≈ -0.44 dB).
- Goal of V18: introduce the minimum architectural change on the AE encoder
  side that produces a downstream-friendlier latent without sacrificing
  reconstruction.
- Core change: split encoder into a supervised phoneme stream and a free
  residual stream; concat to 384d; unchanged decoder.

## User Stories

### US-1: As a TTS system builder, I need a new AE latent that FM-DiT can generate from weak conditions

**Acceptance criteria:**

- WHEN FM-DiT is trained on the V18 latent THEN teacher-forced t=0 SNR
  SHALL be at least 5 dB on the validation set.
- WHEN FM-DiT is sampled without CFG THEN the decoded-audio ASR WER SHALL
  be no worse than V16.3 + FM-DiT baseline.
- WHEN the V18 latent replaces V16.3 latent in the FM-DiT pipeline THEN no
  FM-DiT code change SHALL be required beyond loading the new latent.

### US-2: As a TTS system builder, I need V18 reconstruction to stay close to V16.3

**Acceptance criteria:**

- WHEN V18 is evaluated on the CosyVoice3 4-speaker crossed validation set
  THEN reconstruction SNR SHALL be at least 18 dB.
- WHEN V18 is evaluated on the CosyVoice3 4-speaker crossed validation set
  THEN reconstruction STFT loss SHALL be within 0.1 of V16.3 on the same
  set.
- WHEN V18 training completes THEN the latent effective rank (rank95 on
  pooled latent covariance) SHALL be at least 200.

### US-3: As a researcher, I need the phoneme stream to hold real phonetic content

**Acceptance criteria:**

- WHEN the phoneme encoder output is evaluated on the validation set
  THEN cross-speaker same-text retrieval top1 SHALL be at least 0.30 (on
  par with V17.3 frozen-latent adapter probe).
- WHEN a frame-level CTC head reads from z_phoneme on validation audio
  THEN phoneme frame accuracy SHALL be at least 70% on the validation set.
- WHEN speakers are identified from z_phoneme via a linear probe
  THEN speaker classification accuracy SHALL not exceed 60% (content
  stream should not be speaker-dominated).

### US-4: As a researcher, I need the residual stream to absorb everything the phoneme stream does not cover

**Acceptance criteria:**

- WHEN the residual channels of z are zeroed at inference
  THEN reconstruction SNR SHALL drop by more than 10 dB (residual must be
  carrying real information).
- WHEN speakers are identified from z_residual via a linear probe
  THEN speaker classification accuracy SHALL be at least 90% (residual
  must carry speaker/acoustic detail that phoneme stream does not).

### US-5: As an operator, I need V18 training to be stable on RTX 5090 32GB

**Acceptance criteria:**

- WHEN V18 is trained with `batch_size=1, grad_accum_steps=16,
  segment_length=65536` THEN training SHALL run without OOM.
- WHEN V18 training runs for 40 epochs THEN the reconstruction loss SHALL
  not diverge (no NaN, final loss within expected range of V16.3 training).
- WHEN gradient checkpointing is enabled THEN both phoneme and residual
  encoders SHALL support it.
- WHEN training is interrupted THEN it SHALL resume from `latest.pt`
  without manual intervention.

### US-6: As an operator, I need downstream compatibility with existing FM-DiT code

**Acceptance criteria:**

- WHEN the V18 latent is exported THEN it SHALL have the same shape as
  V16.3 (`[latent_dim=384, frame_rate ≈ 86 Hz]`).
- WHEN FM-DiT loads V18 latents via the existing latent extraction pipeline
  THEN no shape or dtype mismatch SHALL occur.
- WHEN FM-DiT uses the phoneme sub-stream as an explicit condition
  THEN the first 128 channels of the latent SHALL be consistently
  z_phoneme (fixed channel ordering).

## Functional Requirements

### FR-1: Phoneme encoder

- The phoneme encoder SHALL take raw audio `[B, 1, T_samples]` and output
  frame-level embeddings `[B, 128, T_frames]` aligned with the AE latent
  frame rate (T_frames = T_samples / 256).
- The phoneme encoder SHALL support loading pretrained weights from the
  V17.3 Conformer teacher (`outputs/models/v17_3_conformer_teacher_4spk_qc_1000/best.pt`).
- The phoneme encoder SHALL have a detachable CTC head mapping its output
  to pinyin+tone vocabulary.
- The phoneme encoder SHALL be trainable end-to-end during Stage 2 joint
  training.

### FR-2: Residual encoder

- The residual encoder SHALL take raw audio `[B, 1, T_samples]` and output
  frame-level features `[B, 256, T_frames]` aligned with the AE latent
  frame rate.
- The residual encoder architecture SHALL follow V16.3 WavVAEEncoder with
  reduced channels `[96, 192, 384, 768, 768]` and the same
  `strides=[2,4,4,8]`, `dilations=[1,3,9]`.
- The residual encoder SHALL NOT receive any direct supervision beyond
  reconstruction gradient.

### FR-3: Composite autoencoder

- The composite autoencoder SHALL concatenate `[z_phoneme, z_residual]`
  along channel dim to produce `z ∈ R^{B x 384 x T_frames}`.
- The composite autoencoder SHALL feed `z` to a V16.3-derived decoder that
  accepts `[B, 384, T_frames]` input.
- The composite autoencoder SHALL apply V16.3's soft range penalty
  bottleneck on `z` (`margin_mean=6.0, std_min=0.05, std_max=3.0,
  reg_weight=1e-4, reg_warmup_steps=5000`).
- The composite autoencoder SHALL support initializing the decoder from a
  V16.3 checkpoint.

### FR-4: Training loss

- The training loss SHALL include V16.3 reconstruction terms:
  `L1(x_hat, x) * 10.0 + MRSTFT * 0.5 + MultiScaleMel * 0.1`.
- The training loss SHALL include `reg_weight * range_penalty(z)` with V16.3
  schedule.
- The training loss SHALL include `lambda_phoneme * CTC(CTC_head(z_phoneme),
  pinyin_targets)` with `lambda_phoneme=0.3` by default.
- The training loss SHALL not include speaker, prosody, flow, or MI terms
  in V18 core.

### FR-5: Training schedule

- Stage 1 (optional) SHALL train only the phoneme encoder + CTC head with
  the reconstruction path frozen, for up to 2000 steps.
- Stage 2 SHALL train all components jointly for up to 40 epochs on the
  4-speaker crossed training set.
- Training SHALL use `lr=1e-4, lr_warmup_steps=1000, scheduler=cosine,
  grad_clip=1.0`.
- Training SHALL use `batch_size=1, grad_accum_steps=16,
  segment_length=65536, sr=22050`.

### FR-6: Inference and downstream integration

- The V18 model SHALL provide a method to return both `z_phoneme` and
  `z_residual` separately for inference.
- The V18 model SHALL provide a method to encode only the phoneme stream
  (skipping the residual encoder) for analysis.
- A latent-export script SHALL produce `.bin` files compatible with the
  existing FM-DiT `LatentDataset` format.

### FR-7: Diagnostics

- The validation loop SHALL log reconstruction SNR, STFT loss, and PESQ per
  epoch.
- A diagnostic script SHALL report:
  - Phoneme CTC frame accuracy on val
  - Speaker probe accuracy on z_phoneme and z_residual
  - Cross-speaker text retrieval top1 on z_phoneme
  - Latent PCA rank95 on z_phoneme, z_residual, and concatenated z
  - Recon SNR when z_residual is zeroed

## Non-Functional Requirements

### NFR-1: Memory

- V18 training on RTX 5090 32GB SHALL stay within 30 GB memory under the
  prescribed batch/segment settings.
- Gradient checkpointing SHALL be available and enabled by default.

### NFR-2: Total parameter count

- The combined V18 autoencoder (phoneme + residual + decoder) SHALL not
  exceed 160M parameters to stay within the memory envelope that V16.3 ran
  in.

### NFR-3: Codebase compatibility

- V18 SHALL be added without breaking V16.3, V17.1-V17.4 training configs
  or eval scripts.
- V18 model files SHALL live under `autoencoder/models/` alongside
  existing models.
- V18 configs SHALL live under `autoencoder/conf/experiment/` with the
  `v18_` prefix.

### NFR-4: Reproducibility

- Training SHALL log: config, git commit, checkpoint origin, eval metrics
  per epoch, and the seed (if used).
- Best checkpoint SHALL be selected by validation SNR.

## Out of Scope for V18 Core

The following are explicitly deferred and SHALL NOT block V18 completion:

- Speaker encoder as a separate stream
- Prosody encoder or pitch-conditioned generation
- Cyclic ASR-TTS training with shared encoder
- Flow-matching loss within AE training
- Frozen pretrained encoder (RAE-style) as the phoneme source
- FiLM / attention / gated fusion (V18 uses concat)
- Low-rank projections or MI overlap control
- Adaptation for multi-language or non-Mandarin data

These will be addressed in V18.1 through V18.5 if the V18 core hypothesis
holds.

## Acceptance Gate

V18 core is considered **successful** when all of the following hold on
the CosyVoice3 4-speaker crossed validation set:

1. Recon SNR >= 18 dB (US-2)
2. Latent rank95 >= 200 (US-2)
3. Phoneme frame CTC accuracy >= 70% (US-3)
4. Speaker probe on z_phoneme <= 60%, on z_residual >= 90% (US-3, US-4)
5. Zeroing z_residual drops SNR by > 10 dB (US-4)
6. FM-DiT trained on V18 latent achieves teacher-forced t=0 SNR >= 5 dB
   (US-1)

If criteria 1-5 hold but criterion 6 fails, V18 AE is still valid but the
downstream integration needs V18.1 (likely adding speaker stream).

If criteria 1-5 fail, V18 itself is invalidated and the analysis should
inform whether to retreat to frozen-encoder RAE design or push further on
joint supervised training.
