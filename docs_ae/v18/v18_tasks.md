# V18 Implementation Tasks

Derived from `v18_design.md` and `v18_requirements.md`. Each task maps to a concrete file change or training run. Tasks are ordered to match the Day 1-14 plan in the design document. Optional tasks are marked with `*`.

## 1. Scaffolding and data access

- [ ] 1.1 Verify V18 spec directory and docs exist (design.md, requirements.md, tasks.md).
  - _Requirements: NFR-3_
- [ ] 1.2 Verify remote artifacts on active server: V16.3 best.pt, V17.3 Conformer teacher best.pt, and both 4-speaker crossed manifests.
  - _Requirements: FR-1, FR-3_
- [ ] 1.3 Confirm a random training item in the manifest contains valid pinyin tokens and pinyin_durations for CTC targets; fix pipeline if not.
  - _Requirements: FR-4, US-3_

## 2. Phoneme encoder module

- [ ] 2.1 Add V18PhonemeEncoder (Conformer-based) in autoencoder/models/v18_encoder.py, outputs [B, 128, T_frames] aligned to 256x-stride.
  - _Requirements: FR-1_
- [ ] 2.2 Add PhonemeCTCHead (1-2 layer linear) consuming z_phoneme producing pinyin+tone logits.
  - _Requirements: FR-4_
- [ ] 2.3 Implement V17.3 teacher weight loading with backbone key remapping; heads randomly initialized.
  - _Requirements: FR-1_
- [ ] 2.4 Add frame-rate alignment logic; unit-test that T_frames == T_samples / 256 on a reference shape.
  - _Requirements: FR-1, US-6_
- [ ]* 2.5 Optional PhonemeClusterHead predicting soft k-way cluster indices for the weak-discrete prior.
  - _Requirements: FR-4_

## 3. Residual encoder module

- [ ] 3.1 Add V18ResidualEncoder in autoencoder/models/v18_encoder.py reusing V16.3 EncoderBlock with channels [96,192,384,768,768], output 256d.
  - _Requirements: FR-2_
- [ ] 3.2 Enable gradient checkpointing on the residual encoder blocks.
  - _Requirements: NFR-1, US-5_
- [ ] 3.3 Sanity test that residual encoder produces [B, 256, T_frames] on reference input and consumes under 8 GB backward memory at batch 1 segment 65536.
  - _Requirements: NFR-1_

## 4. Composite autoencoder

- [ ] 4.1 Add V18Autoencoder class combining phoneme encoder, residual encoder, V16.3 decoder, and range-penalty bottleneck, with encode/decode/forward methods returning z, z_phoneme, z_residual, x_hat.
  - _Requirements: FR-3_
- [ ] 4.2 Implement decoder weight loading from V16.3 checkpoint with key counting and logging.
  - _Requirements: FR-3_
- [ ] 4.3 Verify a full V18Autoencoder instance can be built with all pretrained init paths resolved and a reference forward pass succeeds.
  - _Requirements: FR-1, FR-3_
- [ ]* 4.4 Optional residual-stream mute mode (z_residual = 0) for Stage 1 ablation.
  - _Requirements: FR-5, US-4_

## 5. Training integration

- [ ] 5.1 Extend autoencoder/train.py to detect model_type == 'v18' and instantiate V18Autoencoder instead of WavVAE.
  - _Requirements: FR-3, NFR-3_
- [ ] 5.2 Extend dataloader collate to carry pinyin ID sequences and frame-aligned lengths for CTC.
  - _Requirements: FR-4_
- [ ] 5.3 Add CTC loss term lambda_phoneme * F.ctc_loss(phoneme_logits, pinyin_targets); zero when lambda_phoneme is 0.
  - _Requirements: FR-4_
- [ ] 5.4 Update checkpoint save/load to serialize V18 state (phoneme encoder, residual encoder, decoder, bottleneck, heads).
  - _Requirements: US-5, NFR-4_
- [ ] 5.5 Add validation-loop logging of recon SNR, STFT, PESQ, and phoneme_ctc_loss using existing evaluate() scaffolding.
  - _Requirements: US-2, FR-7, NFR-4_
- [ ]* 5.6 Optional Stage 1 warmup flag: freeze residual encoder and decoder, train only phoneme encoder + CTC head up to 2000 steps.
  - _Requirements: FR-5_

## 6. Configuration

- [ ] 6.1 Add autoencoder/conf/experiment/v18_base.yaml with model_type v18, architecture dims, V16.3 decoder init path, V17.3 phoneme init path, lambda_phoneme 0.3, V16.3 stability settings, use_checkpoint true.
  - _Requirements: FR-5, FR-3, NFR-1_
- [ ] 6.2 Add autoencoder/conf/experiment/v18_phoneme_warmup.yaml for Stage 1 warmup (freeze residual and decoder, 2000 steps).
  - _Requirements: FR-5_
- [ ]* 6.3 Optional autoencoder/conf/experiment/v18_base_no_phoneme.yaml ablation with lambda_phoneme 0.
  - _Requirements: FR-4_

## 7. Launch scripts

- [ ] 7.1 Add scripts/run_v18_stage1_warmup.sh for Stage 1 warmup remote launch.
  - _Requirements: FR-5_
- [ ] 7.2 Add scripts/run_v18_stage2_joint.sh for Stage 2 joint training remote launch, loading Stage 1 phoneme encoder.
  - _Requirements: FR-5_
- [ ]* 7.3 Optional scripts/run_v18_ablation_no_phoneme.sh for the lambda_phoneme=0 ablation run.

## 8. Training runs

- [ ] 8.1 Stage 1 run: up to 2000 steps of v18_phoneme_warmup; verify CTC loss decreases and val phoneme frame accuracy exceeds 70%.
  - _Requirements: US-3, FR-5_
- [ ] 8.2 Stage 2 run: up to 40 epochs of v18_base; abort if recon SNR stays below 12 dB for 10 consecutive epochs.
  - _Requirements: US-2, FR-5, NFR-1_
- [ ]* 8.3 Optional Stage 3 fine-tune: reduce lambda_phoneme to 0.1, small lr, 10 epochs, only if Stage 2 recon below V16.3 by more than 2 dB.
  - _Requirements: US-2_
- [ ]* 8.4 Optional ablation run v18_base_no_phoneme to quantify the net effect of phoneme supervision.

## 9. Diagnostics on V18 AE

- [ ] 9.1 Add autoencoder/scripts/v18_ae_diagnostics.py reporting recon SNR/STFT/PESQ, phoneme CTC frame accuracy, cross-speaker retrieval on z_phoneme, speaker probe on z_phoneme and z_residual, rank95 on latent streams, and recon SNR with z_residual zeroed; output JSON summary.
  - _Requirements: US-2, US-3, US-4, FR-7_
- [ ] 9.2 Run v18_ae_diagnostics.py on Stage 2 best checkpoint and commit summary JSON plus a section in the V18 report draft.
  - _Requirements: US-2, US-3, US-4_

## 10. Latent export for FM-DiT

- [ ] 10.1 Update or add V18 path to autoencoder/extract_latents.py so exported latents match V16.3 shape [384, T_frames] with a split marker for the phoneme sub-stream.
  - _Requirements: FR-6, US-6_
- [ ] 10.2 Export V18 latents for the LJSpeech FM-DiT training manifests to enable downstream comparison with V16.3.
  - _Requirements: US-1, US-6_

## 11. FM-DiT downstream evaluation

- [ ] 11.1 Train an FM-DiT run on V18-exported latents using the same hyperparameters as the V16.3 FM-DiT reference run.
  - _Requirements: US-1_
- [ ] 11.2 Run teacher-forced t sweep on the V18 FM-DiT model and record t=0 SNR, v_cos, recon_cos, pred_std, target_std on the same t grid used for V16.3.
  - _Requirements: US-1_
- [ ] 11.3 Decode sampled audio, run Whisper or FunASR, compute WER and CER, compare against V16.3 + FM-DiT baseline.
  - _Requirements: US-1_

## 12. Evaluation gate and report

- [ ] 12.1 Draft docs_ae/v18/v18_report.md summarizing V18 AE metrics, FM-DiT metrics, and side-by-side comparison with V16.3.
  - _Requirements: NFR-4_
- [ ] 12.2 Evaluate the six acceptance-gate criteria from requirements with actual numbers; define V18.1 scope if criteria 1-5 pass but criterion 6 fails.
  - _Requirements: Acceptance Gate_
- [ ]* 12.3 If V18 gate passes, draft a V18.x roadmap for speaker, prosody, and cyclic extensions.

## Dependencies

- Tasks 2-4 must complete before task 5.
- Task 5 must complete before tasks 6 and 7.
- Tasks 6 and 7 must complete before task 8.
- Task 8.1 should complete before 8.2 unless V17.3 init quality is sufficient.
- Task 9 depends on task 8.
- Task 10 depends on task 9.
- Task 11 depends on task 10.
- Task 12 depends on task 11.

## Out of Scope (explicit non-tasks)

These are intentionally not V18 core tasks:

- Speaker encoder as a separate stream
- Prosody encoder or pitch-conditioned generation
- Cyclic ASR-TTS training with shared encoder
- Flow-matching loss inside AE training
- Frozen pretrained ASR as phoneme source (RAE-style)
- FiLM, gated, or attention-based fusion (V18 uses concat)
- Low-rank projections or MI overlap control
- Multi-language or non-Mandarin support
- Changes to the FM-DiT architecture

These become V18.x tasks only if V18 core passes or if task 12.2 explicitly requires them.
