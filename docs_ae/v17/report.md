# V17 Semantic Latent Representation Report

Date: 2026-05-04

## Purpose

V17 is motivated by the FM-DiT failure mode observed on V16.3 AE latents:

- V16.3 is a strong reconstruction latent.
- FM-DiT can learn most of the flow path.
- The exact generative start point remains unstable.
- Therefore the latent is not yet a generative-friendly, condition-predictable representation.

V17 should not be framed as simply "make the encoder larger". The target is a latent space whose semantic structure is externally anchored, linearly readable, and useful for downstream generative modeling.

## Evidence From V16.3 And FM-DiT

V16.3 AE is healthy as a reconstruction model:

- 256x, dim384.
- Best AE checkpoint around 20.7 dB train/eval SNR, with recon diagnostics showing clear gains over V16.2.
- Effective rank confirms the latent is used rather than collapsed.
- PCA p95 directions: 234 components cover 95% frame-level variance.

FM-DiT on V16.3 latents exposed a different issue:

- Full-sample conditional DiT training is stable.
- PCA-shaped latent initialization improves early FM loss.
- A staged `t_min` curriculum from large t to zero avoids training collapse.
- Final FM loss can reach about `0.02`.

However pure sampling remains weak:

```text
CFG=1.0: corr=0.0364, std=0.0776
CFG=1.5: corr=0.0236, std=0.0938
CFG=2.0: corr=0.0346, std=0.1028
```

The teacher-forced t sweep isolates the failure:

```text
t      SNR      MSE       v_cos   recon_cos  pred_std / target_std
0      -0.44dB  1.34693   0.585   0.084      0.463 / 1.103
0.025  10.19dB  0.11650   0.969   0.951      1.034 / 1.103
0.05   14.99dB  0.03860   0.989   0.984      1.090 / 1.103
0.1    19.41dB  0.01401   0.996   0.994      1.103 / 1.103
0.15   21.06dB  0.00960   0.997   0.996      1.100 / 1.103
0.25   22.87dB  0.00634   0.997   0.997      1.098 / 1.103
0.5    26.52dB  0.00274   0.997   0.999      1.101 / 1.103
0.8    29.78dB  0.00130   0.992   0.999      1.105 / 1.103
```

### Experiment Provenance

Server:

```text
ssh -p 10891 root@connect.westd.seetacloud.com
cd /root/autodl-tmp/project
```

V16.3 AE evidence:

- Training log: `v16_3_launch.log`
- Best checkpoint: `outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt`
- Effective-rank/PCA files: `docs/v16.3/figures/effective_rank/effective_rank_summary.csv` and `docs/v16.3/figures/latent_pca_directions/v16_3/frame_p95_pca_directions.npz`
- FM-DiT latent export: `fm_dit/data/ljspeech_v16_3_256x_dim384/`
- Latent stats: `fm_dit/data/ljspeech_v16_3_256x_dim384/latent_stats.json`
- Train/val manifests: `fm_dit/data/ljspeech_v16_3_256x_dim384/train.json`, `fm_dit/data/ljspeech_v16_3_256x_dim384/val.json`

FM-DiT run used for the failure diagnosis:

- Train log: `outputs/fm_dit/v1_v16ae_200_cond_pca_twarm_scratch_a/train.log`
- Checkpoint: `outputs/fm_dit/v1_v16ae_200_cond_pca_twarm_scratch_a/fm_dit_v1_v16ae_model.pt`
- Config in checkpoint args:
  - full-sample training: `full_samples=True`, `max_frames=0`
  - condition encoder enabled, duration predictor disabled
  - PCA-shaped source: `noise_init=pca`, `pca_variance_scale=1.0`, `pca_no_mean=false`
  - PCA path: `docs/v16.3/figures/latent_pca_directions/v16_3/frame_p95_pca_directions.npz`
  - staged endpoint curriculum: `t_min_start=0.8`, `t_min_end=0.0`, `t_min_warmup_steps=8000`, cosine schedule
  - train samples: 190

Training evidence from `train.log`:

```text
Step 12000/12000 | FM: 0.0209 | t_min=0.000 | LR: 1.00e-05
[Eval] Step 12000: corr=0.0462 latent_mse=0.050805

Testing...
CFG=1.0: corr=0.0364, std=0.0776
CFG=1.5: corr=0.0236, std=0.0938
CFG=2.0: corr=0.0346, std=0.1028
```

Teacher-forced t sweep:

- JSON: `outputs/fm_dit/v1_v16ae_200_cond_pca_twarm_scratch_a/teacher_forced_t_sweep/teacher_forced_t_sweep.json`
- CSV: `outputs/fm_dit/v1_v16ae_200_cond_pca_twarm_scratch_a/teacher_forced_t_sweep/teacher_forced_t_sweep.csv`
- Script: `fm_dit/scripts/teacher_forced_t_sweep.py`
- Aggregation: mean over 5 samples, each with 8 repeats.

Audit note:

- The CFG sampling table and teacher-forced sweep table above are backed by server artifacts.
- The later `t_start` skipping table should be treated as provisional until its exact output file or command log is recovered.

Interpretation:

- The model has learned the local/interior flow.
- The exact source endpoint is ill-conditioned.
- Adding even 2.5% real latent information makes the velocity field highly predictable.
- Sampling fails because the first segment starts from a condition-independent base distribution that is not coupled to the target latent.

## Why Mimi Latents Did Not Show This Failure

The Mimi/codec latent experiments likely avoided this problem because the latent space was already generation-friendly:

- Codec/RVQ latents are quantized or semi-discrete.
- The latent distribution is more clustered and regularized.
- Semantic and acoustic structure is partially organized by the codec.
- The source-target path has a more natural coupling.
- The target space may not contain as much continuous high-frequency, phase, and decoder-compensation detail.

V16.3 AE latent is different:

```text
Mimi/codec latent: generation-friendly representation
V16.3 AE latent: high-fidelity reconstruction representation
```

This distinction matters. A latent that reconstructs well is not automatically a latent that can be generated from weak external conditions.

## Flow Matching Interpretation

Flow matching itself does not guarantee that every latent space has an easy source endpoint.

The current path is effectively:

```text
condition + independent PCA-shaped noise -> target AE latent
```

At `t=0`, the model receives no target information in `x_t`. The conditional input must determine a high-dimensional continuous acoustic latent direction. With only frame phoneme/pitch/energy conditions and 200 samples, this is underdetermined.

The likely regression effect is:

```text
many valid target directions -> MSE-average velocity -> amplitude contraction
```

This explains the low sample std and the sharp recovery at `t=0.025`.

Skipping `t=0` during sampling was tested and is not enough:

Provenance status: the exact server artifact for this table was not found during the 2026-05-04 audit, so these numbers should be re-run or traced before being used as a formal comparison.

```text
t_start,cfg,corr,raw_std,norm_std
0,2       0.0484, 0.1150, 0.6047
0.025,2   0.0602, 0.0743, 0.3902
0.05,2    0.0726, 0.0578, 0.3034
0.1,2     0.0698, 0.0451, 0.2370
```

Reason: starting integration at `t_start>0` still initializes from `p0`, not from the true `p(t_start)`.

## First Direction: Condition-Dependent Coarse Latent

One direct fix is:

```text
external condition -> coarse latent z0
z0 + structured noise -> DiT refinement -> final latent
```

Variants:

1. Predict full latent mean.
2. Predict PCA coefficients.
3. Predict mean and variance in a low-rank latent subspace.

This direction is practical and likely helps DiT, but it is not the full representation-learning goal. It may produce a useful conditional initializer without making the latent semantically interpretable.

### Active Test: Coarse Latent + Unconditional Residual Flow

Date: 2026-05-04

Server run:

```text
cd /root/autodl-tmp/project
PID: 2784
Log: fm_dit_v1_200_coarse_residual_uncond_a.log
PID file: fm_dit_v1_200_coarse_residual_uncond_a.pid
Output: outputs/fm_dit/v1_v16ae_200_coarse_residual_uncond_a
```

Code changes:

- `fm_dit/train_v1_v16ae.py`
- `fm_dit/scripts/decode_v16_dit_samples.py`
- `fm_dit/scripts/teacher_forced_t_sweep.py`

Experiment form:

```text
condition encoder -> coarse latent z_c
unconditional FM: noise -> residual r = z_target - stopgrad(z_c)
sample: z_sample = z_c + r_sample
```

Key switches:

```text
--coarse-latent
--coarse-hidden-dim 1024
--coarse-layers 4
--coarse-heads 8
--coarse-dim-ff 4096
--residual-flow
--uncond-residual-flow
--coarse-weight 1.0
--cond-drop-prob 0.0
--no-duration-predictor
--full-samples
--batch-size 16
--grad-accum-steps 2
```

Early smoke:

- Server smoke run passed with small model and 2 training steps.
- Full run starts normally.
- Step 50: `FM=1.9460`, `Coarse=0.9921`.
- Step 100: `FM=1.8859`, `Coarse=0.9834`.

Full run result:

```text
Step 12000/12000 | FM: 0.2079 | Coarse: 0.0669 | LR: 1.00e-05
[Eval] Step 12000: corr=0.8876 latent_mse=0.011919

Testing...
CFG=1.0: corr=0.9785, std=0.2114
CFG=1.5: corr=0.9790, std=0.2114
CFG=2.0: corr=0.9793, std=0.2113
```

Checkpoint:

```text
outputs/fm_dit/v1_v16ae_200_coarse_residual_uncond_a/fm_dit_v1_v16ae_model.pt
```

Audit caveat: the final `Testing` block samples from the training split, so it should be read as a fit/interpolation signal rather than evidence of validation generalization.

Decoded waveform readout:

```text
train, indices 0/1/2
AE avg:     snr=22.09  cos=0.9970
CFG=1.0:   snr=19.90  cos=0.9951  mse=0.000096  rms=0.0985
CFG=1.5:   snr=18.85  cos=0.9934  mse=0.000128  rms=0.0987
CFG=2.0:   snr=19.87  cos=0.9951  mse=0.000097  rms=0.0985

val, indices 0/1/2
AE avg:     snr=22.77  cos=0.9974
CFG=1.0:   snr=-0.64  cos=0.0011  mse=0.010994  rms=0.0375
CFG=1.5:   snr=-0.63  cos=0.0008  mse=0.010977  rms=0.0372
CFG=2.0:   snr=-0.64  cos=0.0010  mse=0.011001  rms=0.0376
```

Decode artifacts:

```text
outputs/fm_dit/v1_v16ae_200_coarse_residual_uncond_a/decode_train/decode_report.json
outputs/fm_dit/v1_v16ae_200_coarse_residual_uncond_a/decode_val/decode_report.json
```

Full training-set decode and ASR readout:

```text
Decode dir:
outputs/fm_dit/v1_v16ae_200_coarse_residual_uncond_a/decode_train_all_cfg1

ASR model:
Whisper base.en

Reference:
manifest text

Targets:
original, ae_recon, cfg_1
```

```text
original: n=190  WER mean/median/p90=0.036/0.000/0.118  CER mean/median/p90=0.021/0.000/0.072  exact=126/190
ae_recon: n=190  WER mean/median/p90=0.035/0.000/0.118  CER mean/median/p90=0.020/0.000/0.068  exact=125/190
cfg_1:    n=190  WER mean/median/p90=0.036/0.000/0.118  CER mean/median/p90=0.020/0.000/0.068  exact=120/190
```

ASR artifacts:

```text
outputs/fm_dit/v1_v16ae_200_coarse_residual_uncond_a/decode_train_all_cfg1/decode_report.json
outputs/fm_dit/v1_v16ae_200_coarse_residual_uncond_a/decode_train_all_cfg1/asr_cer_wer.json
outputs/fm_dit/v1_v16ae_200_coarse_residual_uncond_a/decode_train_all_cfg1/asr_cer_wer.log
```

Interpretation: on the training split, the generated `cfg_1` audio preserves lexical content at the same ASR level as original audio and AE reconstruction. The nonzero original WER/CER is mostly Whisper/base normalization and recognition error rather than model error.

Teacher-forced t sweep:

```text
train
t=0      snr=15.44  mse=0.03486  v_cos=0.983  recon_cos=0.986  pred_std=1.096 / target_std=1.103
t=0.025  snr=15.43  mse=0.03489  v_cos=0.982  recon_cos=0.986  pred_std=1.096 / target_std=1.103
t=0.05   snr=15.43  mse=0.03494  v_cos=0.981  recon_cos=0.986  pred_std=1.097 / target_std=1.103
t=0.1    snr=15.43  mse=0.03496  v_cos=0.979  recon_cos=0.986  pred_std=1.097 / target_std=1.103
t=0.15   snr=15.42  mse=0.03502  v_cos=0.976  recon_cos=0.986  pred_std=1.097 / target_std=1.103
t=0.25   snr=15.44  mse=0.03487  v_cos=0.970  recon_cos=0.986  pred_std=1.097 / target_std=1.103
t=0.5    snr=15.86  mse=0.03177  v_cos=0.937  recon_cos=0.987  pred_std=1.099 / target_std=1.103
t=0.8    snr=18.67  mse=0.01675  v_cos=0.779  recon_cos=0.993  pred_std=1.108 / target_std=1.103

val
t=0      snr=-1.47  mse=1.66122  v_cos=0.615  recon_cos=0.009  pred_std=0.690 / target_std=1.088
t=0.025  snr=-1.47  mse=1.66063  v_cos=0.589  recon_cos=0.009  pred_std=0.690 / target_std=1.088
t=0.05   snr=-1.42  mse=1.63471  v_cos=0.568  recon_cos=0.026  pred_std=0.693 / target_std=1.088
t=0.1    snr=-0.86  mse=1.43205  v_cos=0.580  recon_cos=0.182  pred_std=0.744 / target_std=1.088
t=0.15   snr=-0.11  mse=1.20736  v_cos=0.609  recon_cos=0.348  pred_std=0.805 / target_std=1.088
t=0.25   snr=1.30   mse=0.87262  v_cos=0.646  recon_cos=0.568  pred_std=0.888 / target_std=1.088
t=0.5    snr=5.08   mse=0.36593  v_cos=0.673  recon_cos=0.833  pred_std=0.958 / target_std=1.088
t=0.8    snr=12.47  mse=0.06712  v_cos=0.610  recon_cos=0.972  pred_std=1.010 / target_std=1.088
```

Sweep artifacts:

```text
outputs/fm_dit/v1_v16ae_200_coarse_residual_uncond_a/teacher_forced_t_sweep_train/teacher_forced_t_sweep.csv
outputs/fm_dit/v1_v16ae_200_coarse_residual_uncond_a/teacher_forced_t_sweep_train/teacher_forced_t_sweep.json
outputs/fm_dit/v1_v16ae_200_coarse_residual_uncond_a/teacher_forced_t_sweep_val/teacher_forced_t_sweep.csv
outputs/fm_dit/v1_v16ae_200_coarse_residual_uncond_a/teacher_forced_t_sweep_val/teacher_forced_t_sweep.json
```

Primary readout:

- If pure sampling std/corr improves, the main bottleneck is likely endpoint coupling.
- If coarse loss improves but sampling remains weak, residual/detail still carries condition-dependent structure.
- If coarse loss does not improve, the current condition features are insufficient for V16.3 latent prediction.

Observed readout:

- On the training split, coarse latent plus residual flow directly solves the endpoint problem: decoded samples are close to AE reconstruction and teacher-forced `t=0` is no longer pathological.
- On the validation split, it does not generalize: decoded waveform cosine is near zero and teacher-forced `t=0` remains essentially the same failure mode as the previous raw-latent DiT.
- Therefore this test supports the endpoint-coupling diagnosis, but does not yet support the claim that a larger phoneme/condition encoder can solve the bottleneck as a generalizable architecture.
- The likely current failure is that 190 training samples are enough for a large coarse predictor to memorize condition-to-latent mapping, while the V16.3 latent still contains target-specific acoustic/detail information not recoverable from the present conditions.

Recommended next experiment:

1. Add true validation evaluation to the training loop and choose checkpoints by validation, not train sampling.
2. Replace full-latent coarse prediction with a lower-dimensional target first, such as PCA coefficients or a semantic/prosody factor block.
3. Run the same architecture on a substantially larger manifest before judging representation capacity.
4. If data is still limited, reduce coarse predictor capacity and regularize it; otherwise it mainly tests memorization.

## External Encoder Alignment

A broader V17 approach is to align AE latent with external representations:

- Whisper / HuBERT / WavLM / wav2vec features.
- Mel / pitch / energy / duration features.
- Text/phoneme encoders.
- Speaker/style encoders.

Basic form:

```text
audio -> AE encoder -> z
audio/text/prosody -> frozen or supervised encoder -> h
projection(z) aligned with h
```

Candidate losses:

- Cosine or L2 feature alignment.
- InfoNCE.
- JSD mutual-information lower bound.
- Barlow Twins / VICReg-style covariance losses.
- Perceptual decoder-side feature matching.

This is useful, but a pure alignment head is not sufficient if the adapter is deep enough to hide semantics. It can become a post-hoc explanation layer rather than a structural latent representation.

## Disagreement: Orthogonal Semantic Factorization

One proposed direction was fixed semantic blocks:

```text
z = [z_content, z_prosody, z_timbre, z_residual]
```

with block orthogonality and linear probes.

This is too rigid.

The objection is important:

- Real semantics are not orthogonal.
- Phoneme, duration, stress, energy, and articulation are correlated.
- Speaker/timbre can affect articulation.
- Prosody depends on content.

Therefore, forcing clean orthogonal decomposition can produce artificial factors rather than true semantic structure.

This direction should not be the core of V17.

Useful pieces that can be retained:

- Linear probes.
- Leakage diagnostics.
- Factor swap tests.
- Capacity limits on residual/detail information.

But strict orthogonal factorization is not the desired representation.

## Disagreement: Deep Adapter-Based Factor Discovery

Another proposed direction was multiple adapters:

```text
z -> adapter_i(z) -> semantic factor_i
```

This is also insufficient if adapters are expressive.

Problem:

- A nonlinear adapter can discover arbitrary hidden information.
- Semantics may live in the adapter, not in the latent.
- Factor interpretability becomes weak.

Useful retained idea:

- Use shallow, mostly linear filters.
- Keep probes linear.
- Prefer direct latent-space filtering over deep semantic extraction.

## Additional Critique: Clustered Semantics Are Not Linear Semantics

Another discussion raised a sharper objection to cluster-based semantic heads.

If an adapter outputs discrete clusters:

```text
z -> adapter -> cluster
```

the model may learn a semantic classifier rather than a semantic geometry.

This is not enough for V17, because the research goal is not only:

```text
can the representation classify content/speaker/prosody?
```

but:

```text
can semantic changes be represented as continuous, linearly composable directions?
```

Cluster structure and linear semantic structure differ:

| Cluster Objective | Linear Semantic Geometry |
|---|---|
| discrete regions | continuous directions |
| decision boundaries | vector arithmetic |
| classification semantics | computable semantics |
| weak interpolation | explicit interpolation and analogy |

Therefore clusters should not be the only semantic target.

Useful retained role for clusters:

- diagnostics;
- prototype summaries;
- optional label anchors;
- coarse semantic bins for contrastive sampling.

But the main representation should be continuous.

## Additional Critique: Adapter Loss Alone May Not Shape z

If all semantic losses are applied only after adapters:

```text
loss(adapter(z))
```

then the adapter may learn to decode semantics from a still-entangled latent.

This is weaker than requiring the latent itself to have semantic geometry.

V17 should therefore include at least some losses that act directly on `z` or on very shallow linear projections of `z`.

Examples:

```text
distance(z_a, z_b) for controlled pairs
direction consistency in z or shallow-filter space
linear probes directly on shallow factors
```

This does not mean forcing all of `z` to be disentangled. It means the semantics should not live only inside a deep adapter.

## Direction Consistency And Analogy Loss

A critical missing ingredient for linear semantic geometry is direction consistency.

Controlled tuples can define semantic vector analogies:

```text
z(x_1) - z(x_2) ~= z(x_3) - z(x_4)
```

Examples:

- same text, speaker A -> speaker B should induce a consistent speaker-change direction;
- same speaker, text A -> text B should induce a consistent content-change direction;
- same text/speaker, prosody A -> prosody B should induce a consistent prosody-change direction.

This can be applied in:

1. raw latent `z`;
2. shallow factor space `F_g(z)`;
3. coefficient/dictionary space `a`.

The safest first version is to apply it in shallow factor space, because raw `z` still needs to preserve reconstruction detail.

Example loss:

```text
d_12 = normalize(F_s(x_speakerB_text1) - F_s(x_speakerA_text1))
d_34 = normalize(F_s(x_speakerB_text2) - F_s(x_speakerA_text2))
L_analogy = 1 - cos(d_12, d_34)
```

This directly encourages linear directions instead of only classification.

## Predictability Control As A Stable MI Alternative

The full MI-graph idea is attractive but difficult.

An alternative is predictability control:

```text
factor_i should predict label_i well
factor_i should predict label_j only up to the desired level
```

This can replace early MI estimation with controlled linear probes:

```text
Linear(F_content) -> text_id: high accuracy
Linear(F_content) -> speaker_id: low/target accuracy
Linear(F_speaker) -> speaker_id: high accuracy
Linear(F_speaker) -> text_id: low/target accuracy
```

Advantages:

- easier to train than MINE/JSD critics;
- directly measures linear readability;
- supports target ranges instead of full independence;
- can be used as both training loss and diagnostic.

This should be considered a Stage 1/2 alternative to direct pairwise MI lower-bound matching.

## Semantic Dictionary Latent

A better conceptual model is a non-orthogonal semantic dictionary:

```text
z_t ~= sum_k a_{t,k} b_k + r_t
```

where:

- `b_k` are learnable semantic directions in latent space.
- `a_{t,k}` are activation coefficients.
- `r_t` is residual detail.

This respects the key requirement:

```text
semantics can be linearly superposed without requiring clean orthogonal decomposition
```

Constraints should be:

- Sparse activations.
- Temporal smoothness.
- Coherence control, not strict orthogonality.
- External semantic anchoring.
- Residual bottleneck.

This is conceptually strong, but may be more invasive for a first implementation.

## Alternative Route: Emergent Structured Latent

There is another possible route that should be kept separate from explicit factorization.

Instead of decomposing latent semantics, one may train a unified latent:

```text
audio -> encoder -> z
z aligned with text/speaker/prosody/audio SSL
flow matching learns the latent distribution
```

This is a weak-structure-prior plus strong-generative-model route.

It is theoretically plausible. CLIP-like and latent-diffusion systems often develop semantic directions without explicit factor blocks.

However, it is not sufficient by itself for V17's goal.

Risks:

- semantic directions may not become linear;
- semantic variables may stay entangled;
- flow can model the joint distribution without creating controllable coordinates;
- multimodal alignment can make `z` semantic-related but not semantic-structured.

If this route is tested, it must include geometry-inducing constraints:

- direction consistency / analogy loss;
- local linearity constraints;
- information bottleneck or residual capacity control;
- flow loss backpropagated to the encoder so non-flow-friendly latent geometry is penalized.

This route should be treated as an ablation:

```text
Can semantic geometry emerge without explicit factor filters?
```

It should not replace the two-constraint V17 plan until proven.

## Shallow Multi-Group Linear Filter Bank

A practical version of the dictionary idea is a multi-group shallow linear filter bank.

Instead of hard semantic blocks:

```text
s_g = linear_filter_g(z)
```

Each group reads from the same latent through a shallow filter:

```text
z_g = (z * mask_g) @ W_g
```

Properties:

- Groups may overlap.
- Semantics are not forced to be orthogonal.
- The transform remains near-linear.
- Masks make the used latent dimensions inspectable.
- Probes and contrastive objectives operate directly on these shallow activations.

Regularization should avoid degeneracy rather than enforce independence:

- Sparse masks.
- Overlap budget, not zero overlap.
- Group activation balance.
- Residual/detail capacity limit.
- Leakage probes.

This can be kept as an optional diagnostic or adapter implementation detail, but it should not define the V17 core module boundary.

## Controlled Pair Data

The strongest supervision should come from controlled data relationships rather than manual factor definitions.

Using a TTS system, construct samples where some semantic variables are fixed and others vary:

```text
same text, different speaker
same speaker, different text
same text and speaker, different prosody/style
same text/speaker/prosody, different augmentation/noise
```

These define invariances:

| Pair Type | Content | Speaker | Prosody | Detail |
|---|---|---|---|---|
| same text, different speaker | same | different | maybe | different |
| same speaker, different text | different | same | maybe | different |
| same text/speaker, different prosody | same | same | different | different |
| same everything, different augmentation | same | same | same | different |

This is better than assuming semantic orthogonality. It defines what each semantic group should be invariant to and sensitive to.

Recommended initial dataset:

```text
N_text ~= 100
N_speaker ~= 8
N_prosody ~= 3
total ~= 2400 synthetic controlled samples
```

Each sample should store:

- text id
- speaker id
- prosody/style id
- augmentation id
- phoneme sequence
- pitch/energy/duration statistics

## Supervised Contrastive MI Proxy

Before full mutual-information graph estimation, use supervised contrastive losses as a stable MI proxy.

For content factor:

```text
positive = same text_id
negative = different text_id
```

For speaker factor:

```text
positive = same speaker_id
negative = different speaker_id
```

For prosody factor:

```text
positive = same prosody_id or same prosody bin
negative = different prosody_id/bin
```

This gives a lower-bound-like proxy:

```text
I_lb(F_i; label_j) ~= - SupConLoss(F_i, label_j)
```

It is more stable than directly using MINE/JSD critics as the first experiment.

## Mutual Information Graph Control

The complete target is not independence. It is semantic information-geometry control.

Define semantic variables:

```text
Y_content, Y_speaker, Y_prosody, Y_style, Y_detail
```

Estimate a target MI matrix from controlled labels:

```text
M_target[i,j] = I(Y_i; Y_j)
```

Then train latent factors:

```text
F_content, F_speaker, F_prosody, F_style, F_detail
```

so that:

```text
M_latent[i,j] ~= M_target[i,j]
```

This avoids the false assumption that all semantic groups should be independent.

Practical issue:

- True MI is not available.
- JSD/InfoNCE/MINE are lower bounds or proxies.
- Critic capacity and batch size strongly affect the estimate.

Therefore the first version should use interval or ranking constraints, not exact equality:

```text
I_lb(F_i; F_j) should lie in [low_ij, high_ij]
```

Examples:

```text
content-speaker: low
content-prosody: medium allowed
speaker-prosody: low/medium allowed
prosody-style: medium/high allowed if style controls pitch/energy
```

This can later be implemented with pairwise JSD critics:

```text
joint:    (F_i(x), F_j(x))
product:  (F_i(x), F_j(x_shuffled))
```

But this should not be Stage 0.

## Feasibility Assessment

The full system is high value but high risk if attempted in one step.

Main risks:

- MI lower-bound estimators are unstable.
- Multi-objective losses can conflict with reconstruction.
- Synthetic TTS data may contain unintended correlations.
- Residual/detail channels can steal information.
- Factors can collapse or leak information.

Estimated feasibility:

```text
Full MI-graph V17 in one shot: low to medium
Staged V17: medium to high
```

The project should proceed in stages.

## Older Staged Factorization Plan

Status: superseded as the V17 core plan. The useful parts may remain as diagnostics inside the external encoder-adapter constraint, but V17 should not be framed as a filter-bank/factorization architecture.

### Stage 0: Offline Linear Semantic Filter Probe

Do not retrain AE yet.

Use frozen V16.3 AE encoder latents and synthetic controlled data.

Train shallow linear filter banks:

```text
z -> F_content
z -> F_speaker
z -> F_prosody
z -> F_detail
```

Objectives:

- Supervised contrastive content/speaker/prosody losses.
- Linear probes.
- Leakage diagnostics.
- Mask sparsity and overlap budget.

Success criteria:

- Same text/different speaker: content similarity high, speaker similarity low.
- Same speaker/different text: speaker similarity high, content similarity low.
- Linear probes read the intended semantics from the intended factors.
- Unintended semantics are weak in leakage probes.

If Stage 0 fails, the current V16.3 latent does not linearly contain sufficiently accessible semantic factors.

### Stage 1: Add Filter Bank As AE Auxiliary Structure

Train AE with:

```text
L = L_recon + lambda_semantic * L_filter_semantic
```

Keep semantic loss weak initially.

Do not add full MI graph yet.

Success criteria:

- Reconstruction quality does not collapse.
- Factor probes improve over Stage 0.
- Effective rank remains healthy.
- Latent PCA/noise robustness does not degrade sharply.

### Stage 2: Controlled MI Proxy Matrix

Use supervised contrastive/probe objectives to estimate factor-label structure:

```text
A[i,j] ~= I_lb(F_i; Y_j)
```

Match a target high/medium/low matrix rather than exact MI values.

Success criteria:

- Factor-label matrix follows the intended semantic structure.
- Leakage is reduced without forcing unrealistic independence.

### Stage 3: Pairwise Factor JSD Critics

Add pairwise factor-factor JSD MI lower-bound critics:

```text
I_lb(F_i; F_j)
```

Control each pair within target intervals derived from the semantic data graph.

Success criteria:

- MI graph matches intended relation structure.
- Factors remain useful for reconstruction.
- Factor swaps produce interpretable changes.

### Stage 4: DiT On Structured Factors

Train DiT on the structured factor space:

```text
condition -> content/prosody factors
noise/refinement -> residual/detail factors
```

Success criteria:

- `t=0` teacher-forced velocity cosine improves over V16.3 raw latent.
- Pure sampling latent std no longer collapses.
- Decoded audio becomes semantically aligned.

## Optional Diagnostic: Shallow Filter Groups

Use shallow filter groups:

```text
F_g(z) = Linear_g(z * sigmoid(mask_g))
```

Recommended initial groups:

```text
content
speaker/style
prosody
detail/residual
```

For single-speaker LJSpeech-only tests, speaker/style is not meaningful. It requires synthetic or multi-speaker controlled data.

Regularization:

- Mask sparsity.
- Soft overlap budget.
- Activation variance floor.
- Residual/detail bottleneck.
- Factor dropout for diagnostics.

Do not use this as the primary V17 module definition. The current V17 core is the two-constraint design below.

## V17 Core Constraint Modules

Correction: the controlled synthetic corpus is a data source, not a V17 model module. V17 should stay minimal and be defined by two external constraints on the AE latent:

```text
audio -> AE encoder -> z -> AE decoder -> audio
                 |
                 + external modality encoder-adapter constraint
                 + flow / generative constraint
```

### Module 1: External Modality Encoder-Adapter Constraint

This module anchors the AE latent to external, interpretable modalities.

Purpose:

- Force `z` to expose information that external conditions can read.
- Prevent semantics from existing only inside a downstream DiT.
- Keep the AE latent aligned with text/content, speaker, and prosody signals.

Form:

```text
audio/text/prosody/speaker -> frozen_or_supervised external encoder -> h_m
AE latent z -> adapter_m(z) -> h_hat_m
loss_m(h_hat_m, h_m)
```

Possible modalities:

```text
text / phoneme / pinyin-tone content
speaker / prompt voice identity
prosody / duration / pitch / energy / style
audio SSL features if needed
```

Adapter rule:

- The adapter should be shallow enough that the latent, not the adapter, carries the useful structure.
- Start with linear or low-depth MLP adapters.
- Deep adapters are ablations, not the core mechanism.

Losses:

```text
L_content = cosine/contrastive/probe loss against text or phoneme encoder features
L_speaker = speaker embedding or prompt-voice identity loss
L_prosody = pitch/energy/duration/style prediction or alignment loss
```

This is not strict factorization. It is external anchoring:

```text
z should be readable by simple adapters into external modality spaces
```

### Module 2: Flow / Generative Constraint

This module makes the latent generation-friendly.

Purpose:

- Penalize latent geometries that reconstruct well but are hard to generate.
- Directly test whether conditions can couple to the latent endpoint.
- Make V17 optimize for downstream FM-DiT usability, not only AE reconstruction.

Form:

```text
condition c -> flow model -> z
```

or, for a residual variant:

```text
condition c -> coarse z_c
flow noise -> residual r
z = z_c + r
```

Loss:

```text
L_flow = flow_matching_loss(z, c)
```

The key is that `L_flow` should constrain the latent produced by the encoder:

```text
L = L_recon
  + lambda_ext * L_external_adapter
  + lambda_flow * L_flow
```

This makes the AE latent pay a cost when it stores information in a form that cannot be predicted or transported from conditions.

Interpretation:

- External adapter constraint says: the latent must be semantically readable.
- Flow constraint says: the latent must be generatively reachable.
- The two together define V17.

Synthetic CosyVoice3 data is useful because it gives clean external labels and scalable controlled conditions, but it remains the training data pipeline rather than a core model module.

### Implementation Smoke

Date: 2026-05-04

Implemented files:

```text
autoencoder/v17_constraints.py
autoencoder/train.py
autoencoder/conf/experiment/v17_constraints_smoke.yaml
```

Current minimal implementation:

- External adapter constraint:
  - pinyin+tone symbolic external encoder via `pypinyin`;
  - duration-aware frame-aligned pinyin bucket CE from latent frames;
  - shallow temporal latent adapter instead of utterance-level mean pooling for content;
  - speed readout from utterance-level latent pooling;
  - speaker/style heads exist but are disabled in the single-speaker seed config.
- Flow/generative constraint:
  - small conditional latent flow-matching network;
  - condition vector built from pinyin/speaker/style/speed metadata;
  - target detach is enabled in smoke mode to train the flow readout without letting it reshape `z` prematurely.

Smoke command after design fix:

```bash
cd /root/autodl-tmp/project
/root/miniconda3/bin/python -m autoencoder.train \
  experiment=v17_constraints_smoke \
  runtime=remote \
  experiment.train.epochs=1 \
  experiment.train.log_interval=36 \
  experiment.train.eval_batches=1 \
  experiment.train.output_dir=outputs/models/v17_constraints_smoke_duration
```

Smoke result:

```text
V17 constraints: ext_weight=0.05, flow_weight=0.01, params=1,497,057
Train samples: 72, Eval samples: 8

step 36 | G=2.8907 | v17=0.3229 ext=6.2556 flow=1.0135
step 72 | G=2.2063 | v17=0.3138 ext=6.0715 flow=1.0186

Epoch 0/1 | avg_G=2.6606
Eval: SNR=-0.21dB | STFT=2.8506 | KL=0.000000
Training complete.
```

Interpretation:

- The two V17 constraints are now wired into AE training and optimize jointly with reconstruction.
- The content constraint is now a real symbolic external modality target, not a random text hash.
- The smoke is only a functionality check on the seed corpus, not a quality result.

### Design Audit After Smoke

The first smoke implementation should not be treated as a faithful V17 experiment. It exposed several design risks, and the first two have now been corrected in code:

1. The hashed text encoder was only a plumbing placeholder.
   - Fixed: replaced with pinyin+tone tokenization and duration-aware frame-aligned CE.
   - Current duration source: manifest `pinyin_durations` if present; otherwise punctuation/speed heuristic.
   - Remaining limitation: heuristic duration should later be replaced by true frontend/teacher/forced-alignment durations.

2. Speaker/style heads are trivial on the current seed corpus.
   - `seed_zh_v0` has only one speaker and mostly one style.
   - Cross-entropy heads can be solved by bias terms, so they do not prove that `z` carries speaker/style.
   - Fixed for smoke: speaker/style losses are disabled by default.

3. Utterance-level mean pooling is too weak for content alignment.
   - `z.mean(dim=-1)` destroys temporal order.
   - Fixed for content: content now uses a temporal adapter and frame-level pinyin targets.
   - Remaining limitation: current duration-aware alignment is heuristic unless true token durations are provided.

4. Flow gradients into `z` are delicate.
   - If the flow target is not detached, the loss can reshape latent geometry in unintended ways, including amplitude shrinkage or shortcut solutions.
   - The smoke config now defaults to `v17_flow_detach_target: true`; this trains the flow readout against current latents.
   - A later true joint V17 run must deliberately re-enable latent-gradient flow only after reconstruction and external targets are stable.

5. The flow condition currently uses metadata features directly.
   - This tests conditional reachability of `z`, but not necessarily whether the external adapter and flow share the same learned modality representation.
   - A stricter design should route both adapter and flow through the same external condition encoder outputs.

Required next correction before a real run:

```text
add multi-speaker / multi-style data before enabling speaker/style losses
replace heuristic pinyin durations with teacher/frontend/forced-alignment durations
keep flow target detached for diagnostics, then run a deliberate joint-gradient ablation
report adapter readout and flow metrics separately
```

## What Should Not Be Merged Into V17 Core Yet

The following ideas should remain separate until validated:

- Strict orthogonal latent blocks.
- Deep nonlinear adapters for semantic extraction.
- Discrete clusters as the only semantic representation.
- Full pairwise JSD/MINE MI graph from the first run.
- Strong KL/prior matching on raw AE latent.
- Direct full-384 condition-to-latent regression as the main representation objective.
- Pure emergent unified latent without direction/geometry constraints.

They may be useful in ablations but should not define the V17 core.

## Good Ideas That Can Be Fused

The best fused V17 direction is:

```text
controlled paired data
+ external modality encoder-adapter constraint
+ supervised contrastive MI proxy
+ continuous semantic heads
+ direction consistency / analogy loss
+ external encoder/label anchoring
+ soft overlap/coherence control
+ staged MI graph control
+ flow / generative constraint
```

This preserves the key requirements:

- Semantics are not assumed orthogonal.
- Semantics can combine linearly.
- Latent readouts are externally anchored rather than self-invented.
- Adapter readouts should remain shallow enough to be inspectable.
- Mutual information is controlled according to semantic relationships, not blindly minimized.
- Linear semantic directions are explicitly encouraged, not assumed to emerge from clustering.
- The final success criterion is downstream generative usability, especially the FM-DiT `t=0` endpoint.

## Immediate Next Step

Build the two-constraint V17 smoke path:

1. Generate or collect controlled paired audio from CosyVoice3.
2. Encode all audio with the AE.
3. Add external modality encoder-adapter losses on `z`.
4. Add a small flow/generative loss on `z`.
5. Compare against reconstruction-only AE and post-hoc DiT.
6. Report:
   - reconstruction quality
   - adapter readout scores
   - adapter leakage/shortcut checks
   - flow teacher-forced `t=0` metrics
   - sampling std/corr
   - decoded ASR CER/WER

This is the lowest-cost way to test whether the V17 latent becomes both externally readable and generatively reachable.
