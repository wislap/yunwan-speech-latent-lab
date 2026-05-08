# V17.1/V17.2 Latent Structure Probe Report

Date: 2026-05-08

This report summarizes the V17.1/V17.2 experiments on top of the mature V16.3
WavVAE autoencoder. The goal was not to maximize reconstruction alone, but to
test whether shallow external/cross-sample constraints can pull a stronger
phoneme/speaker structure out of the existing latent space without breaking
reconstruction.

## Executive Summary

The current evidence does not support simply increasing the current pairwise
constraint strength. Stronger constraints can move some local training metrics,
but they either collapse the full latent rank or damage reconstruction before
producing useful cross-speaker text retrieval.

The best current baseline is still Guard3:

- it preserves much more full-latent rank than Guard1/Guard2;
- it avoids forcing a two-speaker speaker head into a single positive/negative
  axis;
- it keeps reconstruction usable, though below the V17.1 low-pressure setting.

Guard4 exposed an important implementation/diagnostic issue: storing projected
features in the queue creates stale bank features. Training logs can then look
better than the actual offline geometry. After fixing the queue to store pooled
latents and recompute bank projections with the current head, the hard phoneme
negative becomes a real gradient, but reconstruction drops sharply and offline
phoneme geometry still does not improve.

The most likely interpretation is:

> V16.3 latents contain phoneme information for reconstruction, but the
> information is not organized as a simple utterance-level factor that a shallow
> mean-pooled pairwise head can extract and reshape. The current constraint
> interface is too coarse and pushes against reconstruction-dominant latent
> geometry.

This does not yet prove that we must train from scratch with many speakers.
Before that, we should run frozen-probe and sequence-probe experiments to
separate "latent does not contain accessible structure" from "our adapter/head
is the wrong interface."

## Fixed Experimental Setup

Backbone initialization:

```text
/root/autodl-tmp/project/outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt
```

Main crossed training manifest:

```text
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_2spk_calib_qc_8_13p5_train_phalign_neighbors.jsonl
```

Observed size:

```text
1174 items, 587 text groups, about 3.72 h, currently 2 speakers
```

Validation manifest:

```text
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_2spk_calib_qc_8_13p5_val_phalign_neighbors.jsonl
```

Alignment/diagnostic manifest:

```text
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_align_v0_2spk_calib_train_phalign_neighbors.jsonl
```

All V17.2 runs below are one-epoch probes with V16.3 initialization, not
from-scratch training.

## Metric Meanings

Reconstruction:

- `SNR`: higher is better.
- `STFT`: lower is better.

Structure:

- `phoneme_pair_cos.same_text_diff_speaker`: should be high.
- `phoneme_pair_cos.diff_text_same_speaker`: should be meaningfully lower than
  same-text if phoneme/content is discriminative.
- `speaker_pair_cos.same_text_diff_speaker`: should be low or negative.
- `speaker_pair_cos.diff_text_same_speaker`: should be high.
- `latent_mean_rank.rank95`: full latent should not collapse.
- `latent_mean_rank.top1`: lower is healthier; very high means one dominant
  direction is swallowing the latent space.
- `phoneme_text_cross_speaker_retrieval.top1`: direct test of whether phoneme
  projection retrieves same text across speakers.

Important caveat: very high cosine values can be misleading when all points
collapse into a narrow cone. Retrieval and effective rank are needed to
distinguish real structure from global similarity.

## Results Table

| Run | Main change | SNR | STFT | Ph same-text cross-spk | Ph diff-text same-spk | Sp same-text cross-spk | Latent rank95 | Latent top1 | Ph rank95 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V17.1 baseline | recon only | 23.95 | 0.5911 | n/a | n/a | n/a | n/a | n/a | n/a | Reference reconstruction. |
| V17.1 w01 | low cross-factor pressure | 25.00 | 0.5818 | n/a | n/a | n/a | n/a | n/a | 31 | Best reconstruction in early matrix; structure pressure mild. |
| V17.1 w04 | stronger cross-factor | 24.94 | 0.5982 | n/a | n/a | n/a | about 20 | high risk | 18 | Stronger structure, rank compression risk. |
| V17.1 w04 detach | heads detached from latent | 24.53 | 0.5923 | n/a | n/a | n/a | n/a | n/a | n/a | Weaker than non-detach; non-detach really changes latent. |
| V17.2 Guard1 | paired sampler + rank guard + PCGrad | 24.26 | 0.6221 | 0.925 | n/a | -0.816 | 2 | 0.947 | 21 | Strong pair signals, but full latent collapses badly. |
| V17.2 Guard2 | stronger rank guard, weaker speaker | 24.05 | 0.6670 | 0.919 | n/a | -0.706 | 5 | 0.933 | 16 | Full latent improves from Guard1 but still collapsed; recon worse. |
| V17.2 Guard3 | margin speaker separation | 23.66 | 0.6808 | 0.954 | 0.985 | -0.291 | 21 | 0.813 | 26 | Best structural/recon tradeoff so far; phoneme still not text-discriminative. |
| V17.2 Guard4 stale queue | hard same-speaker phoneme negative | 23.02 | 0.6477 | 0.963 | 0.991 | -0.183 | 22 | 0.785 | 23 | Training logs looked good, but offline phoneme did not improve. |
| V17.2 Guard4 fixed queue | queue stores latent, recomputes heads | 21.88 | 0.6901 | 0.954 | 0.993 | -0.021 | 14 | 0.860 | 20 | Real hard-negative gradient hurts recon and still fails offline phoneme separation. |

## V17.1 Findings

V17.1 established three useful facts.

First, V16.3 is a strong reconstruction base. The recon-only baseline was
already healthy, and the low-pressure V17.1 run (`w01`) did not hurt it. This
means V17-style constraints are not automatically destructive.

Second, non-detached V17 constraints do affect the AE latent. The detached run
was weaker than the non-detached run, which rules out the simplest failure mode
where only the projection heads learn while the latent remains unchanged.

Third, higher cross-factor pressure immediately starts trading reconstruction
for rank compression. The `w04` run kept good SNR, but phoneme rank and latent
rank showed compression risk. This motivated V17.2 rank guards and paired
sampling.

## V17.2 Guard1/Guard2 Findings

Guard1 added:

- paired text sampler, so same-text speaker variants appear close in time;
- same-speaker branch;
- rank guards;
- PCGrad-lite on encoder/bottleneck.

It produced very strong pair metrics, but full latent rank collapsed:

```text
latent rank95 = 2
latent top1 = 0.947
```

This is not acceptable. It means the factor heads can win by compressing the
usable latent geometry, even while reconstruction remains superficially usable.

Guard2 increased rank pressure and reduced speaker pressure. It improved full
latent rank slightly:

```text
latent rank95: 2 -> 5
latent top1: 0.947 -> 0.933
```

But reconstruction degraded:

```text
SNR: 24.26 -> 24.05
STFT: 0.6221 -> 0.6670
```

The speaker head remained essentially rank-1. Guard2 showed that stronger rank
guard alone is not enough; it fights the symptom but does not fix the geometry.

## V17.2 Guard3 Findings

Guard3 replaced hard speaker opposition with a margin:

```text
different speaker: only needs cos <= -0.2
same speaker: only needs cos >= 0.6
```

This was the healthiest change in V17.2. It stopped the speaker objective from
forcing the two-speaker dataset into a single extreme axis.

Main result:

```text
SNR = 23.66
STFT = 0.6808
latent rank95 = 21
latent top1 = 0.813
phoneme rank95 = 26
speaker same-text cross-speaker cos = -0.291
speaker diff-text same-speaker cos = 0.981
```

Interpretation:

- Full latent geometry recovered substantially compared with Guard1/Guard2.
- Speaker margin behaves as intended.
- The phoneme head still does not separate different texts well:

```text
phoneme same-text cross-speaker cos = 0.954
phoneme diff-text same-speaker cos = 0.985
phoneme retrieval top1 = 0.031
```

This says the phoneme projection is still dominated by broad acoustic/speaker
similarity. High same-text cosine is not enough; different text from the same
speaker is even more similar.

Gradient diagnostics for Guard3 also show real conflict:

```text
constraint/recon weighted ratio reaches about 1.27 in late encoder
encoder cosine often negative, roughly -0.15 to -0.30
```

So Guard3 is not solved, but it is the best current baseline because it avoids
catastrophic latent collapse.

## V17.2 Guard4 and the Stale Queue Finding

Guard4 added a targeted hard negative:

```text
same speaker, different text:
  phoneme cosine should stay below pinyin-bag similarity + margin
```

The intent was correct: Guard3's clearest failure was that different text from
the same speaker still had phoneme cosine near 0.985.

The first Guard4 run showed promising online logs:

```text
phhardcos often around 0.3-0.5
same-text phcos often around 0.9+
```

But offline diagnostics still showed:

```text
phoneme diff-text same-speaker cos = 0.991
```

This contradiction exposed a queue problem. The queue stored projected phoneme
and speaker features. Once the head updates, queued features are stale. The
training loss may compare a current projection against an old projection, while
offline diagnostics recompute both sides with the current projection. The
online hard-negative metric can therefore look better than the real geometry.

The fix changed the queue to store pooled latent features and recompute bank
phoneme/speaker projections using the current head.

After this fix, Guard4 became a real constraint, but the result was worse:

```text
SNR = 21.88
STFT = 0.6901
phoneme diff-text same-speaker cos = 0.993
latent rank95 = 14
latent top1 = 0.860
```

Gradient diagnostics also became harsher:

```text
encoder.block.3 constraint/recon weighted ratio = 1.87
encoder.block.3 cosine = -0.296
```

Conclusion: hard same-speaker phoneme negatives, as currently implemented, are
too blunt. They increase pressure on the encoder but do not create useful
offline phoneme discrimination.

## Is the Problem Data Size or Latent Geometry?

The answer is not one-sided.

### What data scarcity explains

The current crossed dataset is small and only has two active speakers. This
makes speaker shortcuts very attractive. With only two voices, the speaker
factor naturally becomes close to a single axis, and cross-sample statistics are
fragile.

More speakers and more matched text groups should help:

- reduce two-speaker axis collapse;
- create richer same-text/different-speaker positives;
- create more diverse same-speaker/different-text negatives;
- make retrieval metrics less noisy.

### What data scarcity does not explain

If this were only a data-size problem, Guard4 fixed should at least move offline
phoneme different-text same-speaker cosine downward. Instead, it stayed near
0.993 while reconstruction degraded. That suggests the current constraint is
not reaching the right structure.

The more likely issue is that V16.3 latent geometry is already shaped by
reconstruction. It contains content information, but not as a clean utterance
mean factor. Mean-pooled shallow heads see stable speaker/acoustic cues more
easily than local phoneme content.

In other words:

```text
content exists in the latent,
but not in the representation format that this constraint is asking for.
```

### Does this mean from-scratch multi-speaker training is required?

Not proven yet.

From-scratch training with many speakers could help if the latent is designed
from the beginning as:

- content/phoneme path;
- speaker path;
- residual/free path;
- decoder composition over the three.

But it is expensive and risky. It may also fail if the content constraint is
still utterance-level and too coarse.

Before committing to from-scratch scale, we need two cheaper probes:

1. Freeze V16.3 and test whether better probes can extract content.
2. Use sequence-level content heads instead of utterance mean pooling.

If frozen sequence probes fail, then from-scratch structured training becomes
more justified.

## Current Best Interpretation

The experiments support this interpretation:

```text
The current V16.3 latent is reconstruction-useful but factor-mixed.
Shallow utterance-level pairwise heads can align broad examples, but they do not
produce a stable phoneme/content geometry. Stronger hard negatives mostly create
gradient conflict with reconstruction.
```

The problem is therefore not simply "too little data." More data is necessary
for a serious result, but the current objective/interface is also wrong or
underpowered.

## Recommended Next Experiments

### 1. Frozen External-Teacher Probe

Implemented entry points:

```text
autoencoder/scripts/v17_teacher_probe.py
scripts/run_v17_teacher_probe.sh
```

This probe freezes:

- V16.3 WavVAE;
- pretrained speech teacher;

and trains only:

- `z_v16.3 -> sequence adapter -> teacher feature`.

Initial HuBERT_BASE results:

```text
run:
/root/autodl-tmp/project/outputs/models/v17_teacher_probe_hubert_100

100 steps, 5 s crops, hidden=256, depth=2, batch=1

eval_loss = 0.731
student retrieval top1 = 0.047
teacher retrieval top1 = 0.172
student same-text cross-speaker cos = 0.991
student diff-text same-speaker cos = 0.998
student rank95 = 4
teacher rank95 = 37
```

With batch pairwise distillation and variance guard:

```text
run:
/root/autodl-tmp/project/outputs/models/v17_teacher_probe_hubert_pair100

100 steps, 5 s crops, hidden=256, depth=2, batch=4

eval_loss = 0.703
student retrieval top1 = 0.031
teacher retrieval top1 = 0.172
student same-text cross-speaker cos = 0.944
student diff-text same-speaker cos = 0.996
student rank95 = 2
teacher rank95 = 37
```

Interpretation:

- The adapter can quickly learn per-sample teacher similarity: pooled cosine
  reaches about 0.9.
- But simple teacher regression still collapses global student geometry.
- HuBERT_BASE itself is only a weak text-retrieval teacher on this Chinese TTS
  alignment set, with top1 about 0.17.
- Therefore this teacher/probe is useful as a plumbing check, but not yet a
  clean content target.

The attempted multilingual `WAV2VEC2_XLSR_300M` download was too slow on the
server during this run and was cancelled before training. It remains the better
candidate teacher if the checkpoint can be pre-fetched or mirrored.

Decision:

```text
Do not use HuBERT_BASE teacher regression as the next V17 backbone loss.
First obtain a stronger multilingual/ASR/phoneme teacher, or add a teacher-side
projection trained for same-text cross-speaker retrieval.
```

### 2. V17.3 From-Scratch Content Teacher

Implemented entry points:

```text
autoencoder/scripts/v17_3_train_content_teacher.py
scripts/run_v17_3_content_teacher.sh
```

This branch does not modify AE. It trains a standalone raw-audio content
teacher from scratch:

```text
audio -> strided conv frontend -> TCN blocks -> attentive pool -> content emb
```

Training objectives:

- supervised contrastive over `same_text_group`;
- same-speaker/different-text hard negative;
- VICReg-style rank guard;
- optional pinyin-bag multilabel prediction from the content embedding.

Early runs:

```text
v17_3_content_teacher_smoke:
  30 steps, weak rank guard
  collapsed: pair cos ~= 1, rank95 = 3

v17_3_content_teacher_rank500:
  stronger VICReg/rank guard, 500 steps
  rank95 = 21
  top1_var = 0.162
  retrieval top1 = 0.031
  same_text_diff_speaker cos = 0.769
  diff_text_same_speaker cos = 0.765

v17_3_content_teacher_text500:
  pinyin-bag auxiliary from pooled hidden, 500 steps
  rank95 = 22
  retrieval top1 = 0.000
  same_text_diff_speaker cos = 0.769
  diff_text_same_speaker cos = 0.795

v17_3_content_teacher_embedtext1000:
  pinyin-bag auxiliary from content projection, 1000 steps
  rank95 = 24
  top1_var = 0.149
  retrieval top1 = 0.031
  same_text_diff_speaker cos = 0.761
  diff_text_same_speaker cos = 0.747
  speaker centroid acc = 0.563

v17_3_content_teacher_ctc001_500:
  weak CTC auxiliary, 500 steps
  rank95 = 21
  top1_var = 0.159
  retrieval top1 = 0.063
  same_text_diff_speaker cos = 0.758
  diff_text_same_speaker cos = 0.774

v17_3_content_teacher_frame500:
  dense frame-level pinyin CE, 500 steps
  rank95 = 22
  top1_var = 0.130
  retrieval top1 = 0.047
  same_text_diff_speaker cos = 0.746
  diff_text_same_speaker cos = 0.748
```

Interpretation:

- The from-scratch teacher can be made non-collapsed with strong rank guard.
- The current utterance-level contrastive objective still does not learn robust
  cross-speaker text retrieval.
- Pinyin-bag auxiliary reduces speaker leakage somewhat and makes training
  healthier, but does not yet create a strong content space.

This is useful negative evidence: even a standalone teacher is not solved by
same-text pair contrastive plus utterance pooling. Weak CTC and dense pinyin
frame CE add real optimization signals, but in the current raw-waveform TCN
teacher they still do not produce robust cross-speaker text retrieval. The next
V17.3 iteration should either use a stronger mature frontend/encoder
(log-mel/Conformer or pretrained multilingual ASR frontend) or train on a
larger/more speaker-diverse crossed set before treating this teacher as a
distillation target.

### 3. Frozen Head Probe

Freeze the V16.3 AE entirely. Train only cross-factor heads.

Purpose:

```text
Can current latents support cross-speaker text retrieval at all without moving
the reconstruction space?
```

Decision:

- If frozen heads fail, current mean-pooled representation is not enough.
- If frozen heads succeed, our previous failures came from gradient conflict,
  not lack of accessible information.

### 4. Frozen Sequence Probe

Freeze V16.3. Replace utterance mean with a lightweight temporal content encoder:

- small temporal conv stack or causal/noncausal attention;
- attentive pooling;
- cross-speaker same-text InfoNCE;
- same-speaker different-text negatives.

Purpose:

```text
Check whether phoneme/content is present locally but lost by mean pooling.
```

This is the most important next step.

### 5. Head-Only Warmup Before Encoder Gradients

For V17.2-style training:

```text
phase 1: train heads only
phase 2: unfreeze bottleneck/high encoder with very small structure weight
phase 3: optionally let structure gradients reach more encoder layers
```

This avoids immediately moving the decoder's familiar latent distribution.

### 6. Larger Multi-Speaker Data Only After the Probe

The next dataset scale should be at least:

```text
4 speakers, about 4 h total, matched text groups
```

But scaling should be paired with the sequence probe. Scaling the current
utterance-mean hard-negative objective is unlikely to be efficient.

## Current Candidate Baseline

Use Guard3 as the current baseline for comparison:

```text
config:
autoencoder/conf/experiment/v17_2_v16_3_cross_factor_paired_margin_guard3_w04.yaml

checkpoint:
/root/autodl-tmp/project/outputs/models/v17_2_guard3_probe_w04_pcgrad/best.pt
```

Do not promote Guard4 fixed as a candidate baseline:

```text
checkpoint:
/root/autodl-tmp/project/outputs/models/v17_2_guard4_fixed_probe/best.pt

reason:
lower SNR, worse latent rank, no offline phoneme discrimination improvement
```

## Practical Notes

Disk was cleaned after these experiments. `latest.pt` files for Guard3/Guard4
temporary runs were removed, while `best.pt` checkpoints were kept.

The queue implementation now stores pooled latent features and recomputes
phoneme/speaker projections from the current head. This should remain the
default for future queue-based factor constraints; projected-feature queues are
too easy to misread in diagnostics.
