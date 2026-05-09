# V17.3 Teacher-Distilled Latent Structure Report

Date: 2026-05-09

This report summarizes the V17.3 experiments after the V17.1/V17.2
cross-factor probes. The focus is narrow: can an external content teacher help
pull cross-speaker text/content structure out of the mature V16.3 WavVAE latent
without destroying reconstruction?

## Executive Summary

The current V17.3 evidence supports the teacher direction, but not the current
direct-to-AE loss design as a finished solution.

What worked:

- The 4-speaker crossed CosyVoice3/AISHELL-style dataset is now structurally
  useful: same text appears under four speaker conditions.
- A Conformer audio teacher trained on this data learns real content structure:
  cross-speaker same-text retrieval reaches roughly `0.47` top-1 on the held-out
  set, while speaker centroid accuracy stays low.
- A frozen teacher can provide a learnable signal to an AE latent adapter:
  pointwise adapter-teacher cosine rises to about `0.63`.
- Queue-based relational/NCE distillation runs with `batch_size=1`, so 10s
  windows remain feasible without OOM.

What did not work yet:

- Raw AE utterance-mean latents remain speaker-dominated.
- The teacher adapter tends to collapse into a narrow cone: same-text and
  different-text cosine are both close to `1.0`.
- Stronger relational/NCE pressure hurts reconstruction before producing
  teacher-like retrieval.
- A lightweight per-step rank guard improves offline adapter rank, but does not
  solve cross-speaker content retrieval.

The most likely interpretation is:

> The teacher contains the desired content structure, but the current shallow
> mean-pooled adapter and direct encoder pressure are not the right interface.
> They can align pointwise direction while failing to preserve the teacher's
> relative geometry. Before further unfreezing the AE, we need a frozen-latent
> probe and a bank-level anti-collapse design.

## Dataset State

Main server for this stage:

```text
ssh -p 14025 root@connect.westc.seetacloud.com
cd /root/autodl-tmp/project
```

The useful 4-speaker dataset is built from the earlier 2-speaker crossed corpus
plus two AISHELL-3 speaker prompts synthesized on the same text groups.

Final manifests:

```text
CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_train_phalign_neighbors.jsonl
CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl
```

Dataset summary:

```text
raw items: 2600
raw same-text groups: 650
QC kept items: 2276
train items: 2164
train groups: 541
train duration: about 6.84 h
val items: 112
val groups: 28
val duration: about 0.36 h
speakers per kept group: 4
```

Speakers:

```text
spk_zero_shot
spk_cross_lingual
aishell3_SSB0631
aishell3_SSB0426
```

Automation:

```text
scripts/build_cosyvoice3_cross_speaker_dataset.py
```

Pipeline notes are in:

```text
docs_ae/v17/cosyvoice3_pipeline.md
```

## Teacher Model

Teacher script:

```text
autoencoder/scripts/v17_3_train_conformer_teacher.py
scripts/run_v17_3_conformer_teacher.sh
```

Teacher checkpoint:

```text
outputs/models/v17_3_conformer_teacher_4spk_qc_1000/best.pt
```

Training log:

```text
outputs/logs/v17_3_conformer_teacher_4spk_20260509_130951.log
```

Teacher result after 1000 steps:

```text
text_cross_speaker_retrieval.top1 = 0.7768  (training/eval summary from teacher run)
mean_rank = 1.4911
same_text_diff_speaker_cos_mean = 0.7127
diff_text_same_speaker_cos_mean = 0.1384
speaker_centroid_acc = 0.0714
```

When re-evaluated through the AE distillation diagnostic on the 112-item val
set, teacher metrics vary slightly by crop/eval path but remain clearly
content-structured:

```text
teacher retrieval top1: about 0.44-0.47
teacher mean rank: about 4.3-4.4
teacher same_text_diff_speaker cos: about 0.73-0.75
teacher diff_text_same_speaker cos: about 0.39-0.47
teacher speaker centroid acc: about 0.19-0.21
```

This is the strongest positive result in V17.3. The external teacher has the
kind of structure we wanted: content is more readable than speaker identity.

## AE Distillation Implementation

Main files changed:

```text
autoencoder/v17_constraints.py
autoencoder/train.py
autoencoder/conf/experiment/v17_3_v16_3_4spk_teacher_distill_smoke.yaml
autoencoder/conf/experiment/v17_3_v16_3_4spk_teacher_relational_smoke.yaml
autoencoder/scripts/v17_3_teacher_distill_diagnostics.py
```

The V17.3 teacher constraint is an external frozen-teacher path:

```text
audio crop -> frozen Conformer teacher -> teacher embedding
AE latent z -> shallow Conv1d adapter + pool + MLP -> adapter embedding
```

The teacher is frozen and run in fp32 to avoid mixed-precision mask issues. The
adapter and, when not detached, the AE latent path receive gradients.

Loss variants tested:

1. Pointwise cosine distillation:

```text
L_point = 1 - cos(adapter(z), teacher(audio))
```

2. Queue-based relational distillation:

```text
adapter-to-bank similarity should match teacher-to-bank similarity
```

3. Teacher-guided InfoNCE:

```text
positive: same_text_group, different speaker
bank: current sample + teacher queue
```

4. Lightweight rank guard:

```text
VICReg-style variance/covariance/top1 guard on the adapter output
```

The queue variant was added because `batch_size=4` on 10s windows OOMed on the
31GB GPU. With `batch_size=1`, paired sampling still places same-text speaker
variants close together, so the teacher queue can provide positive/negative
structure.

## Diagnostics

Diagnostic script:

```text
autoencoder/scripts/v17_3_teacher_distill_diagnostics.py
```

It compares three spaces:

- frozen teacher embedding;
- trained teacher adapter output from AE latent;
- raw AE utterance-mean latent.

Key metrics:

- `text_cross_speaker_retrieval.top1`: can same text be retrieved across
  speakers?
- `same_text_diff_speaker_cos_mean`: should be high for content space.
- `diff_text_same_speaker_cos_mean`: should be lower than same-text if content
  is discriminative.
- `speaker_centroid_acc`: should be low for a content space.
- `rank95` and `top1`: distinguish real geometry from cone collapse.

Important caveat:

High cosine alone is not evidence of useful structure. A collapsed cone can make
almost every pair highly similar. Retrieval and rank are required.

## Results Table

| Run | Main objective | SNR | STFT | Adapter-teacher cos | Adapter retrieval top1 | Adapter same-text cross-spk cos | Adapter diff-text same-spk cos | Adapter rank95 | Adapter top1 eig | Raw latent retrieval top1 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V16.3 baseline diagnostic | no V17.3 adapter | 20.73 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.0179 | Old V16.3 checkpoint evaluated on 4spk val latent means. |
| Pointwise teacher smoke | point cosine only | 19.91 | 0.8129 | 0.630 | 0.0268 | 0.9935 | 0.9982 | 35 | 0.360 | 0.0357 | Learns pointwise direction, but adapter geometry is not content-discriminative. |
| Relational/NCE v1 | point + strong relational + NCE | 16.45 | 0.9353 | 0.598 | 0.0268 | 0.9869 | 0.9981 | 21 | 0.607 | 0.0357 | Structure loss runs, but rank collapses and reconstruction drops. |
| Relational/NCE v2 | lower relational/NCE + rank guard | 17.11 | 0.9009 | 0.634 | 0.0625 | 0.9939 | 0.9984 | 32 | 0.479 | 0.0446 | Rank improves versus v1, retrieval slightly improves, but still not teacher-like. |
| Frozen-latent adapter 300 | AE frozen, train sequence adapter only | n/a | n/a | 0.372 | 0.0357 | 0.8409 | 0.8401 | 50 | 0.120 | 0.0446 | Healthy rank and lower speaker leakage, but content retrieval not yet above raw latent. |
| Frozen-latent adapter 1000 g8 | AE frozen, larger contrastive batch | n/a | n/a | 0.412 | 0.3036 | 0.8968 | 0.4322 | 14 | 0.470 | 0.0446 | Strong evidence that V16.3 latent contains readable content, but extraction is low-rank/compressive. |
| Frozen-latent adapter 1000 g8 rank2 | stronger rank guard retry | n/a | n/a | 0.399 | 0.1161 | 0.7585 | 0.4565 | 36 | 0.386 | 0.0446 | Rank stays near teacher, but retrieval is much weaker than low-rank adapter. |
| Frozen teacher target | teacher itself | n/a | n/a | n/a | 0.44-0.47 | 0.73-0.75 | 0.39-0.47 | about 33 | about 0.30 | n/a | Desired reference geometry. |

Diagnostic artifacts:

```text
outputs/diagnostics/v17_3_teacher_distill_adapter_val/summary.json
outputs/diagnostics/v17_3_teacher_relational_adapter_val/summary.json
outputs/diagnostics/v17_3_teacher_relational_rank_adapter_val/summary.json
outputs/diagnostics/v16_3_baseline_4spk_latent_val/summary.json
outputs/diagnostics/v17_3_frozen_latent_adapter_probe_300/summary.json
outputs/diagnostics/v17_3_frozen_latent_adapter_probe_1000_g8/summary.json
outputs/diagnostics/v17_3_frozen_latent_adapter_probe_1000_g8_rank2_retry/summary.json
```

Model artifacts:

```text
outputs/models/wav_vae_v17.3_v16.3_4spk_teacher_distill_smoke_remote_main/best.pt
outputs/models/wav_vae_v17.3_v16.3_4spk_teacher_relational_smoke_remote_main/best.pt
outputs/models/wav_vae_v17.3_v16.3_4spk_teacher_relational_rank_smoke_remote_main/best.pt
```

Frozen-latent adapter probe script:

```text
autoencoder/scripts/v17_3_train_frozen_latent_teacher_adapter.py
```

## What The Numbers Mean

### The teacher is good enough to use

The frozen teacher is not random and not speaker-dominated. Its held-out
retrieval is far above chance:

```text
teacher top1 about 0.44-0.47 over 28 text groups
```

Its same-text cross-speaker cosine is much higher than different-text
same-speaker cosine. This is exactly the geometry the AE latent lacks.

### Raw AE latent remains speaker-dominated

Across V16.3 and V17.3 runs, raw utterance-mean latent has:

```text
speaker centroid acc = 1.0
diff_text_same_speaker cos much higher than same_text_diff_speaker cos
cross-speaker text retrieval near chance
```

Example from the pointwise V17.3 diagnostic:

```text
same_text_diff_speaker cos = 0.196
diff_text_same_speaker cos = 0.771
retrieval top1 = 0.0357
```

This is the core bottleneck: reconstruction latents clearly encode speaker and
acoustics in a way that overwhelms utterance-level content retrieval.

### Pointwise teacher loss is underconstrained

The pointwise smoke achieved:

```text
adapter_teacher_cos_mean = 0.630
```

But adapter geometry was not useful:

```text
same_text_diff_speaker cos = 0.9935
diff_text_same_speaker cos = 0.9982
retrieval top1 = 0.0268
speaker centroid acc = 1.0
```

This means the adapter can align to the teacher's average direction without
preserving the teacher's relative structure. In other words, pointwise cosine
is too weak and too easy to game.

### Strong relational loss is not automatically better

The strong relational/NCE version was intended to preserve teacher geometry,
but it made two things worse:

```text
SNR: 19.91 -> 16.45
adapter rank95: 35 -> 21
adapter top1 eig: 0.360 -> 0.607
```

The loss did run; `tnce` was nonzero and `tcos` improved during training. The
failure was not implementation emptiness. The issue is that the optimization
found a narrow-cone solution that satisfies some local similarities while
damaging reconstruction.

### Rank guard helps but is not enough

The lower-pressure rank version improved offline adapter rank:

```text
rank95: 21 -> 32
top1 eig: 0.607 -> 0.479
retrieval top1: 0.0268 -> 0.0625
```

But it still did not approximate teacher content geometry:

```text
adapter same_text_diff_speaker cos = 0.9939
adapter diff_text_same_speaker cos = 0.9984
speaker centroid acc = 1.0
```

The current rank guard is also weak in the actual training loop because
`batch_size=1` means per-step rank statistics are not meaningful. The offline
diagnostic sees the real issue; the training-time guard mostly does not.

### Frozen-latent adapter probe changes the diagnosis

The frozen-latent probe freezes both the AE and the Conformer teacher, then
trains only a sequence adapter:

```text
V16.3 AE latent sequence -> residual Conv adapter + attention pooling -> teacher embedding
```

This removes reconstruction gradient conflict from the experiment. It asks a
cleaner question:

```text
Is teacher-readable content already present in the mature V16.3 latent?
```

The short 300-step probe was mostly negative:

```text
adapter top1 = 0.0357
mean rank = 20.75
speaker centroid acc = 0.375
rank95 = 50
top1 eig = 0.120
```

This said the adapter could reduce speaker leakage and keep high rank, but had
not yet learned text retrieval.

The stronger 1000-step, `groups_per_batch=8` probe changed the picture:

```text
adapter top1 = 0.3036
mean rank = 6.607
same_text_diff_speaker cos = 0.8968
diff_text_same_speaker cos = 0.4322
speaker centroid acc = 0.0536
```

Compared with raw V16.3 latent on the same eval:

```text
raw latent top1 = 0.0446
raw latent mean rank = 15.77
raw same_text_diff_speaker cos = 0.323
raw diff_text_same_speaker cos = 0.769
raw speaker centroid acc = 1.0
```

This is a decisive positive diagnostic: the existing V16.3 latent does contain
content information that can be made cross-speaker readable by a trained
sequence adapter.

However the adapter reaches this by compressing the representation:

```text
adapter rank95 = 14
adapter top1 eig = 0.470
teacher rank95 = 39
teacher top1 eig = 0.165
```

So the result is not "the latent is already solved." It is:

> V16.3 latent contains extractable content, but the accessible content manifold
> is weak/entangled enough that the adapter learns a compressed discriminative
> projection rather than a teacher-like full geometry.

## Interpretation

The V17.3 experiments changed the diagnosis from vague to precise.

Earlier V17.1/V17.2 results showed that hand-designed cross-factor constraints
could create attractive pairwise logs while failing offline retrieval and
sometimes collapsing latent rank. V17.3 shows the same risk persists even with a
better teacher: the target geometry is available, but the adapter/AE interface
still allows degenerate solutions.

The likely causes are:

1. The adapter is mean-pooled over time.

   A 10s utterance contains many local phonetic events. Mean pooling may erase
   precisely the sequence-level differences needed for text retrieval, while
   preserving global speaker/acoustic statistics.

2. Direct AE unfreezing mixes incompatible gradients too early.

   Reconstruction wants detailed acoustic information. Teacher content loss
   wants speaker-invariant geometry. Without a proven adapter-only path, the
   encoder receives long-range pressure that can hurt reconstruction before
   structure is cleanly extracted.

3. The anti-collapse objective is not bank-level.

   Batch-size-one training needs rank/variance constraints over a queue or
   memory bank, not only the current batch.

4. InfoNCE positives are sparse and local.

   Same-text cross-speaker positives exist, but with `batch_size=1` they rely on
   queue order. This is workable, but the matching objective needs stronger
   bank bookkeeping and probably teacher-defined hard negatives.

5. The raw V16.3 latent may not expose content through a shallow global adapter.

   This is not yet proven. It requires a frozen-latent probe before concluding
   that from-scratch many-speaker AE training is necessary.

## Current Recommendation

Do not continue by simply increasing V17.3 teacher loss weight.

The next rational experiment is a two-stage probe:

### Stage A: Frozen-AE teacher adapter probe

Freeze the V16.3 AE completely.

Train only a better adapter from latent sequence to teacher embedding:

```text
z sequence -> small Conformer/attention pooling adapter -> teacher content embedding
```

Use:

- pointwise cosine;
- teacher relation matrix loss;
- teacher-guided InfoNCE;
- queue/bank-level variance/rank guard;
- no AE reconstruction gradients.

Question answered:

```text
Does the existing V16.3 latent contain teacher-readable content if the adapter is allowed to be slightly stronger?
```

If no, then direct V16.3 fine-tuning is unlikely to solve the problem cleanly.

If yes, then the current failure is mostly an interface/training-order issue.

### Stage B: Controlled AE unfreeze

Only after Stage A succeeds:

- initialize the AE teacher adapter from Stage A;
- unfreeze encoder lightly;
- keep reconstruction as the primary guardrail;
- use PCGrad or a similar gradient-conflict mechanism;
- use bank-level rank guard;
- monitor reconstruction, adapter retrieval, latent rank, and speaker leakage.

Success criterion should be explicit:

```text
adapter retrieval top1 should move clearly toward teacher, not just above chance
adapter diff_text_same_speaker cos should fall below same_text_diff_speaker cos
SNR should remain close to the pointwise smoke, ideally >= 19 dB on the same eval path
raw latent rank should not collapse
```

## Concrete Next Changes

Recommended implementation tasks:

1. Add `v17_3_train_frozen_latent_teacher_adapter.py`.

   It should load the frozen AE, compute `z`, and train only an adapter against
   the frozen Conformer teacher.

2. Replace mean pooling with a sequence adapter.

   Minimal version:

```text
Conv1d projection -> 2-4 Conformer/Transformer blocks -> attention pooling
```

3. Add bank-level anti-collapse.

   The rank/variance guard should operate on a memory bank of adapter outputs,
   not only the current batch.

4. Add teacher-defined hard negatives.

   Instead of treating all different text equally, use teacher similarity to
   select nearby-but-wrong negatives. This should target local linear content
   geometry more directly.

5. Keep the current direct-AE distillation code, but treat it as Stage B.

   It should not be the first place we test whether content is extractable.

## Bottom Line

V17.3 did not prove that the research direction is wrong. It proved that the
naive direct distillation route is underdesigned.

The useful result is that the external Conformer teacher gives a clean,
measurable target structure. The problem has shifted from "can we define the
structure?" to "can the AE latent expose that structure without reconstruction
gradient conflict?"

The next decisive experiment is frozen-latent adapter training. It will tell us
whether V16.3 already contains accessible content structure or whether v17.x
needs a more fundamental many-speaker/from-scratch latent training recipe.
