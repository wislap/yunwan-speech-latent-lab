# CosyVoice3 Clean TTS Pipeline

## Remote Setup

Server:

```bash
ssh -p 10891 root@connect.westd.seetacloud.com
```

Remote paths:

- Code: `/root/autodl-tmp/project/CosyVoice_main`
- Env: `/root/autodl-tmp/cosyvoice_env`
- Model: `/root/autodl-tmp/project/CosyVoice_main/pretrained_models/Fun-CosyVoice3-0.5B`
- Script: `/root/autodl-tmp/project/CosyVoice_main/scripts/cosyvoice3_batch_synth.py`

Current verified runtime:

- GPU: NVIDIA GeForce RTX 5090
- Torch: `2.10.0+cu128`
- Model size: about 9.1G
- Env size: about 9.0G
- Remaining `/root/autodl-tmp` space after setup: about 20G

## Run Command

Smoke or batch synthesis:

```bash
cd /root/autodl-tmp/project/CosyVoice_main
/root/autodl-tmp/cosyvoice_env/bin/python scripts/cosyvoice3_batch_synth.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
  --out-dir outputs/cosyvoice3_batch \
  --texts-jsonl path/to/texts.jsonl \
  --prompt-wav asset/zero_shot_prompt.wav \
  --prompt-text 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。' \
  --fp16
```

Input JSONL format:

```jsonl
{"id": "utt_000001", "text": "我们需要一批稳定、干净、可控的语音样本。"}
{"id": "utt_000002", "text": "You are a helpful assistant.<|endofprompt|>The quick brown fox jumps over the lazy dog."}
```

The script writes:

- `wav/*.wav`
- `manifest.jsonl`
- `summary.json`

## Verified Smoke

Local copied outputs:

- `outputs/tts/cosyvoice3/smoke_default3/wav/`
- `outputs/tts/cosyvoice3/smoke_default3/manifest.jsonl`
- `outputs/tts/cosyvoice3/smoke_default3/summary.json`

Smoke result:

- 3 items
- 18.88 seconds generated audio
- model load time: 16.58 seconds
- synthesis wall time: 10.46 seconds
- RTF excluding model load: 0.554
- RTF including model load: 1.432
- sample rate: 24000 Hz

## Caveat

`onnxruntime` currently tries `CUDAExecutionProvider` for the speech tokenizer and logs a missing `libcublasLt.so.11` warning, then continues. The pipeline is functional, but part of token extraction is likely falling back to CPU. This is acceptable for the first controlled-data pass; optimize only if large-scale generation becomes the bottleneck.

## Seed Chinese Controlled Corpus V0

Date: 2026-05-04

V17 role:

- This is a seed data source for the V17 external modality encoder-adapter and flow constraints.
- It is not itself a V17 model module.

Purpose:

- Create a tiny but structurally correct CosyVoice3-distilled corpus.
- Preserve dataset labels in the synthesis manifest.
- Smoke-test the path from controlled text metadata to generated wav files.

Local code:

- `tools/tts/cosyvoice3_batch_synth.py`
- `tools/tts/make_cosyvoice3_seed_texts.py`

Remote run:

```bash
cd /root/autodl-tmp/project/CosyVoice_main
/root/autodl-tmp/cosyvoice_env/bin/python scripts/make_cosyvoice3_seed_texts.py \
  --out outputs/tts/cosyvoice3/seed_zh_v0_texts.jsonl

/root/autodl-tmp/cosyvoice_env/bin/python scripts/cosyvoice3_batch_synth.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
  --out-dir outputs/tts/cosyvoice3/seed_zh_v0 \
  --texts-jsonl outputs/tts/cosyvoice3/seed_zh_v0_texts.jsonl \
  --prompt-wav asset/zero_shot_prompt.wav \
  --prompt-text 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。' \
  --fp16
```

Artifacts:

```text
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/seed_zh_v0/wav/
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/seed_zh_v0/manifest.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/seed_zh_v0/manifest_train.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/seed_zh_v0/manifest_val.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/seed_zh_v0/manifest_train_pinyin.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/seed_zh_v0/manifest_val_pinyin.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/seed_zh_v0/summary.json
```

Local copied metadata:

```text
outputs/tts/cosyvoice3/seed_zh_v0/manifest.jsonl
outputs/tts/cosyvoice3/seed_zh_v0/manifest_train.jsonl
outputs/tts/cosyvoice3/seed_zh_v0/manifest_val.jsonl
outputs/tts/cosyvoice3/seed_zh_v0/manifest_train_pinyin.jsonl
outputs/tts/cosyvoice3/seed_zh_v0/manifest_val_pinyin.jsonl
outputs/tts/cosyvoice3/seed_zh_v0/summary.json
```

Summary:

```text
items: 80
wav files: 80
sample rate: 24000 Hz
total audio: 428.96 s = 7.15 min
train / val: 72 / 8
domains: daily=16, wiki=16, numbers=16, questions=16, hard=16
styles: neutral=70, neutral_speed_0.92=5, neutral_speed_1=5
RTF excluding load: 0.5245
RTF including load: 0.5588
GPU: NVIDIA GeForce RTX 5090
```

Caveats:

- The first V0 generator produced only `0.92` and `1.0` speed variants due to the deterministic selection pattern. The local generator has been patched so future runs rotate through `0.92`, `1.0`, and `1.08`.
- CosyVoice3 warns that many synthesis texts are shorter than the prompt text. This is acceptable for this seed pass, but the next prompt set should use shorter prompt text or longer utterances.
- `pypinyin` was installed in the project Python env and used to create pinyin/tone metadata.
- Current `pinyin_durations` are punctuation/speed heuristic labels, not teacher-native durations. Replace them with frontend/teacher/forced-alignment durations when available.

## Crossed Chinese Corpus V0

Date: 2026-05-04

Why this exists:

- A single-speaker corpus does not test the V17 goal.
- V17 needs the same text under multiple external conditions so the latent can be pressured to separate text, speaker, and style/speed factors.
- This crossed corpus is the first useful data shape for the external modality encoder-adapter and flow constraints.

Local code:

- `tools/tts/make_cosyvoice3_crossed_texts.py`
- `tools/tts/split_manifest_by_field.py`
- `tools/tts/add_pinyin_duration_labels.py`
- `tools/tts/cosyvoice3_batch_synth.py`

Design:

```text
24 text prompts x 2 prompt voices x 2 speed/style conditions = 96 utterances
```

Current prompt voices:

```text
spk_zero_shot:
  wav: asset/zero_shot_prompt.wav
  text: You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。

spk_cross_lingual:
  wav: asset/cross_lingual_prompt.wav
  text: You are a helpful assistant.<|endofprompt|>在那之后完全收购那家公司，因此保持管理层的一致性，利益与即将加入家族的资产保持一致，这就是我们有时不买下全部的原因。
```

The `spk_cross_lingual` prompt text was recovered with FunASR. Using the zero-shot prompt text for this wav produced bad smoke samples of only `1.4s` and `0.4s`; after correction the same crossed smoke generated `6.36s` and `5.06s`.

Remote run:

```bash
cd /root/autodl-tmp/project/CosyVoice_main
/root/miniconda3/bin/python scripts/make_cosyvoice3_crossed_texts.py \
  --out outputs/tts/cosyvoice3/crossed_zh_v0_texts.jsonl \
  --max-texts 24

/root/autodl-tmp/cosyvoice_env/bin/python scripts/cosyvoice3_batch_synth.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
  --out-dir outputs/tts/cosyvoice3/crossed_zh_v0 \
  --texts-jsonl outputs/tts/cosyvoice3/crossed_zh_v0_texts.jsonl \
  --fp16

/root/miniconda3/bin/python scripts/split_manifest_by_field.py \
  --in-jsonl outputs/tts/cosyvoice3/crossed_zh_v0/manifest.jsonl \
  --out-dir outputs/tts/cosyvoice3/crossed_zh_v0 \
  --field split \
  --prefix manifest

/root/miniconda3/bin/python scripts/add_pinyin_duration_labels.py \
  --in-jsonl outputs/tts/cosyvoice3/crossed_zh_v0/manifest_train.jsonl \
  --out-jsonl outputs/tts/cosyvoice3/crossed_zh_v0/manifest_train_pinyin.jsonl

/root/miniconda3/bin/python scripts/add_pinyin_duration_labels.py \
  --in-jsonl outputs/tts/cosyvoice3/crossed_zh_v0/manifest_val.jsonl \
  --out-jsonl outputs/tts/cosyvoice3/crossed_zh_v0/manifest_val_pinyin.jsonl
```

Artifacts:

```text
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_v0/wav/
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_v0/manifest.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_v0/manifest_train.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_v0/manifest_val.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_v0/manifest_train_pinyin.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_v0/manifest_val_pinyin.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_v0/summary.json
```

Local copied metadata:

```text
outputs/tts/cosyvoice3/crossed_zh_v0_texts.jsonl
outputs/tts/cosyvoice3/crossed_zh_v0/manifest.jsonl
outputs/tts/cosyvoice3/crossed_zh_v0/manifest_train.jsonl
outputs/tts/cosyvoice3/crossed_zh_v0/manifest_val.jsonl
outputs/tts/cosyvoice3/crossed_zh_v0/manifest_train_pinyin.jsonl
outputs/tts/cosyvoice3/crossed_zh_v0/manifest_val_pinyin.jsonl
outputs/tts/cosyvoice3/crossed_zh_v0/summary.json
```

Summary:

```text
items: 96
sample rate: 24000 Hz
total audio: 543.84 s = 9.06 min
train / val: 84 / 12
train text ids / val text ids: 21 / 3
speaker balance: spk_zero_shot=48, spk_cross_lingual=48
style balance: neutral_normal=48, neutral_fast=48
spk_zero_shot duration: 246.44 s, min=3.00 s, max=7.20 s
spk_cross_lingual duration: 297.40 s, min=4.00 s, max=8.84 s
RTF excluding load: 0.6843
RTF including load: 0.7173
GPU: NVIDIA GeForce RTX 5090
```

V17 crossed smoke:

```bash
cd /root/autodl-tmp/project
/root/miniconda3/bin/python -m autoencoder.train \
  experiment=v17_constraints_crossed_smoke \
  runtime=remote \
  experiment.train.output_dir=outputs/models/v17_constraints_crossed_smoke_test \
  experiment.train.log_interval=10 \
  experiment.train.eval_batches=1
```

Result:

```text
Train samples: 84, Eval samples: 12
V17 constraints params: 1,497,057
step 80 | G=3.2181 | v17=0.3837 ext=7.4694 flow=1.0266
Epoch 0/1 | avg_G=3.2213
Eval: SNR=-0.21dB | STFT=3.1216 | KL=0.000000
```

Notes:

- The train/val split is by held-out `text_id`; all speaker/style variants of a text stay in the same split.
- Current corpus is only a 2-speaker smoke because only two prompt wavs are present locally, and one required ASR recovery. The manifest generator accepts a speaker catalog and is meant to scale to 4/8/16 prompt voices.
- This is still not enough to prove disentanglement. It only proves the data and V17 training path can express the right factorization pressure.

## Crossed Long10 Chinese Corpus V1

Date: 2026-05-05

Goal:

- Use complete Chinese sentences.
- Target each utterance near 10 seconds.
- Keep the crossed text/speaker structure, but avoid speed/style variables until length is calibrated.

Design:

```text
40 complete long texts x 2 prompt voices x 1 neutral_10s style = 80 raw utterances
```

Local code:

- `tools/tts/make_cosyvoice3_crossed_texts.py --text-set long10`
- `tools/tts/filter_manifest_by_duration.py`
- `tools/tts/split_manifest_by_field.py`
- `tools/tts/add_pinyin_duration_labels.py`

Remote run:

```bash
cd /root/autodl-tmp/project/CosyVoice_main
/root/miniconda3/bin/python scripts/make_cosyvoice3_crossed_texts.py \
  --text-set long10 \
  --version crosszh_long10_v1 \
  --max-texts 40 \
  --val-every 8 \
  --out outputs/tts/cosyvoice3/crossed_zh_long10_v1_texts.jsonl

/root/autodl-tmp/cosyvoice_env/bin/python scripts/cosyvoice3_batch_synth.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
  --out-dir outputs/tts/cosyvoice3/crossed_zh_long10_v1 \
  --texts-jsonl outputs/tts/cosyvoice3/crossed_zh_long10_v1_texts.jsonl \
  --fp16
```

Raw summary:

```text
items: 80
texts: 40
speakers: 2
style: neutral_10s
total audio: 787.80 s = 13.13 min
duration min / max / mean / median: 7.00 / 13.24 / 9.85 / 9.80 s
train / val: 70 / 10
RTF excluding load: 0.5458
RTF including load: 0.5697
```

Duration QC:

```bash
/root/miniconda3/bin/python scripts/filter_manifest_by_duration.py \
  --in-jsonl outputs/tts/cosyvoice3/crossed_zh_long10_v1/manifest.jsonl \
  --out-jsonl outputs/tts/cosyvoice3/crossed_zh_long10_v1/manifest_qc_8_13.jsonl \
  --reject-jsonl outputs/tts/cosyvoice3/crossed_zh_long10_v1/manifest_reject_8_13.jsonl \
  --min-seconds 8 \
  --max-seconds 13 \
  --group-field text_id
```

QC summary:

```text
kept: 70 utterances, 35 complete text groups
rejected: 10 utterances, 5 complete text groups
total kept audio: 692.88 s = 11.55 min
duration min / max / mean / median: 8.36 / 12.32 / 9.90 / 9.80 s
train / val: 64 / 6
speaker balance: spk_zero_shot=35, spk_cross_lingual=35
```

QC artifacts:

```text
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_long10_v1/manifest_qc_8_13.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_long10_v1/manifest_qc_8_13_train.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_long10_v1/manifest_qc_8_13_val.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_long10_v1/manifest_qc_8_13_train_pinyin.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_long10_v1/manifest_qc_8_13_val_pinyin.jsonl
```

Local copied metadata:

```text
outputs/tts/cosyvoice3/crossed_zh_long10_v1_texts.jsonl
outputs/tts/cosyvoice3/crossed_zh_long10_v1/
```

V17 long10 QC smoke:

```bash
cd /root/autodl-tmp/project
/root/miniconda3/bin/python -m autoencoder.train \
  experiment=v17_constraints_long10_qc_smoke \
  runtime=remote \
  experiment.train.output_dir=outputs/models/v17_constraints_long10_qc_smoke_test \
  experiment.train.log_interval=10 \
  experiment.train.eval_batches=1
```

Result:

```text
Train samples: 64, Eval samples: 6
step 60 | G=3.7784 | v17=0.3656 ext=7.1104 flow=1.0110
Epoch 0/1 | avg_G=3.1264
Eval: SNR=-0.79dB | STFT=3.1674 | KL=0.000000
```

Notes:

- The raw set is retained, but use the QC manifests for V17 training if the requirement is near-10-second complete utterances.
- QC filtering drops full `text_id` groups, so speaker balance and crossed pairing remain intact.
- The rejected groups are useful feedback for the next text generator pass: some texts are speaker-dependent in duration and need per-speaker speed calibration or replacement.

### Long10x v2 Larger Complete-Sentence Set

Purpose:

- Scale the long complete-sentence crossed set while preserving paired `text_id` groups.
- Keep only one neutral style for now; expand text/time before adding more variables.
- Add a text-quality QC pass before duration QC.

Design:

```text
156 complete long texts x 2 prompt voices x 1 neutral_10s style = 312 raw utterances
```

Generation:

```bash
cd /root/autodl-tmp/project/CosyVoice_main
/root/miniconda3/bin/python scripts/make_cosyvoice3_crossed_texts.py \
  --text-set long10x \
  --version crosszh_long10x_v2 \
  --max-texts 160 \
  --min-chars 40 \
  --max-chars 62 \
  --val-every 10 \
  --out outputs/tts/cosyvoice3/crossed_zh_long10x_v2_texts.jsonl

/root/autodl-tmp/cosyvoice_env/bin/python scripts/cosyvoice3_batch_synth.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
  --out-dir outputs/tts/cosyvoice3/crossed_zh_long10x_v2 \
  --texts-jsonl outputs/tts/cosyvoice3/crossed_zh_long10x_v2_texts.jsonl \
  --fp16
```

Raw summary:

```text
items: 312
texts: 156
speakers: 2
total audio: 3769.56 s = 62.83 min
duration min / max / mean / median: 8.12 / 18.08 / 12.08 / 12.16 s
train / val: 280 / 32
RTF excluding load: 0.4929
RTF including load: 0.4972
```

Text and duration QC:

```text
text QC rejects: 2 utterances, 1 complete text group
duration QC filter: keep full text_id groups only when every speaker variant is 8-13 s
kept: 170 utterances, 85 complete text groups
rejected by duration: 140 utterances, 70 complete text groups
total kept audio: 1891.28 s = 31.52 min
duration min / max / mean / median: 8.12 / 13.00 / 11.13 / 11.32 s
train / val: 148 / 22
speaker balance: spk_zero_shot=85, spk_cross_lingual=85
domain counts: daily=48, questions=34, wiki=34, numbers=28, hard=26
```

QC artifacts:

```text
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_long10x_v2/manifest_qc_8_13.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_long10x_v2/manifest_qc_8_13_train_pinyin.jsonl
/root/autodl-tmp/project/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_long10x_v2/manifest_qc_8_13_val_pinyin.jsonl
```

V17 long10x QC smoke:

```text
config: autoencoder/conf/experiment/v17_constraints_long10x_qc_smoke.yaml
Train samples: 148, Eval samples: 22
step 140 | G=2.4016 | v17=0.3391 ext=6.5813 flow=1.0041
Epoch 0/1 | avg_G=3.0312
Eval: SNR=0.17dB | STFT=2.9727 | KL=0.000000
```

Notes:

- Long10x v2 gives about 2.7x the QC audio of long10 v1 while preserving complete paired text groups.
- The main bottleneck is not synthesis speed; it is duration retention for the slower prompt voice.
- Next expansion should calibrate per-speaker speed or use shorter text candidates for slow voices before adding 16 speakers.

## Long10x Alignment Audit And Long10align

Current `crossed_zh_long10x_v2` is useful for clean long-utterance reconstruction and same-text cross-speaker pairing, but it is not yet the best shape for V17.1 phoneme geometry constraints.

Observed after adding phoneme alignment scaffolds and neighbor indices:

```text
QC train: 148 utterances, 74 complete text groups
QC val: 22 utterances, 11 complete text groups
train duration/rate CV: p50=0.0565, p90=0.1091, max=0.1563
val duration/rate CV: p50=0.0650, max=0.1020
train hard-negative normalized edit distance mean: 0.7734
train easy-negative normalized edit distance mean: 0.9953
val hard-negative normalized edit distance mean: 0.9608
val easy-negative normalized edit distance mean: 0.9794
```

Interpretation:

- Same-text cross-speaker total duration consistency is good enough to start forced alignment.
- The current pinyin spans are only heuristic scaffolds, not true phoneme boundaries.
- The text bank is broad rather than locally clustered. Hard negatives are still far away, so contrastive phoneme constraints receive a weak local geometry signal.
- Before scaling to many speakers, add a clustered text set where each semantic template has anchor, near-substitution, and mid-substitution variants.

Added tools:

```text
tools/tts/add_phoneme_alignment_scaffold.py
tools/tts/add_phoneme_neighbor_index.py
tools/tts/export_forced_alignment_corpus.py
```

Added text set:

```text
tools/tts/make_cosyvoice3_crossed_texts.py --text-set long10align
```

Local generation smoke:

```text
10 phoneme clusters x 5 variants x 2 prompt voices x 1 neutral style = 100 utterances
text groups: 50
characters per text: min=50, max=58, mean=53.68
variant bands: anchor=10, near=20, mid=20
```

Generation command:

```bash
cd /root/autodl-tmp/project/CosyVoice_main
/root/miniconda3/bin/python scripts/make_cosyvoice3_crossed_texts.py \
  --text-set long10align \
  --version crosszh_long10align_v0 \
  --max-texts 0 \
  --min-chars 40 \
  --max-chars 70 \
  --val-every 10 \
  --out outputs/tts/cosyvoice3/crossed_zh_long10align_v0_texts.jsonl

/root/autodl-tmp/cosyvoice_env/bin/python scripts/cosyvoice3_batch_synth.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
  --out-dir outputs/tts/cosyvoice3/crossed_zh_long10align_v0 \
  --texts-jsonl outputs/tts/cosyvoice3/crossed_zh_long10align_v0_texts.jsonl \
  --fp16
```

After synthesis, keep the same QC chain:

```bash
/root/miniconda3/bin/python scripts/filter_manifest_by_text_quality.py ...
/root/miniconda3/bin/python scripts/filter_manifest_by_duration.py ...
/root/miniconda3/bin/python scripts/split_manifest_by_field.py ...
/root/miniconda3/bin/python scripts/add_pinyin_duration_labels.py ...
/root/miniconda3/bin/python scripts/add_phoneme_alignment_scaffold.py ...
/root/miniconda3/bin/python scripts/add_phoneme_neighbor_index.py ...
```

For V17.1, use `long10x` as broad clean coverage and `long10align` as the structured geometry set. The latter should carry more weight in phoneme contrastive/local-linearity diagnostics, while reconstruction should still see both distributions.

## Four-Voice Four-Hour Target

Decision:

```text
voices: 4
target usable audio: about 4 hours
target average utterance length: about 10 seconds
target usable utterances: about 1440
target usable complete text groups: about 360
```

Because earlier long10x QC kept about half of complete text groups, prepare a larger raw set:

```text
raw text groups: 550-650
raw utterances with 4 voices: 2200-2600
expected usable utterances after QC: about 1200-1600
expected usable audio: about 3.3-4.4 hours
```

Recommended mix:

```text
long10scale: broad clean coverage, about 500-550 text groups
long10align: local phoneme-geometry coverage, about 50 text groups
stress/hard cases: included inside long10scale domains
```

Current blocker:

```text
Only 2 prompt wavs are currently present on the server:
asset/zero_shot_prompt.wav
asset/cross_lingual_prompt.wav
```

Before synthesizing the 4-voice set, add two more prompt wavs and exact prompt transcripts. A template is available at:

```text
tools/tts/cosyvoice3_speakers_4voice_template.json
```

The current `spk_cross_lingual` prompt is slower than `spk_zero_shot`, so the template sets:

```text
spk_zero_shot speed: 1.0
spk_cross_lingual speed: 1.1
```

Keep this per-speaker speed calibration in the manifest, because duration QC is performed on complete text groups and a single slow speaker can reject the whole group.

After filling the two candidate voices, copy it to the server and generate the broad set:

```bash
cd /root/autodl-tmp/project/CosyVoice_main

/root/miniconda3/bin/python scripts/make_cosyvoice3_crossed_texts.py \
  --text-set long10scale \
  --speakers-json configs/cosyvoice3_speakers_4voice.json \
  --version crosszh_4spk_4h_scale_v0 \
  --max-texts 650 \
  --min-chars 40 \
  --max-chars 58 \
  --val-every 20 \
  --out outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_texts.jsonl

/root/autodl-tmp/cosyvoice_env/bin/python scripts/cosyvoice3_batch_synth.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
  --out-dir outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0 \
  --texts-jsonl outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_texts.jsonl \
  --fp16
```

Then synthesize the structured alignment set with the same speaker catalog:

```bash
/root/miniconda3/bin/python scripts/make_cosyvoice3_crossed_texts.py \
  --text-set long10align \
  --speakers-json configs/cosyvoice3_speakers_4voice.json \
  --version crosszh_4spk_4h_align_v0 \
  --max-texts 0 \
  --min-chars 40 \
  --max-chars 70 \
  --val-every 10 \
  --out outputs/tts/cosyvoice3/crossed_zh_4spk_4h_align_v0_texts.jsonl

/root/autodl-tmp/cosyvoice_env/bin/python scripts/cosyvoice3_batch_synth.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
  --out-dir outputs/tts/cosyvoice3/crossed_zh_4spk_4h_align_v0 \
  --texts-jsonl outputs/tts/cosyvoice3/crossed_zh_4spk_4h_align_v0_texts.jsonl \
  --fp16
```

Use the existing QC/alignment chain after synthesis. For training, keep the manifests separate at first so `long10align` can be upweighted in phoneme-locality losses without overrepresenting repetitive template text in reconstruction.

Current 2-voice bootstrap:

```text
speaker catalog: configs/cosyvoice3_speakers_2voice_calibrated.json
broad text manifest: outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_texts_2spk_calib.jsonl
broad output dir: outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_2spk_calib
align text manifest: outputs/tts/cosyvoice3/crossed_zh_4spk_4h_align_v0_texts_2spk_calib.jsonl
align output dir: outputs/tts/cosyvoice3/crossed_zh_4spk_4h_align_v0_2spk_calib
```

The calibrated 8-utterance smoke used `max_chars=58` and `spk_cross_lingual speed=1.1`; all eight samples landed in the 10.06-12.84 s range.

## AISHELL-3 Backup Prompt Voices

AISHELL-3 is the preferred public Mandarin source for expanding prompt voices because it is multi-speaker, TTS-oriented, professionally transcribed, and released under Apache-2.0 on OpenSLR/HuggingFace.

Current backup selection:

```text
source: AISHELL-3
selected prompt voices: 32
gender: male=12, female=20
accent: south=12, north=20
prompt duration: min=10.555 s, max=15.480 s, mean=12.524 s
```

Artifacts on server:

```text
/root/autodl-tmp/project_data/aishell3_prompt_candidates_32/
/root/autodl-tmp/project_data/aishell3_prompt_candidates_32/selected_prompts.json
/root/autodl-tmp/project_data/aishell3_prompt_candidates_32/cosyvoice3_speakers_aishell3_32backup.json
/root/autodl-tmp/project/CosyVoice_main/asset/aishell3_prompts/
/root/autodl-tmp/project/CosyVoice_main/configs/cosyvoice3_speakers_aishell3_32backup.json
/root/autodl-tmp/project/CosyVoice_main/configs/aishell3_selected_prompts_32backup.json
```

Immediate 4-voice trial catalog:

```text
/root/autodl-tmp/project/CosyVoice_main/configs/cosyvoice3_speakers_4voice_with_aishell3.json

speakers:
spk_zero_shot
spk_cross_lingual
aishell3_SSB0631
aishell3_SSB0426
```

Selection script:

```text
tools/tts/select_aishell3_prompt_speakers.py
```

The script downloads only the needed AISHELL-3 files from the HuggingFace mirror, filters by available wavs, concatenates same-speaker short clips into 10-15 s prompt wavs, and writes a CosyVoice-compatible speaker catalog.
