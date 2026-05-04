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
