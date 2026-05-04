#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torchaudio
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
import cosyvoice.cli.frontend as cosy_frontend  # noqa: E402
import cosyvoice.utils.file_utils as cosy_file_utils  # noqa: E402


def soundfile_load_wav(wav, target_sr, min_sr=16000):
    data, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
    speech = torch.from_numpy(data).transpose(0, 1).mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate >= min_sr, f"wav sample rate {sample_rate} must be greater than {target_sr}"
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


cosy_file_utils.load_wav = soundfile_load_wav
cosy_frontend.load_wav = soundfile_load_wav

DEFAULT_PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
DEFAULT_TEXTS = [
    {"id": "zh_tongue_twister", "text": "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。"},
    {"id": "zh_clean_001", "text": "我们需要一批稳定、干净、可控的语音样本，用来验证新的语义分解自编码器。"},
    {"id": "en_clean_001", "text": "You are a helpful assistant.<|endofprompt|>The quick brown fox jumps over the lazy dog."},
]

RESERVED_ROW_FIELDS = {"id", "text", "prompt_wav", "prompt_text", "speed"}


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "text" not in row:
                raise ValueError(f"{path}:{line_no} missing text")
            if "id" not in row:
                row["id"] = f"utt_{len(rows):06d}"
            rows.append(row)
    return rows


def safe_name(value):
    keep = []
    for ch in value:
        if ch.isalnum() or ch in "._-":
            keep.append(ch)
        else:
            keep.append("_")
    name = "".join(keep).strip("_")
    return name or "utt"


def main():
    parser = argparse.ArgumentParser(description="Batch synthesize clean data with CosyVoice3 zero-shot inference.")
    parser.add_argument("--model-dir", default="pretrained_models/Fun-CosyVoice3-0.5B")
    parser.add_argument("--out-dir", default="outputs/cosyvoice3_batch")
    parser.add_argument("--texts-jsonl", default=None, help="JSONL with fields: id, text")
    parser.add_argument("--prompt-wav", default="asset/zero_shot_prompt.wav")
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--text-frontend", action="store_true", default=True)
    parser.add_argument("--no-text-frontend", dest="text_frontend", action="store_false")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = read_jsonl(args.texts_jsonl) if args.texts_jsonl else list(DEFAULT_TEXTS)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no text rows to synthesize")

    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    load_start = time.time()
    cosyvoice = AutoModel(model_dir=args.model_dir, fp16=args.fp16)
    load_seconds = time.time() - load_start

    manifest_path = out_dir / "manifest.jsonl"
    records = []
    total_audio_seconds = 0.0
    synth_wall_seconds = 0.0

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for index, row in enumerate(rows):
            utt_id = safe_name(str(row.get("id", f"utt_{index:06d}")))
            text = str(row["text"])
            prompt_wav = str(row.get("prompt_wav", args.prompt_wav))
            prompt_text = str(row.get("prompt_text", args.prompt_text))
            speed = float(row.get("speed", args.speed))
            item_start = time.time()
            chunks = list(
                cosyvoice.inference_zero_shot(
                    text,
                    prompt_text,
                    prompt_wav,
                    stream=False,
                    speed=speed,
                    text_frontend=args.text_frontend,
                )
            )
            if len(chunks) != 1:
                speech = torch.cat([chunk["tts_speech"] for chunk in chunks], dim=1)
            else:
                speech = chunks[0]["tts_speech"]
            item_wall = time.time() - item_start
            wav_path = wav_dir / f"{index:06d}_{utt_id}.wav"
            sf.write(str(wav_path), speech.squeeze(0).detach().cpu().numpy(), cosyvoice.sample_rate)
            audio_seconds = float(speech.shape[1]) / float(cosyvoice.sample_rate)
            rtf = item_wall / audio_seconds if audio_seconds > 0 else None
            record = {
                "id": utt_id,
                "index": index,
                "text": text,
                "wav_path": str(wav_path),
                "sample_rate": int(cosyvoice.sample_rate),
                "audio_seconds": audio_seconds,
                "wall_seconds": item_wall,
                "rtf": rtf,
                "prompt_wav": prompt_wav,
                "prompt_text": prompt_text,
                "speed": speed,
            }
            for key, value in row.items():
                if key not in RESERVED_ROW_FIELDS:
                    record[key] = value
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")
            mf.flush()
            records.append(record)
            total_audio_seconds += audio_seconds
            synth_wall_seconds += item_wall
            print(json.dumps(record, ensure_ascii=False), flush=True)

    summary = {
        "model_dir": args.model_dir,
        "out_dir": str(out_dir),
        "num_items": len(records),
        "sample_rate": int(cosyvoice.sample_rate),
        "load_seconds": load_seconds,
        "total_audio_seconds": total_audio_seconds,
        "synth_wall_seconds": synth_wall_seconds,
        "overall_rtf_excluding_load": synth_wall_seconds / total_audio_seconds if total_audio_seconds > 0 else None,
        "overall_rtf_including_load": (synth_wall_seconds + load_seconds) / total_audio_seconds if total_audio_seconds > 0 else None,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "fp16": bool(args.fp16),
        "seed": args.seed,
        "manifest": str(manifest_path),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
