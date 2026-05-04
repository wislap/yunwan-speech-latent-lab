"""Evaluate ASR CER/WER for decoded FM-DiT audio."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import whisper
from scipy.signal import resample_poly


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def wav_path_for(item_dir: Path, target: str) -> Path:
    if target.startswith("cfg_"):
        cfg = target.removeprefix("cfg_")
        return item_dir / f"dit_cfg{cfg}.wav"
    return item_dir / f"{target}.wav"


def load_whisper_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr).astype(np.float32, copy=False)
    return audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-dir", type=Path, required=True)
    parser.add_argument("--targets", type=str, default="original,ae_recon,cfg_1")
    parser.add_argument("--model", type=str, default="base.en")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    report_path = args.decode_dir / "decode_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    out_json = args.out_json or (args.decode_dir / "asr_cer_wer.json")

    asr = whisper.load_model(args.model, device=args.device)
    rows: list[dict] = []

    for item in report["items"]:
        item_dir = args.decode_dir / f"{int(item['index']):03d}_{item['id']}"
        ref = normalize_text(item.get("text", ""))
        if not ref:
            continue
        for target in targets:
            wav_path = wav_path_for(item_dir, target)
            if not wav_path.exists():
                print(f"missing: {wav_path}", flush=True)
                continue
            audio = load_whisper_audio(wav_path)
            hyp_raw = asr.transcribe(audio, language=args.language, fp16=args.device.startswith("cuda"))["text"]
            hyp = normalize_text(hyp_raw)
            row = {
                "index": int(item["index"]),
                "id": item["id"],
                "target": target,
                "wer": float(jiwer.wer(ref, hyp)),
                "cer": float(jiwer.cer(ref, hyp)),
                "ref": ref,
                "hyp": hyp,
                "hyp_raw": hyp_raw.strip(),
                "wav_path": str(wav_path),
            }
            rows.append(row)
            print(
                f"{item['index']:03d} {item['id']} {target}: "
                f"WER={row['wer']:.3f} CER={row['cer']:.3f}",
                flush=True,
            )

    summary: dict[str, dict] = {}
    for target in targets:
        subset = [r for r in rows if r["target"] == target]
        if not subset:
            continue
        wers = np.array([r["wer"] for r in subset], dtype=np.float64)
        cers = np.array([r["cer"] for r in subset], dtype=np.float64)
        summary[target] = {
            "n": int(len(subset)),
            "wer_mean": float(wers.mean()),
            "wer_median": float(np.median(wers)),
            "wer_p90": float(np.percentile(wers, 90)),
            "cer_mean": float(cers.mean()),
            "cer_median": float(np.median(cers)),
            "cer_p90": float(np.percentile(cers, 90)),
            "exact": int(sum(1 for r in subset if r["wer"] == 0.0 and r["cer"] == 0.0)),
        }

    output = {
        "decode_dir": str(args.decode_dir),
        "model": args.model,
        "targets": targets,
        "summary": summary,
        "items": rows,
    }
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n=== ASR CER/WER Summary ===", flush=True)
    for target, s in summary.items():
        print(
            f"{target}: n={s['n']} "
            f"WER mean/median/p90={s['wer_mean']:.3f}/{s['wer_median']:.3f}/{s['wer_p90']:.3f} "
            f"CER mean/median/p90={s['cer_mean']:.3f}/{s['cer_median']:.3f}/{s['cer_p90']:.3f} "
            f"exact={s['exact']}/{s['n']}",
            flush=True,
        )
    print(f"report: {out_json}", flush=True)


if __name__ == "__main__":
    main()
