#!/usr/bin/env python3
"""Select AISHELL-3 speakers and build prompt wavs for CosyVoice cloning."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import wave
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


def read_spk_info(path: Path) -> dict[str, dict]:
    speakers = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            spk, age, gender, accent = parts[:4]
            speakers[spk] = {"speaker_id": spk, "age_group": age, "gender": gender, "accent": accent}
    return speakers


def extract_chars(label: str) -> str:
    tokens = label.strip().split()
    return "".join(tokens[0::2])


def read_content(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or "\t" not in line:
                continue
            wav_name, label = line.rstrip("\n").split("\t", 1)
            spk = wav_name[:7]
            text = extract_chars(label)
            rows[wav_name] = {"wav_name": wav_name, "speaker_id": spk, "text": text, "char_len": len(text)}
    return rows


def read_available_wavs(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    names = set()
    for item in data:
        if item.get("type") != "file":
            continue
        p = str(item.get("path", ""))
        if p.endswith(".wav"):
            names.add(Path(p).name)
    return names


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def wav_stats(path: Path) -> dict:
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        width = wf.getsampwidth()
        channels = wf.getnchannels()
        sr = wf.getframerate()
    if width != 2:
        return {"sample_rate": sr, "channels": channels, "rms": None, "peak": None}
    import array

    samples = array.array("h")
    samples.frombytes(frames)
    if not samples:
        return {"sample_rate": sr, "channels": channels, "rms": 0.0, "peak": 0.0}
    vals = [s / 32768.0 for s in samples]
    rms = math.sqrt(sum(v * v for v in vals) / len(vals))
    peak = max(abs(v) for v in vals)
    return {"sample_rate": sr, "channels": channels, "rms": rms, "peak": peak}


def download_wav(base_url: str, row: dict, out_dir: Path) -> Path:
    spk = row["speaker_id"]
    name = row["wav_name"]
    out = out_dir / spk / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1024:
        return out
    url = f"{base_url}/train/wav/{spk}/{name}"
    tmp = out.with_suffix(".tmp")
    try:
        urlretrieve(url, tmp)
    except (HTTPError, URLError):
        if tmp.exists():
            tmp.unlink()
        raise
    tmp.rename(out)
    return out


def concat_wavs(paths: list[Path], out_path: Path, silence_ms: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    filter_parts = []
    for idx, path in enumerate(paths):
        cmd.extend(["-i", str(path)])
        filter_parts.append(f"[{idx}:a]")
    silence = f"aevalsrc=0:d={silence_ms / 1000.0}:s=16000"
    cmd.extend(["-f", "lavfi", "-i", silence])
    if len(paths) == 1:
        cmd.extend(["-i", str(paths[0]), "-ar", "16000", "-ac", "1", str(out_path)])
        subprocess.run(cmd, check=True)
        return
    pieces = []
    for idx in range(len(paths)):
        pieces.append(f"[{idx}:a]")
        if idx != len(paths) - 1:
            pieces.append(f"[{len(paths)}:a]")
    filter_graph = "".join(pieces) + f"concat=n={len(pieces)}:v=0:a=1[out]"
    cmd.extend(["-filter_complex", filter_graph, "-map", "[out]", "-ar", "16000", "-ac", "1", str(out_path)])
    subprocess.run(cmd, check=True)


def select_speakers(speakers: dict[str, dict], by_spk: dict[str, list[dict]], num_speakers: int) -> list[str]:
    quotas = [
        ("male", "south", 4),
        ("male", "north", 8),
        ("female", "south", 8),
        ("female", "north", 12),
    ]
    selected: list[str] = []
    selected_set = set()
    for gender, accent, quota in quotas:
        candidates = [
            spk for spk, info in speakers.items()
            if info["gender"] == gender
            and info["accent"] == accent
            and info["age_group"] in {"B", "C"}
            and len(by_spk.get(spk, [])) >= 20
        ]
        candidates.sort(key=lambda s: (-len(by_spk[s]), s))
        for spk in candidates[:quota]:
            if spk not in selected_set:
                selected.append(spk)
                selected_set.add(spk)
    if len(selected) < num_speakers:
        rest = [
            spk for spk, info in speakers.items()
            if spk not in selected_set
            and info["age_group"] in {"B", "C"}
            and len(by_spk.get(spk, [])) >= 20
        ]
        rest.sort(key=lambda s: (-len(by_spk[s]), s))
        for spk in rest:
            selected.append(spk)
            selected_set.add(spk)
            if len(selected) >= num_speakers:
                break
    return selected[:num_speakers]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--asset-prefix", default="asset/aishell3_prompts")
    parser.add_argument("--base-url", default="https://hf-mirror.com/datasets/AISHELL/AISHELL-3/resolve/main")
    parser.add_argument("--num-speakers", type=int, default=32)
    parser.add_argument("--target-min-sec", type=float, default=10.0)
    parser.add_argument("--target-max-sec", type=float, default=15.0)
    parser.add_argument("--max-downloads-per-speaker", type=int, default=12)
    parser.add_argument("--silence-ms", type=int, default=220)
    args = parser.parse_args()

    speakers = read_spk_info(args.meta_dir / "spk-info.txt")
    content = read_content(args.meta_dir / "content.txt")
    available = read_available_wavs(args.meta_dir / "tree_train_wav_all.json")
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for row in content.values():
        if available is not None and row["wav_name"] not in available:
            continue
        if 8 <= row["char_len"] <= 22:
            by_spk[row["speaker_id"]].append(row)
    for rows in by_spk.values():
        rows.sort(key=lambda r: (abs(r["char_len"] - 14), r["wav_name"]))

    selected = select_speakers(speakers, by_spk, args.num_speakers)
    raw_dir = args.out_dir / "raw_wav"
    prompt_dir = args.out_dir / "prompts"
    records = []
    catalog = []
    for idx, spk in enumerate(selected, 1):
        chosen = []
        total = 0.0
        downloaded = 0
        for row in by_spk[spk]:
            if downloaded >= args.max_downloads_per_speaker:
                break
            try:
                path = download_wav(args.base_url, row, raw_dir)
            except (HTTPError, URLError):
                continue
            downloaded += 1
            dur = wav_duration(path)
            stats = wav_stats(path)
            if dur < 1.6 or dur > 6.5:
                continue
            if stats["rms"] is not None and (stats["rms"] < 0.005 or stats["peak"] > 0.999):
                continue
            chosen.append({**row, "path": path, "duration": dur, "stats": stats})
            total += dur
            if total >= args.target_min_sec:
                break
        if total < args.target_min_sec and len(chosen) < 2:
            continue
        prompt_name = f"aishell3_prompt_{idx:02d}_{spk}.wav"
        prompt_path = prompt_dir / prompt_name
        concat_wavs([c["path"] for c in chosen], prompt_path, args.silence_ms)
        prompt_duration = wav_duration(prompt_path)
        prompt_stats = wav_stats(prompt_path)
        prompt_text = "。".join(c["text"] for c in chosen) + "。"
        info = speakers[spk]
        record = {
            "index": idx,
            "speaker_id": f"aishell3_{spk}",
            "source_speaker_id": spk,
            "age_group": info["age_group"],
            "gender": info["gender"],
            "accent": info["accent"],
            "prompt_wav": str(prompt_path),
            "prompt_wav_for_cosyvoice": f"{args.asset_prefix}/{prompt_name}",
            "prompt_text": prompt_text,
            "prompt_duration": prompt_duration,
            "prompt_stats": prompt_stats,
            "source_clips": [
                {
                    "wav_name": c["wav_name"],
                    "text": c["text"],
                    "duration": c["duration"],
                    "rms": c["stats"]["rms"],
                    "peak": c["stats"]["peak"],
                    "source_path": str(c["path"]),
                }
                for c in chosen
            ],
            "license": "Apache-2.0",
            "source": "AISHELL-3",
        }
        records.append(record)
        catalog.append(
            {
                "speaker_id": record["speaker_id"],
                "prompt_wav": record["prompt_wav_for_cosyvoice"],
                "prompt_text": prompt_text,
                "speed": 1.0,
                "source": "AISHELL-3",
                "source_speaker_id": spk,
                "gender": info["gender"],
                "accent": info["accent"],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "selected_prompts.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "cosyvoice3_speakers_aishell3_32backup.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.out_dir / "selected_prompts.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "selected": len(records),
        "requested": args.num_speakers,
        "gender": Counter(r["gender"] for r in records),
        "accent": Counter(r["accent"] for r in records),
        "duration_min": min((r["prompt_duration"] for r in records), default=0.0),
        "duration_max": max((r["prompt_duration"] for r in records), default=0.0),
        "duration_mean": sum(r["prompt_duration"] for r in records) / len(records) if records else 0.0,
        "catalog": str(args.out_dir / "cosyvoice3_speakers_aishell3_32backup.json"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    asset_dir = Path("/root/autodl-tmp/project/CosyVoice_main") / args.asset_prefix
    if asset_dir.parent.exists():
        asset_dir.mkdir(parents=True, exist_ok=True)
        for prompt in prompt_dir.glob("*.wav"):
            shutil.copy2(prompt, asset_dir / prompt.name)


if __name__ == "__main__":
    main()
