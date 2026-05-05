#!/usr/bin/env python3
"""Concatenate paired manifest audio into longer utterances."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_name(value: object) -> str:
    text = str(value)
    keep = [ch if ch.isalnum() or ch in "._-" else "_" for ch in text]
    return "".join(keep).strip("_") or "missing"


def resolve_path(path: str, roots: list[Path]) -> Path:
    p = Path(path)
    if p.exists():
        return p
    for root in roots:
        candidate = root / p
        if candidate.exists():
            return candidate
    return p


def concat_rows(
    rows: list[dict],
    out_path: Path,
    sample_rate: int,
    silence_samples: int,
    roots: list[Path],
) -> tuple[float, list[str]]:
    chunks = []
    paths = []
    silence = np.zeros(silence_samples, dtype=np.float32)
    for i, row in enumerate(rows):
        wav_path = resolve_path(str(row["wav_path"]), roots)
        data, file_sr = sf.read(str(wav_path), dtype="float32")
        if file_sr != sample_rate:
            raise ValueError(f"sample rate mismatch: {wav_path} has {file_sr}, expected {sample_rate}")
        if data.ndim > 1:
            data = data.mean(axis=-1)
        if i > 0 and silence_samples > 0:
            chunks.append(silence)
        chunks.append(data.astype(np.float32, copy=False))
        paths.append(str(row["wav_path"]))

    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sample_rate)
    return float(len(audio) / sample_rate), paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-seconds", type=float, default=45.0)
    parser.add_argument("--silence-seconds", type=float, default=0.25)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--text-id-field", default="text_id")
    parser.add_argument("--speaker-field", default="speaker_id")
    parser.add_argument("--style-field", default="style_id")
    parser.add_argument("--condition-field", default="condition_id")
    parser.add_argument("--path-prefix", default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.in_jsonl)
    if not rows:
        raise ValueError(f"empty manifest: {args.in_jsonl}")

    roots = [Path.cwd(), args.in_jsonl.parent]
    if args.path_prefix:
        roots.insert(0, Path(args.path_prefix))

    by_split_text: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_split_text[(str(row.get(args.split_field, "train")), str(row[args.text_id_field]))].append(row)

    text_groups: dict[str, list[dict]] = {}
    for (split, text_id), group in by_split_text.items():
        speakers = {row[args.speaker_field] for row in group}
        styles = {row.get(args.style_field, "") for row in group}
        if len(speakers) < 2:
            continue
        # Current use case has one style; keeping this strict avoids accidental condition mixing.
        if len(styles) != 1:
            continue
        text_groups[f"{split}\t{text_id}"] = sorted(group, key=lambda r: str(r[args.speaker_field]))

    chunks: list[list[str]] = []
    silence = float(args.silence_seconds)
    target = float(args.target_seconds)

    keys_by_split: dict[str, list[str]] = defaultdict(list)
    for key in text_groups:
        split, _ = key.split("\t", 1)
        keys_by_split[split].append(key)

    for split in sorted(keys_by_split):
        ordered_keys = sorted(
            keys_by_split[split],
            key=lambda key: min(int(row.get("index", 0)) for row in text_groups[key]),
        )
        current: list[str] = []
        current_seconds = 0.0
        for key in ordered_keys:
            group = text_groups[key]
            group_seconds = max(float(row.get("audio_seconds", 0.0) or 0.0) for row in group)
            projected = current_seconds + (silence if current else 0.0) + group_seconds
            if current and projected > target:
                chunks.append(current)
                current = []
                current_seconds = 0.0
            if current:
                current_seconds += silence
            current.append(key)
            current_seconds += group_seconds
        if current:
            chunks.append(current)

    out_rows = []
    by_split_counts: dict[str, int] = defaultdict(int)
    wav_dir = args.out_dir / "wav"
    manifest_index = 0

    for chunk_idx, keys in enumerate(chunks):
        split = keys[0].split("\t", 1)[0]
        rows_by_speaker: dict[str, list[dict]] = defaultdict(list)
        for key in keys:
            for row in text_groups[key]:
                rows_by_speaker[str(row[args.speaker_field])].append(row)

        text_ids = [key.split("\t", 1)[1] for key in keys]
        text = " ".join(text_groups[key][0].get(args.text_field, "") for key in keys)
        domains = sorted({str(row.get("domain", "")) for key in keys for row in text_groups[key] if row.get("domain")})
        style_id = str(text_groups[keys[0]][0].get(args.style_field, "neutral"))

        for speaker_id, speaker_rows in sorted(rows_by_speaker.items()):
            out_name = f"{chunk_idx:04d}_{safe_name(split)}_{safe_name(speaker_id)}_{safe_name(style_id)}.wav"
            out_path = wav_dir / out_name
            audio_seconds, source_wav_paths = concat_rows(
                speaker_rows,
                out_path,
                args.sample_rate,
                int(round(args.silence_seconds * args.sample_rate)),
                roots,
            )
            first = speaker_rows[0]
            rel_wav = out_path.as_posix()
            out_rows.append({
                "id": f"concat_{chunk_idx:04d}_{speaker_id}_{style_id}",
                "index": manifest_index,
                "text": text,
                "wav_path": rel_wav,
                "sample_rate": args.sample_rate,
                "audio_seconds": audio_seconds,
                "language": first.get("language", "zh"),
                "text_set": f"{first.get('text_set', 'unknown')}_concat",
                "domain": "+".join(domains),
                "text_id": f"concat_{chunk_idx:04d}",
                "same_text_group": f"concat_{chunk_idx:04d}",
                "source_text_ids": text_ids,
                "source_ids": [row.get("id") for row in speaker_rows],
                "source_wav_paths": source_wav_paths,
                "speaker_id": speaker_id,
                "same_speaker_group": first.get("same_speaker_group", speaker_id),
                "style_id": style_id,
                "style": first.get("style", style_id),
                "condition_id": first.get(args.condition_field, f"{speaker_id}__{style_id}"),
                "split": split,
                "concat_target_seconds": target,
                "concat_silence_seconds": args.silence_seconds,
            })
            manifest_index += 1
            by_split_counts[split] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "manifest.jsonl", out_rows)
    for split in sorted({row["split"] for row in out_rows}):
        write_jsonl(args.out_dir / f"manifest_{safe_name(split)}.jsonl", [row for row in out_rows if row["split"] == split])

    seconds = sum(float(row["audio_seconds"]) for row in out_rows)
    summary = {
        "input": str(args.in_jsonl),
        "out_dir": str(args.out_dir),
        "items": len(out_rows),
        "chunks": len(chunks),
        "source_text_groups": len(text_groups),
        "source_items": len(rows),
        "target_seconds": target,
        "silence_seconds": args.silence_seconds,
        "total_audio_seconds": seconds,
        "total_audio_minutes": seconds / 60.0,
        "by_split": dict(sorted(by_split_counts.items())),
        "by_speaker": dict(sorted((speaker, sum(1 for row in out_rows if row["speaker_id"] == speaker)) for speaker in {row["speaker_id"] for row in out_rows})),
        "manifest": str(args.out_dir / "manifest.jsonl"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
