#!/usr/bin/env python3
"""Export a manifest as a forced-alignment work directory.

The output is intentionally simple:
- wav/ contains copied or symlinked wav files.
- lab/ contains one text label per wav.
- metadata.jsonl keeps the original manifest rows and export paths.
- lexicon_pinyin.txt maps pinyin tokens to themselves for phone-level aligners.

This prepares the synthetic data for MFA/CTC alignment; it does not run an
aligner by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_name(value: object) -> str:
    text = str(value)
    keep = [ch if ch.isalnum() or ch in "._-" else "_" for ch in text]
    return "".join(keep).strip("_") or "utt"


def resolve_path(path: str, roots: list[Path]) -> Path:
    p = Path(path)
    if p.exists():
        return p
    for root in roots:
        candidate = root / p
        if candidate.exists():
            return candidate
    return p


def lab_from_row(row: dict, mode: str) -> str:
    if mode == "text":
        return str(row.get("canonical_text") or row.get("text", ""))
    tokens = row.get("phoneme_tokens") or row.get("pinyin_tokens") or []
    tokens = [str(t) for t in tokens if not str(t).startswith("<pau:")]
    return " ".join(tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=None)
    parser.add_argument("--wav-field", default="wav_path")
    parser.add_argument("--lab-mode", choices=["pinyin", "text"], default="pinyin")
    parser.add_argument("--copy", action="store_true", help="copy wavs instead of symlinking")
    args = parser.parse_args()

    rows = read_jsonl(args.in_jsonl)
    roots = [Path.cwd(), args.in_jsonl.parent]
    if args.path_root:
        roots.insert(0, args.path_root)

    wav_dir = args.out_dir / "wav"
    lab_dir = args.out_dir / "lab"
    wav_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    vocab = set()
    metadata = []
    for index, row in enumerate(rows):
        utt_id = safe_name(row.get("id", f"utt_{index:06d}"))
        src = resolve_path(str(row[args.wav_field]), roots)
        dst_wav = wav_dir / f"{utt_id}.wav"
        if dst_wav.exists() or dst_wav.is_symlink():
            dst_wav.unlink()
        if args.copy:
            shutil.copy2(src, dst_wav)
        else:
            os.symlink(src.resolve(), dst_wav)
        lab = lab_from_row(row, args.lab_mode)
        (lab_dir / f"{utt_id}.lab").write_text(lab + "\n", encoding="utf-8")
        for token in lab.split():
            vocab.add(token)
        out_row = dict(row)
        out_row.update({
            "alignment_export_id": utt_id,
            "alignment_wav": str(dst_wav),
            "alignment_lab": str(lab_dir / f"{utt_id}.lab"),
            "alignment_lab_mode": args.lab_mode,
        })
        metadata.append(out_row)

    with (args.out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "lexicon_pinyin.txt").open("w", encoding="utf-8") as f:
        for token in sorted(vocab):
            f.write(f"{token}\t{token}\n")

    summary = {
        "input": str(args.in_jsonl),
        "out_dir": str(args.out_dir),
        "items": len(rows),
        "lab_mode": args.lab_mode,
        "copy": args.copy,
        "vocab_size": len(vocab),
        "metadata": str(args.out_dir / "metadata.jsonl"),
        "lexicon": str(args.out_dir / "lexicon_pinyin.txt"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
