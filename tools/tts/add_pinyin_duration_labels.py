#!/usr/bin/env python3
"""Add pinyin+tone tokens and heuristic token durations to a JSONL manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pypinyin import Style, lazy_pinyin


PAUSE_WEIGHTS = {
    "，": 0.8,
    ",": 0.8,
    "、": 0.5,
    "；": 1.0,
    ";": 1.0,
    "：": 0.8,
    ":": 0.8,
    "。": 1.4,
    "！": 1.4,
    "!": 1.4,
    "？": 1.4,
    "?": 1.4,
}


def pinyin_with_chars(text: str) -> tuple[list[str], list[str]]:
    tokens = []
    chars = []
    for ch in text:
        py = lazy_pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True, errors="ignore")
        if py:
            tokens.append(py[0])
            chars.append(ch)
    return tokens, chars


def heuristic_durations(text: str, chars: list[str], speed: float) -> list[float]:
    speed = max(float(speed or 1.0), 0.25)
    weights = []
    char_index = 0
    punctuation_bonus = 0.0
    for ch in text:
        if char_index >= len(chars):
            break
        if ch == chars[char_index]:
            weights.append((1.0 + punctuation_bonus) / speed)
            punctuation_bonus = 0.0
            char_index += 1
        elif ch in PAUSE_WEIGHTS:
            punctuation_bonus += PAUSE_WEIGHTS[ch]
    while len(weights) < len(chars):
        weights.append(1.0 / speed)
    return weights[: len(chars)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    args = parser.parse_args()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.in_jsonl.open("r", encoding="utf-8") as src, args.out_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", ""))
            tokens, chars = pinyin_with_chars(text)
            durations = heuristic_durations(text, chars, float(row.get("speed", 1.0) or 1.0))
            row["pinyin_tokens"] = tokens
            row["pinyin_chars"] = chars
            row["pinyin_durations"] = durations
            row["pinyin_duration_source"] = "heuristic_punctuation_speed"
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"input": str(args.in_jsonl), "output": str(args.out_jsonl), "items": count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
