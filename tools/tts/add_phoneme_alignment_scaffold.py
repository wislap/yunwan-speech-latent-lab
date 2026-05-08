#!/usr/bin/env python3
"""Add canonical phoneme alignment scaffolds and crossed-group QC to manifests.

This is not a forced aligner. It makes the current synthetic corpus safer for
V17.1 by recording a canonical pinyin/tone token sequence, deterministic
heuristic token intervals, latent-frame spans, and group-level duration/rate QC.
Precise boundaries can later overwrite the `phoneme_intervals` field with
`phoneme_alignment_source=forced_alignment`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

try:
    from pypinyin import Style, lazy_pinyin
except ImportError:  # pragma: no cover - exercised in minimal local envs
    Style = None
    lazy_pinyin = None


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


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", "", str(text))
    return text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")


def canonical_pinyin_tokens(text: str, include_pauses: bool, row: dict | None = None) -> tuple[list[str], list[str], list[str]]:
    if lazy_pinyin is None:
        if row is not None and row.get("pinyin_tokens"):
            tokens = [str(t) for t in row.get("pinyin_tokens", [])]
            chars = [str(c) for c in row.get("pinyin_chars", [])]
            if len(chars) != len(tokens):
                chars = ["" for _ in tokens]
            return tokens, chars, ["phone" for _ in tokens]
        raise ImportError("pypinyin is required unless manifest already has pinyin_tokens")
    tokens: list[str] = []
    chars: list[str] = []
    kinds: list[str] = []
    for ch in normalize_text(text):
        if include_pauses and ch in PAUSE_WEIGHTS:
            tokens.append(f"<pau:{ch}>")
            chars.append(ch)
            kinds.append("pause")
            continue
            py = lazy_pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True, errors="ignore")
        if py:
            tokens.append(py[0])
            chars.append(ch)
            kinds.append("phone")
    return tokens, chars, kinds


def token_weights(tokens: list[str], kinds: list[str], speed: float) -> list[float]:
    speed = max(float(speed or 1.0), 0.25)
    weights = []
    for token, kind in zip(tokens, kinds):
        if kind == "pause":
            m = re.match(r"<pau:(.*)>", token)
            punct = m.group(1) if m else "，"
            weights.append(PAUSE_WEIGHTS.get(punct, 0.8) / speed)
        else:
            weights.append(1.0 / speed)
    return weights


def intervals_from_weights(weights: list[float], seconds: float) -> list[dict]:
    total = sum(max(w, 0.0) for w in weights)
    if total <= 0:
        total = float(len(weights) or 1)
        weights = [1.0 for _ in weights]
    cursor = 0.0
    intervals = []
    for idx, weight in enumerate(weights):
        start = cursor
        cursor += seconds * max(weight, 0.0) / total
        intervals.append({"index": idx, "start": start, "end": cursor, "duration": cursor - start})
    if intervals:
        intervals[-1]["end"] = seconds
        intervals[-1]["duration"] = max(0.0, seconds - intervals[-1]["start"])
    return intervals


def add_frame_spans(intervals: list[dict], sample_rate: int, latent_stride: int) -> list[dict]:
    out = []
    for item in intervals:
        start_sample = int(round(item["start"] * sample_rate))
        end_sample = int(round(item["end"] * sample_rate))
        start_frame = start_sample // latent_stride
        end_frame = max(start_frame + 1, math.ceil(end_sample / latent_stride))
        row = dict(item)
        row.update({
            "start_sample": start_sample,
            "end_sample": end_sample,
            "start_frame": start_frame,
            "end_frame": end_frame,
        })
        out.append(row)
    return out


def coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return float(pstdev(values) / m) if m > 1e-8 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--reject-jsonl", type=Path, default=None)
    parser.add_argument("--group-field", default="same_text_group")
    parser.add_argument("--speaker-field", default="speaker_id")
    parser.add_argument("--sample-rate-field", default="sample_rate")
    parser.add_argument("--duration-field", default="audio_seconds")
    parser.add_argument("--target-sample-rate", type=int, default=22050)
    parser.add_argument("--latent-stride", type=int, default=256)
    parser.add_argument("--include-pauses", action="store_true")
    parser.add_argument("--max-duration-cv", type=float, default=0.20)
    parser.add_argument("--max-rate-cv", type=float, default=0.20)
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--drop-bad-groups", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.in_jsonl)
    enriched = []
    for row in rows:
        text = str(row.get("text", ""))
        tokens, chars, kinds = canonical_pinyin_tokens(text, args.include_pauses, row)
        seconds = float(row.get(args.duration_field, 0.0) or 0.0)
        speed = float(row.get("speed", 1.0) or 1.0)
        weights = token_weights(tokens, kinds, speed)
        intervals = add_frame_spans(intervals_from_weights(weights, seconds), args.target_sample_rate, args.latent_stride)
        phone_count = sum(1 for kind in kinds if kind == "phone")
        phone_seconds = sum(item["duration"] for item, kind in zip(intervals, kinds) if kind == "phone")
        speaking_rate = phone_count / max(phone_seconds, 1e-6)
        row = dict(row)
        row.update({
            "canonical_text": normalize_text(text),
            "phoneme_tokens": tokens,
            "phoneme_chars": chars,
            "phoneme_kinds": kinds,
            "phoneme_count": phone_count,
            "pause_count": sum(1 for kind in kinds if kind == "pause"),
            "phoneme_intervals": intervals,
            "phoneme_alignment_source": "heuristic_duration_scaffold",
            "phoneme_alignment_ready": False,
            "phoneme_target_sample_rate": args.target_sample_rate,
            "phoneme_latent_stride": args.latent_stride,
            "phoneme_speaking_rate": speaking_rate,
        })
        enriched.append(row)

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in enriched:
        groups[str(row.get(args.group_field, row.get("text_id", row.get("id"))))].append(row)

    good_groups = set()
    for group_id, group in groups.items():
        speakers = {str(row.get(args.speaker_field, "")) for row in group}
        durations = [float(row.get(args.duration_field, 0.0) or 0.0) for row in group]
        rates = [float(row.get("phoneme_speaking_rate", 0.0) or 0.0) for row in group]
        token_sequences = {tuple(row.get("phoneme_tokens", [])) for row in group}
        duration_cv = coefficient_of_variation(durations)
        rate_cv = coefficient_of_variation(rates)
        ok = (
            len(speakers) >= args.min_speakers
            and len(token_sequences) == 1
            and duration_cv <= args.max_duration_cv
            and rate_cv <= args.max_rate_cv
        )
        if ok:
            good_groups.add(group_id)
        for row in group:
            row.update({
                "alignment_group_id": group_id,
                "alignment_group_speakers": sorted(speakers),
                "alignment_group_size": len(group),
                "alignment_group_duration_cv": duration_cv,
                "alignment_group_rate_cv": rate_cv,
                "alignment_group_same_tokens": len(token_sequences) == 1,
                "alignment_group_qc_pass": ok,
            })

    if args.drop_bad_groups:
        keep = [row for row in enriched if row["alignment_group_id"] in good_groups]
        reject = [row for row in enriched if row["alignment_group_id"] not in good_groups]
    else:
        keep = enriched
        reject = [row for row in enriched if row["alignment_group_id"] not in good_groups]

    write_jsonl(args.out_jsonl, keep)
    if args.reject_jsonl:
        write_jsonl(args.reject_jsonl, reject)

    summary = {
        "input": str(args.in_jsonl),
        "output": str(args.out_jsonl),
        "reject": str(args.reject_jsonl) if args.reject_jsonl else None,
        "items_in": len(rows),
        "items_out": len(keep),
        "groups": len(groups),
        "groups_qc_pass": len(good_groups),
        "groups_qc_reject": len(groups) - len(good_groups),
        "drop_bad_groups": args.drop_bad_groups,
        "alignment_source": "heuristic_duration_scaffold",
        "include_pauses": args.include_pauses,
        "target_sample_rate": args.target_sample_rate,
        "latent_stride": args.latent_stride,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
