#!/usr/bin/env python3
"""Filter a JSONL manifest by duration, optionally dropping full groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--reject-jsonl", type=Path, default=None)
    parser.add_argument("--min-seconds", type=float, default=8.0)
    parser.add_argument("--max-seconds", type=float, default=13.0)
    parser.add_argument("--duration-field", default="audio_seconds")
    parser.add_argument("--group-field", default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.in_jsonl)

    def in_range(row: dict) -> bool:
        value = float(row.get(args.duration_field, 0.0) or 0.0)
        return args.min_seconds <= value <= args.max_seconds

    if args.group_field:
        bad_groups = {row.get(args.group_field) for row in rows if not in_range(row)}
        keep = [row for row in rows if row.get(args.group_field) not in bad_groups]
        reject = [row for row in rows if row.get(args.group_field) in bad_groups]
    else:
        keep = [row for row in rows if in_range(row)]
        reject = [row for row in rows if not in_range(row)]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in keep:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.reject_jsonl:
        args.reject_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.reject_jsonl.open("w", encoding="utf-8") as f:
            for row in reject:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.in_jsonl),
        "output": str(args.out_jsonl),
        "reject": str(args.reject_jsonl) if args.reject_jsonl else None,
        "items_in": len(rows),
        "items_keep": len(keep),
        "items_reject": len(reject),
        "min_seconds": args.min_seconds,
        "max_seconds": args.max_seconds,
        "group_field": args.group_field,
    }
    if args.group_field:
        summary["groups_keep"] = len({row.get(args.group_field) for row in keep})
        summary["groups_reject"] = len({row.get(args.group_field) for row in reject})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
