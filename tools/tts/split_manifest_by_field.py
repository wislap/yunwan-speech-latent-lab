#!/usr/bin/env python3
"""Split a JSONL manifest into one JSONL file per field value."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def safe_name(value: object) -> str:
    text = str(value)
    keep = [ch if ch.isalnum() or ch in "._-" else "_" for ch in text]
    return "".join(keep).strip("_") or "missing"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--field", default="split")
    parser.add_argument("--prefix", default="manifest")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    counts: dict[str, int] = defaultdict(int)
    try:
        with args.in_jsonl.open("r", encoding="utf-8") as src:
            for line_no, line in enumerate(src, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                value = safe_name(row.get(args.field, "missing"))
                if value not in handles:
                    handles[value] = (args.out_dir / f"{args.prefix}_{value}.jsonl").open("w", encoding="utf-8")
                handles[value].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[value] += 1
    finally:
        for handle in handles.values():
            handle.close()

    print(
        json.dumps(
            {
                "input": str(args.in_jsonl),
                "out_dir": str(args.out_dir),
                "field": args.field,
                "counts": dict(sorted(counts.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
