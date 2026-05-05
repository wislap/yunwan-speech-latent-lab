#!/usr/bin/env python3
"""Filter a JSONL manifest by simple text-quality rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_BAD_SUBSTRINGS = (
    "同时同时",
    "先先",
    "再再",
)


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
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--group-field", default=None)
    parser.add_argument("--bad-substring", action="append", default=[])
    args = parser.parse_args()

    bad_substrings = tuple(args.bad_substring) or DEFAULT_BAD_SUBSTRINGS
    rows = read_jsonl(args.in_jsonl)

    def is_bad(row: dict) -> bool:
        text = str(row.get(args.text_field, "") or "")
        return any(pattern in text for pattern in bad_substrings)

    if args.group_field:
        bad_groups = {row.get(args.group_field) for row in rows if is_bad(row)}
        keep = [row for row in rows if row.get(args.group_field) not in bad_groups]
        reject = [row for row in rows if row.get(args.group_field) in bad_groups]
    else:
        keep = [row for row in rows if not is_bad(row)]
        reject = [row for row in rows if is_bad(row)]

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
        "bad_substrings": bad_substrings,
        "group_field": args.group_field,
    }
    if args.group_field:
        summary["groups_keep"] = len({row.get(args.group_field) for row in keep})
        summary["groups_reject"] = len({row.get(args.group_field) for row in reject})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
