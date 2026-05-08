#!/usr/bin/env python3
"""Add pinyin edit-distance hard/easy negative text-group indices to a manifest."""

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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def edit_distance(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def token_jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--group-field", default="same_text_group")
    parser.add_argument("--token-field", default="phoneme_tokens")
    parser.add_argument("--hard-k", type=int, default=8)
    parser.add_argument("--easy-k", type=int, default=8)
    parser.add_argument("--min-hard-distance", type=float, default=0.08)
    args = parser.parse_args()

    rows = read_jsonl(args.in_jsonl)
    group_tokens: dict[str, list[str]] = {}
    for row in rows:
        group = str(row.get(args.group_field) or row.get("text_id") or row.get("id"))
        tokens = [str(t) for t in row.get(args.token_field, row.get("pinyin_tokens", [])) if not str(t).startswith("<pau:")]
        group_tokens.setdefault(group, tokens)

    groups = sorted(group_tokens)
    neighbors: dict[str, dict[str, list[dict]]] = {}
    for group in groups:
        src = group_tokens[group]
        scored = []
        for other in groups:
            if other == group:
                continue
            tgt = group_tokens[other]
            dist = edit_distance(src, tgt)
            norm = dist / max(len(src), len(tgt), 1)
            jac = token_jaccard(src, tgt)
            scored.append({
                "text_id": other,
                "edit_distance": dist,
                "normalized_edit_distance": norm,
                "token_jaccard": jac,
            })
        hard = [
            item
            for item in sorted(scored, key=lambda x: (x["normalized_edit_distance"], -x["token_jaccard"]))
            if item["normalized_edit_distance"] >= args.min_hard_distance
        ][: args.hard_k]
        easy = sorted(scored, key=lambda x: (-x["normalized_edit_distance"], x["token_jaccard"]))[: args.easy_k]
        neighbors[group] = {"hard": hard, "easy": easy}

    out = []
    for row in rows:
        row = dict(row)
        group = str(row.get(args.group_field) or row.get("text_id") or row.get("id"))
        row["phoneme_hard_negatives"] = neighbors[group]["hard"]
        row["phoneme_easy_negatives"] = neighbors[group]["easy"]
        out.append(row)
    write_jsonl(args.out_jsonl, out)

    hard_d = [n["normalized_edit_distance"] for group in neighbors.values() for n in group["hard"]]
    easy_d = [n["normalized_edit_distance"] for group in neighbors.values() for n in group["easy"]]
    summary = {
        "input": str(args.in_jsonl),
        "output": str(args.out_jsonl),
        "items": len(rows),
        "groups": len(groups),
        "hard_k": args.hard_k,
        "easy_k": args.easy_k,
        "hard_distance_mean": sum(hard_d) / len(hard_d) if hard_d else 0.0,
        "easy_distance_mean": sum(easy_d) / len(easy_d) if easy_d else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
