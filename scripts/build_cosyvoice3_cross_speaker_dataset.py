#!/usr/bin/env python3
"""Build a crossed-speaker CosyVoice3 dataset from anchor manifests.

The important invariant is that new speaker rows are derived from the anchor
manifests' real same_text_group values. This avoids silently generating a
different shuffled text set and then pretending the result is four-speaker
paired data.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SYNTH_DROP_FIELDS = {
    "index",
    "wav_path",
    "sample_rate",
    "audio_seconds",
    "wall_seconds",
    "rtf",
    "canonical_text",
    "pinyin_tokens",
    "pinyin_chars",
    "pinyin_durations",
    "pinyin_duration_source",
    "phoneme_tokens",
    "phoneme_chars",
    "phoneme_kinds",
    "phoneme_count",
    "pause_count",
    "phoneme_intervals",
    "phoneme_alignment_source",
    "phoneme_alignment_ready",
    "phoneme_target_sample_rate",
    "phoneme_latent_stride",
    "phoneme_speaking_rate",
    "alignment_group_id",
    "alignment_group_speakers",
    "alignment_group_size",
    "alignment_group_duration_cv",
    "alignment_group_rate_cv",
    "alignment_group_same_tokens",
    "alignment_group_qc_pass",
    "phoneme_hard_negatives",
    "phoneme_easy_negatives",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            paths.extend(Path(p) for p in matched)
        else:
            paths.append(Path(pattern))
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    missing = [str(p) for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError("missing manifest(s): " + ", ".join(missing))
    return out


def load_unique_rows(paths: list[Path]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            row_id = str(row.get("id", ""))
            if not row_id:
                raise ValueError(f"row without id in {path}")
            if row_id in by_id:
                raise ValueError(f"duplicate id {row_id} while reading {path}")
            by_id[row_id] = row
    rows = list(by_id.values())
    rows.sort(key=lambda r: (str(r.get("same_text_group") or r.get("text_id") or r["id"]), str(r.get("speaker_id")), str(r["id"])))
    return rows


def group_rows(rows: list[dict[str, Any]], group_field: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = str(row.get(group_field) or row.get("text_id") or row["id"])
        groups[group].append(row)
    return groups


def choose_anchor_groups(
    rows: list[dict[str, Any]],
    group_field: str,
    required_speakers: set[str],
    min_anchor_speakers: int,
) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    groups = group_rows(rows, group_field)
    chosen = []
    rejected = Counter()
    for group_id, group in groups.items():
        speakers = {str(row.get("speaker_id")) for row in group}
        if required_speakers and speakers != required_speakers:
            rejected["speaker_set"] += 1
            continue
        if len(speakers) < min_anchor_speakers:
            rejected["min_anchor_speakers"] += 1
            continue
        text_values = {str(row.get("text", "")) for row in group}
        if len(text_values) != 1:
            rejected["mixed_text"] += 1
            continue
        base = sorted(group, key=lambda r: (str(r.get("speaker_id")), str(r.get("id"))))[0]
        chosen.append((group_id, base, group))
    chosen.sort(key=lambda item: (str(item[1].get("text_id", item[0])), item[0]))
    print(
        json.dumps(
            {
                "anchor_items": len(rows),
                "anchor_groups": len(groups),
                "chosen_groups": len(chosen),
                "rejected_groups": dict(rejected),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return chosen


def make_new_speaker_rows(
    groups: list[tuple[str, dict[str, Any], list[dict[str, Any]]]],
    speakers: list[dict[str, Any]],
    dataset_id: str,
    style_id: str,
) -> list[dict[str, Any]]:
    out = []
    for group_index, (_, base, _) in enumerate(groups):
        for speaker in speakers:
            speaker_id = str(speaker["speaker_id"])
            speed = float(speaker.get("speed", base.get("speed", 1.0)) or 1.0)
            speed_tag = str(speed).replace(".", "p")
            row = {k: v for k, v in base.items() if k not in SYNTH_DROP_FIELDS and not k.startswith("_")}
            row.update(
                {
                    "id": f"{dataset_id}_paired_{group_index:04d}_{speaker_id}_{style_id}",
                    "speaker_id": speaker_id,
                    "same_speaker_group": speaker_id,
                    "prompt_wav": str(speaker["prompt_wav"]),
                    "prompt_text": str(speaker["prompt_text"]),
                    "speed": speed,
                    "speaker_speed": speed,
                    "style_id": style_id,
                    "style": style_id,
                    "style_speed": 1.0,
                    "condition_id": f"{speaker_id}__{style_id}__paired_calib{speed_tag}",
                }
            )
            out.append(row)
    return out


def split_by_speaker(rows: list[dict[str, Any]], out_prefix: Path) -> dict[str, Path]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row["speaker_id"])].append(row)
    paths = {}
    for speaker_id, speaker_rows in sorted(by_speaker.items()):
        path = out_prefix.with_name(f"{out_prefix.name}_{speaker_id}.jsonl")
        write_jsonl(path, speaker_rows)
        paths[speaker_id] = path
    return paths


def synth_speakers(
    per_speaker_texts: dict[str, Path],
    out_root: Path,
    model_dir: str,
    python_bin: str,
    script_path: Path,
    dataset_id: str,
    fp16: bool,
    parallel: int,
) -> dict[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    pending = list(sorted(per_speaker_texts.items()))
    running: list[tuple[str, subprocess.Popen, Path]] = []
    outputs: dict[str, Path] = {}

    def launch(speaker_id: str, text_path: Path) -> tuple[str, subprocess.Popen, Path]:
        out_dir = out_root / f"{dataset_id}_addspk_paired_calib_{speaker_id}"
        log_path = out_root / "logs" / f"{dataset_id}_paired_{speaker_id}_{int(time.time())}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            python_bin,
            str(script_path),
            "--model-dir",
            model_dir,
            "--out-dir",
            str(out_dir),
            "--texts-jsonl",
            str(text_path),
        ]
        if fp16:
            cmd.append("--fp16")
        print(f"launch {speaker_id}: {' '.join(cmd)} > {log_path}", flush=True)
        log_handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
        proc._cosy_log_handle = log_handle  # type: ignore[attr-defined]
        outputs[speaker_id] = out_dir / "manifest.jsonl"
        return speaker_id, proc, log_path

    while pending or running:
        while pending and len(running) < max(1, parallel):
            running.append(launch(*pending.pop(0)))
        time.sleep(5)
        still_running = []
        for speaker_id, proc, log_path in running:
            code = proc.poll()
            if code is None:
                still_running.append((speaker_id, proc, log_path))
                continue
            proc._cosy_log_handle.close()  # type: ignore[attr-defined]
            if code != 0:
                raise RuntimeError(f"synthesis failed for {speaker_id}; see {log_path}")
            print(f"done {speaker_id}: {outputs[speaker_id]}", flush=True)
        running = still_running
    return outputs


def duration_group_filter(
    rows: list[dict[str, Any]],
    group_field: str,
    min_seconds: float,
    max_seconds: float,
    expected_speakers: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups = group_rows(rows, group_field)
    keep_groups = set()
    reject_reason = Counter()
    for group_id, group in groups.items():
        speakers = {str(row.get("speaker_id")) for row in group}
        if expected_speakers and speakers != expected_speakers:
            reject_reason["speaker_set"] += 1
            continue
        bad_duration = [
            row
            for row in group
            if not (min_seconds <= float(row.get("audio_seconds", 0.0) or 0.0) <= max_seconds)
        ]
        if bad_duration:
            reject_reason["duration"] += 1
            continue
        keep_groups.add(group_id)
    keep = [row for row in rows if str(row.get(group_field) or row.get("text_id") or row["id"]) in keep_groups]
    reject = [row for row in rows if str(row.get(group_field) or row.get("text_id") or row["id"]) not in keep_groups]
    summary = {
        "items_in": len(rows),
        "groups_in": len(groups),
        "items_keep": len(keep),
        "groups_keep": len(keep_groups),
        "items_reject": len(reject),
        "groups_reject": len(groups) - len(keep_groups),
        "reject_reason": dict(reject_reason),
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
        "expected_speakers": sorted(expected_speakers),
    }
    return keep, reject, summary


def combine_anchor_and_generated(
    anchor_rows: list[dict[str, Any]],
    generated_manifests: list[Path],
) -> list[dict[str, Any]]:
    rows = list(anchor_rows)
    for path in generated_manifests:
        rows.extend(read_jsonl(path))
    by_id = {}
    for row in rows:
        row_id = str(row["id"])
        if row_id in by_id:
            raise ValueError(f"duplicate id after combine: {row_id}")
        by_id[row_id] = row
    out = list(by_id.values())
    out.sort(key=lambda r: (str(r.get("same_text_group") or r.get("text_id") or r["id"]), str(r.get("speaker_id")), str(r["id"])))
    return out


def postprocess_split(
    split_name: str,
    in_jsonl: Path,
    out_prefix: Path,
    scripts_dir: Path,
    python_bin: str,
    scaffold_min_speakers: int,
) -> Path:
    pinyin = out_prefix.with_name(f"{out_prefix.name}_{split_name}_pinyin.jsonl")
    scaffold = out_prefix.with_name(f"{out_prefix.name}_{split_name}_phalign_scaffold.jsonl")
    reject = out_prefix.with_name(f"{out_prefix.name}_{split_name}_phalign_reject.jsonl")
    neighbors = out_prefix.with_name(f"{out_prefix.name}_{split_name}_phalign_neighbors.jsonl")
    run([python_bin, str(scripts_dir / "add_pinyin_duration_labels.py"), "--in-jsonl", str(in_jsonl), "--out-jsonl", str(pinyin)])
    run(
        [
            python_bin,
            str(scripts_dir / "add_phoneme_alignment_scaffold.py"),
            "--in-jsonl",
            str(pinyin),
            "--out-jsonl",
            str(scaffold),
            "--reject-jsonl",
            str(reject),
            "--min-speakers",
            str(scaffold_min_speakers),
            "--drop-bad-groups",
        ]
    )
    run([python_bin, str(scripts_dir / "add_phoneme_neighbor_index.py"), "--in-jsonl", str(scaffold), "--out-jsonl", str(neighbors)])
    return neighbors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True, help="Stable prefix for generated ids and outputs.")
    parser.add_argument("--anchor-manifest", action="append", required=True, help="Anchor manifest path/glob. Repeatable.")
    parser.add_argument("--extra-speakers-json", type=Path, required=True)
    parser.add_argument(
        "--extra-speaker-id",
        action="append",
        default=[],
        help="Speaker id to keep from --extra-speakers-json. Repeatable. If omitted, all JSON speakers are used.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tts/cosyvoice3"))
    parser.add_argument("--model-dir", default="pretrained_models/Fun-CosyVoice3-0.5B")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--synth-script", type=Path, default=Path("scripts/cosyvoice3_batch_synth.py"))
    parser.add_argument("--scripts-dir", type=Path, default=Path("scripts"))
    parser.add_argument("--group-field", default="same_text_group")
    parser.add_argument("--required-anchor-speakers", default="")
    parser.add_argument("--min-anchor-speakers", type=int, default=2)
    parser.add_argument("--style-id", default="neutral_10s")
    parser.add_argument("--min-seconds", type=float, default=8.0)
    parser.add_argument("--max-seconds", type=float, default=13.5)
    parser.add_argument("--parallel-synth", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Only write paired text manifests, then exit.")
    parser.add_argument("--skip-synth", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")
    args = parser.parse_args()

    anchor_paths = expand_paths(args.anchor_manifest)
    anchor_rows = load_unique_rows(anchor_paths)
    required_anchor_speakers = {s.strip() for s in args.required_anchor_speakers.split(",") if s.strip()}
    anchor_groups = choose_anchor_groups(anchor_rows, args.group_field, required_anchor_speakers, args.min_anchor_speakers)
    if not anchor_groups:
        raise SystemExit("no usable anchor groups")

    extra_speakers = read_json(args.extra_speakers_json)
    if not isinstance(extra_speakers, list) or not extra_speakers:
        raise ValueError("--extra-speakers-json must contain a non-empty JSON list")
    if args.extra_speaker_id:
        keep_ids = set(args.extra_speaker_id)
        extra_speakers = [speaker for speaker in extra_speakers if str(speaker.get("speaker_id")) in keep_ids]
        missing_ids = sorted(keep_ids - {str(speaker.get("speaker_id")) for speaker in extra_speakers})
        if missing_ids:
            raise ValueError("extra speaker id(s) not found: " + ", ".join(missing_ids))
    new_rows = make_new_speaker_rows(anchor_groups, extra_speakers, args.dataset_id, args.style_id)
    text_manifest = args.out_dir / f"{args.dataset_id}_texts_addspk_paired_calib.jsonl"
    write_jsonl(text_manifest, new_rows)
    per_speaker = split_by_speaker(new_rows, text_manifest.with_suffix(""))
    print(
        json.dumps(
            {
                "text_manifest": str(text_manifest),
                "new_rows": len(new_rows),
                "new_speakers": sorted(per_speaker),
                "per_speaker_texts": {k: str(v) for k, v in per_speaker.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.prepare_only:
        return

    if args.skip_synth:
        generated_manifests = [
            args.out_dir / f"{args.dataset_id}_addspk_paired_calib_{speaker_id}" / "manifest.jsonl"
            for speaker_id in sorted(per_speaker)
        ]
    else:
        generated = synth_speakers(
            per_speaker,
            args.out_dir,
            args.model_dir,
            args.python_bin,
            args.synth_script,
            args.dataset_id,
            args.fp16,
            args.parallel_synth,
        )
        generated_manifests = [generated[speaker_id] for speaker_id in sorted(generated)]

    missing_generated = [str(p) for p in generated_manifests if not p.exists()]
    if missing_generated:
        raise FileNotFoundError("generated manifest(s) missing: " + ", ".join(missing_generated))

    selected_anchor_ids = {row["id"] for _, _, group in anchor_groups for row in group}
    selected_anchor_rows = [row for row in anchor_rows if row["id"] in selected_anchor_ids]
    combined = combine_anchor_and_generated(selected_anchor_rows, generated_manifests)
    raw_manifest = args.out_dir / f"{args.dataset_id}_4spk_raw.jsonl"
    write_jsonl(raw_manifest, combined)

    expected_speakers = {str(row.get("speaker_id")) for row in selected_anchor_rows} | {str(s["speaker_id"]) for s in extra_speakers}
    keep, reject, qc_summary = duration_group_filter(combined, args.group_field, args.min_seconds, args.max_seconds, expected_speakers)
    qc_manifest = args.out_dir / f"{args.dataset_id}_4spk_qc_{str(args.min_seconds).replace('.', 'p')}_{str(args.max_seconds).replace('.', 'p')}.jsonl"
    reject_manifest = args.out_dir / f"{args.dataset_id}_4spk_qc_reject.jsonl"
    write_jsonl(qc_manifest, keep)
    write_jsonl(reject_manifest, reject)
    print(json.dumps({"raw_manifest": str(raw_manifest), "qc_manifest": str(qc_manifest), **qc_summary}, ensure_ascii=False, indent=2), flush=True)

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in keep:
        by_split[str(row.get("split", "train"))].append(row)
    split_outputs = {}
    for split_name, split_rows in sorted(by_split.items()):
        split_path = qc_manifest.with_name(f"{qc_manifest.stem}_{split_name}.jsonl")
        write_jsonl(split_path, split_rows)
        split_outputs[split_name] = str(split_path)
    print(json.dumps({"splits": split_outputs}, ensure_ascii=False, indent=2), flush=True)

    final_outputs = {}
    if not args.skip_postprocess:
        out_prefix = qc_manifest.with_suffix("")
        for split_name, split_path in split_outputs.items():
            final_outputs[split_name] = str(
                postprocess_split(
                    split_name,
                    Path(split_path),
                    out_prefix,
                    args.scripts_dir,
                    args.python_bin,
                    scaffold_min_speakers=len(expected_speakers),
                )
            )
        print(json.dumps({"final_outputs": final_outputs}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
