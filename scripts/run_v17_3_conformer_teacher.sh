#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/project}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-$PROJECT_ROOT/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_2spk_calib_qc_8_13p5_train_phalign_neighbors.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-$PROJECT_ROOT/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_align_v0_2spk_calib_train_phalign_neighbors.jsonl}"
PATH_ROOT="${PATH_ROOT:-$PROJECT_ROOT/CosyVoice_main}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/models/v17_3_conformer_teacher_smoke}"

STEPS="${STEPS:-1000}"
EVAL_ITEMS="${EVAL_ITEMS:-64}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-8.0}"
GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
EMB_DIM="${EMB_DIM:-192}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-4}"
FFN_DIM="${FFN_DIM:-1024}"
LR="${LR:-2e-4}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
HARD_NEG_WEIGHT="${HARD_NEG_WEIGHT:-1.0}"
HARD_NEG_MARGIN="${HARD_NEG_MARGIN:-0.2}"
RANK_WEIGHT="${RANK_WEIGHT:-2.0}"
RANK_COV_WEIGHT="${RANK_COV_WEIGHT:-0.01}"
TEXT_WEIGHT="${TEXT_WEIGHT:-0.5}"
CTC_WEIGHT="${CTC_WEIGHT:-0.0}"
FRAME_WEIGHT="${FRAME_WEIGHT:-1.0}"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -u -m autoencoder.scripts.v17_3_train_conformer_teacher \
  --train-manifest "$TRAIN_MANIFEST" \
  --eval-manifest "$EVAL_MANIFEST" \
  --path-root "$PATH_ROOT" \
  --out "$OUT_DIR" \
  --steps "$STEPS" \
  --eval-items "$EVAL_ITEMS" \
  --segment-seconds "$SEGMENT_SECONDS" \
  --groups-per-batch "$GROUPS_PER_BATCH" \
  --hidden-dim "$HIDDEN_DIM" \
  --emb-dim "$EMB_DIM" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --ffn-dim "$FFN_DIM" \
  --lr "$LR" \
  --log-interval "$LOG_INTERVAL" \
  --eval-interval "$EVAL_INTERVAL" \
  --hard-neg-weight "$HARD_NEG_WEIGHT" \
  --hard-neg-margin "$HARD_NEG_MARGIN" \
  --rank-weight "$RANK_WEIGHT" \
  --rank-cov-weight "$RANK_COV_WEIGHT" \
  --text-weight "$TEXT_WEIGHT" \
  --ctc-weight "$CTC_WEIGHT" \
  --frame-weight "$FRAME_WEIGHT"
