#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/project}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$PROJECT_ROOT/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_2spk_calib_qc_8_13p5_train_phalign_neighbors.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-$PROJECT_ROOT/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_align_v0_2spk_calib_train_phalign_neighbors.jsonl}"
PATH_ROOT="${PATH_ROOT:-$PROJECT_ROOT/CosyVoice_main}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/models/v17_teacher_probe_xlsr300m}"
TEACHER_BUNDLE="${TEACHER_BUNDLE:-WAV2VEC2_XLSR_300M}"

EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
TRAIN_ITEMS="${TRAIN_ITEMS:-0}"
EVAL_ITEMS="${EVAL_ITEMS:-64}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-10.0}"
LR="${LR:-2e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
DEPTH="${DEPTH:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-1.0}"
VARIANCE_WEIGHT="${VARIANCE_WEIGHT:-0.1}"
MIN_STD="${MIN_STD:-0.05}"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -u -m autoencoder.scripts.v17_teacher_probe \
  --checkpoint "$CHECKPOINT" \
  --train-manifest "$TRAIN_MANIFEST" \
  --eval-manifest "$EVAL_MANIFEST" \
  --path-root "$PATH_ROOT" \
  --out "$OUT_DIR" \
  --teacher-bundle "$TEACHER_BUNDLE" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --train-items "$TRAIN_ITEMS" \
  --eval-items "$EVAL_ITEMS" \
  --log-interval "$LOG_INTERVAL" \
  --eval-interval "$EVAL_INTERVAL" \
  --segment-seconds "$SEGMENT_SECONDS" \
  --lr "$LR" \
  --hidden-dim "$HIDDEN_DIM" \
  --depth "$DEPTH" \
  --batch-size "$BATCH_SIZE" \
  --pairwise-weight "$PAIRWISE_WEIGHT" \
  --variance-weight "$VARIANCE_WEIGHT" \
  --min-std "$MIN_STD"
