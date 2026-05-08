#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/project}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
EXPERIMENT="${EXPERIMENT:-v17_2_v16_3_cross_factor_paired_rank_guard_w04}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-$PROJECT_ROOT/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_2spk_calib_qc_8_13p5_train_phalign_neighbors.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$PROJECT_ROOT/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_2spk_calib_qc_8_13p5_val_phalign_neighbors.jsonl}"
ALIGN_MANIFEST="${ALIGN_MANIFEST:-$PROJECT_ROOT/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_align_v0_2spk_calib_train_phalign_neighbors.jsonl}"
PATH_ROOT="${PATH_ROOT:-$PROJECT_ROOT/CosyVoice_main}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/models/v17_2_probe_paired_rank_guard_w04}"
DIAG_DIR="${DIAG_DIR:-$PROJECT_ROOT/outputs/diagnostics/v17_2_probe_paired_rank_guard_w04}"

EPOCHS="${EPOCHS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
EVAL_BATCHES="${EVAL_BATCHES:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -u -m autoencoder.train \
  "experiment=$EXPERIMENT" \
  runtime=remote \
  "experiment.train.output_dir=$OUT_DIR" \
  "experiment.train.data_path=$TRAIN_MANIFEST" \
  "experiment.train.eval_data_path=$VAL_MANIFEST" \
  "experiment.train.epochs=$EPOCHS" \
  "experiment.train.log_interval=$LOG_INTERVAL" \
  "experiment.train.eval_batches=$EVAL_BATCHES" \
  "experiment.train.grad_accum_steps=$GRAD_ACCUM_STEPS" \
  "experiment.train.num_workers=$NUM_WORKERS"

if [[ "$RUN_DIAGNOSTICS" == "1" ]]; then
  "$PYTHON_BIN" -m autoencoder.scripts.v17_gradient_alignment \
    --constraint cross_factor \
    --checkpoint "$OUT_DIR/latest.pt" \
    --manifest "$ALIGN_MANIFEST" \
    --path-root "$PATH_ROOT" \
    --out "${DIAG_DIR}_grad_align" \
    --cross-factor-weight 0.4 \
    --cross-factor-speaker-same-weight 0.35 \
    --cross-factor-rank-guard-min-std 0.03 \
    --cross-factor-rank-guard-max-top1 0.70 \
    --cross-factor-phoneme-rank-guard-weight 0.04 \
    --cross-factor-speaker-rank-guard-weight 0.06 \
    --cross-factor-latent-rank-guard-weight 0.02 \
    --n-pairs 12

  "$PYTHON_BIN" -m autoencoder.scripts.v17_cross_factor_diagnostics \
    --checkpoint "$OUT_DIR/latest.pt" \
    --manifest "$ALIGN_MANIFEST" \
    --path-root "$PATH_ROOT" \
    --out "${DIAG_DIR}_factor_align" \
    --n-eval 64
fi
