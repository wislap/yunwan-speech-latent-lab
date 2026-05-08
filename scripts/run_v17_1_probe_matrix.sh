#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/project}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASE_EXPERIMENT="${BASE_EXPERIMENT:-v17_v16_3_long10x_qc_cross_factor_soft_w04}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-$PROJECT_ROOT/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_2spk_calib_qc_8_13p5_train_phalign_neighbors.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$PROJECT_ROOT/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_2spk_calib_qc_8_13p5_val_phalign_neighbors.jsonl}"
ALIGN_MANIFEST="${ALIGN_MANIFEST:-$PROJECT_ROOT/CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_align_v0_2spk_calib_train_phalign_neighbors.jsonl}"
PATH_ROOT="${PATH_ROOT:-$PROJECT_ROOT/CosyVoice_main}"

EPOCHS="${EPOCHS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
EVAL_BATCHES="${EVAL_BATCHES:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RUNS="${RUNS:-baseline_recon,cross_w01,cross_w04,cross_w08,cross_w04_detach}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"

cd "$PROJECT_ROOT"

IFS=',' read -r -a RUN_LIST <<< "$RUNS"

run_train() {
  local name="$1"
  local weight="$2"
  local enable="$3"
  local detach="$4"
  local out_dir="$PROJECT_ROOT/outputs/models/v17_1_probe_${name}"

  echo "==> training $name"
  "$PYTHON_BIN" -u -m autoencoder.train \
    "experiment=$BASE_EXPERIMENT" \
    runtime=remote \
    "experiment.train.output_dir=$out_dir" \
    "experiment.train.data_path=$TRAIN_MANIFEST" \
    "experiment.train.eval_data_path=$VAL_MANIFEST" \
    "experiment.train.epochs=$EPOCHS" \
    "experiment.train.log_interval=$LOG_INTERVAL" \
    "experiment.train.eval_batches=$EVAL_BATCHES" \
    "experiment.train.grad_accum_steps=$GRAD_ACCUM_STEPS" \
    "experiment.train.num_workers=$NUM_WORKERS" \
    "experiment.train.v17_constraints_enable=$enable" \
    "experiment.train.v17_cross_factor_weight=$weight" \
    "experiment.train.v17_cross_factor_detach_latent=$detach"
}

run_diagnostics() {
  local name="$1"
  local weight="$2"
  local checkpoint="$PROJECT_ROOT/outputs/models/v17_1_probe_${name}/latest.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "skip diagnostics for $name: missing $checkpoint"
    return 0
  fi
  if [[ "$name" == "baseline_recon" ]]; then
    echo "skip cross-factor diagnostics for baseline_recon: checkpoint has no V17 heads"
    return 0
  fi

  echo "==> diagnostics $name"
  "$PYTHON_BIN" -m autoencoder.scripts.v17_gradient_alignment \
    --constraint cross_factor \
    --checkpoint "$checkpoint" \
    --manifest "$ALIGN_MANIFEST" \
    --path-root "$PATH_ROOT" \
    --out "$PROJECT_ROOT/outputs/diagnostics/v17_1_probe_${name}_grad_align" \
    --cross-factor-weight "$weight" \
    --n-pairs 12

  "$PYTHON_BIN" -m autoencoder.scripts.v17_cross_factor_diagnostics \
    --checkpoint "$checkpoint" \
    --manifest "$ALIGN_MANIFEST" \
    --path-root "$PATH_ROOT" \
    --out "$PROJECT_ROOT/outputs/diagnostics/v17_1_probe_${name}_factor_align" \
    --n-eval 64
}

for run in "${RUN_LIST[@]}"; do
  case "$run" in
    baseline_recon)
      run_train "$run" "0.0" "false" "false"
      [[ "$RUN_DIAGNOSTICS" == "1" ]] && run_diagnostics "$run" "0.0"
      ;;
    cross_w01)
      run_train "$run" "0.1" "true" "false"
      [[ "$RUN_DIAGNOSTICS" == "1" ]] && run_diagnostics "$run" "0.1"
      ;;
    cross_w04)
      run_train "$run" "0.4" "true" "false"
      [[ "$RUN_DIAGNOSTICS" == "1" ]] && run_diagnostics "$run" "0.4"
      ;;
    cross_w08)
      run_train "$run" "0.8" "true" "false"
      [[ "$RUN_DIAGNOSTICS" == "1" ]] && run_diagnostics "$run" "0.8"
      ;;
    cross_w04_detach)
      run_train "$run" "0.4" "true" "true"
      [[ "$RUN_DIAGNOSTICS" == "1" ]] && run_diagnostics "$run" "0.4"
      ;;
    *)
      echo "unknown run: $run" >&2
      exit 2
      ;;
  esac
done

echo "done"
