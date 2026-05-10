#!/bin/bash
# V17.4 Phase 0: Flow subspace probe on 4-speaker crossed dataset.
#
# Run on server: ssh -p 14025 root@connect.westc.seetacloud.com
# cd /root/autodl-tmp/project
# bash scripts/run_v17_4_flow_subspace_probe.sh

set -e

PYTHON=/root/miniconda3/bin/python
PROJECT=/root/autodl-tmp/project
DATA_ROOT=$PROJECT/CosyVoice_main/outputs/tts/cosyvoice3

TRAIN_MANIFEST=$DATA_ROOT/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_train_phalign_neighbors.jsonl
VAL_MANIFEST=$DATA_ROOT/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl
AE_CHECKPOINT=$PROJECT/outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt
OUTPUT_DIR=$PROJECT/outputs/models/v17_4_flow_subspace_probe

echo "=== V17.4 Phase 0: Flow Subspace Probe ==="
echo "AE checkpoint: $AE_CHECKPOINT"
echo "Train data: $TRAIN_MANIFEST"
echo "Val data: $VAL_MANIFEST"
echo "Output: $OUTPUT_DIR"
echo ""

cd $PROJECT

$PYTHON -m autoencoder.scripts.v17_4_flow_subspace_probe \
  --ae-checkpoint "$AE_CHECKPOINT" \
  --data-path "$TRAIN_MANIFEST" \
  --eval-data-path "$VAL_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --latent-dim 384 \
  --strides 2 4 4 8 \
  --encoder-channels 128 256 512 1024 1024 \
  --dilations 1 3 9 \
  --proj-dim 64 \
  --flow-hidden-dim 256 \
  --flow-depth 4 \
  --cond-emb-dim 64 \
  --total-steps 2000 \
  --lr 3e-4 \
  --warmup-steps 100 \
  --segment-samples 220500 \
  --sr 22050 \
  --log-interval 50 \
  --eval-interval 500 \
  2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "=== Done ==="
