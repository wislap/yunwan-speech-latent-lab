#!/bin/bash
# V17.4 Phase 1: Joint AE + Flow + Analogy training.
#
# Server: ssh -p 14025 root@connect.westc.seetacloud.com
# cd /root/autodl-tmp/project
# bash scripts/run_v17_4_joint_phase1.sh

set -e
cd /root/autodl-tmp/project

PYTHON=/root/miniconda3/bin/python
DATA_ROOT=CosyVoice_main/outputs/tts/cosyvoice3
AE_CKPT=outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt
PROBE_CKPT=outputs/models/v17_4_flow_probe_v5_spk_centered/best.pt
TRAIN_DATA=$DATA_ROOT/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_train_phalign_neighbors.jsonl
VAL_DATA=$DATA_ROOT/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl
OUTPUT=outputs/models/v17_4_joint_phase1

mkdir -p $OUTPUT

echo "=== V17.4 Phase 1: Joint AE + Flow + Analogy ==="
echo "AE: $AE_CKPT"
echo "Probe: $PROBE_CKPT"
echo "Output: $OUTPUT"
echo ""

$PYTHON -m autoencoder.scripts.v17_4_joint_train \
  --ae-checkpoint "$AE_CKPT" \
  --probe-checkpoint "$PROBE_CKPT" \
  --data-path "$TRAIN_DATA" \
  --eval-data-path "$VAL_DATA" \
  --output-dir "$OUTPUT" \
  --latent-dim 384 \
  --strides 2 4 4 8 \
  --encoder-channels 128 256 512 1024 1024 \
  --dilations 1 3 9 \
  --proj-dim 64 \
  --proj-hidden-dim 256 \
  --proj-depth 3 \
  --flow-hidden-dim 256 \
  --flow-depth 4 \
  --cond-emb-dim 128 \
  --lambda-flow 0.05 \
  --lambda-analogy 0.1 \
  --analogy-min-cos 0.3 \
  --l1-weight 10.0 \
  --stft-weight 0.5 \
  --mel-weight 0.1 \
  --total-steps 10000 \
  --encoder-freeze-steps 1000 \
  --encoder-lr 1e-5 \
  --decoder-lr 3e-5 \
  --struct-lr 3e-4 \
  --warmup-steps 200 \
  --batch-groups 8 \
  --grad-clip 1.0 \
  --segment-samples 220500 \
  --sr 22050 \
  --log-interval 20 \
  --eval-interval 500 \
  2>&1 | tee "$OUTPUT/train.log"
