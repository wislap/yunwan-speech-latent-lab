#!/bin/bash
# V18 Stage 2: joint training of phoneme + residual encoders with decoder.

cd /root/autodl-tmp/project
pkill -f "autoencoder.train" 2>/dev/null || true
sleep 1
OUT=outputs/models/wav_vae_v18_base_remote_main
mkdir -p "$OUT"
rm -f "$OUT/latest.pt"
nohup /root/miniconda3/bin/python -m autoencoder.train \
  experiment=v18_base \
  runtime=remote \
  > "$OUT/train.log" 2>&1 &
echo "PID=$!"
