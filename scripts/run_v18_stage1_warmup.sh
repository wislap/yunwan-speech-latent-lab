#!/bin/bash
# V18 Stage 1: phoneme encoder warmup. Residual + decoder frozen.

cd /root/autodl-tmp/project
pkill -f "autoencoder.train" 2>/dev/null || true
sleep 1
OUT=outputs/models/wav_vae_v18_phoneme_warmup_remote_main
mkdir -p "$OUT"
rm -f "$OUT/latest.pt"
nohup /root/miniconda3/bin/python -m autoencoder.train \
  experiment=v18_phoneme_warmup \
  runtime=remote \
  > "$OUT/train.log" 2>&1 &
echo "PID=$!"
