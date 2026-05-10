#!/bin/bash
# V17.4 Phase 1: Joint AE + Flow Subspace + Analogy training.
#
# Initializes from V16.3 checkpoint. Warmup 2000 steps (encoder frozen,
# only projection + flow + analogy train). Then unfreeze encoder, joint train.

cd /root/autodl-tmp/project
pkill -f "autoencoder.train" 2>/dev/null
sleep 1
mkdir -p outputs/models/wav_vae_v17_4_phase1_joint_remote_main
nohup /root/miniconda3/bin/python -m autoencoder.train \
  experiment=v17_4_phase1_joint \
  runtime=remote \
  > outputs/models/wav_vae_v17_4_phase1_joint_remote_main/train.log 2>&1 &
echo "PID=$!"
