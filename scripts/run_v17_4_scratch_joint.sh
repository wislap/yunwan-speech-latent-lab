#!/bin/bash
# V17.4 From-scratch joint training.
# No V16.3 init, no probe init. Encoder+Decoder+V17.4 all from scratch.

cd /root/autodl-tmp/project
pkill -f "autoencoder.train" 2>/dev/null
sleep 1
mkdir -p outputs/models/wav_vae_v17_4_scratch_joint_remote_main
rm -f outputs/models/wav_vae_v17_4_scratch_joint_remote_main/latest.pt
nohup /root/miniconda3/bin/python -m autoencoder.train \
  experiment=v17_4_scratch_joint \
  runtime=remote \
  > outputs/models/wav_vae_v17_4_scratch_joint_remote_main/train.log 2>&1 &
echo "PID=$!"
