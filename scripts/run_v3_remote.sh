#!/bin/bash
cd /root/autodl-tmp/project
pkill -f v17_4_flow_subspace 2>/dev/null
sleep 1
mkdir -p outputs/models/v17_4_flow_probe_v5_spk_centered
nohup /root/miniconda3/bin/python -m autoencoder.scripts.v17_4_flow_subspace_probe \
  --ae-checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
  --data-path CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_train_phalign_neighbors.jsonl \
  --eval-data-path CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl \
  --output-dir outputs/models/v17_4_flow_probe_v5_spk_centered \
  --proj-dim 64 --proj-hidden-dim 256 --proj-depth 3 \
  --flow-hidden-dim 256 --flow-depth 4 --cond-emb-dim 128 \
  --batch-size 32 \
  --total-steps 4000 --lr 5e-4 --warmup-steps 200 \
  --segment-samples 220500 --sr 22050 \
  --log-interval 50 --eval-interval 500 \
  > outputs/models/v17_4_flow_probe_v5_spk_centered/train.log 2>&1 &
echo "PID=$!"
