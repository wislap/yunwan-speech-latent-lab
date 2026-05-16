#!/bin/bash
# Run B1/B2/B2.5-S2/B3 on the V18 checkpoint for B6 comparison.
# This is an as-is V18 anatomy, NOT a controlled supervision-lambda
# comparison. Confounds: CosyVoice data vs LJSpeech, dual-encoder vs
# single-encoder architecture, different training length, different
# initialization. See docs_ae/v16_3_latent_anatomy/b6_v18_confounds.md.
set -e

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/project}
cd "$PROJECT_ROOT"

CKPT=outputs/models/wav_vae_v18_base_remote_main/best.pt
MANIFEST=CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl
OUT=docs_ae/v16_3_latent_anatomy/data/b6_v18
PY=/root/miniconda3/bin/python

mkdir -p "$OUT"

echo "=== B1 (spectrum + TwoNN) on V18 ==="
$PY -u -m autoencoder.scripts.anatomy.spectrum \
  --checkpoint "$CKPT" \
  --dataset-name v18_cosyvoice \
  --manifest "$MANIFEST" \
  --path-root CosyVoice_main --path-root . \
  --n-samples 112 --segment-samples 131072 \
  --twonn-subsample 8000 --twonn-bootstrap 5 \
  --max-lag 16 \
  --tag v18 \
  --out "$OUT" --verbose

echo "=== B2 (PCA vs semantic labels) on V18 ==="
$PY -u -m autoencoder.scripts.anatomy.pca_semantics \
  --checkpoint "$CKPT" \
  --dataset-name v18_cosyvoice \
  --manifest "$MANIFEST" \
  --path-root CosyVoice_main --path-root . \
  --n-samples 112 --segment-samples 131072 \
  --k-directions 32 --n-folds 5 \
  --tag v18 \
  --out "$OUT" --verbose

echo "=== B2.5-S2 (full-384 linear probe) on V18 ==="
$PY -u -m autoencoder.scripts.anatomy.b2_5_full_linear \
  --checkpoint "$CKPT" \
  --manifest "$MANIFEST" \
  --path-root CosyVoice_main --path-root . \
  --n-samples 112 --segment-samples 131072 \
  --ridge-alpha 10.0 --logistic-alpha 1.0 --logistic-iter 400 \
  --tag v18 \
  --out "$OUT" --verbose

echo "=== B3 (supervised subspaces + principal angles) on V18 ==="
$PY -u -m autoencoder.scripts.anatomy.subspace \
  --checkpoint "$CKPT" \
  --dataset-name v18_cosyvoice \
  --manifest "$MANIFEST" \
  --path-root CosyVoice_main --path-root . \
  --n-samples 112 --segment-samples 131072 \
  --phoneme-min-count 100 --variance-target 0.90 \
  --n-shuffles 1000 \
  --tag v18 \
  --out "$OUT" --verbose

echo "ALL_DONE"
