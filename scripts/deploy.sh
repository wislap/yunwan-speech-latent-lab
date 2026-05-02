#!/bin/bash
# ============================================================
# Wav-VAE — 一键部署
#
# 用法:  bash scripts/deploy.sh 26713 connect.westd.seetacloud.com
#
# 目录规划:
#   持久盘 (autodl-fs, 网络存储, 少量大文件):
#     /root/autodl-fs/wav-vae/code/          代码
#     /root/autodl-fs/wav-vae/checkpoints/   模型存档
#
#   本地盘 (autodl-tmp, SSD, 大量小文件 + 训练 I/O):
#     /root/autodl-tmp/wav-vae/data/         数据集
#     /root/autodl-tmp/wav-vae/runs/         训练输出
# ============================================================
set -euo pipefail

PORT="${1:?用法: bash scripts/deploy.sh <port> <host>}"
HOST="${2:?用法: bash scripts/deploy.sh <port> <host>}"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p $PORT root@$HOST"

FS="/root/autodl-fs/wav-vae"
TMP="/root/autodl-tmp/wav-vae"
PY="/root/miniconda3/bin/python"

echo "══════════════════════════════════════════"
echo "  部署 → $HOST:$PORT"
echo "══════════════════════════════════════════"

# ── 1. 依赖 ─────────────────────────────────────────────────
echo "[1/6] Python 依赖..."
$SSH "$PY -m pip install -q torchaudio hydra-core omegaconf soundfile -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -1"
echo "  done"

# ── 2. 目录 ─────────────────────────────────────────────────
echo "[2/6] 目录结构..."
$SSH "mkdir -p $FS/{code,checkpoints} $TMP/{data,runs}"
echo "  done"

# ── 3. 代码 → 持久盘 ────────────────────────────────────────
echo "[3/6] 同步代码..."
rsync -az --delete \
  --exclude='.git' --exclude='outputs' --exclude='__pycache__' \
  --exclude='.kiro' --exclude='*.pyc' --exclude='docs' \
  -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p $PORT" \
  ./ "root@$HOST:$FS/code/"
echo "  done"

# ── 4. LJSpeech → 本地快盘 ──────────────────────────────────
echo "[4/6] LJSpeech 数据..."
$SSH "
if [ -d $TMP/data/LJSpeech-1.1/wavs ]; then
  n=\$(ls $TMP/data/LJSpeech-1.1/wavs/ | wc -l)
  echo \"  已存在 (\$n files)\"
else
  echo '  下载中 (~2.6GB)...'
  cd $TMP/data
  wget -q https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
  echo '  解压中...'
  tar xjf LJSpeech-1.1.tar.bz2
  rm LJSpeech-1.1.tar.bz2
  n=\$(ls LJSpeech-1.1/wavs/ | wc -l)
  echo \"  完成 (\$n files)\"
fi
"

# ── 5. Manifest + Chunks ────────────────────────────────────
echo "[5/6] Manifest + 30s chunks..."
$SSH "
export PYTHONPATH=$FS/code

if [ -f $TMP/data/ljspeech_chunks/manifest.json ]; then
  echo '  已存在, 跳过'
else
  $PY - <<'PYEOF'
import json, random, numpy as np, soundfile as sf
from pathlib import Path

TMP = '$TMP'
sr = 22050
stride = 512

# train/val split
wavs = sorted(Path(f'{TMP}/data/LJSpeech-1.1/wavs').glob('*.wav'))
items = [{'audio_path': str(p)} for p in wavs]
random.seed(42)
random.shuffle(items)
split = int(len(items) * 0.95)
train_items = items[:split]

with open(f'{TMP}/data/ljspeech_train.json', 'w') as f:
    json.dump(train_items, f, indent=2)
with open(f'{TMP}/data/ljspeech_val.json', 'w') as f:
    json.dump(items[split:], f, indent=2)
print(f'  Split: {split} train, {len(items)-split} val')

# concat 30s chunks
target = (int(30.0 * sr) // stride) * stride
silence = np.zeros(int(0.1 * sr), dtype=np.float32)
out_dir = Path(f'{TMP}/data/ljspeech_chunks')
out_dir.mkdir(parents=True, exist_ok=True)

idx = 0
buf = np.zeros(0, dtype=np.float32)
manifest = []
for item in train_items:
    try:
        d, _ = sf.read(item['audio_path'])
        a = d.astype(np.float32)
        if a.ndim > 1: a = a.mean(axis=-1)
    except: continue
    if len(buf) > 0: buf = np.concatenate([buf, silence])
    buf = np.concatenate([buf, a])
    while len(buf) >= target:
        c = buf[:target]
        cl = (len(c) // stride) * stride
        p = out_dir / f'chunk_{idx:04d}.wav'
        sf.write(str(p), c[:cl], sr)
        manifest.append({'audio_path': str(p)})
        idx += 1
        buf = buf[target:]
if len(buf) > stride * 10:
    cl = (len(buf) // stride) * stride
    p = out_dir / f'chunk_{idx:04d}.wav'
    sf.write(str(p), buf[:cl], sr)
    manifest.append({'audio_path': str(p)})

with open(out_dir / 'manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'  Chunks: {len(manifest)}')
PYEOF
fi
"

# ── 6. Sanity check ─────────────────────────────────────────
echo "[6/6] Sanity check..."
$SSH "
export PYTHONPATH=$FS/code
$PY -c \"
import torch
from autoencoder.models.autoencoder import WavVAE
m = WavVAE(latent_dim=256, strides=[2,4,4,4,4],
    encoder_channels=[128,256,512,512,1024,2048],
    use_hybrid_decoder=True, hybrid_vocos_hidden=1024,
    hybrid_vocos_layers=8, hybrid_n_fft=512, hybrid_hop=128).cuda()
x = torch.randn(1,1,176128).cuda()
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    o = m(x)
n = sum(p.numel() for p in m.parameters())
print(f'  {n/1e6:.1f}M params | stride={m.total_stride} | z={o[\\\"z\\\"].shape}')
print('  ✓ OK')
\"
"

echo ""
echo "══════════════════════════════════════════"
echo "  部署完成"
echo ""
echo "  启动训练:"
echo "    ssh -p $PORT root@$HOST"
echo "    cd $FS/code"
echo "    export PYTHONPATH=$FS/code"
echo "    nohup $PY -m autoencoder.train experiment=v16_hybrid runtime=remote > $TMP/runs/train.log 2>&1 &"
echo "══════════════════════════════════════════"
