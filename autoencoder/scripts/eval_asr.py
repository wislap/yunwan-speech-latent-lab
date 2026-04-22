"""ASR evaluation: measure WER on original vs reconstructed audio.

Compares Whisper transcription of original audio against reconstructed audio
to measure how much intelligibility the autoencoder preserves.

Usage:
  python -u -m autoencoder.scripts.eval_asr
"""

import torch
import json
import math
import numpy as np
import soundfile as sf
from pathlib import Path


def load_model(ckpt_path, device):
    from autoencoder.models.autoencoder import WavVAE
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint: epoch={ckpt.get('epoch')}, snr={ckpt.get('best_snr', 0):.2f}")
    model = WavVAE(
        latent_dim=128,
        encoder_channels=(128, 256, 512, 512, 1024, 1024),
        strides=(2, 4, 4, 4, 4),
        dilations=(1, 3, 9),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def main():
    import whisper

    device = torch.device("cuda")
    sr = 22050
    stride = 512

    # Load autoencoder
    ckpt_path = "outputs/models/wav_vae_v14.4_full_remote_main/best.pt"
    model = load_model(ckpt_path, device)

    # Load Whisper
    print("Loading Whisper base.en...")
    asr = whisper.load_model("base.en", device=device)

    # Load test samples (use first 50 from training set as proxy)
    with open("data/ljspeech_train_splits.json") as f:
        items = json.load(f)

    n_eval = 50
    results = []

    for i, item in enumerate(items[:n_eval]):
        audio_path = item["audio_path"]
        data, file_sr = sf.read(audio_path)
        audio = torch.from_numpy(data.astype(np.float32))

        # Align to stride
        T = (audio.shape[0] // stride) * stride
        if T < stride * 2:
            continue
        audio = audio[:T]

        # Reconstruct
        audio_t = audio.unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(audio_t)["x_hat"].squeeze().cpu().float().numpy()
        orig = audio.numpy()

        # Whisper transcribe both (needs 16kHz)
        # Whisper handles resampling internally when given audio array
        result_orig = asr.transcribe(orig, language="en", fp16=True)
        result_recon = asr.transcribe(recon, language="en", fp16=True)

        text_orig = result_orig["text"].strip().lower()
        text_recon = result_recon["text"].strip().lower()

        # Simple WER (word-level edit distance)
        words_orig = text_orig.split()
        words_recon = text_recon.split()

        if len(words_orig) == 0:
            continue

        # Levenshtein distance
        n, m = len(words_orig), len(words_recon)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for j in range(m + 1): dp[0][j] = j
        for i2 in range(n + 1): dp[i2][0] = i2
        for i2 in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if words_orig[i2-1] == words_recon[j-1] else 1
                dp[i2][j] = min(dp[i2-1][j]+1, dp[i2][j-1]+1, dp[i2-1][j-1]+cost)
        wer = dp[n][m] / n

        results.append({
            "file": Path(audio_path).stem,
            "wer": wer,
            "n_words": n,
            "text_orig": text_orig[:80],
            "text_recon": text_recon[:80],
            "match": text_orig == text_recon,
        })

        if i < 5 or wer > 0.3:
            print(f"  [{i}] WER={wer:.2f} ({n} words) match={text_orig == text_recon}")
            if text_orig != text_recon:
                print(f"    orig:  {text_orig[:100]}")
                print(f"    recon: {text_recon[:100]}")

    # Summary
    wers = [r["wer"] for r in results]
    matches = sum(1 for r in results if r["match"])
    print(f"\n=== ASR Evaluation ({len(results)} samples) ===")
    print(f"  WER: mean={np.mean(wers):.3f}, median={np.median(wers):.3f}")
    print(f"  WER: p10={np.percentile(wers, 10):.3f}, p90={np.percentile(wers, 90):.3f}")
    print(f"  Exact match: {matches}/{len(results)} ({matches/len(results)*100:.1f}%)")
    print(f"  WER=0: {sum(1 for w in wers if w == 0)}/{len(results)}")
    print(f"  WER<0.1: {sum(1 for w in wers if w < 0.1)}/{len(results)}")
    print(f"  WER<0.2: {sum(1 for w in wers if w < 0.2)}/{len(results)}")
    print(f"  WER>0.5: {sum(1 for w in wers if w > 0.5)}/{len(results)}")


if __name__ == "__main__":
    main()
