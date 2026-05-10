"""V18 smoke test: architecture shape and memory sanity.

Runs a forward + backward pass with reference shapes matching the V18 config:
  batch_size=1
  segment_length=65536 (V16.3 stable)
  sr=22050
  total_stride=256 -> T_frames=256

Verifies:
  - z_phoneme shape [1, 128, 256]
  - z_residual shape [1, 256, 256]
  - z shape [1, 384, 256]
  - x_hat shape [1, 1, 65536]
  - ctc_log_probs shape [1, 256, vocab]
  - Forward + backward runs without OOM on CPU or GPU
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from autoencoder.models.v18_encoder import V18Autoencoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--segment-length", type=int, default=65536)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--backward", action="store_true", help="Run backward to test gradient memory")
    parser.add_argument("--load-phoneme", type=str, default="",
                        help="Path to V17.3 teacher checkpoint (optional)")
    parser.add_argument("--load-decoder", type=str, default="",
                        help="Path to V16.3 AE checkpoint (optional)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    model = V18Autoencoder(
        sr=args.sr,
        use_checkpoint=args.use_checkpoint,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_phoneme = sum(p.numel() for p in model.phoneme_encoder.parameters())
    n_residual = sum(p.numel() for p in model.residual_encoder.parameters())
    n_decoder = sum(p.numel() for p in model.decoder.parameters())
    n_ctc = sum(p.numel() for p in model.ctc_head.parameters())
    print(f"Total params: {n_params:,}")
    print(f"  Phoneme encoder: {n_phoneme:,}")
    print(f"  Residual encoder: {n_residual:,}")
    print(f"  Decoder: {n_decoder:,}")
    print(f"  CTC head: {n_ctc:,}")
    print(f"  total_stride: {model.total_stride}, latent_dim: {model.latent_dim}")

    if args.load_phoneme:
        info = model.load_phoneme_pretrained(args.load_phoneme)
        print(f"Phoneme pretrained load: {info}")
    if args.load_decoder:
        info = model.load_decoder_pretrained(args.load_decoder)
        print(f"Decoder pretrained load: {info}")

    audio = torch.randn(args.batch_size, 1, args.segment_length, device=device)

    if args.backward:
        model.train()
    else:
        model.eval()

    print(f"\nInput audio: {audio.shape}")
    t0 = time.time()
    with torch.set_grad_enabled(args.backward):
        out = model(audio)
    t1 = time.time()
    print(f"Forward: {t1 - t0:.2f}s")
    print(f"  z_phoneme: {out['z_phoneme'].shape}")
    print(f"  z_residual: {out['z_residual'].shape}")
    print(f"  z: {out['z'].shape}")
    print(f"  x_hat: {out['x_hat'].shape}")
    print(f"  ctc_log_probs: {out['ctc_log_probs'].shape}")
    print(f"  reg_loss: {out['reg_loss'].item():.6f}")

    expected_frames = args.segment_length // model.total_stride
    assert out["z_phoneme"].shape == (args.batch_size, 128, expected_frames), (
        f"z_phoneme shape mismatch: got {out['z_phoneme'].shape}, "
        f"expected ({args.batch_size}, 128, {expected_frames})"
    )
    assert out["z_residual"].shape == (args.batch_size, 256, expected_frames), (
        f"z_residual shape mismatch: got {out['z_residual'].shape}"
    )
    assert out["z"].shape == (args.batch_size, 384, expected_frames), (
        f"z shape mismatch: got {out['z'].shape}"
    )
    assert out["x_hat"].shape == (args.batch_size, 1, args.segment_length), (
        f"x_hat shape mismatch: got {out['x_hat'].shape}"
    )
    assert out["ctc_log_probs"].shape == (args.batch_size, expected_frames, 512), (
        f"ctc_log_probs shape mismatch: got {out['ctc_log_probs'].shape}"
    )

    if args.backward:
        # Synthetic loss: recon L1 + small CTC dummy loss
        target = torch.randn_like(audio)
        recon_loss = F.l1_loss(out["x_hat"], target)
        # CTC fake: log-probs already log-softmax
        log_probs = out["ctc_log_probs"].transpose(0, 1)  # [T, B, vocab]
        targets = torch.randint(1, 512, (args.batch_size, 16), device=device)
        input_lengths = torch.full((args.batch_size,), expected_frames, device=device, dtype=torch.long)
        target_lengths = torch.full((args.batch_size,), 16, device=device, dtype=torch.long)
        ctc_loss = F.ctc_loss(
            log_probs.float(), targets, input_lengths, target_lengths,
            blank=0, reduction="mean", zero_infinity=True,
        )
        total = recon_loss + 0.1 * ctc_loss + out["reg_loss"]
        t2 = time.time()
        total.backward()
        t3 = time.time()
        print(f"\nBackward: {t3 - t2:.2f}s")
        print(f"  recon_loss: {recon_loss.item():.4f}")
        print(f"  ctc_loss: {ctc_loss.item():.4f}")
        print(f"  total: {total.item():.4f}")

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Peak CUDA memory: {peak_mem:.2f} GB")

    print("\nV18 smoke test PASSED.")


if __name__ == "__main__":
    main()
