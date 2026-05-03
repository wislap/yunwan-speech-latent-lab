# V16.1 High-Fidelity Time AE Report

Date: 2026-05-03

## Summary

V16.1 is the first version that looks usable as a latent representation baseline.

The key change was not a new decoder trick. It was relaxing the temporal compression from 512x to 128x and training a deterministic, pure time-domain autoencoder. This sharply improved both objective metrics and listening quality.

Main conclusion:

> The previous failure mode was dominated by an overly narrow 512x information channel. Pure time-domain AE is viable when the latent frame rate is high enough.

V16.1 should become the current high-fidelity reference baseline. 256x should be tested next as a compression boundary experiment, but not treated as a replacement until listening quality is confirmed.

## Context

Earlier V14/V15 experiments repeatedly saturated around 11-12 dB SNR at 512x total stride. Several decoder and loss variants failed to break the quality ceiling:

- phase losses and PhaseRefiner only gave small improvements
- Vocos/iSTFT paths were hard to optimize from abstract latents
- 256-dimensional latent at 512x did not solve the decoder/quality problem
- V16A, a shallow 512x time-domain test, improved training loss but still gave poor eval quality

The practical issue was that 12 dB reconstructions were still too noisy and high-frequency dirty for DiT latent representation research. AE artifacts made it impossible to judge whether downstream DiT errors came from the generator or from the AE target itself.

## V16A Result

V16A tested:

- total stride: 512x
- latent dim: 256
- strides: `[8, 8, 8]`
- pure time-domain decoder
- deterministic generation path was not the focus

Results:

| Eval epoch | SNR | STFT |
|---:|---:|---:|
| 4 | 6.30 dB | 2.0419 |
| 9 | 7.19 dB | 1.7084 |

Interpretation:

- V16A trained stably and loss decreased.
- However, eval SNR improved too slowly.
- This did not support 512x shallow time-domain as a high-fidelity baseline.
- V16A was stopped and preserved as a negative compression-path result.

## V16.1 Setup

Configuration:

- config: `autoencoder/conf/experiment/v16_1_hf_time.yaml`
- total stride: 128x
- strides: `[2, 4, 4, 4]`
- latent dim: 256
- latent frame rate: about 172 Hz at 22.05 kHz
- model: pure time-domain encoder/decoder
- bottleneck: deterministic, `sample_latent: false`
- loss: L1 + STFT + mel
- no GAN
- no Vocos
- no PhaseRefiner
- no multi-band complex STFT in the first run

Parameter count:

- total: 104,788,481
- encoder: 48,816,512
- decoder: 55,971,969

Runtime:

- GPU: RTX 5090 32GB
- memory: about 8.1 GB
- epoch time: about 15 minutes

## V16.1 Metrics

Training/eval log:

| Eval epoch | SNR | STFT | Notes |
|---:|---:|---:|---|
| 1 | 7.55 dB | 1.3949 | first eval, already better than V16A epoch 9 |
| 3 | 12.80 dB | 0.9752 | crosses old 512x plateau very early |
| 5 | 14.28 dB | 0.8854 | clear positive trend |
| 7 | 19.08 dB | 0.6518 | strong reconstruction quality |
| 9 | 18.27 dB | 0.6703 | slight SNR dip |
| 11 | 23.21 dB | 0.5217 | current best checkpoint |
| 13 | 22.82 dB | 0.5067 | comparable quality, lower STFT |

Checkpoint:

- best: `/root/autodl-tmp/project/outputs/models/wav_vae_v16.1_hf_time_remote_main/best.pt`
- best metric: 23.21 dB SNR at epoch 11

Eval audio copied locally:

- audio copies are not kept under `docs/`; use the remote model output directory when listening samples are needed

## Listening Result

Listening feedback:

> Almost perfectly preserves semantics. Only slight articulation blur remains.

This is the most important result. V16.1 is qualitatively different from the 512x runs: the reconstruction is no longer dominated by obvious AE noise or dirty high-frequency artifacts.

Remaining issue:

- slight mouth/phoneme articulation blur
- likely high-frequency consonant and transient detail loss

This is acceptable for a first high-fidelity AE baseline, but it should be tracked before using the latent for serious DiT quality claims.

## Interpretation

The main bottleneck was temporal compression, not simply decoder architecture.

At 512x:

- latent frame rate is about 43 Hz
- each latent frame must carry about 23 ms of waveform detail
- high-frequency phase and articulation are too compressed
- decoder/loss patches could not recover missing detail

At 128x:

- latent frame rate is about 172 Hz
- the same time-domain architecture becomes viable
- deterministic training removes sampling noise from the reconstruction target
- SNR and listening quality improve quickly

This supports a new baseline strategy:

1. Use 128x as the high-fidelity anchor.
2. Treat 256x as a compression boundary test.
3. Treat 512x as currently too aggressive for the baseline.

## Recommended Baseline Status

V16.1 should be the current AE baseline for latent representation research, subject to listening review of more samples.

Use it for:

- latent extraction experiments
- first DiT representation tests
- reconstruction/listening references
- comparison target for 256x variants

Do not yet use it to claim final codec quality, because:

- only one eval sample has been listened to so far
- articulation blur remains
- high-frequency band metrics are not yet separated
- PESQ is effectively unusable in the current logs

## Next Experiment: V16.2 256x

The next question is whether 256x keeps enough quality while reducing DiT sequence length.

Why test 256x:

- 128x at 22.05 kHz gives about 172 latent frames/s
- 30 seconds gives about 5160 frames
- 256x would reduce this to about 86 frames/s
- DiT sequence cost would be roughly halved

V16.2 should change only the compression schedule, keeping all other variables as close to V16.1 as possible.

Candidate setup:

- total stride: 256x
- latent dim: 256
- `sample_latent: false`
- pure time-domain decoder
- same loss as V16.1
- no GAN
- no Vocos
- no PhaseRefiner
- no multi-band complex STFT initially

Candidate stride schedules:

- `[2, 4, 4, 8]`
- `[4, 4, 4, 4]`

Recommended first choice:

- `[2, 4, 4, 8]`

Reason:

- preserves the gentle early downsampling from V16.1
- pushes the extra compression later, where features are more abstract
- changes fewer early signal-processing assumptions than `[4, 4, 4, 4]`

Decision rule:

- If V16.2 reaches near-V16.1 listening quality by epoch 5-10, 256x can become the DiT main baseline candidate.
- If semantics remain but articulation/high-frequency detail degrades, keep 128x as the main baseline and consider high-frequency loss or wider latents for 256x.
- If it regresses toward V16A behavior, pause 256x and use 128x for representation learning.

Launch status:

- Config added: `autoencoder/conf/experiment/v16_2_256x_time.yaml`
- Doc added: `docs/v16.2/256x_time.md`
- Remote log: `/root/autodl-tmp/project/v16_2_launch.log`
- Remote output dir: `/root/autodl-tmp/project/outputs/models/wav_vae_v16.2_256x_time_remote_main`
- Smoke result: input 8192 samples -> latent shape `[1, 256, 32]`, output length aligned, deterministic encode confirmed.
- Parameter count: 125,768,193 total; encoder 57,205,120; decoder 68,563,073.
- Training started on 2026-05-03 via port 10891.

## Action Items

1. Listen to V16.1 eval epochs 7, 11, and 13.
2. Copy or preserve the V16.1 best checkpoint.
3. Add a small fixed listening set, not just one eval sample.
4. Monitor V16.2 256x through epochs 3, 5, 7, and 11.
5. Add band-wise/high-frequency diagnostics before making DiT conclusions.
