"""Per-frame label extraction aligned to AE frame rate.

For each audio clip, produce label arrays at the same frame rate as the
autoencoder latent: one label per `total_stride` samples.

Labels produced:
- phoneme: integer id per frame (CosyVoice only; -1 elsewhere)
- speaker: single id per utterance (from manifest)
- f0: Hz per frame (NaN in unvoiced frames)
- voiced: bool per frame
- energy: log(1 + rms) per frame
- centroid: spectral centroid in kHz per frame

All functions operate on mono float32 numpy arrays at TARGET_SR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.signal import stft

from .common import TARGET_SR, V16_3_TOTAL_STRIDE

# ---------------------------------------------------------------------------
# Frame indexing
# ---------------------------------------------------------------------------


def ae_frame_count(n_samples: int, total_stride: int = V16_3_TOTAL_STRIDE) -> int:
    return n_samples // total_stride


# ---------------------------------------------------------------------------
# Energy and centroid
# ---------------------------------------------------------------------------


def frame_energy(audio: np.ndarray, total_stride: int = V16_3_TOTAL_STRIDE) -> np.ndarray:
    """log(1 + RMS) per AE frame. Shape: [T_frames]."""
    n_frames = ae_frame_count(len(audio), total_stride)
    trimmed = audio[: n_frames * total_stride]
    reshaped = trimmed.reshape(n_frames, total_stride)
    rms = np.sqrt(np.mean(reshaped.astype(np.float64) ** 2, axis=1) + 1e-12)
    return np.log1p(rms).astype(np.float32)


def frame_spectral_centroid(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    total_stride: int = V16_3_TOTAL_STRIDE,
    n_fft: int = 1024,
) -> np.ndarray:
    """Spectral centroid in kHz per AE frame. Uses STFT centered on each AE frame.

    Returns shape [T_frames].
    """
    n_frames = ae_frame_count(len(audio), total_stride)
    # STFT hop matches AE stride so frames correspond one-to-one (approximately)
    freqs, _, spec = stft(
        audio,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - total_stride,
        boundary="zeros",
        padded=True,
    )
    mag2 = np.abs(spec) ** 2  # [F, T_stft]
    centroid_hz = (freqs[:, None] * mag2).sum(axis=0) / (mag2.sum(axis=0) + 1e-12)
    # Align to AE frame count
    if len(centroid_hz) >= n_frames:
        centroid_hz = centroid_hz[:n_frames]
    else:
        centroid_hz = np.concatenate(
            [centroid_hz, np.full(n_frames - len(centroid_hz), centroid_hz[-1] if len(centroid_hz) else 0.0)]
        )
    return (centroid_hz / 1000.0).astype(np.float32)


# ---------------------------------------------------------------------------
# F0 / voicedness via autocorrelation
# ---------------------------------------------------------------------------


def frame_f0(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    total_stride: int = V16_3_TOTAL_STRIDE,
    fmin: float = 50.0,
    fmax: float = 500.0,
    window_mult: int = 4,
    voicing_thresh: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation-based F0 estimator.

    For each AE frame, use a window of size window_mult * total_stride centered
    on the frame, compute autocorrelation, pick the max in the lag range
    [sr/fmax, sr/fmin]. Return (f0_hz with NaN for unvoiced, voiced_bool).

    This is deliberately simple. It is not meant to be ASR-grade F0; it is
    consistent and produces a frame-aligned voicedness signal which is what
    B2/B4 need.
    """
    n_frames = ae_frame_count(len(audio), total_stride)
    window_size = total_stride * window_mult
    half = window_size // 2
    min_lag = max(1, int(math.floor(sr / fmax)))
    max_lag = min(half, int(math.ceil(sr / fmin)))
    f0_hz = np.full(n_frames, np.nan, dtype=np.float32)
    voiced = np.zeros(n_frames, dtype=bool)

    padded = np.pad(audio.astype(np.float64), (half, half), mode="constant")
    for t in range(n_frames):
        center = t * total_stride + total_stride // 2
        w = padded[center : center + window_size]
        w = w - w.mean()
        norm = float(np.sqrt(np.dot(w, w) + 1e-12))
        if norm < 1e-6:
            continue
        # autocorrelation via FFT
        fw = np.fft.rfft(w, n=2 * window_size)
        ac = np.fft.irfft(fw * np.conj(fw), n=2 * window_size)[:window_size]
        ac0 = ac[0]
        if ac0 < 1e-6:
            continue
        ac_norm = ac[min_lag : max_lag + 1] / ac0
        if ac_norm.size == 0:
            continue
        best_lag = int(np.argmax(ac_norm))
        peak = float(ac_norm[best_lag])
        if peak < voicing_thresh:
            continue
        lag = min_lag + best_lag
        f0_hz[t] = float(sr) / lag
        voiced[t] = True
    return f0_hz, voiced


# ---------------------------------------------------------------------------
# Phoneme labels from CosyVoice pinyin_durations
# ---------------------------------------------------------------------------


def phoneme_vocabulary(items: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Build a pinyin vocabulary across all items.

    Supports two manifest formats:
    - pinyin_tokens: list[str] + pinyin_durations: list[float]
    - pinyin_durations: list[dict] with "pinyin" key

    0 is reserved for silence/blank, ids start at 1 for seen tokens.
    """
    vocab: dict[str, int] = {"<sil>": 0}
    for item in items:
        tokens = item.get("pinyin_tokens")
        if tokens:
            for t in tokens:
                if t and t not in vocab:
                    vocab[t] = len(vocab)
            continue
        pds = item.get("pinyin_durations") or []
        for entry in pds:
            if isinstance(entry, dict):
                token = entry.get("pinyin")
                if token and token not in vocab:
                    vocab[token] = len(vocab)
    return vocab


def frame_phoneme_from_pinyin_durations(
    item: dict[str, Any],
    n_frames: int,
    vocab: dict[str, int],
    sr: int = TARGET_SR,
    total_stride: int = V16_3_TOTAL_STRIDE,
) -> np.ndarray:
    """Convert an item's pinyin tokens+durations to a per-frame integer label
    array.

    Supports:
    - item["pinyin_tokens"] = list[str] and item["pinyin_durations"] = list[float_sec]
    - item["pinyin_durations"] = list[dict] with "pinyin" and "duration"/"start"+"end"
    - item["phoneme_intervals"] = list[dict] with "pinyin"/"token" and time info

    Frames outside any interval become the silence id (0).
    """
    labels = np.zeros(n_frames, dtype=np.int64)
    frames_per_sec = sr / total_stride

    def _write(start_sec: float, end_sec: float, token_id: int) -> None:
        if end_sec <= start_sec:
            return
        fs = max(0, int(math.floor(start_sec * frames_per_sec)))
        fe = min(n_frames, int(math.ceil(end_sec * frames_per_sec)))
        if fe > fs and token_id != 0:
            labels[fs:fe] = token_id

    tokens = item.get("pinyin_tokens")
    durations = item.get("pinyin_durations")
    if tokens and durations and isinstance(durations[0], (int, float)):
        cursor = 0.0
        for tok, dur in zip(tokens, durations):
            dur_f = float(dur)
            tid = vocab.get(tok, 0) if tok else 0
            _write(cursor, cursor + dur_f, tid)
            cursor += dur_f
        return labels

    # Fallback 1: pinyin_durations as list of dicts
    if durations and isinstance(durations[0], dict):
        cursor = 0.0
        for entry in durations:
            token = entry.get("pinyin") or entry.get("token")
            tid = vocab.get(token, 0) if token else 0
            if "start" in entry and "end" in entry:
                _write(float(entry["start"]), float(entry["end"]), tid)
            else:
                dur_f = float(entry.get("duration", 0.0))
                _write(cursor, cursor + dur_f, tid)
                cursor += dur_f
        return labels

    # Fallback 2: phoneme_intervals with explicit time
    intervals = item.get("phoneme_intervals")
    if intervals:
        for entry in intervals:
            token = entry.get("pinyin") or entry.get("token")
            tid = vocab.get(token, 0) if token else 0
            if "start" in entry and "end" in entry:
                _write(float(entry["start"]), float(entry["end"]), tid)
        return labels

    return labels


# ---------------------------------------------------------------------------
# Public bundle
# ---------------------------------------------------------------------------


@dataclass
class FrameLabels:
    """Per-frame label arrays for one audio clip.

    phoneme: int array [T], 0 = sil/unknown. -1 means phoneme not available.
    speaker: str or None. Single value for the clip.
    f0: float32 [T], NaN in unvoiced frames.
    voiced: bool [T].
    energy: float32 [T].
    centroid: float32 [T], kHz.
    n_frames: int
    sample_id: str
    """

    phoneme: np.ndarray
    speaker: str | None
    f0: np.ndarray
    voiced: np.ndarray
    energy: np.ndarray
    centroid: np.ndarray
    n_frames: int
    sample_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    def save_npz(self, path) -> None:
        np.savez_compressed(
            path,
            phoneme=self.phoneme,
            f0=self.f0,
            voiced=self.voiced,
            energy=self.energy,
            centroid=self.centroid,
            speaker=np.asarray(self.speaker if self.speaker is not None else "", dtype=object),
            n_frames=np.asarray(self.n_frames),
            sample_id=np.asarray(self.sample_id, dtype=object),
        )

    @classmethod
    def load_npz(cls, path) -> "FrameLabels":
        data = np.load(path, allow_pickle=True)
        spk = data["speaker"].item()
        return cls(
            phoneme=data["phoneme"],
            speaker=spk if spk else None,
            f0=data["f0"],
            voiced=data["voiced"].astype(bool),
            energy=data["energy"],
            centroid=data["centroid"],
            n_frames=int(data["n_frames"]),
            sample_id=str(data["sample_id"].item()),
        )


def compute_frame_labels(
    audio: np.ndarray,
    item: dict[str, Any],
    phoneme_vocab: dict[str, int] | None = None,
    sr: int = TARGET_SR,
    total_stride: int = V16_3_TOTAL_STRIDE,
    sample_id: str | None = None,
) -> FrameLabels:
    """Compute all per-frame labels for one audio clip."""
    n_frames = ae_frame_count(len(audio), total_stride)
    if sample_id is None:
        sample_id = str(item.get("id", item.get("audio_path", "unknown")))

    speaker = item.get("speaker") or item.get("spk_id") or item.get("speaker_id")
    if speaker is not None:
        speaker = str(speaker)

    energy = frame_energy(audio, total_stride)[:n_frames]
    centroid = frame_spectral_centroid(audio, sr, total_stride)[:n_frames]
    f0, voiced = frame_f0(audio, sr, total_stride)

    if phoneme_vocab is not None and (
        ("pinyin_tokens" in item and item["pinyin_tokens"])
        or ("pinyin_durations" in item and item["pinyin_durations"])
        or ("phoneme_intervals" in item and item["phoneme_intervals"])
    ):
        phoneme = frame_phoneme_from_pinyin_durations(
            item, n_frames, phoneme_vocab, sr, total_stride
        )
    else:
        phoneme = np.full(n_frames, -1, dtype=np.int64)

    return FrameLabels(
        phoneme=phoneme,
        speaker=speaker,
        f0=f0,
        voiced=voiced,
        energy=energy,
        centroid=centroid,
        n_frames=n_frames,
        sample_id=sample_id,
        meta={"dataset": item.get("dataset", "unknown")},
    )


__all__ = [
    "FrameLabels",
    "ae_frame_count",
    "frame_energy",
    "frame_spectral_centroid",
    "frame_f0",
    "phoneme_vocabulary",
    "frame_phoneme_from_pinyin_durations",
    "compute_frame_labels",
]
