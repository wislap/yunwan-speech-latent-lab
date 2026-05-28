"""Dataset for generation-friendly latent training.

Loads synthesized clean audio with phoneme alignment and applies low-pass filtering.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import butter, sosfilt
from torch.utils.data import Dataset


def trim_silence(wav: np.ndarray, sr: int, top_db: float = 30.0, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Remove long silence segments from audio, keeping short pauses.

    Scans for contiguous silence regions longer than 0.3s and truncates them.
    Preserves speech and short natural pauses.

    Args:
        wav: Audio signal (1D numpy array).
        sr: Sample rate.
        top_db: Threshold in dB below peak to consider as silence.
        frame_length: Frame length for energy computation.
        hop_length: Hop length for energy computation.

    Returns:
        Audio with long silences truncated.
    """
    if len(wav) == 0:
        return wav

    max_silence_sec = 0.3

    # Compute frame energies
    n_frames = 1 + (len(wav) - frame_length) // hop_length
    if n_frames <= 0:
        return wav

    energies = np.array([
        np.sqrt(np.mean(wav[i * hop_length : i * hop_length + frame_length] ** 2))
        for i in range(n_frames)
    ])

    # Threshold
    peak_energy = energies.max()
    if peak_energy < 1e-8:
        return wav

    threshold = peak_energy * (10 ** (-top_db / 20))
    is_speech = energies > threshold

    max_silence_frames = int(max_silence_sec * sr / hop_length)

    # Find contiguous silence regions and truncate long ones
    output_segments = []
    i = 0
    while i < n_frames:
        if is_speech[i]:
            # Speech frame - find end of speech segment
            j = i
            while j < n_frames and is_speech[j]:
                j += 1
            start_sample = i * hop_length
            end_sample = min(j * hop_length + frame_length, len(wav))
            output_segments.append(wav[start_sample:end_sample])
            i = j
        else:
            # Silence frame - find end of silence segment
            j = i
            while j < n_frames and not is_speech[j]:
                j += 1
            silence_len = j - i
            # Truncate if too long
            keep_frames = min(silence_len, max_silence_frames)
            start_sample = i * hop_length
            end_sample = min((i + keep_frames) * hop_length + frame_length, len(wav))
            output_segments.append(wav[start_sample:end_sample])
            i = j

    if not output_segments:
        return wav

    return np.concatenate(output_segments)


class SynthesizedAudioDataset(Dataset):
    """Dataset for clean synthesized speech with phoneme alignment.

    Supports low-pass filtering at configurable cutoff frequency.
    Can also work with any audio + phoneme alignment JSONL manifest.

    Args:
        data_jsonl: Path to JSONL manifest file.
        sr: Target sample rate.
        segment_length: Audio segment length in samples (for random crop).
        lpf_cutoff_hz: Low-pass filter cutoff. None = no filtering.
        total_stride: AE total stride (for frame count alignment).
        path_root: Optional root prefix for audio paths in manifest.
    """

    def __init__(
        self,
        data_jsonl: str,
        sr: int = 22050,
        segment_length: int = 65536,
        lpf_cutoff_hz: float | None = 4000.0,
        total_stride: int = 256,
        path_root: str | None = None,
        trim_silence_db: float = 30.0,
    ):
        self.sr = sr
        self.segment_length = segment_length
        self.lpf_cutoff_hz = lpf_cutoff_hz
        self.total_stride = total_stride
        self.path_root = Path(path_root) if path_root else None
        self.trim_silence_db = trim_silence_db

        # Load manifest
        self.items = []
        with open(data_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

        # Build phoneme vocabulary from data
        self._phoneme_vocab = self._build_phoneme_vocab()

        # Pre-compute LPF coefficients
        self._sos = None
        if lpf_cutoff_hz is not None and lpf_cutoff_hz < sr / 2:
            nyquist = sr / 2
            normalized_cutoff = lpf_cutoff_hz / nyquist
            self._sos = butter(5, normalized_cutoff, btype="low", output="sos")

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_path(self, audio_path: str) -> str:
        if self.path_root is not None:
            return str(self.path_root / audio_path)
        return audio_path

    def _apply_lpf(self, audio: np.ndarray) -> np.ndarray:
        """Apply low-pass filter to audio."""
        if self._sos is None:
            return audio
        return sosfilt(self._sos, audio).astype(np.float32)

    def _build_phoneme_vocab(self):
        """Build phoneme token to ID mapping from dataset."""
        vocab = {"<pad>": 0, "<unk>": 1}
        for item in self.items:
            tokens = item.get("phoneme_tokens", [])
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        return vocab

    def _get_frame_durations_from_intervals(
        self, intervals: list[dict], phoneme_tokens: list[str],
        start_sample: int, end_sample: int
    ) -> tuple[list[int], list[int]]:
        """Extract phoneme IDs and frame-level durations from intervals.

        Clips intervals to the [start_sample, end_sample) crop window.

        Returns:
            (phoneme_ids, frame_durations) - lists of same length
        """
        phoneme_ids = []
        frame_durations = []

        for iv in intervals:
            iv_start = iv["start_sample"]
            iv_end = iv["end_sample"]

            # Clip to crop window
            clipped_start = max(iv_start, start_sample)
            clipped_end = min(iv_end, end_sample)

            if clipped_start >= clipped_end:
                continue

            # Convert to frames within the crop
            frame_start = (clipped_start - start_sample) // self.total_stride
            frame_end = (clipped_end - start_sample + self.total_stride - 1) // self.total_stride

            n_frames = frame_end - frame_start
            if n_frames <= 0:
                continue

            # Get phoneme token from the phoneme_tokens list
            idx = iv.get("index", 0)
            if idx < len(phoneme_tokens):
                token = phoneme_tokens[idx]
            else:
                token = "<unk>"

            ph_id = self._phoneme_vocab.get(token, 1)  # 1 = <unk>
            phoneme_ids.append(ph_id)
            frame_durations.append(n_frames)

        return phoneme_ids, frame_durations

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]

        # Resolve audio path - support both "audio_path" and "wav_path" keys
        raw_path = item.get("audio_path") or item.get("wav_path", "")
        audio_path = self._resolve_path(raw_path)

        # Load audio
        wav_np, file_sr = sf.read(audio_path, dtype="float32")
        if wav_np.ndim > 1:
            wav_np = wav_np[:, 0]  # mono
        if file_sr != self.sr:
            wav_np = torchaudio.functional.resample(
                torch.from_numpy(wav_np), file_sr, self.sr
            ).numpy()

        # Remove long silence segments (keep max 0.3s pauses)
        if self.trim_silence_db > 0:
            wav_np = trim_silence(wav_np, self.sr, top_db=self.trim_silence_db)

        wav = torch.from_numpy(wav_np)
        # wav is [T]

        total_samples = wav.shape[0]

        # Random crop to segment_length
        if total_samples > self.segment_length:
            start = torch.randint(0, total_samples - self.segment_length, (1,)).item()
            # Align start to stride boundary
            start = (start // self.total_stride) * self.total_stride
            end = start + self.segment_length
            wav = wav[start:end]
        else:
            start = 0
            end = total_samples
            # Pad if shorter
            pad_len = self.segment_length - total_samples
            wav = torch.nn.functional.pad(wav, (0, pad_len))

        # Ensure length is multiple of total_stride
        remainder = wav.shape[0] % self.total_stride
        if remainder != 0:
            wav = wav[: wav.shape[0] - remainder]

        # Apply low-pass filter
        if self._sos is not None:
            wav_np = wav.numpy()
            wav_np = self._apply_lpf(wav_np)
            wav = torch.from_numpy(wav_np)

        # Get phoneme info from intervals or fallback
        n_frames = wav.shape[0] // self.total_stride
        intervals = item.get("phoneme_intervals")

        if intervals:
            # Use frame-level intervals with crop clipping
            phoneme_tokens = item.get("phoneme_tokens", [])
            phoneme_ids, frame_durations = self._get_frame_durations_from_intervals(
                intervals, phoneme_tokens, start, end
            )
            if not phoneme_ids:
                # Fallback if no intervals overlap with crop
                phoneme_ids = [1]
                frame_durations = [n_frames]
        elif "phoneme_ids" in item:
            # Direct phoneme_ids + phoneme_durations format
            phoneme_ids = item["phoneme_ids"]
            frame_durations = item.get(
                "phoneme_durations", [n_frames // max(len(phoneme_ids), 1)] * len(phoneme_ids)
            )
        else:
            # Fallback: use phoneme_tokens with vocab
            tokens = item.get("phoneme_tokens", ["<unk>"])
            phoneme_ids = [self._phoneme_vocab.get(t, 1) for t in tokens]
            # Uniform duration
            frame_durations = [n_frames // max(len(phoneme_ids), 1)] * len(phoneme_ids)

        phoneme_ids_t = torch.tensor(phoneme_ids, dtype=torch.long)
        frame_durations_t = torch.tensor(frame_durations, dtype=torch.long)

        # Adjust durations to match actual frame count
        dur_sum = frame_durations_t.sum().item()
        if dur_sum != n_frames and dur_sum > 0:
            scale = n_frames / dur_sum
            frame_durations_t = (frame_durations_t.float() * scale).long()
            # Ensure minimum 1 frame per phoneme
            frame_durations_t = frame_durations_t.clamp(min=1)
            # Fix rounding error on last phoneme
            diff = n_frames - frame_durations_t.sum().item()
            if diff != 0:
                frame_durations_t[-1] += diff

        return {
            "audio": wav.unsqueeze(0),  # [1, T]
            "phoneme_ids": phoneme_ids_t,
            "phoneme_durations": frame_durations_t,
        }
