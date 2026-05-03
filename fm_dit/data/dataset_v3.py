"""TTS Dataset v3: 支持 phoneme-level duration

新格式:
{
    "phoneme_ids": [1, 2, 3, ...],        # phoneme-level
    "phoneme_durations": [5, 3, 8, ...],  # 每个音素的帧数
    "frame_phoneme_ids": [...],           # frame-level (兼容)
    "pitch": [...],                        # frame-level
    "energy": [...],                       # frame-level
}
"""

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


class TTSDatasetV3(Dataset):
    """TTS 数据集 v3: 支持 phoneme-level duration"""
    
    def __init__(
        self,
        manifest_path: str,
        max_frames: int = 256,
        min_frames: int = 16,
        latent_stats_path: Optional[str] = None,
        preload_latents: bool = False,  # 预加载 latent 到内存
        preload_to_gpu: bool = False,  # 预加载到 GPU 显存
    ):
        self.manifest_path = Path(manifest_path)
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.preload_latents = preload_latents
        self.preload_to_gpu = preload_to_gpu
        self.gpu_device = torch.device('cuda') if preload_to_gpu and torch.cuda.is_available() else None
        self.latent_dim = None
        
        # 加载 latent 归一化统计量
        self.latent_mean = 0.0
        self.latent_std = 1.0
        if latent_stats_path and Path(latent_stats_path).exists():
            with open(latent_stats_path, "r") as f:
                stats = json.load(f)
                # Prefer per-channel stats when present. V16 AE latents have
                # meaningful channel scale differences, so scalar stats are
                # only a fallback for older manifests.
                if isinstance(stats.get("mean"), list) and isinstance(stats.get("std"), list):
                    self.latent_dim = len(stats["mean"])
                    self.latent_mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, -1)
                    self.latent_std = torch.tensor(stats["std"], dtype=torch.float32).view(1, -1)
                elif "global_mean" in stats:
                    self.latent_mean = stats["global_mean"]
                    self.latent_std = stats["global_std"]
                else:
                    self.latent_mean = stats.get("mean", 0.0)
                    self.latent_std = stats.get("std", 1.0)
        
        # 加载 manifest
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        
        # 过滤太短的样本
        self.items = [
            item for item in self.items 
            if item.get("duration", float("inf")) >= min_frames
        ]
        
        # 预加载 latent 到内存/显存 (预先 pad 到 max_frames)
        self.latent_cache = {}
        self.cond_cache = {}  # 条件数据缓存
        if preload_latents:
            device_str = "GPU" if preload_to_gpu else "memory"
            print(f"Preloading {len(self.items)} samples to {device_str} (pre-padded to {max_frames})...")
            for item in self.items:
                # 加载 latent
                latent = torch.load(item["latent_path"], weights_only=True)
                latent = self._to_time_channel(latent)
                T = latent.shape[0]
                if T > max_frames:
                    latent = latent[:max_frames]
                    T = max_frames
                elif T < max_frames:
                    pad = torch.zeros(max_frames - T, latent.shape[1])
                    latent = torch.cat([latent, pad], dim=0)
                
                # 加载条件数据并 pad
                frame_phoneme_ids = torch.tensor(item["frame_phoneme_ids"][:max_frames], dtype=torch.long)
                pitch = torch.tensor(item["pitch"][:max_frames], dtype=torch.float32)
                energy = torch.tensor(item["energy"][:max_frames], dtype=torch.float32)
                
                if len(frame_phoneme_ids) < max_frames:
                    pad_len = max_frames - len(frame_phoneme_ids)
                    frame_phoneme_ids = torch.cat([frame_phoneme_ids, torch.zeros(pad_len, dtype=torch.long)])
                    pitch = torch.cat([pitch, torch.zeros(pad_len)])
                    energy = torch.cat([energy, torch.zeros(pad_len)])
                
                # 归一化 latent
                latent = self._normalize_latent(latent)
                
                # 移动到 GPU
                if self.gpu_device is not None:
                    latent = latent.to(self.gpu_device)
                    frame_phoneme_ids = frame_phoneme_ids.to(self.gpu_device)
                    pitch = pitch.to(self.gpu_device)
                    energy = energy.to(self.gpu_device)
                
                # 计算 frame_duration 和 frame_position (v4.7)
                phoneme_durations_tensor = torch.tensor(item["phoneme_durations"], dtype=torch.float32)
                frame_duration, frame_position = self._compute_frame_duration_position(phoneme_durations_tensor, max_frames)
                
                # 预加载 phoneme_ids 和 phoneme_durations (避免每次 __getitem__ 重新创建)
                phoneme_ids = torch.tensor(item["phoneme_ids"], dtype=torch.long)
                phoneme_durations = torch.tensor(item["phoneme_durations"], dtype=torch.float32)
                
                if self.gpu_device is not None:
                    frame_duration = frame_duration.to(self.gpu_device)
                    frame_position = frame_position.to(self.gpu_device)
                    phoneme_ids = phoneme_ids.to(self.gpu_device)
                    phoneme_durations = phoneme_durations.to(self.gpu_device)
                
                self.latent_cache[item["id"]] = latent
                self.cond_cache[item["id"]] = {
                    "frame_phoneme_ids": frame_phoneme_ids,
                    "pitch": pitch,
                    "energy": energy,
                    "frame_duration": frame_duration,
                    "frame_position": frame_position,
                    "phoneme_ids": phoneme_ids,
                    "phoneme_durations": phoneme_durations,
                }
            print(f"Preloaded {len(self.latent_cache)} samples")
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        
        # 如果数据已预加载到 GPU，直接返回（不需要 clone，因为是只读）
        if self.preload_latents and item["id"] in self.cond_cache:
            cond = self.cond_cache[item["id"]]
            # 直接返回缓存的数据，避免每次创建 tensor
            return {
                "id": item["id"],
                "latent": self.latent_cache[item["id"]],
                "frame_phoneme_ids": cond["frame_phoneme_ids"],
                "pitch": cond["pitch"],
                "energy": cond["energy"],
                "frame_duration": cond["frame_duration"],    # v4.7
                "frame_position": cond["frame_position"],    # v4.7
                "phoneme_ids": cond["phoneme_ids"],
                "phoneme_durations": cond["phoneme_durations"],
            }
        
        # 加载 latent (从缓存或文件)
        if self.preload_latents and item["id"] in self.latent_cache:
            latent = self.latent_cache[item["id"]].clone()
        else:
            latent = torch.load(item["latent_path"], weights_only=True)
            latent = self._to_time_channel(latent)
        
        # 归一化 latent
        latent = self._normalize_latent(latent)
        
        # 加载 phoneme-level 数据
        phoneme_ids = torch.tensor(item["phoneme_ids"], dtype=torch.long)
        phoneme_durations = torch.tensor(item["phoneme_durations"], dtype=torch.float)
        
        # 加载 frame-level 数据
        frame_phoneme_ids = torch.tensor(item.get("frame_phoneme_ids", []), dtype=torch.long)
        pitch = torch.tensor(item.get("pitch", []), dtype=torch.float)
        energy = torch.tensor(item.get("energy", []), dtype=torch.float)
        
        # 确保 frame-level 长度一致
        T = latent.shape[0]
        
        if len(frame_phoneme_ids) != T:
            # 重新生成 frame_phoneme_ids
            frame_phoneme_ids = self._expand_phonemes(phoneme_ids, phoneme_durations, T)
        
        if len(pitch) != T:
            if len(pitch) > T:
                pitch = pitch[:T]
            else:
                pitch = torch.cat([pitch, torch.zeros(T - len(pitch))])
        
        if len(energy) != T:
            if len(energy) > T:
                energy = energy[:T]
            else:
                energy = torch.cat([energy, torch.zeros(T - len(energy))])
        
        # 随机截取 (训练时)
        if T > self.max_frames:
            start = random.randint(0, T - self.max_frames)
            end = start + self.max_frames
            
            latent = latent[start:end]
            frame_phoneme_ids = frame_phoneme_ids[start:end]
            pitch = pitch[start:end]
            energy = energy[start:end]
            
            # 重新计算截取后的 phoneme-level 数据
            phoneme_ids, phoneme_durations = self._extract_phoneme_level(
                frame_phoneme_ids, phoneme_ids
            )
        
        # 计算 frame_duration 和 frame_position (v4.7)
        T = latent.shape[0]
        frame_duration, frame_position = self._compute_frame_duration_position(phoneme_durations, T)
        
        return {
            "id": item["id"],
            "latent": latent,                       # [T, 512]
            "phoneme_ids": phoneme_ids,             # [N_phone]
            "phoneme_durations": phoneme_durations, # [N_phone]
            "frame_phoneme_ids": frame_phoneme_ids, # [T]
            "pitch": pitch,                         # [T]
            "energy": energy,                       # [T]
            "frame_duration": frame_duration,       # [T] v4.7: 每帧所属 phoneme 的总时长
            "frame_position": frame_position,       # [T] v4.7: 每帧在 phoneme 内的相对位置
        }
    
    def _expand_phonemes(
        self,
        phoneme_ids: torch.Tensor,
        durations: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """将 phoneme-level 展开为 frame-level"""
        frame_ids = []
        for pid, dur in zip(phoneme_ids.tolist(), durations.tolist()):
            frame_ids.extend([pid] * int(dur))
        
        frame_ids = torch.tensor(frame_ids, dtype=torch.long)
        
        # 调整长度
        if len(frame_ids) > target_len:
            frame_ids = frame_ids[:target_len]
        elif len(frame_ids) < target_len:
            frame_ids = torch.cat([
                frame_ids,
                torch.full((target_len - len(frame_ids),), frame_ids[-1].item(), dtype=torch.long)
            ])
        
        return frame_ids

    def _normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Normalize [T, C] latent with scalar or per-channel stats."""
        if isinstance(self.latent_std, torch.Tensor):
            mean = self.latent_mean.to(device=latent.device, dtype=latent.dtype)
            std = self.latent_std.to(device=latent.device, dtype=latent.dtype).clamp_min(1e-6)
            return (latent - mean) / std
        if self.latent_std > 0:
            return (latent - self.latent_mean) / self.latent_std
        return latent

    def _to_time_channel(self, latent: torch.Tensor) -> torch.Tensor:
        """Convert latent saved as [C, T] or [T, C] to [T, C]."""
        if latent.dim() != 2:
            raise ValueError(f"expected 2D latent, got shape {tuple(latent.shape)}")
        if self.latent_dim is not None:
            if latent.shape[0] == self.latent_dim:
                return latent.T
            if latent.shape[1] == self.latent_dim:
                return latent
        # V16 AE sequences usually have many more frames than channels.
        return latent.T if latent.shape[0] < latent.shape[1] else latent
    
    def _compute_frame_duration_position(
        self,
        phoneme_durations: torch.Tensor,
        target_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算帧级的 duration 和 position
        
        Args:
            phoneme_durations: [N_phonemes] 每个 phoneme 的帧数
            target_len: 目标帧数
            
        Returns:
            frame_duration: [T] 每帧所属 phoneme 的总时长
            frame_position: [T] 每帧在 phoneme 内的相对位置 [0, 1]
        """
        frame_duration = []
        frame_position = []
        
        for dur in phoneme_durations.tolist():
            dur = int(dur)
            for i in range(dur):
                frame_duration.append(dur)
                # 相对位置: 0 = 开始, 1 = 结束
                frame_position.append(i / max(dur - 1, 1))
        
        frame_duration = torch.tensor(frame_duration, dtype=torch.float32)
        frame_position = torch.tensor(frame_position, dtype=torch.float32)
        
        # 调整长度
        if len(frame_duration) > target_len:
            frame_duration = frame_duration[:target_len]
            frame_position = frame_position[:target_len]
        elif len(frame_duration) < target_len:
            pad_len = target_len - len(frame_duration)
            # 用最后一个值填充
            last_dur = frame_duration[-1] if len(frame_duration) > 0 else 1.0
            frame_duration = torch.cat([frame_duration, torch.full((pad_len,), last_dur)])
            frame_position = torch.cat([frame_position, torch.ones(pad_len)])  # 填充为 1.0 (结束位置)
        
        return frame_duration, frame_position
    
    def _extract_phoneme_level(
        self,
        frame_phoneme_ids: torch.Tensor,
        original_phoneme_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从截取后的 frame-level 数据提取 phoneme-level"""
        # 找出连续的 phoneme 段
        phoneme_ids = []
        durations = []
        
        if len(frame_phoneme_ids) == 0:
            return torch.tensor([1], dtype=torch.long), torch.tensor([1.0])
        
        current_id = frame_phoneme_ids[0].item()
        current_dur = 1
        
        for i in range(1, len(frame_phoneme_ids)):
            if frame_phoneme_ids[i].item() == current_id:
                current_dur += 1
            else:
                phoneme_ids.append(current_id)
                durations.append(current_dur)
                current_id = frame_phoneme_ids[i].item()
                current_dur = 1
        
        # 最后一个
        phoneme_ids.append(current_id)
        durations.append(current_dur)
        
        return (
            torch.tensor(phoneme_ids, dtype=torch.long),
            torch.tensor(durations, dtype=torch.float),
        )


def collate_fn_v3(batch: list[dict]) -> dict:
    """Collate function for v3 dataset"""
    # 快速路径: 如果所有样本长度相同 (预加载模式)，直接 stack
    first_T = batch[0]["latent"].shape[0]
    all_same_length = all(item["latent"].shape[0] == first_T for item in batch)
    
    if all_same_length:
        # 预加载模式: 所有数据已 pad 到相同长度，直接 stack
        device = batch[0]["latent"].device
        return {
            "ids": [item["id"] for item in batch],
            "latent": torch.stack([item["latent"] for item in batch]),
            "phoneme_ids": torch.nn.utils.rnn.pad_sequence(
                [item["phoneme_ids"] for item in batch], batch_first=True, padding_value=0
            ),
            "phoneme_durations": torch.nn.utils.rnn.pad_sequence(
                [item["phoneme_durations"] for item in batch], batch_first=True, padding_value=0
            ),
            "frame_phoneme_ids": torch.stack([item["frame_phoneme_ids"] for item in batch]),
            "pitch": torch.stack([item["pitch"] for item in batch]),
            "energy": torch.stack([item["energy"] for item in batch]),
            "frame_duration": torch.stack([item["frame_duration"] for item in batch]),
            "frame_position": torch.stack([item["frame_position"] for item in batch]),
            "frame_lengths": torch.tensor([first_T] * len(batch), device=device),
            "phoneme_lengths": torch.tensor([len(item["phoneme_ids"]) for item in batch], device=device),
        }
    
    # 慢速路径: 需要 padding (非预加载模式)
    max_frames = max(item["latent"].shape[0] for item in batch)
    max_phonemes = max(len(item["phoneme_ids"]) for item in batch)
    
    latents = []
    phoneme_ids_list = []
    phoneme_durations_list = []
    frame_phoneme_ids_list = []
    pitches = []
    energies = []
    frame_durations = []  # v4.7
    frame_positions = []  # v4.7
    frame_lengths = []
    phoneme_lengths = []
    ids = []
    
    for item in batch:
        T = item["latent"].shape[0]
        latent_dim = item["latent"].shape[1]
        N = len(item["phoneme_ids"])
        
        frame_lengths.append(T)
        phoneme_lengths.append(N)
        ids.append(item["id"])
        
        # Pad frame-level (确保在同一设备上)
        device = item["latent"].device
        pad_frames = max_frames - T
        if pad_frames > 0:
            latent = torch.cat([item["latent"], torch.zeros(pad_frames, latent_dim, device=device)], dim=0)
            frame_phone = torch.cat([item["frame_phoneme_ids"], torch.zeros(pad_frames, dtype=torch.long, device=device)])
            pitch = torch.cat([item["pitch"], torch.zeros(pad_frames, device=device)])
            energy = torch.cat([item["energy"], torch.zeros(pad_frames, device=device)])
            # v4.7: frame_duration 和 frame_position
            frame_dur = torch.cat([item["frame_duration"], torch.ones(pad_frames, device=device)])
            frame_pos = torch.cat([item["frame_position"], torch.ones(pad_frames, device=device)])
        else:
            latent = item["latent"]
            frame_phone = item["frame_phoneme_ids"]
            pitch = item["pitch"]
            energy = item["energy"]
            frame_dur = item["frame_duration"]
            frame_pos = item["frame_position"]
        
        # Pad phoneme-level
        pad_phonemes = max_phonemes - N
        if pad_phonemes > 0:
            phone_device = item["phoneme_ids"].device
            phone_ids = torch.cat([item["phoneme_ids"], torch.zeros(pad_phonemes, dtype=torch.long, device=phone_device)])
            phone_durs = torch.cat([item["phoneme_durations"], torch.zeros(pad_phonemes, device=phone_device)])
        else:
            phone_ids = item["phoneme_ids"]
            phone_durs = item["phoneme_durations"]
        
        latents.append(latent)
        phoneme_ids_list.append(phone_ids)
        phoneme_durations_list.append(phone_durs)
        frame_phoneme_ids_list.append(frame_phone)
        pitches.append(pitch)
        energies.append(energy)
        frame_durations.append(frame_dur)
        frame_positions.append(frame_pos)
    
    return {
        "ids": ids,
        "latent": torch.stack(latents),                     # [B, T, 512]
        "phoneme_ids": torch.stack(phoneme_ids_list),       # [B, N_phone]
        "phoneme_durations": torch.stack(phoneme_durations_list),  # [B, N_phone]
        "frame_phoneme_ids": torch.stack(frame_phoneme_ids_list),  # [B, T]
        "pitch": torch.stack(pitches),                      # [B, T]
        "energy": torch.stack(energies),                    # [B, T]
        "frame_duration": torch.stack(frame_durations),     # [B, T] v4.7
        "frame_position": torch.stack(frame_positions),     # [B, T] v4.7
        "frame_lengths": torch.tensor(frame_lengths),       # [B]
        "phoneme_lengths": torch.tensor(phoneme_lengths),   # [B]
    }


def get_dataloader_v3(
    manifest_path: str,
    batch_size: int = 32,
    max_frames: int = 256,
    min_frames: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    latent_stats_path: Optional[str] = None,
    preload_latents: bool = False,
    preload_to_gpu: bool = False,
) -> DataLoader:
    """创建 v3 DataLoader"""
    dataset = TTSDatasetV3(
        manifest_path=manifest_path,
        max_frames=max_frames,
        min_frames=min_frames,
        latent_stats_path=latent_stats_path,
        preload_latents=preload_latents,
        preload_to_gpu=preload_to_gpu,
    )
    
    # 如果预加载到 GPU，不需要 num_workers
    if preload_to_gpu:
        num_workers = 0
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_v3,
        pin_memory=not preload_to_gpu,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def test_dataset_v3():
    """测试 v3 数据集"""
    dataset = TTSDatasetV3(
        manifest_path="data/ljspeech_duration_asr/train.json",
        latent_stats_path="data/ljspeech_duration_asr/latent_stats.json",
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    item = dataset[0]
    print(f"\nSample 0:")
    print(f"  latent: {item['latent'].shape}")
    print(f"  phoneme_ids: {item['phoneme_ids'].shape}")
    print(f"  phoneme_durations: {item['phoneme_durations'].shape}")
    print(f"  frame_phoneme_ids: {item['frame_phoneme_ids'].shape}")
    print(f"  pitch: {item['pitch'].shape}")
    print(f"  energy: {item['energy'].shape}")
    print(f"  sum(durations): {item['phoneme_durations'].sum():.0f}")
    
    # 测试 DataLoader
    loader = get_dataloader_v3(
        manifest_path="data/ljspeech_duration_asr/train.json",
        latent_stats_path="data/ljspeech_duration_asr/latent_stats.json",
        batch_size=4,
    )
    
    batch = next(iter(loader))
    print(f"\nBatch:")
    print(f"  latent: {batch['latent'].shape}")
    print(f"  phoneme_ids: {batch['phoneme_ids'].shape}")
    print(f"  phoneme_durations: {batch['phoneme_durations'].shape}")
    print(f"  frame_phoneme_ids: {batch['frame_phoneme_ids'].shape}")
    print(f"  pitch: {batch['pitch'].shape}")
    print(f"  energy: {batch['energy'].shape}")
    print(f"  frame_lengths: {batch['frame_lengths']}")
    print(f"  phoneme_lengths: {batch['phoneme_lengths']}")
    
    print("\nTest passed!")


if __name__ == "__main__":
    test_dataset_v3()
