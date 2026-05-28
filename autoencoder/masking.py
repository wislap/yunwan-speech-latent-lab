"""Block masking for masked latent prediction training."""

import torch


class BlockMaskGenerator:
    """Generates block masks for masked latent prediction.

    Produces large contiguous masked blocks to force the predictor to rely on
    global semantic context (phonemes) rather than local interpolation.

    Args:
        mask_ratio: Target fraction of frames to mask (>= 0.5).
        min_block_frames: Minimum contiguous block length.
        max_block_frames: Maximum contiguous block length.
    """

    def __init__(
        self,
        mask_ratio: float = 0.6,
        min_block_frames: int = 400,
        max_block_frames: int = 800,
    ):
        assert 0.5 <= mask_ratio <= 0.95, "mask_ratio must be in [0.5, 0.95]"
        assert min_block_frames > 0
        assert max_block_frames >= min_block_frames
        self.mask_ratio = mask_ratio
        self.min_block_frames = min_block_frames
        self.max_block_frames = max_block_frames

    def generate_mask(self, seq_len: int, batch_size: int = 1) -> torch.Tensor:
        """Generate block mask.

        Args:
            seq_len: Total number of frames in the sequence.
            batch_size: Number of sequences in the batch.

        Returns:
            Boolean tensor [B, T] where True = visible, False = masked.
        """
        masks = []
        for _ in range(batch_size):
            mask = self._generate_single_mask(seq_len)
            masks.append(mask)
        return torch.stack(masks)

    def _generate_single_mask(self, seq_len: int) -> torch.Tensor:
        """Generate a single mask for one sequence."""
        # Start with all visible
        mask = torch.ones(seq_len, dtype=torch.bool)

        # Adjust block sizes if sequence is shorter than min_block_frames
        min_block = min(self.min_block_frames, seq_len // 2)
        max_block = min(self.max_block_frames, seq_len - 1)
        if max_block < min_block:
            max_block = min_block

        target_masked = int(seq_len * self.mask_ratio)
        masked_count = 0

        # Place blocks until we reach target
        attempts = 0
        max_attempts = 100
        while masked_count < target_masked and attempts < max_attempts:
            attempts += 1
            # Random block length
            block_len = torch.randint(min_block, max_block + 1, (1,)).item()
            # Don't exceed target
            remaining = target_masked - masked_count
            block_len = min(block_len, remaining)
            if block_len < min_block and remaining >= min_block:
                block_len = min_block

            # Random start position
            max_start = seq_len - block_len
            if max_start < 0:
                break
            start = torch.randint(0, max_start + 1, (1,)).item()

            # Apply mask
            mask[start : start + block_len] = False
            masked_count = (~mask).sum().item()

        # Ensure at least one frame is visible
        if mask.sum() == 0:
            # Unmask a small segment at the beginning
            unmask_len = max(1, seq_len // 10)
            mask[:unmask_len] = True

        # Ensure mask ratio is at least 50%
        actual_ratio = (~mask).float().mean().item()
        if actual_ratio < 0.5:
            # Mask more frames from the end
            visible_indices = mask.nonzero(as_tuple=True)[0]
            n_to_mask = int(seq_len * 0.5) - (~mask).sum().item()
            if n_to_mask > 0 and len(visible_indices) > 1:
                # Mask from the tail of visible indices
                to_mask = visible_indices[-n_to_mask:]
                mask[to_mask] = False

        return mask
