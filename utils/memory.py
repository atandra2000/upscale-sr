"""channels_last helper (Blackwell sm_120) + VRAM sanity estimation."""
from __future__ import annotations

import torch
import torch.nn as nn


def apply_channels_last(model: nn.Module) -> nn.Module:
    """Move conv-heavy modules to channels_last before the first forward.

    On Blackwell (sm_120) the channels_last memory format lets cuDNN pick the
    NHWC-fused conv kernels, which is the single biggest conv speedup on the
    5090.  Must be applied BEFORE the first forward so the layout sticks.
    (CLAUDE.md §1.)
    """
    return model.to(memory_format=torch.channels_last)


def gpu_vram_gb(device: torch.device) -> float:
    """Reserved VRAM in GB (rank-local)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved(device) / 1024**3