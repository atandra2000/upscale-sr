"""Memory helpers — channels_last on Ampere+/Blackwell only.

``apply_channels_last`` is the single source of truth for the channels_last
memory format.  On pre-Ampere (sm_75) cuDNN cannot dispatch depthwise convs
under channels_last + BF16 autocast, so we silently skip the conversion.
"""
from __future__ import annotations

import torch


def gpu_vram_gb(device: torch.device) -> float:
    """Reserved VRAM in GB (rank-local)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved(device) / 1024**3


def cuda_compute_capability(device: torch.device | None = None) -> tuple[int, int]:
    """Return (major, minor) of the active CUDA device.  Defaults to cuda:0."""
    if not torch.cuda.is_available():
        return (0, 0)
    idx = device.index if device is not None and device.index is not None else 0
    return torch.cuda.get_device_capability(idx)


def channels_last_supported(device: torch.device | None = None) -> bool:
    """True if the active CUDA device supports the channels_last + autocast
    depthwise-conv path (Ampere / sm_80+).  False on pre-Ampere or CPU."""
    if not torch.cuda.is_available():
        return False
    return cuda_compute_capability(device)[0] >= 8


def apply_channels_last(*modules: torch.nn.Module,
                        device: torch.device | None = None) -> None:
    """Convert each module to ``channels_last`` memory format.

    No-op on CPU, on pre-Ampere CUDA, or when the format is already applied.
    """
    if not channels_last_supported(device):
        return
    for m in modules:
        m.to(memory_format=torch.channels_last)
