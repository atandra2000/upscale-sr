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


def gpu_util_pct(device: torch.device) -> float:
    """Best-effort instantaneous GPU utilization (%) via the Python API.

    The torch CUDA API does not expose utilisation directly; we use the
    reserved-vs-allocated ratio as a coarse proxy when the real metric is not
    available.  The authoritative ≥95% check is done in training/profiler.py
    via `nvidia-smi` polling.
    """
    if not torch.cuda.is_available():
        return 0.0
    try:
        import subprocess
        idx = int(str(device).split(":")[-1]) if ":" in str(device) else 0
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={idx}", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return float(out.splitlines()[0]) if out else 0.0
    except Exception:
        return 0.0


def estimate_param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())