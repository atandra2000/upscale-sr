"""LR schedule: linear warmup → cosine decay, with full RNG-safe state."""
from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def build_schedule(
    optimizer: Optimizer,
    total_iters: int,
    warmup_iters: int,
    eta_min_factor: float = 0.01,
) -> SequentialLR:
    """Warmup (linear, 1%→100% of peak lr) then cosine decay to lr·eta_min_factor.

    Mirrors the SD_Train.py schedule but in iteration space (not epoch space).
    """
    warmup_iters = min(max(1, warmup_iters), total_iters // 10)
    decay_iters = max(1, total_iters - warmup_iters)
    return SequentialLR(
        optimizer,
        milestones=[warmup_iters],
        schedulers=[
            LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_iters),
            CosineAnnealingLR(optimizer, T_max=decay_iters,
                               eta_min=optimizer.defaults["lr"] * eta_min_factor),
        ],
    )