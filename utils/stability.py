"""Training-stability helpers: EMA, grad-clip, NaN guard."""
from __future__ import annotations

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# EMA (GPU-resident; 32 GB/GPU is plenty — no CPU round-trip). Ported from the
# portfolio's SD_Train.py with the warmup formula d=min(decay,(1+s)/(10+s)).
# ─────────────────────────────────────────────────────────────────────────────
class EMA:
    """Exponential moving average of trainable weights.

    Shadow weights are kept on-GPU for zero-latency updates.  The warmup
    formula ramps EMA gently at the start so early noisy updates don't
    pollute the shadow copy.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        self.decay = decay
        self.device = device or next(model.parameters()).device
        self.step_count = 0
        self.shadow = {
            n: p.detach().clone().to(self.device)
            for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step_count += 1
        d = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].lerp_(p.detach().to(self.device), 1.0 - d)

    def apply_shadow(self, model: nn.Module) -> dict:
        """Swap live weights → shadow. Returns a backup dict to restore later."""
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model: nn.Module, backup: dict) -> None:
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

    def state_dict(self) -> dict:
        return {
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
            "step_count": self.step_count,
            "decay": self.decay,
        }

    def load_state_dict(self, state: dict, device=None) -> None:
        dev = device or self.device
        self.shadow = {k: v.to(dev) for k, v in state["shadow"].items()}
        self.step_count = state["step_count"]
        self.decay = state.get("decay", self.decay)


# ─────────────────────────────────────────────────────────────────────────────
# Grad-clip with a NaN/Inf guard. Returns the pre-clip grad-norm (for logging).
# ─────────────────────────────────────────────────────────────────────────────
def clip_and_guard(params, max_norm: float = 1.0, nan_guard: bool = True) -> float:
    """Clip gradients to ``max_norm``; if ``nan_guard`` and any grad is
    NaN/Inf, zero ALL grads for this step and skip the optimizer step.

    Returns the grad-norm measured before clipping (NaN-safe: NaN→report 0).
    """
    if nan_guard:
        bad = False
        for p in params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad = True
                break
        if bad:
            for p in params:
                if p.grad is not None:
                    p.grad.detach_().zero_()
            return 0.0
    grad_norm = torch.nn.utils.clip_grad_norm_(list(params), max_norm=max_norm)
    return float(grad_norm) if torch.isfinite(torch.as_tensor(grad_norm)) else 0.0