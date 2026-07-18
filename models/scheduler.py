"""Noise schedulers for latent-diffusion SR.

``DDPMScheduler``  — forward noising for training (ε-prediction).
``DDIMScheduler``  — deterministic sampler for inference (30 steps default,
                     10 in ``--fast`` mode).  Ported from the portfolio's
                     ``Vision/StableDiffusion/SD_Model.py`` (same SD 1.x
                     scaled-linear β schedule) with the cosine-schedule option
                     used by Upscale-SR.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# β schedules
# ─────────────────────────────────────────────────────────────────────────────
def _betas(steps: int, beta_start: float, beta_end: float, schedule: str) -> torch.Tensor:
    if schedule == "scaled_linear":
        # SD 1.x: interpolate √β linearly, then square.  Concentrates small β
        # near t=0 for fine-detail preservation.
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, steps) ** 2
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, steps)
    if schedule == "cosine":
        # Nichol & Dhariwal cosine schedule (nicer for SR's short horizons).
        s = 0.008
        f = lambda t: math.cos((t / steps + s) / (1 + s) * math.pi / 2) ** 2
        betas = []
        for i in range(steps):
            t1 = f(i) / f(0)
            t2 = f(i + 1) / f(0)
            betas.append(max(1e-5, 1 - t2 / t1))
        return torch.tensor(betas, dtype=torch.float32)
    raise ValueError(f"unknown schedule '{schedule}'")


# ─────────────────────────────────────────────────────────────────────────────
# DDPM — training-time forward noising
# ─────────────────────────────────────────────────────────────────────────────
class DDPMScheduler:
    """ε-prediction forward noising: z_t = √ᾱ_t·z_0 + √(1−ᾱ_t)·ε."""

    def __init__(self, steps: int = 1000, beta_start: float = 0.00085,
                 beta_end: float = 0.012, schedule: str = "scaled_linear"):
        self.num_train_timesteps = steps
        betas = _betas(steps, beta_start, beta_end, schedule)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()

    def add_noise(self, x: torch.Tensor, t: torch.Tensor,
                  noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x)
        device = x.device
        sa = self.sqrt_alphas_cumprod.to(device)[t].reshape(-1, 1, 1, 1)
        ss = self.sqrt_one_minus_alphas_cumprod.to(device)[t].reshape(-1, 1, 1, 1)
        return sa * x + ss * noise, noise

    def to(self, device):
        for k in ("betas", "alphas", "alphas_cumprod",
                  "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"):
            setattr(self, k, getattr(self, k).to(device))
        return self


# ─────────────────────────────────────────────────────────────────────────────
# DDIM — deterministic inference sampler
# ─────────────────────────────────────────────────────────────────────────────
class DDIMScheduler:
    """DDIM (Song et al., 2020).  eta=0 → fully deterministic.

    Update at each step:
        x̂_0 = (x_t − √(1−ᾱ_t)·ε_θ) / √ᾱ_t
        x_{t-1} = √ᾱ_{t-1}·x̂_0 + √(1−ᾱ_{t-1})·ε_θ
    """

    def __init__(self, steps: int = 1000, beta_start: float = 0.00085,
                 beta_end: float = 0.012, schedule: str = "scaled_linear"):
        self.num_train_timesteps = steps
        betas = _betas(steps, beta_start, beta_end, schedule)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, 0)
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None

    def to(self, device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def set_timesteps(self, num_steps: int, device: torch.device):
        self.num_inference_steps = num_steps
        ratio = self.num_train_timesteps // num_steps
        self.timesteps = (torch.arange(0, num_steps) * ratio).flip(0).long().to(device)

    def step(self, noise_pred: torch.Tensor, t, x_t: torch.Tensor,
             eta: float = 0.0) -> torch.Tensor:
        assert self.num_inference_steps is not None, "call set_timesteps() first"
        device = x_t.device
        t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        step_size = self.num_train_timesteps // self.num_inference_steps
        prev_t = t_int - step_size
        alpha_t = self.alphas_cumprod[t_int].to(device)
        alpha_prev = (self.alphas_cumprod[prev_t].to(device)
                      if prev_t >= 0 else torch.ones(1, device=device))
        pred_x0 = (x_t - (1.0 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        pred_x0 = pred_x0.clamp(-1.0, 1.0)
        dir_xt = (1.0 - alpha_prev).sqrt() * noise_pred
        x_prev = alpha_prev.sqrt() * pred_x0 + dir_xt
        if eta > 0.0:
            sigma_t = eta * (((1.0 - alpha_prev) / (1.0 - alpha_t)) *
                             (1.0 - alpha_t / alpha_prev)).clamp(min=0.0).sqrt()
            x_prev = x_prev + sigma_t * torch.randn_like(x_t)
        return x_prev


def build_ddpm(cfg: dict) -> DDPMScheduler:
    s = cfg.get("scheduler", cfg)
    return DDPMScheduler(steps=s.get("train_steps", 1000),
                         beta_start=s.get("beta_start", 0.00085),
                         beta_end=s.get("beta_end", 0.012),
                         schedule=s.get("schedule", "scaled_linear"))


def build_ddim(cfg: dict) -> DDIMScheduler:
    s = cfg.get("scheduler", cfg)
    return DDIMScheduler(steps=s.get("train_steps", 1000),
                         beta_start=s.get("beta_start", 0.00085),
                         beta_end=s.get("beta_end", 0.012),
                         schedule=s.get("schedule", "scaled_linear"))