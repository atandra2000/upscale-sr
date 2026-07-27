"""Noise schedulers for latent-diffusion SR (DDPM train, DDIM/DPM-Solver++ infer)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def _betas(steps: int, beta_start: float, beta_end: float, schedule: str) -> torch.Tensor:
    if schedule == "scaled_linear":
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, steps) ** 2
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, steps)
    if schedule == "cosine":
        s = 0.008
        f = lambda t: math.cos((t / steps + s) / (1 + s) * math.pi / 2) ** 2
        betas = []
        for i in range(steps):
            t1 = f(i) / f(0)
            t2 = f(i + 1) / f(0)
            betas.append(max(1e-5, 1 - t2 / t1))
        return torch.tensor(betas, dtype=torch.float32)
    raise ValueError(f"unknown schedule '{schedule}'")


def _logit_normal_timesteps(num_steps: int, train_timesteps: int,
                             device: torch.device) -> torch.Tensor:
    """Logit-normal sampling schedule, descending (T → 0).  No RNG.

    # ponytail: O(num_steps), add quadratic spacing if you measure a gap.
    """
    if num_steps > train_timesteps:
        raise ValueError(f"num_steps ({num_steps}) > train_timesteps ({train_timesteps})")
    u = torch.linspace(0.0, 1.0, num_steps + 2)[1:-1]
    s = torch.sigmoid(6.0 * (u - 0.5))                  # logit-normal around 0.5
    ts = (s * train_timesteps).long().clamp(0, train_timesteps - 1).unique()
    while ts.numel() < num_steps:                       # backfill on collisions
        extra = torch.linspace(0, train_timesteps - 1, num_steps + 1)[1:-1].long()
        ts = torch.unique(torch.cat([ts, extra]))
    return ts.flip(0).to(device)


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


class DDIMScheduler:
    """DDIM (Song et al., 2020), eta=0 → deterministic."""

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
        self.timesteps = _logit_normal_timesteps(num_steps, self.num_train_timesteps, device)

    def step(self, noise_pred: torch.Tensor, t, x_t: torch.Tensor) -> torch.Tensor:
        """Deterministic (eta=0) DDIM update."""
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


class DPMSolverMultistepScheduler:
    """DPM-Solver++ (2M) multistep deterministic sampler (12–15 steps)."""

    def __init__(self, steps: int = 1000, beta_start: float = 0.00085,
                 beta_end: float = 0.012, schedule: str = "scaled_linear"):
        self.num_train_timesteps = steps
        betas = _betas(steps, beta_start, beta_end, schedule)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, 0)
        self.lambda_t = 0.5 * torch.log(self.alphas_cumprod / (1.0 - self.alphas_cumprod).clamp(min=1e-12))
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None
        self.old_pred_x0: Optional[torch.Tensor] = None
        self.old_h: Optional[float] = None

    def to(self, device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.lambda_t = self.lambda_t.to(device)
        return self

    def set_timesteps(self, num_steps: int, device: torch.device):
        self.num_inference_steps = num_steps
        self.timesteps = _logit_normal_timesteps(num_steps, self.num_train_timesteps, device)
        self.old_pred_x0 = None
        self.old_h = None

    def step(self, noise_pred: torch.Tensor, t, x_t: torch.Tensor) -> torch.Tensor:
        """DPM-Solver++ 2M update step."""
        assert self.num_inference_steps is not None, "call set_timesteps() first"
        device = x_t.device
        t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        step_size = self.num_train_timesteps // self.num_inference_steps
        prev_t = t_int - step_size

        alpha_t = self.alphas_cumprod[t_int].to(device)
        alpha_prev = (self.alphas_cumprod[prev_t].to(device)
                      if prev_t >= 0 else torch.ones(1, device=device))

        lambda_curr = self.lambda_t[t_int].to(device)
        lambda_prev = (self.lambda_t[prev_t].to(device)
                       if prev_t >= 0 else self.lambda_t[0].to(device))
        h = float((lambda_prev - lambda_curr).item())

        pred_x0 = (x_t - (1.0 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        if self.old_pred_x0 is None or prev_t < 0:
            d_i = pred_x0
        else:
            r_i = self.old_h / h if (self.old_h is not None and abs(h) > 1e-8) else 1.0
            d_i = (1.0 + 0.5 / r_i) * pred_x0 - (0.5 / r_i) * self.old_pred_x0

        self.old_pred_x0 = pred_x0
        self.old_h = h

        exp_minus_h = math.exp(-h)
        x_prev = (alpha_prev.sqrt() / alpha_t.sqrt()) * x_t - alpha_prev.sqrt() * (exp_minus_h - 1.0) * d_i
        return x_prev


def build_dpm_solver(cfg: dict) -> DPMSolverMultistepScheduler:
    s = cfg.get("scheduler", cfg)
    return DPMSolverMultistepScheduler(steps=s.get("train_steps", 1000),
                                        beta_start=s.get("beta_start", 0.00085),
                                        beta_end=s.get("beta_end", 0.012),
                                        schedule=s.get("schedule", "scaled_linear"))