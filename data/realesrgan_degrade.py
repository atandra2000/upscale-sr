"""Real-ESRGAN stochastic degradation (arXiv:2107.10833), applied on-the-fly per crop.

First-order: blur → resize → noise → JPEG → sinc.  Second-order shuffles the
stage order with prob ``shuffle_prob`` and inserts a second blur with prob
``second_blur_prob``.  Deterministic given a seed (RNG save/seed/restore).
"""
from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_kernel2d(sigma: float, kernel_size: int, device, dtype) -> torch.Tensor:
    """Isotropic 2-D Gaussian kernel (normalised), as (k, 1, s, s) conv weight."""
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    g1d = torch.exp(-(ax ** 2) / (2 * sigma * sigma + 1e-12))
    g1d = g1d / g1d.sum()
    k2d = g1d[:, None] * g1d[None, :]
    return k2d[None, None].expand(3, 1, -1, -1).contiguous()


def _gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """img: (B,3,H,W) in [0,1] float32 → blurred (B,3,H,W)."""
    if sigma < 0.05:
        return img
    ks = max(3, int(2 * round(3 * sigma) + 1))
    if ks % 2 == 0:
        ks += 1
    w = _gaussian_kernel2d(sigma, ks, img.device, img.dtype)
    return F.conv2d(img, w, padding=ks // 2, groups=3)


def _aniso_blur(img: torch.Tensor, sigma_x: float, sigma_y: float) -> torch.Tensor:
    """Anisotropic Gaussian (different σ in x and y)."""
    if sigma_x < 0.05 and sigma_y < 0.05:
        return img
    kx = max(3, int(2 * round(3 * sigma_x) + 1)); kx += (kx % 2 == 0)
    ky = max(3, int(2 * round(3 * sigma_y) + 1)); ky += (ky % 2 == 0)
    ax = kx // 2; ay = ky // 2
    dev, dt = img.device, img.dtype
    gx = torch.exp(-(torch.arange(kx, device=dev, dtype=dt) - ax) ** 2 /
                   (2 * sigma_x * sigma_x + 1e-12)); gx = gx / gx.sum()
    gy = torch.exp(-(torch.arange(ky, device=dev, dtype=dt) - ay) ** 2 /
                   (2 * sigma_y * sigma_y + 1e-12)); gy = gy / gy.sum()
    k2d = gx[:, None] * gy[None, :]                       # (kx, ky) = (kh, kw)
    w = k2d[None, None].expand(3, 1, -1, -1).contiguous()
    return F.conv2d(img, w, padding=(ax, ay), groups=3)


def _sinc_kernel(cutoff: float, kernel_size: int, device, dtype) -> torch.Tensor:
    """1-D sinc filter kernel (jinc-free, separable). cutoff in (0, 1]."""
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    x = cutoff * ax
    sinc = torch.where(x == 0, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x))
    win = torch.hamming_window(kernel_size, periodic=False, device=device, dtype=dtype)
    k = sinc * win
    k = k / k.sum()
    return k


def _sinc_filter(img: torch.Tensor, cutoff: float, kernel_size: int = 21) -> torch.Tensor:
    """Apply a separable sinc low-pass (the Real-ESRGAN "sinc" stage)."""
    if cutoff >= 0.999:
        return img
    k = _sinc_kernel(cutoff, kernel_size, img.device, img.dtype)
    w2d = k[:, None] * k[None, :]
    w = w2d[None, None].expand(3, 1, -1, -1).contiguous()
    return F.conv2d(img, w, padding=kernel_size // 2, groups=3)


def _jpeg_compress(img: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG compress a (B,3,H,W) [0,1] tensor → (B,3,H,W) [0,1] via torchvision or PIL fallback."""
    try:
        from torchvision.io import encode_jpeg, decode_jpeg
        out = []
        for b in range(img.shape[0]):
            arr = (img[b].clamp(0, 1) * 255).round().to(torch.uint8)
            enc = encode_jpeg(arr, int(quality))
            dec = decode_jpeg(enc).to(img.dtype) / 255.0
            out.append(dec.to(img.device, img.dtype))
        return torch.stack(out, dim=0)
    except ImportError:
        import io
        import numpy as np
        from PIL import Image
        out = []
        for b in range(img.shape[0]):
            arr = (img[b].clamp(0, 1) * 255).round().to(torch.uint8).cpu().permute(1, 2, 0).numpy()
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
            buf.seek(0)
            dec = np.array(Image.open(buf), dtype=np.float32) / 255.0
            dec_t = torch.from_numpy(dec).permute(2, 0, 1).to(img.device, img.dtype)
            out.append(dec_t)
        return torch.stack(out, dim=0)


def _add_noise(img: torch.Tensor, sigma: float, poisson: bool, rng) -> torch.Tensor:
    """Gaussian (σ) or Poisson noise.  σ in [0, 55] (Real-ESRGAN range)."""
    if sigma <= 0.01 and not poisson:
        return img
    if poisson:
        lam = max(1e-3, sigma * sigma)
        noisy = torch.poisson(img.clamp(min=0.0) * lam) / lam
        return noisy.clamp(0, 1)
    return (img + torch.randn_like(img) * sigma).clamp(0, 1)


def _resize_stage(img: torch.Tensor, rng, scale: float) -> torch.Tensor:
    """Resize to (H/s, W/s) then back to (H, W) with a random interpolation."""
    if abs(scale - 1.0) < 1e-3:
        return img
    H, W = img.shape[-2:]
    nh, nw = max(1, int(round(H / scale))), max(1, int(round(W / scale)))
    modes = ["bilinear", "bicubic", "area"]
    mode = rng.choice(modes)
    down = F.interpolate(img, size=(nh, nw), mode=mode if mode != "area" else "area",
                        align_corners=None if mode != "area" else None)
    align = None if mode == "nearest" or mode == "area" else False
    return F.interpolate(down, size=(H, W), mode=mode if mode != "area" else "area",
                         align_corners=align)


class RealESRGANDegrader:
    """Stochastic Real-ESRGAN degradation.  ``degrade(hr, seed)`` → LR image.

    ``scale`` here is the *intermediate* resize factor, not the final SR scale.
    """

    def __init__(self, cfg: dict):
        d = cfg.get("degradation", cfg)
        self.blur_sigma = tuple(d.get("blur_sigma", [0.2, 3.0]))
        self.noise_sigma = tuple(d.get("noise_sigma", [1.0, 30.0]))
        self.jpeg_quality = tuple(d.get("jpeg_quality", [30, 95]))
        self.sinc_prob = float(d.get("sinc_prob", 0.15))
        self.resize_prob = float(d.get("resize_prob", 0.25))
        self.second_blur_prob = float(d.get("second_blur_prob", 0.25))
        self.shuffle_prob = float(d.get("shuffle_prob", 0.15))

    def _rng(self, seed: int) -> random.Random:
        return random.Random(int(seed))

    @torch.no_grad()
    def degrade(self, hr: torch.Tensor, seed: int) -> torch.Tensor:
        """Apply the stochastic pipeline. Deterministic given ``seed``.

        Saves + locally seeds the torch/numpy global RNGs and restores them
        at the end so the pipeline is reproducible without disturbing the
        caller's RNG state.
        """
        rng = self._rng(seed)
        torch_state = torch.get_rng_state()
        torch_cuda_state = (torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else None)
        np_state = np.random.get_state()
        torch.manual_seed(int(seed)); np.random.seed(int(seed) % (2**32 - 1))
        try:
            img = self._degrade_body(hr, rng)
        finally:
            torch.set_rng_state(torch_state)
            if torch_cuda_state is not None:
                torch.cuda.set_rng_state_all(torch_cuda_state)
            np.random.set_state(np_state)
        return img.clamp(0, 1)

    def _degrade_body(self, img: torch.Tensor, rng: random.Random) -> torch.Tensor:
        sigma1 = rng.uniform(*self.blur_sigma)
        img = _gaussian_blur(img, sigma1)

        # Stage-order shuffle (Real-ESRGAN "second-order" degradation).
        # When shuffle_prob hits, the four stages are reordered; otherwise
        # they run in the canonical order: resize → noise → JPEG → sinc.
        stages = [
            (1, lambda: self._stage_resize(img, rng)),
            (2, lambda: self._stage_noise(img, rng)),
            (3, lambda: self._stage_jpeg(img, rng)),
            (4, lambda: self._stage_sinc(img, rng)),
        ]
        if rng.random() < self.shuffle_prob:
            rng.shuffle(stages)
            second_blur = rng.random() < self.second_blur_prob
        else:
            second_blur = False
        for _, fn in stages:
            img = fn()

        if second_blur:
            sx = rng.uniform(*self.blur_sigma)
            sy = rng.uniform(*self.blur_sigma)
            img = _aniso_blur(img, sx, sy)

        return img.clamp(0, 1)

    def _stage_resize(self, img: torch.Tensor, rng: random.Random) -> torch.Tensor:
        if rng.random() < self.resize_prob:
            scale = rng.uniform(0.15, 1.5)
            return _resize_stage(img, rng, scale)
        return img

    def _stage_noise(self, img: torch.Tensor, rng: random.Random) -> torch.Tensor:
        sigma_n = rng.uniform(*self.noise_sigma)
        poisson = rng.random() < 0.5
        return _add_noise(img, sigma_n / 255.0, poisson, rng)

    def _stage_jpeg(self, img: torch.Tensor, rng: random.Random) -> torch.Tensor:
        q = rng.randint(*self.jpeg_quality)
        return _jpeg_compress(img, q)

    def _stage_sinc(self, img: torch.Tensor, rng: random.Random) -> torch.Tensor:
        if rng.random() < self.sinc_prob:
            cutoff = rng.uniform(0.6, 0.99)
            return _sinc_filter(img, cutoff)
        return img


def make_lr_pair(hr_patch: torch.Tensor, degrader: RealESRGANDegrader,
                 scale: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """HR patch (B,3,H,W) in [0,1] → (LR, HR) at the SR scale.  HR target is
    the original clean patch; LR is degraded + bicubic-downsampled by ``scale``."""
    degraded = degrader.degrade(hr_patch, seed=seed)
    H, W = hr_patch.shape[-2:]
    nh, nw = H // scale, W // scale
    lr = F.interpolate(degraded, size=(nh, nw), mode="bicubic",
                       align_corners=False).clamp(0, 1)
    return lr, hr_patch