"""Loss helpers: differentiable SSIM + lazy LPIPS (with torchvision fallback)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SSIM (differentiable, no external dep) — used by the refiner loss.
# ─────────────────────────────────────────────────────────────────────────────
def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    ax = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma * sigma))
    return g / g.sum()


def _ssim_map(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor,
              data_range: float = 2.0) -> torch.Tensor:
    """Per-pixel SSIM map between x, y (B,C,H,W) in the same range."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    pad = window.shape[-1] // 2
    w = window[None, None].expand(x.shape[1], 1, -1, -1).contiguous()
    mu_x = F.conv2d(x, w, padding=pad, groups=x.shape[1])
    mu_y = F.conv2d(y, w, padding=pad, groups=y.shape[1])
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, w, padding=pad, groups=x.shape[1]) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, padding=pad, groups=y.shape[1]) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=pad, groups=x.shape[1]) - mu_xy
    ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
    return ssim


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11,
              sigma: float = 1.5) -> torch.Tensor:
    """1 - mean SSIM between x and y (B,C,H,W). Lower = more similar."""
    win = _gaussian_window(window_size, sigma, x.device, x.dtype)
    win = win[:, None] * win[None, :]                    # 2-D separable window
    return 1.0 - _ssim_map(x, y, win).mean()


# ─────────────────────────────────────────────────────────────────────────────
# LPIPS — uses the `lpips` package when available; falls back to a VGG-feature
# L2 via torchvision (DESIGN §8: VGG feature extractor mismatch → use
# torchvision.models.vgg16 features).  Both are perceptual losses only.
# ─────────────────────────────────────────────────────────────────────────────
class _LPIPSVGGFallback(torch.nn.Module):
    """L2 distance between VGG16 conv3_3 features of x and y (B,3,H,W) in [-1,1]."""

    def __init__(self, device):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        weights = VGG16_Weights.IMAGENET1K_V1
        vgg = vgg16(weights=weights).features[:16].eval()  # up to conv3_3
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg.to(device)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406],
                                                   device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225],
                                                  device=device).view(1, 3, 1, 1))

    @torch.no_grad()
    def _feat(self, x):
        x = (x * 0.5 + 0.5).clamp(0, 1)            # [-1,1] → [0,1]
        x = (x - self.mean) / self.std
        return self.vgg(x)

    def forward(self, x, y):
        return (self._feat(x) - self._feat(y)).pow(2).mean()


def build_lpips(device) -> torch.nn.Module:
    """Return a perceptual-loss module. Tries `lpips.LPIPS`, else VGG fallback."""
    try:
        import lpips as lpips_pkg  # type: ignore
        loss = lpips_pkg.LPIPS(net="vgg").to(device).eval()
        for p in loss.parameters():
            p.requires_grad_(False)
        return loss
    except Exception:
        return _LPIPSVGGFallback(device)


def lpips_loss_fn(lpips: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Dispatch to `lpips.LPIPS` (expects [-1,1]) or the VGG fallback."""
    if isinstance(lpips, _LPIPSVGGFallback):
        return lpips(x, y)
    with torch.no_grad() if False else torch.enable_grad():
        return lpips(x, y).mean()