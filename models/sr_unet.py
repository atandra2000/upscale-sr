"""Latent-diffusion SR U-Net with FlashAttention-2 (DESIGN §1.2).

Conditioning is concat-based (9-channel input `[lr_lat_up | z_t | lr_mask]`).
FA2 self-attention on Ampere/Blackwell; sdpa flash-backend fallback on others.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ── FlashAttention-2 availability probe (done once) ──────────────────────────
try:
    from flash_attn import flash_attn_func as _fa2_func  # type: ignore
    _HAS_FA2 = True
except Exception:  # pragma: no cover — env-dependent
    _fa2_func = None  # type: ignore
    _HAS_FA2 = False


def fa2_available() -> bool:
    return _HAS_FA2


# ═══════════════════════════════════════════════════════════════════════════════
# FlashAttention-2 self-attention block for 2D feature maps
# ═══════════════════════════════════════════════════════════════════════════════
class FA2SelfAttention(nn.Module):
    """Multi-head self-attention on (B, C, H, W) via FA2 (sdpa flash-backend fallback)."""

    def __init__(self, channels: int, num_heads: int = 16):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels={channels} not divisible by num_heads={num_heads}")
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _fa2(self, q, k, v, B, H, W, C):
        # flash_attn_func expects (B, S, H, D) with D = head_dim
        q = q.view(B, H * W, self.num_heads, self.head_dim)
        k = k.view(B, H * W, self.num_heads, self.head_dim)
        v = v.view(B, H * W, self.num_heads, self.head_dim)
        out = _fa2_func(q, k, v, softmax_scale=self.scale, causal=False)
        return out.view(B, H * W, C).transpose(1, 2).reshape(B, C, H, W)

    def _sdpa(self, q, k, v, B, H, W, C):
        # (B, num_heads, head_dim, H*W) → (B, num_heads, H*W, head_dim)
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return out.transpose(2, 3).reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)
        if _HAS_FA2 and x.is_cuda:
            out = self._fa2(q, k, v, b, h, w, c)
        else:
            out = self._sdpa(q, k, v, b, h, w, c)
        return self.proj(out) + x  # residual


class SRResBlock(nn.Module):
    """GN → SiLU → Conv + time-bias → GN → SiLU → Conv + skip. Optional attn."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int,
                 use_attn: bool = False, heads: int = 16, grad_ckpt: bool = False):
        super().__init__()
        self.use_attn = use_attn
        self.grad_ckpt = grad_ckpt
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        if use_attn:
            self.attn = FA2SelfAttention(out_ch, heads)
        # zero-init final conv → identity residual at init (stable)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def _forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x))) + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        if self.use_attn:
            h = self.attn(h)
        return h + self.skip(x)

    def forward(self, x, t_emb):
        if self.grad_ckpt and self.training:
            return checkpoint(self._forward, x, t_emb, use_reentrant=False)
        return self._forward(x, t_emb)


class SRUNet(nn.Module):
    """Conditional denoising U-Net in VAE latent space: (B,9,H/8,W/8) → (B,4,H/8,W/8)."""

    def __init__(
        self,
        in_ch: int = 9,
        out_ch: int = 4,
        base_ch: int = 256,
        ch_mults: tuple = (1, 2, 4, 4),
        res_blks: int = 2,
        attn_levels: tuple = (1, 2, 3),
        heads: int = 16,
        t_dim: int = 256,
        grad_ckpt: bool = False,
    ):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.t_dim = t_dim
        from .cond import LRUpConv
        self.lr_up = LRUpConv(channels=4)
        self.t_emb = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim),
        )
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.downs = nn.ModuleList()
        ch_list, cur = [], base_ch
        for i, mult in enumerate(ch_mults):
            nxt = base_ch * mult
            for _ in range(res_blks):
                blk = SRResBlock(cur, nxt, t_dim, use_attn=(i in attn_levels),
                                 heads=heads, grad_ckpt=grad_ckpt)
                self.downs.append(blk)
                cur, _ = nxt, ch_list.append(nxt)
            if i < len(ch_mults) - 1:
                self.downs.append(nn.Conv2d(cur, cur, 3, stride=2, padding=1))

        self.mid = nn.ModuleList([
            SRResBlock(cur, cur, t_dim, use_attn=True, heads=heads, grad_ckpt=grad_ckpt),
            SRResBlock(cur, cur, t_dim, use_attn=False, heads=heads, grad_ckpt=grad_ckpt),
        ])

        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            nxt = base_ch * mult
            for _ in range(res_blks):
                skip_ch = ch_list.pop()
                blk = SRResBlock(cur + skip_ch, nxt, t_dim,
                                 use_attn=(i in attn_levels),
                                 heads=heads, grad_ckpt=grad_ckpt)
                self.ups.append(blk)
                cur = nxt
            if i > 0:
                self.ups.append(nn.ConvTranspose2d(cur, cur, 4, stride=2, padding=1))

        self.norm_out = nn.GroupNorm(32, base_ch)
        self.conv_out = nn.Conv2d(base_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def enable_gradient_checkpointing(self):
        self.grad_ckpt = True
        for m in self.modules():
            if isinstance(m, SRResBlock):
                m.grad_ckpt = True

    def _sinusoidal_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.t_dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(math.log(10000) / (half - 1))
        )
        angles = t.float()[:, None] * freq[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Args: x (B,9,H,W) noisy latent concat, t (B,) timestep indices.
        Returns predicted noise ε (B,4,H,W)."""
        t_emb = self.t_emb(self._sinusoidal_time_embedding(t))
        h = self.conv_in(x)
        skips = []
        for m in self.downs:
            if isinstance(m, SRResBlock):
                h = m(h, t_emb)
                skips.append(h)
            else:
                h = m(h)
        for m in self.mid:
            h = m(h, t_emb)
        for m in self.ups:
            if isinstance(m, SRResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = m(h, t_emb)
            else:
                h = m(h)
        return self.conv_out(F.silu(self.norm_out(h)))


def build_sr_unet(cfg: dict) -> SRUNet:
    u = cfg.get("sr_unet", cfg)
    return SRUNet(
        in_ch=u.get("in_ch", 9), out_ch=u.get("out_ch", 4),
        base_ch=u.get("base_ch", 256), ch_mults=tuple(u.get("ch_mults", (1, 2, 4, 4))),
        res_blks=u.get("res_blks", 2), attn_levels=tuple(u.get("attn_levels", (1, 2, 3))),
        heads=u.get("heads", 16), t_dim=u.get("t_dim", 256),
        grad_ckpt=u.get("grad_ckpt", True),
    )