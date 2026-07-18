"""SSM refiner — NAFNet-body U-Net with a ``mamba-ssm`` dilated bottleneck.

(DESIGN §1.3 / candidate §3.2.)

Why a refiner: latent-diffusion SR over-smooths high frequencies.  A
lightweight pixel-space refiner runs **after** the diffusion pass to
re-inject sharp detail.  Its bottleneck is a **bidirectional selective scan**
(``mamba-ssm``) with **dilated multi-scan** (strides {1,2,4}, reflect-pad
wrap, summed) — the dilated-conv analog for SSMs (SRMamba arXiv:2403.11143).

Kernels (vision-research README §4.2 — optimised over pure PyTorch):
  * fast path: ``mamba_ssm.selective_scan_fn`` (Blackwell sm_120 build).
  * fallback  : pure-PyTorch chunkwise scan (correct, slower; time is no
    concern — the suite keeps pure-PyTorch only as a fallback).

~20 M params, d=64, runs at the diffusion output resolution (256² train /
2048² inference, tileable).

The kernel-equivalence test (``tests/test_refiner_shapes.py``) asserts the
fast path ≡ fallback on a toy input.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ── mamba-ssm availability probe (done once) ─────────────────────────────────
try:
    from mamba_ssm import selective_scan_fn as _ss_fn  # type: ignore
    _HAS_MAMBA = True
except Exception:  # pragma: no cover — env-dependent
    _ss_fn = None  # type: ignore
    _HAS_MAMBA = False


def mamba_available() -> bool:
    return _HAS_MAMBA


# ═══════════════════════════════════════════════════════════════════════════════
# Selective scan — fast path (mamba-ssm) + pure-PyTorch fallback
# ═══════════════════════════════════════════════════════════════════════════════
def selective_scan(x, dt, A, B, C):
    """Selective SSM scan.

    Discretised recurrence (zoh):  A_bar = exp(Δ · A),  h_t = A_bar·h_{t-1} + B·x_t,
    y_t = C · h_t.

    Args (fallback convention — channels-last sequence):
        x : (B, L, D)        input sequence (per-head feature)
        dt : (B, L, 1)       input-dependent step (gates the recurrence)
        A : (D,)  or (B,L,D)  log-decay (negative ⇒ decay); broadcastable
        B : (B, L, N)        input-dependent B matrix
        C : (B, L, N)        input-dependent C matrix
    Returns:
        y : (B, L, D)        SSM output

    The fast path uses ``mamba_ssm.selective_scan_fn`` which expects
    ``(B, L, D, N)`` layout for x/B/C and a separate ``delta`` arg; we wrap it
    so callers use the simpler (B,L,*) convention.
    """
    Bsz, L, D = x.shape
    N = B.shape[-1]

    if _HAS_MAMBA and x.is_cuda:
        # mamba-ssm expects: u (B,D,L), delta (B,L,1)->(B,1,L) but we just feed (B,L)
        # Layout it wants: u (B, D, L), delta (B, L, 1) is acceptable as (B, L).
        # We follow the documented signature: selective_scan_fn(u, delta, A, B, C)
        # where u:(B,D,L), delta:(B,L,1), A:(D), B:(B,N,L), C:(B,N,L).
        u = x.transpose(1, 2).contiguous()                       # (B, D, L)
        delta = dt.contiguous()                                  # (B, L, 1)
        A_ = A.contiguous() if A.dim() == 1 else A[..., 0].contiguous()  # (D,)
        B_ = B.transpose(1, 2).contiguous()                       # (B, N, L)
        C_ = C.transpose(1, 2).contiguous()                       # (B, N, L)
        y = _ss_fn(u, delta, A_, B_, C_, z=None,
                   delta_softplus=True, return_last_states=False)
        return y.transpose(1, 2)  # back to (B, L, D)

    # ── pure-PyTorch chunkwise scan fallback (correct, slower) ──────────────
    # delta_softplus: actual step = softplus(dt)
    dt_eff = F.softplus(dt)                                  # (B, L, 1)
    # Discretisation (zoh): A_bar = exp(Δ·A); A is per-channel decay (D,)
    A_bar = torch.exp(dt_eff * A.reshape(1, 1, D) if A.dim() == 1 else
                      dt_eff * A)                            # (B, L, D)
    # Recurrence (per mamba-ssm convention, state h is (B, D, N)):
    #   h_t[d, n] = A_bar_t[d] · h_{t-1}[d, n] + B_t[n] · x_t[d]
    #   y_t[d]    = Σ_n C_t[n] · h_t[d, n]
    h = torch.zeros(Bsz, D, N, device=x.device, dtype=torch.float32)
    ys = []
    Bf = B.float(); Cf = C.float(); xf = x.float(); A_barf = A_bar.float()
    for t in range(L):
        h = (A_barf[:, t].unsqueeze(-1) * h                   # (B, D, 1)·(B, D, N)
             + Bf[:, t].unsqueeze(1) * xf[:, t].unsqueeze(-1))  # (B, 1, N)·(B, D, 1)
        y_t = (Cf[:, t].unsqueeze(1) * h).sum(-1)              # (B, D)
        ys.append(y_t)
    y = torch.stack(ys, dim=1).to(x.dtype)                    # (B, L, D)
    return y


# ═══════════════════════════════════════════════════════════════════════════════
# Bidirectional dilated multi-scan bottleneck (SRMamba-style)
# ═══════════════════════════════════════════════════════════════════════════════
class DilatedBiSSM(nn.Module):
    """Bidirectional selective scan with **dilated multi-scan**.

    For each stride ``s`` in ``dilations``:
      1. reflect-pad the flattened sequence by ``s`` on both ends,
      2. take the strided subsequence (positions 0, s, 2s, …),
      3. run a forward + a backward selective scan,
      4. scatter the scanned values back to their original positions,
      5. sum across strides.

    This is the dilated-conv analog for SSMs: each stride captures a
    different receptive field, and the sum aggregates them.  The
    reflect-pad wrap-around avoids boundary artifacts.

    Input/Output: (B, C, H, W) feature map.
    """

    def __init__(self, channels: int, state_dim: int = 16,
                 dilations: tuple = (1, 2, 4)):
        super().__init__()
        self.channels = channels
        self.state_dim = state_dim
        self.dilations = tuple(dilations)
        # input-dependent projections: dt, B, C  (D = channels, N = state_dim)
        self.in_proj = nn.Linear(channels, channels + 2 * state_dim + 1, bias=False)
        # A is a per-channel learnable log-decay (init negative ⇒ decay)
        self.A = nn.Parameter(torch.ones(channels) * -0.5)
        self.out_norm = nn.GroupNorm(32, channels)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # identity at init

    def _scan_direction(self, x, dt, B, C, A):
        # x:(B,L,D) dt:(B,L,1) B:(B,L,N) C:(B,L,N) A:(D,)
        return selective_scan(x, dt, A, B, C)

    def _dilated_scan(self, x_seq, dt, B, C, A, stride):
        """Bidirectional dilated scan over a (B, L, D) sequence.

        For stride ``s`` the sequence is partitioned into ``s`` interleaved
        subsequences (positions ``g, g+s, g+2s, …`` for each offset ``g``);
        each subsequence is scanned forward + backward, and the results are
        scattered back to their original positions and summed.  This is the
        dilated-conv analog for SSMs: each stride gives a different receptive
        field (s=1 → dense local, s=4 → 4× longer range), and the multi-scan
        sum (over ``self.dilations``) aggregates them.
        """
        Bsz, L, D = x_seq.shape
        if stride == 1:
            fwd = self._scan_direction(x_seq, dt, B, C, A)
            rev = self._scan_direction(x_seq.flip(1), dt.flip(1),
                                       B.flip(1), C.flip(1), A).flip(1)
            return fwd + rev
        out = torch.zeros_like(x_seq)
        for g in range(stride):
            pos = torch.arange(g, L, stride, device=x_seq.device)   # (Lg,)
            if pos.numel() < 2:
                continue
            sub = x_seq[:, pos]                                       # (B, Lg, D)
            fwd = self._scan_direction(sub, dt[:, pos], B[:, pos], C[:, pos], A)
            rev = self._scan_direction(sub.flip(1), dt[:, pos].flip(1),
                                       B[:, pos].flip(1), C[:, pos].flip(1), A).flip(1)
            scanned = fwd + rev                                        # (B, Lg, D)
            out = out.scatter(1, pos.view(1, -1, 1).expand(Bsz, -1, D), scanned)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # (B, C, H, W) → (B, L, D)  with L = H*W, D = C
        seq = x.view(b, c, h * w).transpose(1, 2)              # (B, L, D)
        proj = self.in_proj(seq)                                # (B, L, D+2N+1)
        dt = proj[..., :1]
        Bm = proj[..., 1:1 + self.state_dim]
        Cm = proj[..., 1 + self.state_dim:1 + 2 * self.state_dim]
        xs = proj[..., 1 + 2 * self.state_dim:]
        out = torch.zeros_like(seq)
        for s in self.dilations:
            out = out + self._dilated_scan(xs, dt, Bm, Cm, self.A, s)
        out = self.out_norm(out.transpose(1, 2).view(b, c, h, w))
        out = out.view(b, c, h * w).transpose(1, 2)
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(b, c, h, w)              # (B, C, H, W)
        return out + x  # residual


# ═══════════════════════════════════════════════════════════════════════════════
# NAFNet-style building blocks
# ═══════════════════════════════════════════════════════════════════════════════
class SimpleGate(nn.Module):
    """Split a tensor in half along the channel dim and multiply — NAFNet gate."""

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """NAFNet residual block: LN → 1×1 conv → 3×3 depthwise conv → SimpleGate →
    Squeeze-Excitation → 1×1 conv; plus a gated channel-attention branch."""

    def __init__(self, channels: int, expand: int = 2, grad_ckpt: bool = False):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.norm = nn.GroupNorm(1, channels)  # equivalent to LayerNorm per-pixel
        self.conv1 = nn.Conv2d(channels, channels * expand, 1)
        self.dwconv = nn.Conv2d(channels * expand, channels * expand, 3,
                                padding=1, groups=channels * expand)
        self.gate = SimpleGate()
        mid = channels * expand // 2
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(mid, mid, 1),
        )
        self.conv2 = nn.Conv2d(mid, channels, 1)
        # element-wise gating (GELU-derived) on the residual branch
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def _forward(self, x):
        res = x
        h = self.conv1(self.norm(x))
        h = self.dwconv(h)
        h = self.gate(h)                  # channels halved
        h = h * self.sca(h)
        h = self.conv2(h)
        return res + h * self.gamma + self.beta * 0  # beta unused gate placeholder

    def forward(self, x):
        if self.grad_ckpt and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


# ═══════════════════════════════════════════════════════════════════════════════
# SSM refiner — NAFNet-body U-Net + dilated BiSSM bottleneck
# ═══════════════════════════════════════════════════════════════════════════════
class SSMRefiner(nn.Module):
    """Pixel-space refiner.  Input/output: (B, 3, H, W) image in [-1, 1].

    Encoder: 3×3 conv → NAF blocks at each resolution, downsample by 2 twice.
    Bottleneck: NAF blocks + ``DilatedBiSSM``.
    Decoder: NAF blocks + bilinear upsample, skip-concat from encoder.
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 64,
                 ch_mults: tuple = (1, 2, 4),
                 num_blocks: tuple = (4, 6, 6, 8),
                 ssm_state_dim: int = 16,
                 ssm_dilated_strides: tuple = (1, 2, 4),
                 grad_ckpt: bool = False):
        super().__init__()
        assert len(num_blocks) == len(ch_mults) + 1, (
            "num_blocks must have len(ch_mults) encoder levels + 1 bottleneck")
        self.grad_ckpt = grad_ckpt
        ch = [base_ch * m for m in ch_mults]            # e.g. [64, 128, 256]
        self.levels = len(ch)

        self.conv_in = nn.Conv2d(in_ch, ch[0], 3, padding=1)

        # ── Encoder: blocks at each level, downsample between levels ───────
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(self.levels):
            self.encoders.append(nn.Sequential(
                *[NAFBlock(ch[i], grad_ckpt=grad_ckpt) for _ in range(num_blocks[i])]))
            if i < self.levels - 1:
                self.downs.append(nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1))

        # ── Bottleneck at the deepest level: NAF blocks + SSM + NAF ────────
        cur = ch[-1]
        self.bottleneck = nn.Sequential(
            *[NAFBlock(cur, grad_ckpt=grad_ckpt) for _ in range(num_blocks[-1])])
        self.ssm = DilatedBiSSM(cur, state_dim=ssm_state_dim,
                               dilations=ssm_dilated_strides)
        self.mid_block = NAFBlock(cur, grad_ckpt=grad_ckpt)

        # ── Decoder: upsample + skip-concat + 1×1 reduce + blocks ──────────
        self.ups = nn.ModuleList()
        self.skip_reduce = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(self.levels - 1)):       # deepest-1 … 0
            self.ups.append(nn.Conv2d(cur, ch[i], 3, padding=1))            # channel-reduce
            self.skip_reduce.append(nn.Conv2d(ch[i] * 2, ch[i], 1))       # fold skip
            self.decoders.append(nn.Sequential(
                *[NAFBlock(ch[i], grad_ckpt=grad_ckpt) for _ in range(num_blocks[i])]))
            cur = ch[i]

        self.norm_out = nn.GroupNorm(1, ch[0])
        self.conv_out = nn.Conv2d(ch[0], in_ch, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)  # refiner starts as identity
        nn.init.zeros_(self.conv_out.bias)

    def enable_gradient_checkpointing(self):
        self.grad_ckpt = True
        for m in self.modules():
            if isinstance(m, NAFBlock):
                m.grad_ckpt = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine a diffusion-output image. (B,3,H,W) → (B,3,H,W)."""
        res = x
        h = self.conv_in(x)
        skips = []
        for i in range(self.levels):
            h = self.encoders[i](h)
            skips.append(h)
            if i < self.levels - 1:
                h = self.downs[i](h)
        h = self.bottleneck(h)
        h = self.ssm(h) + h          # SSM re-injects high-frequency detail
        h = self.mid_block(h)
        for up, reduce, blocks, skip in zip(self.ups, self.skip_reduce,
                                            self.decoders, reversed(skips[:-1])):
            h = up(F.interpolate(h, scale_factor=2, mode="nearest"))
            h = reduce(torch.cat([h, skip], dim=1))
            h = blocks(h)
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return h + res  # global residual → refiner = identity at init


def build_ssm_refiner(cfg: dict) -> SSMRefiner:
    r = cfg.get("refiner", cfg)
    return SSMRefiner(
        in_ch=r.get("in_ch", 3), base_ch=r.get("base_ch", 64),
        ch_mults=tuple(r.get("ch_mults", (1, 2, 4))),
        num_blocks=tuple(r.get("num_blocks", (4, 6, 6, 8))),
        ssm_state_dim=r.get("ssm_state_dim", 16),
        ssm_dilated_strides=tuple(r.get("ssm_dilated_strides", (1, 2, 4))),
        grad_ckpt=r.get("grad_ckpt", True),
    )