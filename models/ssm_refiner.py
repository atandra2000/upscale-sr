"""SSM refiner — NAFNet-body U-Net with a mamba-ssm dilated bidirectional bottleneck.

~23 M params, runs at the diffusion output resolution (256² train / 2048²
inference, tileable).  Fast path: ``mamba_ssm.selective_scan_fn``; fallback:
pure-PyTorch chunkwise scan (kernel-equivalence tested in test_refiner_shapes).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
    """Selective SSM scan (zoh): A_bar = exp(Δ·A), h_t = A_bar·h_{t-1} + B·x_t, y_t = C·h_t.

    x:(B,L,D), dt:(B,L,1), A:(D,) or (B,L,D), B:(B,L,N), C:(B,L,N) → y:(B,L,D).
    Fast path wraps ``mamba_ssm.selective_scan_fn`` (transposes to its layout).
    """
    Bsz, L, D = x.shape
    N = B.shape[-1]

    if _HAS_MAMBA and x.is_cuda:
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


class DilatedBiSSM(nn.Module):
    """Bidirectional selective scan with dilated multi-scan (the dilated-conv
    analog for SSMs).  Input/output: (B, C, H, W)."""

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
        """Bidirectional dilated scan over (B, L, D): stride-s interleaved
        subsequences, each scanned fwd+rev, scattered back and summed."""
        Bsz, L, D = x_seq.shape
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
        # 1. Horizontal scan sequence (B, L, D)  with L = H*W, D = C
        seq = x.view(b, c, h * w).transpose(1, 2)
        proj = self.in_proj(seq)
        dt = proj[..., :1]
        Bm = proj[..., 1:1 + self.state_dim]
        Cm = proj[..., 1 + self.state_dim:1 + 2 * self.state_dim]
        xs = proj[..., 1 + 2 * self.state_dim:]
        out_h = torch.zeros_like(seq)
        for s in self.dilations:
            out_h = out_h + self._dilated_scan(xs, dt, Bm, Cm, self.A, s)

        # 2. Vertical scan sequence (B, L, D) via grid transpose (H, W) → (W, H)
        v_x = x.transpose(2, 3).reshape(b, c, h * w).transpose(1, 2)
        v_proj = self.in_proj(v_x)
        v_dt = v_proj[..., :1]
        v_Bm = v_proj[..., 1:1 + self.state_dim]
        v_Cm = v_proj[..., 1 + self.state_dim:1 + 2 * self.state_dim]
        v_xs = v_proj[..., 1 + 2 * self.state_dim:]
        out_v = torch.zeros_like(v_x)
        for s in self.dilations:
            out_v = out_v + self._dilated_scan(v_xs, v_dt, v_Bm, v_Cm, self.A, s)
        out_v_seq = out_v.transpose(1, 2).view(b, c, w, h).transpose(2, 3).reshape(b, c, h * w).transpose(1, 2)

        out = out_h + out_v_seq
        out = self.out_norm(out.transpose(1, 2).view(b, c, h, w))
        out = out.view(b, c, h * w).transpose(1, 2)
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(b, c, h, w)
        return out + x  # residual


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
        return res + h * self.gamma

    def forward(self, x):
        if self.grad_ckpt and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class SSMRefiner(nn.Module):
    """Pixel-space refiner: (B,3,H,W) → (B,3,H,W).  NAFNet-body U-Net + DilatedBiSSM bottleneck."""

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

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(self.levels):
            self.encoders.append(nn.Sequential(
                *[NAFBlock(ch[i], grad_ckpt=grad_ckpt) for _ in range(num_blocks[i])]))
            if i < self.levels - 1:
                self.downs.append(nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1))

        cur = ch[-1]
        self.bottleneck = nn.Sequential(
            *[NAFBlock(cur, grad_ckpt=grad_ckpt) for _ in range(num_blocks[-1])])
        self.ssm = DilatedBiSSM(cur, state_dim=ssm_state_dim,
                               dilations=ssm_dilated_strides)
        self.mid_block = NAFBlock(cur, grad_ckpt=grad_ckpt)

        self.ups = nn.ModuleList()
        self.skip_reduce = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(self.levels - 1)):
            self.ups.append(nn.Conv2d(cur, ch[i], 3, padding=1))
            self.skip_reduce.append(nn.Conv2d(ch[i] * 2, ch[i], 1))
            self.decoders.append(nn.Sequential(
                *[NAFBlock(ch[i], grad_ckpt=grad_ckpt) for _ in range(num_blocks[i])]))
            cur = ch[i]

        self.norm_out = nn.GroupNorm(1, ch[0])
        self.conv_out = nn.Conv2d(ch[0], in_ch, 3, padding=1)
        self.gamma_hf = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def enable_gradient_checkpointing(self):
        self.grad_ckpt = True
        for m in self.modules():
            if isinstance(m, NAFBlock):
                m.grad_ckpt = True

    def _haar_dwt_high_pass(self, x: torch.Tensor) -> torch.Tensor:
        """2D Haar DWT high-pass extraction (LH, HL, HH sub-bands)."""
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            return torch.zeros_like(x)
        x00, x10 = x[..., 0::2, 0::2], x[..., 1::2, 0::2]
        x01, x11 = x[..., 0::2, 1::2], x[..., 1::2, 1::2]
        lh = (-x00 - x10 + x01 + x11) * 0.5
        hl = (-x00 + x10 - x01 + x11) * 0.5
        hh = (x00 - x10 - x01 + x11) * 0.5
        return F.interpolate(lh + hl + hh, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine a diffusion-output image. (B,3,H,W) → (B,3,H,W)."""
        res = x
        hf_detail = self._haar_dwt_high_pass(x)
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
        return h + res + self.gamma_hf * hf_detail  # global residual + learned Haar detail


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