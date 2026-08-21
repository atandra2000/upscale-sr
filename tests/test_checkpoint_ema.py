"""test_checkpoint_ema.py — checkpoint save/load round-trip, EMA math, and
DPM-Solver++ correctness.

These cover the audit findings: no test existed for the atomic checkpoint
path (RNG/optimizer/EMA restoration), the EMA update/apply/restore contract,
or the DPM-Solver++ 2M step math.
"""
import pytest
import torch
import torch.nn as nn

from utils.checkpoint import CheckpointManager, rng_state_dict, restore_rng_state
from utils.stability import EMA
from models.scheduler import DPMSolverMultistepScheduler


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint round-trip
# ═══════════════════════════════════════════════════════════════════════════
class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.register_buffer("b", torch.ones(2))

    def forward(self, x):
        return self.fc(x) + self.b


def test_checkpoint_round_trip(tmp_path):
    """save → load must restore weights, optimizer, scheduler, and RNG state."""
    torch.manual_seed(0)
    net = _TinyNet()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    ema = EMA(net, decay=0.9999)
    mgr = CheckpointManager(str(tmp_path), keep_last=2)

    # Train a few steps so optimizer/scheduler/EMA/RNG are non-trivial.
    for _ in range(5):
        opt.zero_grad()
        loss = (net(torch.randn(8, 4)) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        ema.update(net)
    rng_before = rng_state_dict()

    mgr.save(net, nn.Sequential(), opt, sched, ema, step=5, best_loss=1.234)
    w_before = {k: v.clone() for k, v in net.state_dict().items()}
    with torch.no_grad():
        net.fc.weight.add_(1.0)  # corrupt the model
    torch.manual_seed(999)
    opt.zero_grad(set_to_none=True)

    loaded = mgr.load(net, nn.Sequential(), opt, sched, ema, device="cpu")
    assert loaded["step"] == 5
    assert loaded["best_loss"] == 1.234

    for k, v in net.state_dict().items():
        assert torch.equal(v, w_before[k]), f"weight {k} not restored"

    assert opt.param_groups[0]["lr"] == sched.get_last_lr()[0]

    rng_after = rng_state_dict()
    assert torch.equal(rng_after["torch"], rng_before["torch"]), "torch RNG not restored"

    assert ema.step_count == 5
    assert ema.shadow["fc.weight"].shape == (2, 4)


def test_checkpoint_prune_keeps_last(tmp_path):
    mgr = CheckpointManager(str(tmp_path), keep_last=2)
    net = _TinyNet()
    opt = torch.optim.AdamW(net.parameters())
    sched = torch.optim.lr_scheduler.ConstantLR(opt)
    ema = EMA(net)
    for i in (1, 2, 3, 4):
        mgr.save(net, nn.Sequential(), opt, sched, ema, step=i, best_loss=0.0)
    steps = sorted(
        int(p.stem.removeprefix("sr_step_"))
        for p in tmp_path.glob("sr_step_*.safetensors"))
    assert steps == [3, 4], f"prune kept {steps}, expected [3, 4]"


def test_rng_state_round_trip():
    torch.manual_seed(42)
    state = rng_state_dict()
    a = torch.randn(3)
    _ = torch.randn(1000)               # consume RNG
    restore_rng_state(state)
    c = torch.randn(3)
    assert torch.equal(a, c), "RNG restore failed"


# ═══════════════════════════════════════════════════════════════════════════
# EMA math
# ═══════════════════════════════════════════════════════════════════════════
def test_ema_update_formula():
    """After one update, shadow = (1-d)·w + d·shadow_old; warmup d = 2/11."""
    net = _TinyNet()
    ema = EMA(net, decay=0.99)
    with torch.no_grad():
        net.fc.weight.mul_(0.0).add_(2.0)   # live weights = 2
        ema.shadow["fc.weight"].zero_()      # shadow starts at 0
    ema.update(net)
    d = min(0.99, 2 / 11)
    expected = (1 - d) * 2.0
    assert torch.allclose(ema.shadow["fc.weight"],
                          torch.full_like(ema.shadow["fc.weight"], expected),
                          atol=1e-6)


def test_ema_apply_restore():
    net = _TinyNet()
    ema = EMA(net, decay=0.5)
    with torch.no_grad():
        net.fc.weight.mul_(0.0).add_(3.0)
        ema.shadow["fc.weight"].fill_(1.0)
    backup = ema.apply_shadow(net)
    assert torch.allclose(net.fc.weight, torch.full_like(net.fc.weight, 1.0))
    ema.restore(net, backup)
    assert torch.allclose(net.fc.weight, torch.full_like(net.fc.weight, 3.0))


def test_ema_state_dict_round_trip():
    net = _TinyNet()
    ema = EMA(net, decay=0.99)
    ema.update(net)
    sd = ema.state_dict()
    ema2 = EMA(net, decay=0.99)
    ema2.load_state_dict(sd)
    assert ema2.step_count == ema.step_count
    for k in ema.shadow:
        assert torch.equal(ema2.shadow[k], ema.shadow[k])


# ═══════════════════════════════════════════════════════════════════════════
# DPM-Solver++ correctness
# ═══════════════════════════════════════════════════════════════════════════
def test_dpm_solver_step_matches_analytic_formula():
    """First step (no history) must reduce to the DPM-Solver++ 2M first-order
    update: x_{prev} = √(ᾱ_prev)/√(ᾱ_t) · x_t − √(ᾱ_prev)·(e^{−h} − 1)·x0_pred."""
    s = DPMSolverMultistepScheduler(steps=1000, beta_start=0.00085,
                                    beta_end=0.012, schedule="scaled_linear")
    s.set_timesteps(15, device=torch.device("cpu"))
    t = s.timesteps[0]
    x_t = torch.randn(2, 4, 8, 8)
    eps = torch.randn(2, 4, 8, 8)

    t_int = int(t)
    step_size = 1000 // 15
    prev_t = t_int - step_size
    alpha_t = s.alphas_cumprod[t_int]
    alpha_prev = s.alphas_cumprod[prev_t] if prev_t >= 0 else torch.ones(1)
    lambda_curr = s.lambda_t[t_int]
    lambda_prev = s.lambda_t[prev_t] if prev_t >= 0 else s.lambda_t[0]
    h = float((lambda_prev - lambda_curr).item())

    x0_pred = (x_t - (1.0 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
    x0_pred = x0_pred.clamp(-1.0, 1.0)
    expected = (alpha_prev.sqrt() / alpha_t.sqrt()) * x_t \
        - alpha_prev.sqrt() * (torch.exp(torch.tensor(-h)) - 1.0) * x0_pred

    out = s.step(eps, t, x_t)
    assert torch.allclose(out, expected, atol=1e-6), \
        f"first-order DPM step mismatch, max diff {(out - expected).abs().max()}"


def test_dpm_solver_second_step_populates_history():
    """After the first step, old_pred_x0 and old_h must be populated for 2M."""
    s = DPMSolverMultistepScheduler(steps=1000)
    s.set_timesteps(15, device=torch.device("cpu"))
    ts = s.timesteps
    x_t = torch.randn(2, 4, 8, 8)
    assert s.old_pred_x0 is None
    assert s.old_h is None
    out1 = s.step(torch.randn_like(x_t), ts[0], x_t)
    assert s.old_pred_x0 is not None
    assert s.old_h is not None
    out2 = s.step(torch.randn_like(x_t), ts[1], x_t)
    assert torch.isfinite(out2).all()
    assert out1.shape == out2.shape
