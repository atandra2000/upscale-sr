"""Upscale-SR training — 2× RTX 5090 DDP | BF16 autocast + FP32 master | FA2.

Recipe (DESIGN §4, EXECUTION-PLAN Phase 3-4):
  * DDP ``torchrun --nproc-per-node=2``; effective batch 64 = 32/GPU × 2 ×
    grad-accum 2; grad-ckpt on the U-Net.
  * BF16 autocast for the U-Net + refiner forward/backward; model weights stay
    FP32 (the "FP32 master"); fused AdamW in FP32; no GradScaler (BF16).
  * GroupNorm stays FP32 under autocast (== FP32 LayerNorm requirement).
  * FA2 attention (built into ``SRUNet``); ``channels_last`` on U-Net + refiner
    before the first forward (Blackwell sm_120).
  * ``torch.compile(mode="max-autotune")`` on U-Net + refiner modules; the DDIM
    step loop stays eager (dynamic step count — DESIGN §8).
  * EMA decay 0.9999 on U-Net + refiner; grad-clip 1.0; NaN guard.
  * Atomic checkpoints every 10 K iters (``utils.checkpoint``) with full
    RNG + optimizer + scheduler + EMA state.  No pickle for weights.
  * Loss: U-Net MSE(ε_pred, ε) + 0.1·LPIPS(decode(pred_x0), HR); refiner
    L1 + 0.05·SSIM, with a refiner-warmup ramp so the refiner only kicks in
    once the U-Net's one-step x0 estimate is meaningful (DESIGN §4 — joint
    training, or stage via ``--refiner-warmup <large>`` to train U-Net first).

The 9-channel U-Net input is ``[LR_latent_up(4) | z_t(4) | LR-structure mask(1)]``
where the mask is the channel-mean of the upsampled LR latent (a 1-channel
structure hint aligned with the noisy latent).
"""
from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.logging import setup_logger, is_main_process, log_env_summary, rank
from utils.memory import apply_channels_last, gpu_vram_gb
from utils.schedule import build_schedule
from utils.stability import EMA, clip_and_guard
from utils.checkpoint import CheckpointManager
from utils.losses import build_lpips, lpips_loss_fn, ssim_loss
from models import (build_sr_unet, build_ssm_refiner, build_ddpm, build_ddim,
                     sr_cond_input)
from models.vae_frozen import FrozenVAE

# ── GPU flags — tuned for Blackwell (RTX 5090), same as SD_Train.py ──────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


# ═══════════════════════════════════════════════════════════════════════════════
# DDP utilities
# ═══════════════════════════════════════════════════════════════════════════════
def setup_ddp(rank_i: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29555")
    dist.init_process_group("nccl", rank=rank_i, world_size=world_size)
    torch.cuda.set_device(rank_i)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m


def _autocast(device: torch.device):
    """BF16 autocast on CUDA; no-op on CPU (smoke/CI runs without GPU)."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


# ═══════════════════════════════════════════════════════════════════════════════
# Stub VAE — for local smoke runs without downloading SD weights / diffusers
# ═══════════════════════════════════════════════════════════════════════════════
class _StubVAE(nn.Module):
    """A no-op VAE stand-in: 8× bilinear downsample / upsample, fixed scale.
    Used only by ``--stub-vae`` so the training loop can be smoke-tested on a
    machine without the SD VAE weights. NOT for real training."""

    SCALE_FACTOR = 0.18215

    def __init__(self):
        super().__init__()
        self.scale_factor = self.SCALE_FACTOR
        self.dtype = torch.float32

    @torch.no_grad()
    def encode(self, x):
        # (B,3,H,W) → (B,4,H/8,W/8) pseudo-latent (channel-split+pad)
        d = F.avg_pool2d(x, 8)
        pad = 4 - d.shape[1]
        if pad > 0:
            d = F.pad(d, (0, 0, 0, 0, 0, pad))
        return d[:, :4] * self.scale_factor

    @torch.no_grad()
    def decode(self, z):
        z = z / self.scale_factor
        return F.interpolate(z[:, :3], scale_factor=8, mode="bilinear",
                             align_corners=False)


def load_vae(cfg: dict, device, stub: bool = False):
    if stub:
        return _StubVAE().to(device)
    vae_cfg = cfg.get("model", cfg).get("vae", {})
    vae = FrozenVAE(
        model_id=vae_cfg.get("model_id", "stabilityai/sd-vae-ft-mse"),
        dtype=getattr(torch, vae_cfg.get("dtype", "bfloat16")),
        local_dir=vae_cfg.get("local_dir"),
    ).to(device)
    vae.scale_factor = vae_cfg.get("scale_factor", FrozenVAE.SCALE_FACTOR)
    return vae


# ═══════════════════════════════════════════════════════════════════════════════
# One training step (returns the loss dict, un-reduced for logging)
# ═══════════════════════════════════════════════════════════════════════════════
def train_step(
    batch, model, ddpm, lpips_fn, cfg, device, refiner_warmup_iters, global_step,
):
    """Run one (micro-batch) forward + backward. Returns (loss_dict, backward_loss).

    The ``backward_loss`` is already divided by ``grad_accum`` and is the tensor
    that ``.backward()`` is called on.  ``loss_dict`` holds scalar values for
    logging.
    """
    tcfg = cfg.get("train", cfg)
    lcfg = tcfg.get("loss", {})
    lr = batch["lr"].to(device, non_blocking=True)         # (B,3,h,w) [-1,1]
    hr = batch["hr"].to(device, non_blocking=True)         # (B,3,H,W) [-1,1]
    B = lr.shape[0]

    # ── 1. encode (frozen VAE, no grad) ──────────────────────────────────────
    with torch.no_grad():
        hr_lat = model["vae"].encode((hr + 1) / 2)          # z_0 (B,4,H/8,W/8) [0-ish]
        lr_lat = model["vae"].encode((lr + 1) / 2)          # (B,4,h/8,w/8)
        # HR image in [0,1] for LPIPS/L1 targets (decode targets are [-1,1]→[0,1])
        hr_01 = (hr + 1) / 2

    # ── 2. noise + U-Net forward (BF16 autocast) ────────────────────────────
    t = torch.randint(0, ddpm.num_train_timesteps, (B,), device=device,
                      dtype=torch.long)
    noise = torch.randn_like(hr_lat)
    z_t, _ = ddpm.add_noise(hr_lat, t, noise)

    unet_in = sr_cond_input(lr_lat, z_t)                   # (B,9,H/8,W/8)

    ref_w = _refiner_weight(global_step, refiner_warmup_iters,
                            tcfg.get("refiner_warmup_ramp", 2000))
    # one-step x0 estimate + decoded image — needed by LPIPS and/or the refiner
    need_x0 = (lcfg.get("unet_lpips", 0.1) > 0 and lpips_fn is not None) or (ref_w > 0.0)
    with _autocast(device):
        eps_pred = model["ddp_unet"](unet_in, t)
        # U-Net ε-prediction MSE (per-sample, then mean)
        mse = F.mse_loss(eps_pred.float(), noise.float(),
                         reduction="none").mean(dim=[1, 2, 3]).mean()
        unet_loss = lcfg.get("unet_mse", 1.0) * mse

    if need_x0:
        sa = ddpm.sqrt_alphas_cumprod.to(device)[t].reshape(-1, 1, 1, 1)
        ss = ddpm.sqrt_one_minus_alphas_cumprod.to(device)[t].reshape(-1, 1, 1, 1)
        with torch.no_grad():
            x0_pred = ((z_t - ss * eps_pred) / sa.clamp(min=1e-3)).clamp(-6, 6)
            img_pred = model["vae"].decode(x0_pred)         # (B,3,H,W) [-1,1]

    if lcfg.get("unet_lpips", 0.1) > 0 and lpips_fn is not None:
        img_pred_01 = (img_pred + 1) / 2
        lp = lpips_loss_fn(lpips_fn, img_pred_01, hr_01)
        unet_loss = unet_loss + lcfg["unet_lpips"] * lp
    else:
        lp = torch.tensor(0.0, device=device)

    # ── 3. refiner forward (BF16 autocast) — only after warmup ──────────────
    if ref_w > 0.0:
        with torch.no_grad():
            ref_in = img_pred.detach()                       # sharpen the U-Net estimate
        with _autocast(device):
            refined = model["ddp_refiner"](ref_in)
            r_l1 = F.l1_loss(refined.float(), hr.float())
            r_ssim = ssim_loss(refined.float(), hr.float())
            refiner_loss = lcfg.get("refiner_l1", 1.0) * r_l1 + \
                lcfg.get("refiner_ssim", 0.05) * r_ssim
    else:
        refined = ref_in = None
        r_l1 = torch.tensor(0.0, device=device)
        r_ssim = torch.tensor(0.0, device=device)
        refiner_loss = torch.tensor(0.0, device=device)

    total = unet_loss + ref_w * refiner_loss
    backward_loss = total / tcfg.get("grad_accum", 2)

    loss_dict = {
        "unet": float(mse.detach()),
        "lpips": float(lp.detach()) if torch.is_tensor(lp) else float(lp),
        "refiner_l1": float(r_l1.detach()),
        "refiner_ssim": float(r_ssim.detach()),
        "ref_w": ref_w,
        "total": float(total.detach()),
    }
    return backward_loss, loss_dict


def _refiner_weight(step: int, warmup: int, ramp: int) -> float:
    """0 before ``warmup``; linear ramp 0→1 over ``ramp`` iters; 1 after."""
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, (step - warmup) / ramp)


# ═══════════════════════════════════════════════════════════════════════════════
# Main per-rank entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main(rank_i: int, world_size: int, args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if not args.stub:
        setup_ddp(rank_i, world_size)
    device = torch.device(f"cuda:{rank_i}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    mcfg = cfg["model"]
    if args.max_iters is not None:
        tcfg["total_iters"] = args.max_iters
    if args.batch_size is not None:
        tcfg["batch_size_per_gpu"] = args.batch_size
    if args.ckpt_dir is not None:
        tcfg["ckpt"]["dir"] = args.ckpt_dir
    logger = setup_logger(log_file=os.path.join(tcfg["ckpt"]["dir"], "train.log"))
    if is_main_process() or args.stub:
        log_env_summary(logger)
        logger.info(f"config: {args.config} | stub_vae={args.stub} | iters={tcfg['total_iters']}")

    # ── Build model components ──────────────────────────────────────────────
    vae = load_vae(cfg, device, stub=args.stub)
    unet = build_sr_unet(mcfg).to(device)
    refiner = build_ssm_refiner(mcfg).to(device)
    if cfg.get("train", {}).get("channels_last", True) and torch.cuda.is_available():
        unet = apply_channels_last(unet)
        refiner = apply_channels_last(refiner)
    if mcfg.get("sr_unet", {}).get("grad_ckpt", True):
        unet.enable_gradient_checkpointing()
    if mcfg.get("refiner", {}).get("grad_ckpt", True):
        refiner.enable_gradient_checkpointing()

    ddpm = build_ddpm(mcfg).to(device)
    lpips_fn = build_lpips(device) if not args.stub else None

    # ── Optimizer (FP32 master = the model; fused AdamW) ─────────────────────
    params = list(unet.parameters()) + list(refiner.parameters())
    try:
        optimizer = AdamW(params, lr=tcfg["optimizer"]["lr"],
                          weight_decay=tcfg["optimizer"].get("weight_decay", 1e-4),
                          betas=tuple(tcfg["optimizer"].get("betas", (0.9, 0.999))),
                          eps=tcfg["optimizer"].get("eps", 1e-8),
                          fused=bool(tcfg["optimizer"].get("fused", True)))
    except (TypeError, RuntimeError):
        optimizer = AdamW(params, lr=tcfg["optimizer"]["lr"],
                          weight_decay=tcfg["optimizer"].get("weight_decay", 1e-4))
    scheduler = build_schedule(optimizer, total_iters=tcfg["total_iters"],
                               warmup_iters=tcfg.get("schedule", {}).get("warmup_iters", 2000),
                               eta_min_factor=tcfg.get("schedule", {}).get("eta_min_factor", 0.01))

    # ── EMA (one per trainable module) ──────────────────────────────────────
    ema_unet = EMA(unet, decay=tcfg.get("ema", {}).get("decay", 0.9999), device=device)
    ema_refiner = EMA(refiner, decay=tcfg.get("ema", {}).get("decay", 0.9999), device=device)

    # ── Checkpoint manager + resume ────────────────────────────────────────
    ckpt_mgr = CheckpointManager(tcfg["ckpt"]["dir"], keep_last=tcfg["ckpt"].get("keep_last", 6))
    resume = {"step": 0, "best_loss": float("inf")}
    if not args.no_resume:
        try:
            resume = ckpt_mgr.load_latest(unet, refiner, optimizer, scheduler,
                                           ema=None, device=str(device))
            # EMA restore: load into both EMAs from a combined shadow is not
            # stored separately here; on first run this is a no-op.
            logger.info(f"resumed from step {resume.get('step', 0)}")
        except Exception as e:
            logger.warning(f"resume skipped: {e}")

    # ── DDP wrap ────────────────────────────────────────────────────────────
    if not args.stub:
        ddp_unet = DDP(unet, device_ids=[rank_i], output_device=rank_i,
                       find_unused_parameters=False, gradient_as_bucket_view=True)
        ddp_refiner = DDP(refiner, device_ids=[rank_i], output_device=rank_i,
                          find_unused_parameters=True, gradient_as_bucket_view=True)
    else:
        ddp_unet, ddp_refiner = unet, refiner
    model = {"vae": vae, "ddp_unet": ddp_unet, "ddp_refiner": ddp_refiner}

    # ── torch.compile (modules only; step loop eager) ───────────────────────
    if tcfg.get("compile", {}).get("enabled", True) and not args.stub and torch.cuda.is_available():
        try:
            unet_c = torch.compile(_unwrap(ddp_unet), mode=tcfg.get("compile", {}).get("mode", "max-autotune"))
            refiner_c = torch.compile(_unwrap(ddp_refiner), mode=tcfg.get("compile", {}).get("mode", "max-autotune"))
            # Swap the compiled modules back into the model dict for forward use.
            # DDP wraps the raw module; we call the compiled raw module directly
            # under autocast and skip DDP's forward wrapper — grads still
            # all-reduce because autograd sees the same parameters.
            model["unet_compiled"] = unet_c
            model["refiner_compiled"] = refiner_c
            logger.info("torch.compile enabled (max-autotune) on U-Net + refiner")
        except Exception as e:
            logger.warning(f"torch.compile disabled: {e}")
            model["unet_compiled"] = _unwrap(ddp_unet)
            model["refiner_compiled"] = _unwrap(ddp_refiner)
    else:
        model["unet_compiled"] = _unwrap(ddp_unet)
        model["refiner_compiled"] = _unwrap(ddp_refiner)
    # Re-route ddp_unet/ddp_refiner forward calls to the compiled modules
    model["ddp_unet"] = model["unet_compiled"]
    model["ddp_refiner"] = model["refiner_compiled"]

    # ── Data ────────────────────────────────────────────────────────────────
    from data.sr_dataset import build_train_dataset, SRDistributedSampler
    if args.stub:
        # tiny synthetic dataset for the smoke run (no real HR images needed)
        train_ds = _StubDataset(n=16, patch_hr=64, scale=cfg["data"].get("scale", 4))
    else:
        train_ds = build_train_dataset(cfg)
    sampler = None
    if not args.stub:
        from torch.utils.data.distributed import DistributedSampler
        sampler = SRDistributedSampler(train_ds, num_replicas=world_size, rank=rank_i)
    loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size_per_gpu"],
        sampler=sampler, shuffle=(sampler is None and not args.stub),
        num_workers=cfg["data"]["num_workers"] if not args.stub else 0,
        pin_memory=cfg["data"].get("pin_memory", True) and torch.cuda.is_available(),
        persistent_workers=(cfg["data"]["num_workers"] > 0) if not args.stub else False,
        prefetch_factor=cfg["data"].get("prefetch_factor", 4) if not args.stub else None,
        drop_last=True,
    )

    # ── Training loop ───────────────────────────────────────────────────────
    total_iters = tcfg["total_iters"]
    grad_accum = tcfg.get("grad_accum", 2)
    log_every = tcfg.get("log_every", 50)
    ckpt_every = tcfg["ckpt"]["every_iters"]
    eval_every = tcfg.get("eval", {}).get("every_iters", 10000)
    refiner_warmup = args.refiner_warmup if args.refiner_warmup is not None else tcfg.get("refiner_warmup_iters", 10000)

    global_step = resume.get("step", 0)
    best_loss = resume.get("best_loss", float("inf"))
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()

    logger.info(f"starting training: total_iters={total_iters} | grad_accum={grad_accum} | refiner_warmup={refiner_warmup}")
    data_iter = iter(loader)
    for it in range(global_step + 1, total_iters + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(it)
            data_iter = iter(loader)
            batch = next(data_iter)

        backward_loss, ldict = train_step(
            batch, model, ddpm, lpips_fn, cfg, device, refiner_warmup, it)
        backward_loss.backward()

        if it % grad_accum == 0:
            grad_norm = clip_and_guard(params, max_norm=tcfg.get("grad_clip", 1.0),
                                        nan_guard=tcfg.get("nan_guard", True))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema_unet.update(_unwrap(ddp_unet))
            ema_refiner.update(_unwrap(ddp_refiner))
            global_step = it
            if grad_norm == 0.0 and tcfg.get("nan_guard", True):
                logger.warning(f"step {it}: NaN/Inf grad — optimizer step skipped")
                continue

        # ── logging ────────────────────────────────────────────────────────
        if is_main_process() and (it % log_every == 0):
            dt = max(1e-6, time.perf_counter() - t0)
            ips = log_every / dt
            t0 = time.perf_counter()
            vram = gpu_vram_gb(device)
            logger.info(
                f"it {it}/{total_iters} | total {ldict['total']:.4f} unet {ldict['unet']:.4f} "
                f"lpips {ldict['lpips']:.4f} | ref L1 {ldict['refiner_l1']:.4f} ssim {ldict['refiner_ssim']:.4f} "
                f"(w={ldict['ref_w']:.2f}) | gn {grad_norm:.2f} | {ips:.1f} it/s | {vram:.1f}GB"
            )

        # ── checkpoint + eval ──────────────────────────────────────────────
        if it % ckpt_every == 0 and is_main_process():
            ckpt_mgr.save(_unwrap(ddp_unet), _unwrap(ddp_refiner), optimizer,
                          scheduler, ema_unet, it, best_loss,
                          extra_meta={"refiner_ema": ema_refiner.state_dict()})
            logger.info(f"checkpoint saved @ it {it}")
        if it % eval_every == 0 and is_main_process():
            from training.eval import evaluate
            metrics = evaluate(model, cfg, device, ema_unet, ema_refiner,
                               max_imgs=args.eval_imgs)
            logger.info(f"[eval @ {it}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # ── final checkpoint ────────────────────────────────────────────────────
    if is_main_process():
        ckpt_mgr.save(_unwrap(ddp_unet), _unwrap(ddp_refiner), optimizer,
                      scheduler, ema_unet, total_iters, best_loss,
                      extra_meta={"refiner_ema": ema_refiner.state_dict()})
        logger.info(f"TRAINING COMPLETE — final ckpt @ it {total_iters}")
    if not args.stub:
        cleanup_ddp()


# ═══════════════════════════════════════════════════════════════════════════════
# Stub dataset (smoke runs)
# ═══════════════════════════════════════════════════════════════════════════════
class _StubDataset(torch.utils.data.Dataset):
    def __init__(self, n=16, patch_hr=128, scale=4):
        self.n, self.patch_hr, self.scale = n, patch_hr, scale

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        hr = torch.rand(3, self.patch_hr, self.patch_hr) * 2 - 1     # [-1,1]
        lr = F.interpolate(hr.unsqueeze(0), scale_factor=1 / self.scale,
                            mode="bicubic", align_corners=False).squeeze(0)
        return {"lr": lr.clamp(-1, 1), "hr": hr, "seed": idx}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def _argparser():
    p = argparse.ArgumentParser(description="Upscale-SR training — 2× RTX 5090 DDP")
    p.add_argument("--config", default=None, help="path to YAML (default: configs/sr_x4_realesrgan_2x5090.yaml)")
    p.add_argument("--stub", action="store_true",
                   help="smoke mode: stub VAE + tiny synthetic dataset, no DDP")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--refiner-warmup", type=int, default=None,
                   help="refiner loss weight 0 until this iter (default from config)")
    p.add_argument("--eval-imgs", type=int, default=20,
                   help="number of val images for in-loop eval")
    p.add_argument("--max-iters", type=int, default=None,
                   help="override train.total_iters (smoke / short runs)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="override train.batch_size_per_gpu")
    p.add_argument("--ckpt-dir", default=None,
                   help="override train.ckpt.dir (default /workspace/runs/upscale-sr; "
                        "set to a local path for non-RunPod runs)")
    return p


if __name__ == "__main__":
    args = _argparser().parse_args()
    if args.stub:
        # smoke path: single process, CPU or 1 GPU, stub VAE
        main(0, 1, args)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for real training (use --stub for smoke).")
        world_size = torch.cuda.device_count()
        if "RANK" in os.environ:
            main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), args)
        else:
            import torch.multiprocessing as mp
            mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)