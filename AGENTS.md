# AGENTS.md — upscale-sr

> Read root `AGENTS.md` and `self.md` first. Workspace rules are authoritative;
> this file adds **project-specific** rules only. **This file wins over root
> for `upscale-sr/` only.** Design authority: `vision-research/DESIGN-upscale-sr.md`
> + `vision-research/EXECUTION-PLAN-upscale-sr.md` +
> `vision-research/candidates/01-upscale-sr.md`.

> **Project:** `upscale-sr/` · **Type:** 4× real-world photo super-resolution
> **product** (latent-diffusion SR U-Net + SSM refiner + Real-ESRGAN
> degradation) · **Hardware:** 2× RTX 5090 32 GB (Blackwell sm_120) DDP, no
> time cap · **Stack:** PyTorch 2.12, BF16, FA2/sdpa, mamba-ssm/pure-PyTorch
> scan, channels_last, torch.compile, EMA, atomic safetensors ckpts.

This file is the **developer reference** for the project. The design doc
holds the *what*; this file holds the *where-it-lives + how-to-touch-it*.

---

## 0. Operating principles (non-negotiable — from root `AGENTS.md §1`)

- **Raw PyTorch first.** No HF Trainer / Lightning. The training loop is
  hand-written DDP in `training/train.py`.
- **First-principles debugging.** Trace NaN / OOM / slow-throughput to math,
  memory-layout, or architecture — never superficial patches. See §6 below.
- **BF16 over FP16** (Ampere/Blackwell) → **no `GradScaler`**.
- **`channels_last`** on U-Net + refiner convs before first forward
  (`utils/memory.apply_channels_last`).
- **Atomic checkpoints:** `safetensors` weights + `torch.save` state → `.tmp`
  → `os.replace`; full RNG + optim + sched + EMA state. **No `pickle` for
  weights.** (`utils/checkpoint.py`.)
- **No magic numbers.** Architectural constants live in
  `configs/sr_x4_realesrgan_2x5090.yaml` and are named + documented.
- **Prefer `tools/` and `scripts/`** over one-off shells.

---

## 1. Quick commands

```bash
# ── environment (RunPod 2× 5090) ──────────────────────────────────────────
bash scripts/env_check.sh                      # verify deps + sm_120 + kernels
pip install -r requirements.txt                # torch, safetensors, diffusers, lpips, gradio, onnx
# kernels (Blackwell sm_120 builds; if a build fails, the fallback path runs):
pip install flash-attn mamba-ssm causal-conv1d

# ── data ───────────────────────────────────────────────────────────────────
python3 data/build_div2k_flickr.py --out /workspace/data/sr   # DIV2K+Flickr2K+DF2K → HR shards

# ── smoke (local Mac OK — uses stub VAE + fallback kernels) ────────────────
bash scripts/smoke.sh                          # SMOKE_REAL=0 stub 8-iter gate
SMOKE_REAL=1 bash scripts/smoke.sh              # real-VAE 2-iter gate (needs diffusers)

# ── tests ──────────────────────────────────────────────────────────────────
PYTHONPATH=. python3 -m pytest tests/ -q        # 23 tests (4 files); 2 skip on Mac

# ── full training (RunPod 2× 5090, ~2–3 days, 100 K iters) ──────────────────
bash scripts/launch_2x5090.sh                   # DDP via torchrun + bg profiler
CKPT_DIR=/workspace/runs/upscale-sr/v2 bash scripts/launch_2x5090.sh

# ── eval + product surface ─────────────────────────────────────────────────
python3 -m training.eval --ckpt sr_latest.safetensors --val-dir /workspace/data/sr/div2k_val
python3 infer.py --in lr.jpg --out sr.png --scale 4 --steps 30
python3 infer.py --in lr.jpg --out sr.png --scale 4 --fast        # 10-step mode
python3 demo_gradio.py                                          # browser before/after demo

# ── ship ──────────────────────────────────────────────────────────────────
python3 export/to_safetensors.py --ckpt sr_latest.safetensors --out sr_x4_final.safetensors
python3 export/to_onnx.py        --ckpt sr_latest.safetensors --out refiner_x4.onnx
bash scripts/dump_samples.sh /workspace/data/sr/div2k_val sr_x4_final.safetensors
```

---

## 2. Source tree → responsibility

```
upscale-sr/
├── configs/sr_x4_realesrgan_2x5090.yaml   # single source of truth for shapes/recipe
├── data/
│   ├── build_div2k_flickr.py              # DIV2K+Flickr2K+DF2K → webdataset HR shards
│   ├── realesrgan_degrade.py              # ★ stochastic blur/noise/JPEG/sinc (deterministic given seed)
│   └── sr_dataset.py                      # SRDataset (on-the-fly degrade) + SRDistributedSampler
├── models/
│   ├── vae_frozen.py                      # frozen SD1.5 VAE (encode/decode, SCALE_FACTOR=0.18215)
│   ├── sr_unet.py                         # ★ 9-ch latent-diffusion SR U-Net (FA2 / sdpa fallback)
│   ├── ssm_refiner.py                     # ★ NAFNet-body + bidirectional dilated mamba-ssm bottleneck
│   ├── scheduler.py                       # DDPMScheduler (train) + DDIMScheduler (infer, eta=0)
│   └── __init__.py                        # build_model / build_sr_unet / build_ssm_refiner / build_ddim
├── training/
│   ├── train.py                            # DDP main loop (BF16+FP32 master, EMA, grad-clip, atomic ckpt)
│   ├── eval.py                             # PSNR/LPIPS on DIV2K-val (EMA swap, DDIM+refiner)
│   └── profiler.py                         # nvidia-smi util ≥ 95% probe
├── infer.py                                # ★ product CLI: --in/--out/--scale/--steps/--fast/--crop/--stub-vae
├── demo_gradio.py                          # browser before/after (bicubic vs Upscale-SR)
├── export/
│   ├── to_safetensors.py                   # ship sr_x4_final.safetensors (EMA-swapped, unet.+refiner. prefixed)
│   └── to_onnx.py                          # refiner ONNX export + onnxruntime round-trip check
├── utils/{config,checkpoint,logging,memory,schedule,stability,losses}.py
├── scripts/{env_check,smoke,launch_2x5090,dump_samples}.sh
└── tests/{test_realesrgan_deg,test_refiner_shapes,test_fa2_unet,test_infer_cli}.py
```

---

## 3. Architecture at a glance

**Two-stage product.** A frozen SD1.5 VAE encodes the LR image to latent
space `(B,3,H,W)→(B,4,H/8,W/8)`. A **conditional latent-diffusion U-Net**
(`sr_unet.py`, ~52 M) denoises in latent space, conditioned on a **9-channel
input** = concat(`lr_lat_upsampled`, `z_t`, `lr_structure_mask`) + a
sinusoidal **timestep embedding**. A **DDIM sampler** (`scheduler.py`,
30 steps, η=0) walks the reverse process. The decoded HR image is then
refined in **pixel space** by the **SSM refiner** (`ssm_refiner.py`, ~23 M) —
a NAFNet-body U-Net whose bottleneck is a **bidirectional `mamba-ssm`
selective scan with dilated multi-scan {1,2,4}** (the dilated-conv analog for
SSMs). The refiner re-injects high-frequency detail that diffusion blurs.

**Training pair generator.** `data/realesrgan_degrade.py` synthesises
real-world LR from HR (DIV2K + Flickr2K + DF2K): blur (gaussian iso/aniso)
→ resize (sinc/area/bicubic) → noise (gaussian/poisson) → JPEG → sinc, with
a second-order stage shuffle. **Deterministic given a seed** (RNG
save+seed+restore — tested in `tests/test_realesrgan_deg.py`).

**Conditioning input** (`models/cond.py:sr_cond_input`, used by
`train.py`/`infer.py`/`eval.py`): `cat(lr_lat_up, z_t, lr_lat_up.mean(dim=1,keepdim))` →
9 channels.

**Selective-scan recurrence** (`models/ssm_refiner.py:selective_scan`):
`A_bar = exp(softplus(dt)·A)`; `h_t = A_bar·h_{t-1} + B·x_t`;
`y_t = Σ_n C·h`. Fast path = `mamba_ssm.selective_scan_fn` (CUDA); fallback =
pure-PyTorch chunkwise loop with correct broadcast. Kernel-equivalence is
tested (`tests/test_refiner_shapes.py::test_selective_scan_kernel_equiv`).

---

## 4. Config → knob map (the single source of truth is the YAML)

| Knob | YAML path | Default | Notes |
|------|-----------|---------|-------|
| U-Net width | `model.sr_unet.base_ch` | 96 | ch_mults (1,2,4,4) → 384 deep, ~52 M |
| U-Net heads | `model.sr_unet.heads` | 8 | all stage channels divisible by 8 |
| Refiner width | `model.refiner.base_ch` | 128 | ch_mults (1,2,4) → 512 deep, ~23 M |
| SSM strides | `model.refiner.ssm_dilated_strides` | [1,2,4] | dilated multi-scan |
| Schedule | `model.scheduler.schedule` | scaled_linear | β_start 8.5e-4 → β_end 1.2e-2 |
| DDIM steps | `model.scheduler.infer_steps` | 30 | `fast_steps` 10 for `--fast` |
| Effective batch | `train.batch_size_per_gpu × grad_accum × world` | 32·2·2 = 128 | spec targets 64; raise grad_accum to taste |
| LR / wd | `train.optimizer.{lr,weight_decay}` | 1e-4 / 1e-4 | fused AdamW |
| EMA decay | `train.ema.decay` | 0.9999 | warmup `min(decay,(1+s)/(10+s))` |
| Grad-clip | `train.grad_clip` | 1.0 | NaN guard zeros grads if non-finite |
| Compile | `train.compile.mode` | max-autotune | on U-Net + refiner; **DDIM step loop eager** |
| Ckpt every | `train.ckpt.every_iters` | 10000 | `keep_last` 6, atomic `.tmp→os.replace` |
| Eval every | `train.eval.every_iters` | 10000 | `fixed_seed` 42 → reproducible PSNR/LPIPS |
| Refiner warmup | CLI `--refiner-warmup N` | config-driven | refiner loss weight 0→1 ramp over N |

> **Never hard-code a shape or LR in Python.** Read it from the config via
> `utils/config.load_config()` and the `build_*` helpers in `models/__init__.py`.

---

## 5. Precision, kernels, memory-format (the stability contract)

- **BF16 autocast** on U-Net + refiner forward; **FP32 master weights**
  (`utils.stability.FP32Master`); **FP32 LayerNorm/GroupNorm** (stable under
  BF16). **No `GradScaler`** (BF16, not FP16).
- **FA2** (`flash_attn.flash_attn_func`) on U-Net self-attention; falls back
  to `F.scaled_dot_product_attention` (flash backend where available, else
  math). Probe: `models.sr_unet.fa2_available()`.
- **`mamba-ssm`** selective scan on the refiner bottleneck; falls back to the
  pure-PyTorch chunkwise scan (correct, slower). Probe:
  `models.ssm_refiner.mamba_available()`.
- **`channels_last`** on U-Net + refiner convs before the first forward
  (`utils.memory.apply_channels_last`) — Blackwell sm_120 mandate.
- **`torch.compile(mode="max-autotune")`** on U-Net + refiner modules. The
  **DDIM step loop stays eager** (dynamic step count → recompiles otherwise).
- **EMA** shadow on U-Net + refiner (GPU-resident); eval swaps EMA in.
- **Grad-clip 1.0** + **NaN guard** (`utils.stability.clip_and_guard`):
  zeros grads if any non-finite, skipping the step.

---

## 6. Failure-mode triage (first-principles — DESIGN §8)

| Symptom | Check first |
|---------|-------------|
| `mamba-ssm` import fails on Blackwell | no sm_120 build → pure-PyTorch fallback (correct, slower); time is not a concern. Verify via `models.ssm_refiner.mamba_available()` |
| `flash_attn` import fails | sdpa flash-backend fallback runs; verify `models.sr_unet.fa2_available()` |
| NaN loss | `clip_and_guard` already zeros grads; trace to dt/A overflow in the scan or VAE decode overflow. Drop to FP32 master-only for one step to localise |
| OOM at 32/GPU | enable `grad_ckpt` on U-Net; raise `grad_accum` to 2/4; fall back to patch 192² |
| PSNR low / over-smooth | diffusion over-smoothing → raise refiner loss weight / shorten `--refiner-warmup`; the refiner is the fix (report PSNR with/without) |
| Degradation too weak (trains on easy pairs) | verify Real-ESRGAN severity ranges in the YAML; smoke-test determinism (`tests/test_realesrgan_deg.py`) |
| DDIM + compile recompiles | keep the step loop eager; compile only U-Net + refiner modules |
| LPIPS stuck | VGG feature-extractor mismatch → `utils.losses._LPIPSVGGFallback` uses `vgg16` IMAGENET1K_V1 features |
| Reflect-pad crash on tiny input | `infer._safe_reflect_pad` reflect-then-constant pads; inputs < scale·8 are supported |

---

## 7. Subagent routing (when to spawn which)

Spawn via the Agent tool when the trigger matches (root `AGENTS.md §2`):

| Agent | Trigger for this project |
|-------|--------------------------|
| `pytorch-debugger` | NaN loss, OOM, slow throughput, grad issues, ckpt corruption in `training/train.py` or the models |
| `data-pipeline-engineer` | shard build (`data/build_div2k_flickr.py`), `webdataset` loader, Real-ESRGAN degradation tuning |
| `ml-benchmarks` | tokens/sec-equiv (imgs/sec), memory profiling, ≥95% util probe, with/without-refiner PSNR |
| `project-finder` | "which file owns the SSM scan?", "where is the conditioning concat?" |
| `research-engineer-dev` | (only if extending the platform itself — not this project) |

> For architecture / attention / scan / memory-layout questions, prefer the
> `.agents/skills/llm-architecture` and `computer-vision-multimodal` skills
> and `generative-vision-gans-vaes` over re-deriving from scratch.

---

## 8. Hard don'ts (from root `AGENTS.md §6`)

- Don't bypass this file for `upscale-sr/` questions.
- Don't hard-code shapes / LRs / schedule constants in Python — read the YAML.
- Don't `pickle` checkpoints — `safetensors` weights + `torch.save` state.
- Don't rewrite in JAX/TF.
- Don't add HF Trainer / Lightning — hand-written DDP only.
- Don't compile the DDIM step loop — keep it eager.
- Don't ship a checkpoint without the EMA swap (`export/to_safetensors.py`).

---

## 9. Vault sync (root `AGENTS.md §8`)

`upscale-sr/` lives under `CoreProjects/` → the workspace Stop hook
auto-mirrors every new/modified `.md` into `~/Documents/obsidian`.
**Create new `.md` here, never in the vault.** Never hand-edit mirror files
(next sync overwrites). Excluded dirs (`.venv`, `__pycache__`, etc.) are
skipped automatically.