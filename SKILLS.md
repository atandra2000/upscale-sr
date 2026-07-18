# SKILLS.md — upscale-sr

> Read root `AGENTS.md` and `self.md` first, then `upscale-sr/AGENTS.md`.
> Workspace rules are authoritative; this file adds **project-specific
> workflows** — the interactive playbooks that aren't obvious from a single
> `AGENTS.md` read. Design authority: `vision-research/DESIGN-upscale-sr.md`.

This is the companion to `AGENTS.md` (this folder): that file holds the
architecture / source-tree / config-knob reference; **this file holds the
hands-on workflows** (smoke, train, eval, ship, debug).

---

## Skill 1: Local smoke-test on a Mac (no CUDA)

The whole pipeline runs on CPU via fallback kernels (sdpa attention,
pure-PyTorch selective scan, stub VAE). This is the gate before any GPU run.

```bash
cd upscale-sr
SMOKE_REAL=0 bash scripts/smoke.sh        # 8-iter stub gate → atomic safetensors ckpt, no NaN
PYTHONPATH=. python3 -m pytest tests/ -q   # 23 tests (2 skip on Mac: mamba-ssm kernel, channels_last)
```

**What it proves:** shapes are correct end-to-end, the SSM scan broadcast is
right, degradation is deterministic given a seed, the atomic checkpoint
writes `sr_step_*.safetensors` + `.state.pt` with no NaN, and `infer.py`
round-trips 16²→64². **Exit code 0 = green; do not push to the GPU pod
without it.**

To smoke-test the **real** SD1.5 VAE (needs `diffusers` + the weights):

```bash
SMOKE_REAL=1 bash scripts/smoke.sh
```

---

## Skill 2: Launch the full 100 K-iter training run (RunPod 2× 5090)

```bash
# Phase 0 + 1: pod env + data shards (one-time)
bash scripts/env_check.sh                                     # sm_120 + kernels + deps
python3 data/build_div2k_flickr.py --out /workspace/data/sr   # DIV2K+Flickr2K+DF2K HR shards

# Phase 4: the run (~2–3 days, ≥95% util target, ckpt every 10 K)
CKPT_DIR=/workspace/runs/upscale-sr bash scripts/launch_2x5090.sh
```

`launch_2x5090.sh` runs `torchrun --nproc-per-node=2` + a background
`training.profiler` polling GPU-0 util every 5 s → `util.log` (the ≥95%
acceptance evidence). Watch:

- `diffusion loss` + `refiner loss` decreasing
- `imgs/sec` and `GPU util` (target ≥95%)
- **30 K-iter early signal:** spot-check a DIV2K-val upscale → PSNR ≥ 26 dB.
  If PSNR is still climbing at 100 K, extend (no time cap).
- If util < 85% → CUDA graphs already on; consider raising `batch_size_per_gpu`.

**Resume from crash:** the loop auto-resumes from `sr_latest.*` (atomic
state + RNG). Just re-launch the same command.

---

## Skill 3: Evaluate on DIV2K-val (the product benchmark)

```bash
python3 -m training.eval \
    --ckpt /workspace/runs/upscale-sr/sr_latest.safetensors \
    --val-dir /workspace/data/sr/div2k_val \
    --eval-imgs 100
```

This swaps the **EMA weights** in, runs **DDIM (30 steps) + refiner** on each
of the 100 DIV2K-val images under **fixed-seed Real-ESRGAN degradation**
(seed 42 → reproducible), and prints **mean PSNR / LPIPS**. Acceptance:
**PSNR ≥ 27 dB / LPIPS ≤ 0.82**. Log the numbers into `README.md` §Results.

**Ablation (proves the refiner earns its keep):** re-run with
`--no-refiner` and report PSNR with/without — the refiner should lift PSNR
and visibly sharpen high-frequency detail.

---

## Skill 4: Ship the product (safetensors + ONNX + samples)

```bash
# 1. swap EMA weights in → ship weights
python3 export/to_safetensors.py \
    --ckpt /workspace/runs/upscale-sr/sr_latest.safetensors \
    --out sr_x4_final.safetensors

# 2. refiner ONNX export (+ onnxruntime round-trip check)
python3 export/to_onnx.py \
    --ckpt /workspace/runs/upscale-sr/sr_latest.safetensors \
    --out refiner_x4.onnx

# 3. 8 LR→SR before/after PNGs (LR | bicubic | Upscale-SR) for the README
bash scripts/dump_samples.sh /workspace/data/sr/div2k_val sr_x4_final.safetensors
```

**Verify the CLI end-to-end** (the product contract):

```bash
python3 infer.py --in samples/0001_lr.jpg --out samples/0001_sr.png --scale 4 --steps 30
python3 infer.py --in big.jpg --out big_sr.png --scale 4 --crop 512   # tile large images
python3 infer.py --in lr.jpg --out sr.png --scale 4 --fast            # 10-step fast mode
```

`--crop 512` tiles large images (hann-window blend, 25% overlap) to bound
VRAM; `--crop 0` runs the whole image at once.

---

## Skill 5: Run the Gradio before/after demo

```bash
python3 demo_gradio.py
# → opens a browser: drop an image, see bicubic baseline vs Upscale-SR side-by-side
```

Use this for live demos / recruiter screens — it loads `sr_x4_final.safetensors`
+ the frozen VAE and runs the same `infer.upscale` path as the CLI.

---

## Skill 6: Debug NaN / OOM / slow throughput (first-principles)

| Symptom | First-principles move |
|---------|------------------------|
| **NaN loss** | `clip_and_guard` already zeros grads. Drop to FP32-master-only for one step (`train.precision.autocast: float32`) to localise. Inspect `dt`/`A` in `selective_scan` for overflow (`softplus(dt)·A`); check VAE decode range. |
| **OOM @ 32/GPU** | `grad_ckpt: true` on U-Net (already on); raise `grad_accum` to 2/4; fall back to `patch_hr: 192`. Profile with `torch.cuda.memory._record_memory_history` if persistent. |
| **util < 85%** | CUDA graphs already on; ensure `num_workers: 8` + `prefetch_factor: 4` (data-bound?); check the DDIM step loop is **not** compiled (recompiles kill util). |
| **slow throughput** | confirm `channels_last` applied *before* first forward; confirm FA2 path (`fa2_available()`), not sdpa-math; confirm fused AdamW (`optimizer.fused: true`). |
| **ckpt corruption** | atomic write is `.tmp → os.replace`; if a `.tmp` lingers, the prior write was killed mid-flight — delete it and resume from `sr_latest`. Never `pickle`. |
| **reflect-pad crash on tiny input** | `infer._safe_reflect_pad` already reflect-then-constant pads; if you hit it, you bypassed `upscale()` — use `upscale()`, not raw `F.pad`. |

**Rule:** trace to math / memory-layout / architecture. Never paper over with
a `try/except` or a `.clamp()` you can't justify.

---

## Skill 7: Tune the Real-ESRGAN degradation (the fiddly part)

The degradation severity is the single biggest lever on real-photo
generalisation. Knobs live in `configs/...yaml → data.degradation`:

| Knob | Range | Effect |
|------|-------|--------|
| `blur_sigma` | [0.2, 3.0] | wider → blurrier LR → harder task |
| `noise_sigma` | [1, 30] | 1≈clean, 30=heavy grain |
| `jpeg_quality` | [30, 95] | lower → more compression artifacts |
| `sinc_prob` | 0.15 | final sinc ringing probability |
| `resize_prob` | 0.25 | intermediate resize (sinc/area/bicubic) probability |
| `second_blur_prob` | 0.25 | second-order blur (the "2nd-order" stage) |
| `shuffle_prob` | 0.15 | stage-order shuffle (Real-ESRGAN 2nd-order) |

**Determinism invariant:** `RealESRGANDegrader.degrade(hr, seed)` is
bit-identical for the same seed and does **not** perturb the caller's RNG
state (`tests/test_realesrgan_deg.py::test_rng_state_restored`). Preserve
this when editing — always save/seed/restore the torch+numpy RNG inside
`degrade()`.

**Tuning loop:** if PSNR on real photos lags bicubic, the degradation is too
severe; if it's only good on bicubic-downsampled pairs, it's too weak. Match
the distribution of your target real photos.

---

## Skill 8: Reuse from the portfolio (don't reinvent)

- **SD1.5 VAE** — `models/vae_frozen.py` loads `stabilityai/sd-vae-ft-mse`
  (or copy weights from `Vision/StableDiffusion/`). Frozen, reused, not
  retrained.
- **Atomic ckpt / `channels_last` / logging / EMA** patterns — mirror
  `LLM/LLaMA-3-Lite/utils/`.
- **DDP launch + DDIM sampler** patterns — mirror `Vision/StableDiffusion/`.
- **LPIPS** — `lpips` package, with a `torchvision.vgg16` (IMAGENET1K_V1)
  fallback in `utils/losses._LPIPSVGGFallback` for boxes without `lpips`.

---

## 9. Cross-references

- **Design:** `vision-research/DESIGN-upscale-sr.md` (the *how-to-build* spec)
- **Execution plan:** `vision-research/EXECUTION-PLAN-upscale-sr.md` (phased)
- **Candidate / fit:** `vision-research/candidates/01-upscale-sr.md`
- **Workspace rules:** root `AGENTS.md` (§1 rules, §3 decision tree)
- **Personal tone:** `self.md`
- **Sibling skills:** `.agents/skills/{computer-vision-multimodal,
  generative-vision-gans-vaes, llm-architecture}/SKILL.md`