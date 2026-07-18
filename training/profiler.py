"""GPU utilization profiler — asserts ≥ 95% util (EXECUTION-PLAN acceptance).

Polls ``nvidia-smi`` once per ``interval_s`` for ``duration_s`` and reports
the mean / min / p50 / p95 GPU-uti% and memory.  Used as a Phase-3 / Phase-4
knob: if mean util < 95%, enable CUDA graphs or increase the per-GPU batch.

Can be run standalone (alongside training, on rank 0):
    python -m training.profiler --duration 60 --interval 1 --gpu 0
"""
from __future__ import annotations

import argparse
import subprocess
import time
from statistics import mean, median

from utils.logging import setup_logger


def _query(gpu: int) -> tuple[float, float]:
    """Return (utilisation%, memory-used-GiB) for ``gpu`` via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}",
         "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    util_s, mem_s = [x.strip() for x in out.split(",")]
    return float(util_s), float(mem_s) / 1024.0


def profile(gpu: int = 0, duration_s: int = 60, interval_s: float = 1.0,
            target_util: float = 95.0) -> dict:
    """Poll GPU utilisation for ``duration_s``. Returns a metrics dict."""
    logger = setup_logger()
    samples = []
    mem_samples = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        try:
            u, m = _query(gpu)
        except Exception as e:
            logger.warning(f"nvidia-smi query failed: {e}")
            break
        samples.append(u)
        mem_samples.append(m)
        time.sleep(interval_s)
    if not samples:
        return {"mean_util": 0.0, "n": 0, "meets_target": False}
    samples_sorted = sorted(samples)
    p95 = samples_sorted[min(len(samples_sorted) - 1, int(0.95 * len(samples_sorted)))]
    metrics = {
        "mean_util": mean(samples),
        "min_util": min(samples),
        "p50_util": median(samples),
        "p95_util": p95,
        "mean_mem_gib": mean(mem_samples),
        "max_mem_gib": max(mem_samples),
        "n": len(samples),
        "target_util": target_util,
        "meets_target": mean(samples) >= target_util,
    }
    logger.info(
        f"[profiler gpu={gpu}] util mean={metrics['mean_util']:.1f}% "
        f"p50={metrics['p50_util']:.1f}% p95={metrics['p95_util']:.1f}% "
        f"min={metrics['min_util']:.1f}% | mem mean={metrics['mean_mem_gib']:.1f}GiB "
        f"max={metrics['max_mem_gib']:.1f}GiB | "
        f"{'PASS ≥' if metrics['meets_target'] else 'FAIL <'}{target_util:.0f}%"
    )
    return metrics


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--target-util", type=float, default=95.0)
    args = ap.parse_args()
    profile(args.gpu, args.duration, args.interval, args.target_util)


if __name__ == "__main__":
    _main()