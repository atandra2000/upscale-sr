"""Rank-aware logging + is_main_process helper for 2× RTX 5090 DDP."""
from __future__ import annotations

import logging
import sys  # ponytail: os removed (unused)
from pathlib import Path

import torch
import torch.distributed as dist


def is_main_process() -> bool:
    """True on rank 0 (or when DDP is not initialized)."""
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def rank() -> int:
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0


def setup_logger(name: str = "upscale-sr", log_file: str | None = None) -> logging.Logger:
    """File + stream logger; only rank 0 writes the file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] | %(message)s")
    if is_main_process():
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        logger.addHandler(stream)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, mode="a")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    else:
        # Null handler on non-rank-0 to avoid stdout spam from every GPU.
        logger.addHandler(logging.NullHandler())
    return logger


def log_env_summary(logger: logging.Logger) -> None:
    """Log the GPU / torch / kernel availability summary once on rank 0."""
    if not is_main_process():
        return
    logger.info("=" * 72)
    logger.info("Upscale-SR — 2× RTX 5090 DDP | BF16 | FA2 | mamba-ssm")
    logger.info("=" * 72)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {p.name} | {p.total_memory / 1e9:.1f} GB | sm_{p.major}{p.minor}")
    else:
        logger.info("CUDA not available — running on CPU (smoke/dev only).")
    # Kernel availability
    for mod, label in [("flash_attn", "flash-attn"), ("mamba_ssm", "mamba-ssm"),
                       ("causal_conv1d", "causal-conv1d")]:
        try:
            __import__(mod)
            logger.info(f"kernel: {label} — available")
        except Exception:
            logger.info(f"kernel: {label} — MISSING (fallback path will be used)")
    logger.info(f"torch={torch.__version__} | world_size={_world_size()}")


def _world_size() -> int:
    return dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1