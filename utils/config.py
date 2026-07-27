"""YAML config loader — single source of truth for the run config."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CFG = Path(__file__).resolve().parent.parent / "configs" / "sr_x4_realesrgan_2x5090.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    p = Path(path) if path else _DEFAULT_CFG
    with open(p, "r") as f:
        return yaml.safe_load(f)