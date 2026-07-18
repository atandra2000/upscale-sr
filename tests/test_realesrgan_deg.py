"""test_realesrgan_deg.py — Real-ESRGAN degradation is deterministic given a seed.

EXECUTION-PLAN Phase 2 acceptance: "degradation pipeline is deterministic given
seed".  Also asserts size preservation and a valid [0,1] range.
"""
import pytest
import torch

from data.realesrgan_degrade import RealESRGANDegrader, make_lr_pair

CFG = {"blur_sigma": [0.2, 3.0], "noise_sigma": [1, 30], "jpeg_quality": [30, 95],
       "sinc_prob": 0.5, "resize_prob": 0.5, "second_blur_prob": 0.5, "shuffle_prob": 0.3}


def _degrader():
    return RealESRGANDegrader(CFG)


def test_determinism_same_seed():
    deg = _degrader()
    hr = torch.rand(2, 3, 128, 128)
    a = deg.degrade(hr, seed=42)
    b = deg.degrade(hr, seed=42)
    assert a.shape == hr.shape == b.shape
    assert torch.allclose(a, b, atol=1e-6), "same seed must give identical output"


def test_different_seed_differs():
    deg = _degrader()
    hr = torch.rand(2, 3, 128, 128)
    a = deg.degrade(hr, seed=42)
    c = deg.degrade(hr, seed=7)
    assert not torch.allclose(a, c, atol=1e-4), "different seeds should differ"


def test_size_preserved_all_stages():
    """Even with the anisotropic blur + resize + sinc, output size == input."""
    deg = _degrader()
    for seed in range(20):
        hr = torch.rand(1, 3, 128, 128)
        out = deg.degrade(hr, seed=seed)
        assert out.shape == hr.shape, f"size changed at seed={seed}: {out.shape}"


def test_range_and_no_nan():
    deg = _degrader()
    hr = torch.rand(1, 3, 64, 64)
    out = deg.degrade(hr, seed=123)
    assert not torch.isnan(out).any()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_make_lr_pair_shapes():
    deg = _degrader()
    hr = torch.rand(1, 3, 128, 128)
    lr, hr2 = make_lr_pair(hr, deg, scale=4, seed=42)
    assert lr.shape == (1, 3, 32, 32)
    assert hr2.shape == (1, 3, 128, 128)
    assert torch.allclose(hr2, hr)  # HR target is the original clean patch


def test_rng_state_restored():
    """degrade() must not perturb the caller's torch RNG state."""
    deg = _degrader()
    hr = torch.rand(1, 3, 64, 64)
    torch.manual_seed(0)
    after_a = torch.randint(0, 1000, (8,))
    deg.degrade(hr, seed=99)  # should restore RNG state
    torch.manual_seed(0)
    after_b = torch.randint(0, 1000, (8,))
    assert torch.allclose(after_a, after_b), "degrade leaked RNG state"