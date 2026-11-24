"""Sanity checks for geometry, denoising and the evaluator interface."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from curate import DuplicateFilter, crop, square_for_head
from catdog_eval.data import default_pool, rebuild
from catdog_eval.manifest import check_manifest
from submission.generate import CHECKPOINT, Diffusion, generate, load_model, noisy_with_velocity
from train import loss_for_batch


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workload():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_manifest_geometry_and_official_crop():
    pixels = np.arange(120 * 160 * 3, dtype=np.uint8).reshape(120, 160, 3)
    original = Image.fromarray(pixels)

    class FixturePool:
        def image(self, pool_id):
            assert pool_id == "fixture"
            return original

    x, y, side = square_for_head([50, 20, 94, 64], 160, 120, 0.70)
    assert side == 63
    assert (x, y) == (40, 10)
    assert square_for_head([0, 10, 32, 42], 160, 120, 0.70) is None
    assert square_for_head([25, 30, 55, 70], 160, 120, 0.70) is None
    row = {"pool_id": "fixture", "x0": x, "y0": y, "size": side, "angle": 0, "label": 1}
    batch, labels = rebuild([row], FixturePool())
    expected = torch.from_numpy(np.array(crop(original, x, y, side))).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(batch[0], expected, rtol=0, atol=0)
    assert labels.tolist() == [1]


def test_hash_filter_rejects_near_duplicates():
    hashes = DuplicateFilter()
    assert hashes.add(0)
    assert not hashes.add((1 << 5) | (1 << 27) | (1 << 50))
    assert hashes.add((1 << 5) | (1 << 27) | (1 << 50) | (1 << 61))


def test_velocity_parameterization():
    clean = torch.tensor([0.25, -0.5])
    noise = torch.tensor([0.75, 0.5])
    alpha = torch.tensor([0.1, 0.8])
    noisy, velocity = noisy_with_velocity(clean, noise, alpha)
    torch.testing.assert_close(alpha.sqrt() * noisy - (1 - alpha).sqrt() * velocity, clean)
    torch.testing.assert_close((1 - alpha).sqrt() * noisy + alpha.sqrt() * velocity, noise)


def test_overfit_one_fixed_denoising_batch():
    try:
        pool = default_pool()
    except FileNotFoundError:
        pytest.skip("the raw pool is needed for this training sanity check")
    manifest = Path(__file__).resolve().parents[1] / "submission" / "manifest.csv"
    checked = check_manifest(manifest, pool)
    assert checked["errors"] == []
    rows = [next(row for row in checked["rows"] if row["label"] == label) for label in (0, 1)]
    photos, labels = rebuild(rows, pool)
    # Use one documented crop per class. The test model is never exported.
    torch.manual_seed(3024)
    model = Diffusion(width=8)
    clean = F.pixel_unshuffle(photos * 2 - 1, 2)
    noise = torch.randn_like(clean)
    steps = torch.tensor([400, 800])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004)
    before = loss_for_batch(model, clean, labels, noise, steps).item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss_for_batch(model, clean, labels, noise, steps).backward()
        optimizer.step()
    assert loss_for_batch(model, clean, labels, noise, steps).item() < 0.45 * before


@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_shape_range_determinism_and_batch_independence(scale):
    torch.manual_seed(3024)
    model = Diffusion(width=8, sampling_steps=4, guidance_scale=scale).eval()
    torch.nn.init.normal_(model.out[-1].weight, std=0.02)
    labels = torch.tensor([0, 1, 0, 1])
    first = generate(model, labels, 19)
    assert first.shape == (4, 3, 32, 32)
    assert first.isfinite().all() and 0 <= first.min() <= first.max() <= 1
    # The CPU evaluator clamps the returned tensor in place after generate returns.
    first.clamp_(0, 1)
    assert torch.equal(first, generate(model, labels, 19))
    assert (first - generate(model, labels, 20)).abs().mean() > 0.001
    changed = generate(model, torch.tensor([0, 0, 0, 0, 1, 1]), 19)
    assert torch.equal(first[[0, 2]], changed[[0, 2]])
    assert torch.equal(first[:1], generate(model, labels[:1], 19))


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="trained weights are not exported yet")
def test_exported_model_on_cpu():
    model = load_model(torch.device("cpu"))
    imgs = generate(model, torch.tensor([0, 1]), 3024)
    assert imgs.shape == (2, 3, 32, 32)
    assert imgs.isfinite().all() and 0 <= imgs.min() <= imgs.max() <= 1
    assert torch.equal(imgs, generate(model, torch.tensor([0, 1]), 3024))
