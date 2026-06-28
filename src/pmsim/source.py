"""Extended-source point sampling for partial-coherence simulation."""

from __future__ import annotations

import numpy as np

from .constants import fwhm_to_sigma


def _halton(n: int, base: int) -> np.ndarray:
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        f = 1.0
        r = 0.0
        k = i + 1
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        out[i] = r
    return out


def sample_source(
    shape: str,
    size_um: float,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Return an (n_samples, 2) array of source-point offsets in metres.

    `size_um` is the FWHM for gaussian, the diameter for uniform_disk, and the
    side length for uniform_square. `shape == "point"` (or n_samples == 1 with
    size_um == 0) returns a single (0, 0) sample.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if shape == "point" or size_um == 0.0:
        return np.zeros((n_samples, 2), dtype=np.float64)

    size_m = size_um * 1e-6

    if seed >= 0:
        # Halton sequence for fast convergence
        u1 = _halton(n_samples, 2)
        u2 = _halton(n_samples, 3)
    else:
        rng = np.random.default_rng(None)
        u1 = rng.random(n_samples)
        u2 = rng.random(n_samples)

    if shape == "gaussian":
        sigma = fwhm_to_sigma(size_m)
        # Box-Muller from (u1, u2)
        r = np.sqrt(-2.0 * np.log(np.clip(u1, 1e-12, 1.0)))
        theta = 2.0 * np.pi * u2
        xs = sigma * r * np.cos(theta)
        ys = sigma * r * np.sin(theta)
    elif shape == "uniform_disk":
        radius = 0.5 * size_m
        r = radius * np.sqrt(u1)
        theta = 2.0 * np.pi * u2
        xs = r * np.cos(theta)
        ys = r * np.sin(theta)
    elif shape == "uniform_square":
        half = 0.5 * size_m
        xs = (u1 - 0.5) * 2.0 * half
        ys = (u2 - 0.5) * 2.0 * half
    else:
        raise ValueError(f"Unknown focal-spot shape: {shape}")

    return np.stack([xs, ys], axis=1)
