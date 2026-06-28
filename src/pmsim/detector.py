"""Detector model: efficiency, photon shot noise, Gaussian read noise."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def lookup_efficiency(
    efficiency: float | Sequence[Sequence[float]],
    energy_keV: float,
) -> float:
    """Return detector efficiency at ``energy_keV``.

    ``efficiency`` is either a scalar in [0, 1] or a sorted table of
    ``[[E_keV, eff], ...]`` which is linearly interpolated.
    """
    if isinstance(efficiency, (int, float)):
        return float(efficiency)
    table = np.asarray(efficiency, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError("efficiency table must have shape (N, 2)")
    E = table[:, 0]
    eff = table[:, 1]
    order = np.argsort(E)
    return float(np.interp(energy_keV, E[order], eff[order]))


def apply_detector(
    intensity: np.ndarray,
    n_photons_per_pixel: float,
    efficiency: float = 1.0,
    read_noise_e: float = 0.0,
    include_poisson: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Convert a unit-flat-field intensity map into a noisy photon-count image.

    ``intensity`` is expected to be normalised such that an empty beam reads ~1.
    """
    rng = np.random.default_rng(seed)
    mean_counts = intensity * n_photons_per_pixel * efficiency
    if include_poisson:
        # rng.poisson rejects negatives; clip to be safe (numerical jitter)
        mean_counts = np.clip(mean_counts, 0.0, None)
        counts = rng.poisson(mean_counts).astype(np.float32)
    else:
        counts = mean_counts.astype(np.float32)
    if read_noise_e > 0.0:
        counts = counts + rng.normal(0.0, read_noise_e, size=counts.shape).astype(np.float32)
    return counts
