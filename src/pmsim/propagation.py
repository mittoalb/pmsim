"""Coherent Fresnel propagation in the angular-spectrum form.

Backend-agnostic: works with NumPy arrays on CPU and CuPy arrays on CUDA.
The choice is implicit — whichever module owns the input ``field`` is used for
the FFTs, the chirp construction, and the temporaries. No copies are made
between host and device inside this function.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.fft import fft2 as _scipy_fft2, ifft2 as _scipy_ifft2  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - scipy is a hard dep but be defensive
    _scipy_fft2 = _scipy_ifft2 = None  # type: ignore[assignment]


def _array_module(arr):
    """Return the numpy-compatible module that owns `arr` (numpy or cupy)."""
    try:
        import cupy as cp  # type: ignore[import-not-found]
        if isinstance(arr, cp.ndarray):
            return cp
    except Exception:
        pass
    return np


def _fft2(xp, a):
    if xp is np and _scipy_fft2 is not None:
        return _scipy_fft2(a)
    return xp.fft.fft2(a)


def _ifft2(xp, a):
    if xp is np and _scipy_ifft2 is not None:
        return _scipy_ifft2(a)
    return xp.fft.ifft2(a)


def fresnel_sampling_product(
    n_samples: int, pixel_size_m: float, wavelength_m: float, distance_m: float
) -> float:
    """Dimensionless Fresnel-sampling product Q = N·dx² / (λ·|d|).

    The angular-spectrum propagator's frequency-domain chirp samples without
    aliasing iff Q ≥ 1. For Q < 1 the chirp wraps around — split the
    propagation distance into ``K = ceil(1/Q)`` substeps to recover well-
    sampled propagation. Use Q to diagnose / drive the auto-step logic in
    ``fresnel_propagate``.
    """
    if distance_m == 0.0:
        return np.inf
    return (n_samples * pixel_size_m * pixel_size_m) / (wavelength_m * abs(distance_m))


def fresnel_propagate(
    field: np.ndarray,
    wavelength_m: float,
    distance_m: float,
    pixel_size_m: float,
    pad_factor: int = 2,
    n_steps: int | None = None,
    safety: float = 1.0,
) -> np.ndarray:
    """Propagate a 2D complex field by ``distance_m`` (Fresnel / angular spectrum).

    Implements ``U(d) = IFFT{ FFT{U(0)} · exp(-i π λ d (u²+v²)) }`` per step.

    Zero-pads by ``pad_factor`` to suppress periodic wrap-around, then crops back.
    Set ``distance_m = 0`` for a no-op.

    ``n_steps`` selects how many serial Fresnel substeps to do. ``None`` (default)
    automatically picks ``K = max(1, ceil(safety / Q))`` where Q is the
    Fresnel-sampling product of the *padded* grid; this guarantees each substep
    is well-sampled in frequency space. ``safety=1.0`` is the Nyquist boundary;
    use ``safety=2.0`` or higher to be conservative (more substeps, slower).
    """
    if distance_m == 0.0:
        return field.copy()
    if field.ndim != 2:
        raise ValueError(f"field must be 2D, got shape {field.shape}")

    xp = _array_module(field)
    ny, nx = field.shape
    pad_factor = max(1, int(pad_factor))
    py, px = ny * pad_factor, nx * pad_factor
    cdtype = field.dtype if field.dtype in (xp.complex64, xp.complex128) else xp.complex64

    if pad_factor > 1:
        padded = xp.zeros((py, px), dtype=cdtype)
        y0 = (py - ny) // 2
        x0 = (px - nx) // 2
        padded[y0:y0 + ny, x0:x0 + nx] = field
    else:
        padded = field

    # Pick number of substeps so each substep is well-sampled.
    if n_steps is None:
        Q = fresnel_sampling_product(min(py, px), pixel_size_m, wavelength_m, distance_m)
        n_steps = max(1, int(np.ceil(safety / Q))) if np.isfinite(Q) else 1
    n_steps = max(1, int(n_steps))
    d_step = distance_m / n_steps

    fy = xp.fft.fftfreq(py, d=pixel_size_m).astype(xp.float64)
    fx = xp.fft.fftfreq(px, d=pixel_size_m).astype(xp.float64)
    FX, FY = xp.meshgrid(fx, fy, indexing="xy")
    H_step = xp.exp(-1j * np.pi * wavelength_m * d_step * (FX * FX + FY * FY)).astype(cdtype)

    for _ in range(n_steps):
        padded = _ifft2(xp, _fft2(xp, padded) * H_step)

    if pad_factor > 1:
        out = padded[y0:y0 + ny, x0:x0 + nx]
    else:
        out = padded
    return out.astype(field.dtype, copy=False)
