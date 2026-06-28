"""Cone-beam projection helpers built on top of the Fresnel scaling theorem.

For a point source at distance ``R1`` from the sample and a detector at distance
``R2`` from the sample, the cone-beam geometry is equivalent to a parallel-beam
simulation at the sample plane with

* magnification ``M = (R1 + R2) / R1``
* equivalent propagation distance ``d_eff = R1 * R2 / (R1 + R2)`` (i.e. ``R2/M``)
* simulation pixel size ``dx_eff = dx_det / M``

For an *off-axis* source point at ``(x_s, y_s, -R1)`` the equivalent wave at the
sample plane is a tilted plane wave with direction cosines
``alpha = (x_s, y_s) / R1``. Multiplying the sample-exit field by the carrier
``exp(-i k (alpha_x x + alpha_y y))`` and Fresnel-propagating by ``d_eff``
correctly reproduces the geometric image shift of ``-x_s R2 / R1`` on the
detector (after re-applying the magnification).

The in-sample shear ``alpha . z`` is neglected here. For typical lab-microscope
parameters this term is well below one voxel pitch (e.g. ``5 um / 100 mm * 100 um
= 5 nm`` for a 100 µm-thick sample) and the dominant partial-coherence effect is
the propagation-side blur captured by summing source points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


@dataclass(frozen=True)
class ConeBeam:
    R1_m: float
    R2_m: float

    @property
    def M(self) -> float:
        return (self.R1_m + self.R2_m) / self.R1_m

    @property
    def d_eff_m(self) -> float:
        return self.R1_m * self.R2_m / (self.R1_m + self.R2_m)

    def dx_eff_m(self, dx_det_m: float) -> float:
        return dx_det_m / self.M


def resample_centered(
    image: np.ndarray,
    dx_in_m: float,
    out_shape: tuple[int, int],
    dx_out_m: float,
    interp_order: int = 3,
    anti_alias: bool = True,
    xp=None,
) -> np.ndarray:
    """Resample a 2D image. Both grids are centred on their geometric centre.

    Out-of-bounds samples are filled with zero (phase / μt = 0 → transmission 1).

    When ``dx_out_m << dx_in_m`` (input voxels much coarser than the simulation
    pixel) ``interp_order=1`` produces piecewise-bilinear surfaces whose C0
    gradient seams at every voxel boundary act as a periodic grating and
    Talbot-image themselves after Fresnel propagation. Defaults are tuned to
    avoid that:

    * ``interp_order=3`` — cubic B-spline (C2 continuous) instead of bilinear.
    * ``anti_alias=True`` — when upsampling (``dx_out_m < dx_in_m``), apply a
      Gaussian low-pass to the *input* with σ chosen so the cutoff sits at the
      input's Nyquist frequency. This bandlimits the projection map to what
      the voxel grid can actually describe, killing the periodic seam
      structure regardless of interp order.

    Set ``anti_alias=False`` and ``interp_order=1`` to recover the legacy
    behaviour (useful for tests / debugging).
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")
    ny_in, nx_in = image.shape
    ny_out, nx_out = out_shape

    # The resample runs ONCE per energy (outside the Fresnel hot loop), so we
    # stay on the CPU here even when the simulator is GPU-backed — this avoids
    # `cupyx.scipy.ndimage` JIT-compilation issues that show up on some CUDA
    # toolkits / driver combinations. The Fresnel loop, which dominates the
    # wall time, still runs on the requested device. The returned array is
    # NumPy; the caller may move it to the device with `to_device(...)`.
    src = np.asarray(image)
    if anti_alias and dx_out_m < dx_in_m:
        src = gaussian_filter(
            src,
            sigma=1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0))),
            mode="constant", cval=0.0,
        )

    y_phys = (np.arange(ny_out, dtype=np.float64) - (ny_out - 1) / 2.0) * dx_out_m
    x_phys = (np.arange(nx_out, dtype=np.float64) - (nx_out - 1) / 2.0) * dx_out_m

    y_idx = y_phys / dx_in_m + (ny_in - 1) / 2.0
    x_idx = x_phys / dx_in_m + (nx_in - 1) / 2.0

    YY, XX = np.meshgrid(y_idx, x_idx, indexing="ij")
    coords = np.stack([YY, XX], axis=0)
    return map_coordinates(
        src, coords, order=interp_order, mode="constant", cval=0.0
    ).astype(image.dtype)


def tilt_carrier(
    shape: tuple[int, int],
    pixel_size_m: float,
    alpha_x: float,
    alpha_y: float,
    wavelength_m: float,
    dtype: np.dtype = np.complex64,
    xp=None,
) -> np.ndarray:
    """Return ``exp(-i k (alpha_x x + alpha_y y))`` on a centred grid.

    Multiplying the sample-exit field by this carrier and then Fresnel-propagating
    by ``d_eff`` produces a lateral shift of ``-alpha * d_eff`` in the demagnified
    frame. Pass ``xp=cupy`` to build the carrier directly on the GPU.
    """
    if xp is None:
        xp = np
    if alpha_x == 0.0 and alpha_y == 0.0:
        return xp.ones(shape, dtype=dtype)
    ny, nx = shape
    y = (xp.arange(ny, dtype=xp.float64) - (ny - 1) / 2.0) * pixel_size_m
    x = (xp.arange(nx, dtype=xp.float64) - (nx - 1) / 2.0) * pixel_size_m
    k = 2.0 * np.pi / wavelength_m
    phase = -k * (alpha_x * x[None, :] + alpha_y * y[:, None])
    return xp.exp(1j * phase).astype(dtype)
