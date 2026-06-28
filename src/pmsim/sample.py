"""Load a 3D voxel sample (delta + beta) from disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass
class VoxelSample:
    delta: np.ndarray  # shape (nz, ny, nx)
    beta: np.ndarray   # shape (nz, ny, nx)
    voxel_size_m: float  # isotropic voxel pitch in metres

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return self.delta.shape  # type: ignore[return-value]

    @property
    def extent_m(self) -> tuple[float, float, float]:
        nz, ny, nx = self.shape_zyx
        v = self.voxel_size_m
        return (nz * v, ny * v, nx * v)


def _read_volume(path: str | Path) -> np.ndarray:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        arr = tifffile.imread(str(p))
    elif suffix == ".npy":
        arr = np.load(p)
    elif suffix in {".h5", ".hdf5"}:
        import h5py  # optional dependency

        with h5py.File(p, "r") as fh:
            keys = list(fh.keys())
            if not keys:
                raise ValueError(f"{p} contains no datasets")
            arr = fh[keys[0]][...]
    else:
        raise ValueError(f"Unsupported volume extension: {suffix}")
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D array, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def load_sample(
    delta_path: str | Path,
    beta_path: str | Path | None,
    voxel_size_um: float,
    axis_order: str = "zyx",
) -> VoxelSample:
    delta = _read_volume(delta_path)
    beta = _read_volume(beta_path) if beta_path is not None else np.zeros_like(delta)
    if delta.shape != beta.shape:
        raise ValueError(
            f"delta shape {delta.shape} differs from beta shape {beta.shape}"
        )
    if axis_order == "xyz":
        delta = np.transpose(delta, (2, 1, 0))
        beta = np.transpose(beta, (2, 1, 0))
    elif axis_order != "zyx":
        raise ValueError(f"Unsupported axis_order: {axis_order}")

    return VoxelSample(
        delta=np.ascontiguousarray(delta),
        beta=np.ascontiguousarray(beta),
        voxel_size_m=voxel_size_um * 1e-6,
    )


def project_axis_aligned(
    sample: VoxelSample,
    wavelength_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned (zero-tilt) projection.

    Returns the phase shift (rad) and absorption (linear, μ·t) maps on the
    sample's native (ny, nx) grid.
    """
    dz = sample.voxel_size_m
    delta_proj = sample.delta.sum(axis=0) * dz
    beta_proj = sample.beta.sum(axis=0) * dz
    k = 2.0 * np.pi / wavelength_m
    phase = -k * delta_proj
    absorb = 2.0 * k * beta_proj  # μ·t
    return phase.astype(np.float32), absorb.astype(np.float32)
