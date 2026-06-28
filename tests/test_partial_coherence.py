"""Phase-contrast fringe contrast must decrease monotonically with focal spot size.

A point-like source produces sharp interference fringes at a phase edge. As the
source extends, the fringes are washed out (incoherent sum over source points).
We verify this with a thin phase-only object and the peak fringe overshoot
above unity as the metric.
"""

import json

import numpy as np
import tifffile

from pmsim import Simulator, load_config


def _build_thin_phase_edge(tmp_path):
    """A 2-voxel-thick phase-only slab covering half the volume in x."""
    nz, ny, nx = 4, 32, 64
    delta = np.zeros((nz, ny, nx), dtype=np.float32)
    delta[:2, :, : nx // 2] = 5.0e-7  # only first two z-slices have material
    beta = np.zeros_like(delta)
    dpath = tmp_path / "d.tif"
    bpath = tmp_path / "b.tif"
    tifffile.imwrite(str(dpath), delta)
    tifffile.imwrite(str(bpath), beta)
    return dpath, bpath


def _config(tmp_path, dpath, bpath, focal_spot_um, source_samples):
    cfg = {
        "beam": {
            "energy_keV": 20.0,
            "focal_spot": {"size_um": focal_spot_um, "shape": "uniform_disk"},
        },
        "detector": {
            "pixels": [128, 64],
            "pixel_size_um": 5.0,
            "efficiency": 1.0,
            "include_poisson": False,
        },
        "geometry": {"source_to_sample_mm": 50.0, "sample_to_detector_mm": 200.0},
        "sample": {
            "type": "voxel",
            "delta_path": str(dpath),
            "beta_path": str(bpath),
            "voxel_size_um": 1.0,
        },
        "simulation": {
            "source_samples": source_samples,
            "n_photons_per_pixel": 1.0e6,
            "seed": 42,
            "fft_padding": 2,
        },
        "output": {"path": str(tmp_path / "out.tif")},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path


def _fringe_power(image: np.ndarray, flatfield: np.ndarray) -> float:
    """Integrated squared deviation from flat-field along the central row.

    This is more robust than a peak metric: a few residual fringe lobes are
    still penalised proportionally instead of cancelling at a specific pixel.
    """
    row = image[image.shape[0] // 2, :].astype(np.float64)
    flat = flatfield[image.shape[0] // 2, :].astype(np.float64)
    ratio = row / np.clip(flat, 1e-6, None)
    return float(np.sum((ratio - np.mean(ratio)) ** 2))


def test_fringe_contrast_decreases_with_focal_spot(tmp_path):
    dpath, bpath = _build_thin_phase_edge(tmp_path)

    contrasts = []
    for fs in (0.1, 8.0, 30.0):
        cfg = load_config(_config(tmp_path, dpath, bpath, fs, source_samples=128))
        result = Simulator(cfg).run()
        contrasts.append(_fringe_power(result.image, result.flatfield))

    assert contrasts[0] > contrasts[1] > contrasts[2], (
        f"expected monotonic fringe-power decrease, got {contrasts}"
    )
