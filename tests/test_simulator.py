"""End-to-end sphere phantom simulation."""

import json

import numpy as np
import tifffile

from pmsim import Simulator, load_config


def _sphere_phantom(tmp_path, n=64, radius_vox=18.0, delta=3e-7, beta=2e-10):
    g = np.arange(n) - (n - 1) / 2.0
    Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
    mask = (X * X + Y * Y + Z * Z <= radius_vox * radius_vox).astype(np.float32)
    dpath = tmp_path / "d.tif"
    bpath = tmp_path / "b.tif"
    tifffile.imwrite(str(dpath), (mask * delta).astype(np.float32))
    tifffile.imwrite(str(bpath), (mask * beta).astype(np.float32))
    return dpath, bpath


def _config(tmp_path, dpath, bpath):
    cfg = {
        "beam": {
            "energy_keV": 30.0,
            "focal_spot": {"size_um": 1.0, "shape": "gaussian"},
        },
        "detector": {
            "pixels": [128, 128],
            "pixel_size_um": 5.0,
            "efficiency": 1.0,
            "include_poisson": False,
        },
        "geometry": {"source_to_sample_mm": 100.0, "sample_to_detector_mm": 300.0},
        "sample": {
            "type": "voxel",
            "delta_path": str(dpath),
            "beta_path": str(bpath),
            "voxel_size_um": 0.5,
        },
        "simulation": {
            "source_samples": 4,
            "n_photons_per_pixel": 1.0e6,
            "seed": 1,
            "fft_padding": 2,
        },
        "output": {"path": str(tmp_path / "out.tif")},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path


def test_end_to_end_runs(tmp_path):
    dpath, bpath = _sphere_phantom(tmp_path)
    cfg = load_config(_config(tmp_path, dpath, bpath))
    sim = Simulator(cfg)
    result = sim.run()

    assert result.image.shape == (128, 128)
    assert result.flatfield.shape == (128, 128)
    # bright background near unity (×n_photons): mean of border pixels
    border = np.concatenate([result.image[0, :], result.image[-1, :],
                             result.image[:, 0], result.image[:, -1]])
    flat_border = np.concatenate([result.flatfield[0, :], result.flatfield[-1, :],
                                  result.flatfield[:, 0], result.flatfield[:, -1]])
    ratio = border.mean() / flat_border.mean()
    assert 0.95 < ratio < 1.05, f"border ratio {ratio}"

    # Phase contrast: ratio image should show fringe overshoot above 1 just outside the sphere edge
    ratio_img = result.image.astype(np.float64) / np.clip(result.flatfield.astype(np.float64), 1e-6, None)
    centre_row = ratio_img[ratio_img.shape[0] // 2, :]
    assert centre_row.max() > 1.005, "expected a bright phase-contrast fringe"


def test_save_writes_outputs(tmp_path):
    dpath, bpath = _sphere_phantom(tmp_path)
    cfg = load_config(_config(tmp_path, dpath, bpath))
    sim = Simulator(cfg)
    out_path = sim.save(sim.run())
    assert out_path.exists()
    assert out_path.with_name(out_path.stem + "_flatfield" + out_path.suffix).exists()
