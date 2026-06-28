import numpy as np
import tifffile

from pmsim.sample import load_sample, project_axis_aligned


def _write_sphere(tmpdir, n=64, radius_vox=20.0, delta_val=1e-7, beta_val=2e-10):
    g = np.arange(n) - (n - 1) / 2.0
    Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
    mask = (X * X + Y * Y + Z * Z <= radius_vox * radius_vox).astype(np.float32)
    dpath = tmpdir / "d.tif"
    bpath = tmpdir / "b.tif"
    tifffile.imwrite(str(dpath), (mask * delta_val).astype(np.float32))
    tifffile.imwrite(str(bpath), (mask * beta_val).astype(np.float32))
    return dpath, bpath


def test_sphere_projection_matches_chord_length(tmp_path):
    n = 64
    R_vox = 20.0
    delta = 1e-7
    voxel_um = 0.5
    dpath, bpath = _write_sphere(tmp_path, n=n, radius_vox=R_vox, delta_val=delta)
    s = load_sample(dpath, bpath, voxel_size_um=voxel_um)
    phase, _ = project_axis_aligned(s, wavelength_m=1e-10)

    # At the sphere centre, projected thickness = 2 * R = 2 * R_vox * voxel
    R_m = R_vox * voxel_um * 1e-6
    expected_thickness = 2.0 * R_m
    k = 2.0 * np.pi / 1e-10
    expected_phase_centre = -k * delta * expected_thickness
    centre = phase[n // 2, n // 2]

    # voxel discretisation error is ~ 1 voxel of thickness on a sphere this size
    tol_thickness = 1.5 * voxel_um * 1e-6
    rel = abs(centre - expected_phase_centre) / abs(expected_phase_centre)
    assert rel < tol_thickness / expected_thickness * 2.0, (
        f"centre phase {centre} vs expected {expected_phase_centre}"
    )


def test_load_sample_axis_orders(tmp_path):
    n = 16
    arr = np.random.default_rng(0).random((n, n, n)).astype(np.float32)
    dpath = tmp_path / "d.tif"
    tifffile.imwrite(str(dpath), arr)
    s_zyx = load_sample(dpath, None, voxel_size_um=1.0, axis_order="zyx")
    s_xyz = load_sample(dpath, None, voxel_size_um=1.0, axis_order="xyz")
    assert s_zyx.delta.shape == (n, n, n)
    assert s_xyz.delta.shape == (n, n, n)
    # zyx leaves the array unchanged
    assert np.allclose(s_zyx.delta, arr)
    # xyz transposes (xyz → zyx) so should equal arr.transpose(2,1,0)
    assert np.allclose(s_xyz.delta, np.transpose(arr, (2, 1, 0)))
