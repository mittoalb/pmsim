import numpy as np

from pmsim.propagation import fresnel_propagate


def test_propagation_zero_distance_is_identity():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((64, 64)).astype(np.complex64) + 1j * rng.standard_normal((64, 64)).astype(np.complex64)
    out = fresnel_propagate(f, wavelength_m=1e-10, distance_m=0.0, pixel_size_m=1e-6)
    assert np.allclose(out, f)


def test_propagation_roundtrip_recovers_input():
    """Propagate forward then backward should return the original field.

    No padding: the angular-spectrum propagator is unitary on a periodic grid.
    """
    rng = np.random.default_rng(1)
    n = 128
    f = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(np.complex128)

    lam = 1.0e-10  # 12.4 keV
    dx = 1.0e-6
    d = 0.05  # 5 cm

    forward = fresnel_propagate(f, lam, d, dx, pad_factor=1)
    back = fresnel_propagate(forward, lam, -d, dx, pad_factor=1)
    err = np.max(np.abs(back - f))
    assert err < 1e-10, f"max round-trip error {err}"


def test_propagation_plane_wave_intensity_preserved():
    """A uniform plane wave has uniform intensity after Fresnel propagation.

    Use pad_factor=1: padding a uniform field with zeros creates an artificial
    edge that leaks into the cropped output.
    """
    n = 64
    f = np.ones((n, n), dtype=np.complex64)
    out = fresnel_propagate(f, wavelength_m=1e-10, distance_m=0.1, pixel_size_m=2e-6, pad_factor=1)
    I = (out.real ** 2 + out.imag ** 2)
    assert np.allclose(I, 1.0, atol=1e-4)
