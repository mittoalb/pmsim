"""Physical constants and small unit-conversion helpers."""

from __future__ import annotations

import numpy as np

H_PLANCK = 6.62607015e-34  # J*s
C_LIGHT = 2.99792458e8  # m/s
E_CHARGE = 1.602176634e-19  # C
R_E = 2.8179403262e-15  # m, classical electron radius


def keV_to_joule(energy_keV: float) -> float:
    return energy_keV * 1.0e3 * E_CHARGE


def wavelength_m(energy_keV: float) -> float:
    """X-ray wavelength in metres for a photon energy in keV."""
    return H_PLANCK * C_LIGHT / keV_to_joule(energy_keV)


def fwhm_to_sigma(fwhm: float) -> float:
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
