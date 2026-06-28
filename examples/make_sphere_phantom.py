"""Generate a small polystyrene-sphere phantom (delta + beta) at 30 keV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

# polystyrene at 30 keV (rough, from CXRO)
DELTA_PS_30keV = 3.5e-7
BETA_PS_30keV = 2.0e-10


def make_sphere(n: int, radius_vox: float) -> np.ndarray:
    """Return a (n,n,n) float32 mask of a sphere centred in the volume."""
    g = np.arange(n) - (n - 1) / 2.0
    Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
    r2 = X * X + Y * Y + Z * Z
    return (r2 <= radius_vox * radius_vox).astype(np.float32)


def main(out_dir: str = "examples", n: int = 128, radius_um: float = 25.0,
         voxel_size_um: float = 0.5) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    radius_vox = radius_um / voxel_size_um
    mask = make_sphere(n, radius_vox)
    delta = mask * DELTA_PS_30keV
    beta = mask * BETA_PS_30keV

    delta_path = out / "phantom_delta.tif"
    beta_path = out / "phantom_beta.tif"
    tifffile.imwrite(str(delta_path), delta.astype(np.float32))
    tifffile.imwrite(str(beta_path), beta.astype(np.float32))
    print(f"wrote {delta_path} and {beta_path}  shape={delta.shape}  voxel={voxel_size_um} um")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="examples")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--radius-um", type=float, default=25.0)
    p.add_argument("--voxel-size-um", type=float, default=0.5)
    args = p.parse_args()
    main(args.out_dir, args.n, args.radius_um, args.voxel_size_um)
