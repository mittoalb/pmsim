"""Generate a bundle of randomly-oriented nylon wires as a (delta, beta) voxel volume.

Wires are infinite cylinders clipped to the volume. Their start points are drawn
uniformly inside the volume and their directions uniformly on the sphere. The
voxel mask of each cylinder is the set of voxels whose centre lies within
``radius`` of the cylinder axis.

Default material: nylon-6,6 at 30 keV — delta = 2.7e-7, beta = 1.5e-10
(CXRO; density 1.14 g/cm^3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

# nylon-6,6, density 1.14 g/cm^3, at 30 keV (CXRO)
DELTA_NYLON_30keV = 2.7e-7
BETA_NYLON_30keV = 1.5e-10


def _random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform directions on the unit sphere."""
    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def make_wire_bundle(
    n_voxels: int = 256,
    voxel_size_um: float = 0.5,
    n_wires: int = 25,
    diameter_um: float = 5.0,
    bundle_radius_um: float | None = None,
    seed: int = 7,
) -> np.ndarray:
    """Build a boolean (nz, ny, nx) mask containing the union of all wires.

    ``bundle_radius_um`` constrains the wire start points to a cylinder of that
    radius around the volume centre (so the wires look bundled, not scattered).
    Defaults to one-quarter of the volume edge.
    """
    rng = np.random.default_rng(seed)
    vox = voxel_size_um  # microns per voxel
    n = n_voxels

    if bundle_radius_um is None:
        bundle_radius_um = 0.25 * n * vox

    radius_vox = 0.5 * diameter_um / vox

    # Volume coordinate grids (in voxel units, centred on the volume centre)
    g = np.arange(n) - (n - 1) / 2.0
    Z, Y, X = np.meshgrid(g, g, g, indexing="ij")  # each (n, n, n)

    mask = np.zeros((n, n, n), dtype=bool)

    # Random start points inside a central cylinder oriented along z
    br_vox = bundle_radius_um / vox
    angles = rng.uniform(0.0, 2.0 * np.pi, n_wires)
    radii = br_vox * np.sqrt(rng.uniform(0.0, 1.0, n_wires))
    p0 = np.stack(
        [
            rng.uniform(-n / 2, n / 2, n_wires),       # z
            radii * np.sin(angles),                    # y
            radii * np.cos(angles),                    # x
        ],
        axis=1,
    )
    dirs = _random_unit_vectors(n_wires, rng)

    r2_thresh = radius_vox * radius_vox
    for p, d in zip(p0, dirs):
        # vector from each voxel centre to wire start
        vz = Z - p[0]
        vy = Y - p[1]
        vx = X - p[2]
        # parallel component: (v · d)
        t = vz * d[0] + vy * d[1] + vx * d[2]
        # perpendicular component vector
        pz = vz - t * d[0]
        py = vy - t * d[1]
        px = vx - t * d[2]
        d2 = pz * pz + py * py + px * px
        mask |= d2 <= r2_thresh

    return mask


def main(out_dir: str, n_voxels: int, voxel_size_um: float,
         n_wires: int, diameter_um: float, bundle_radius_um: float | None,
         delta: float, beta: float, seed: int) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mask = make_wire_bundle(
        n_voxels=n_voxels,
        voxel_size_um=voxel_size_um,
        n_wires=n_wires,
        diameter_um=diameter_um,
        bundle_radius_um=bundle_radius_um,
        seed=seed,
    )
    print(f"wire volume fraction: {mask.mean():.4f}")

    delta_vol = (mask * delta).astype(np.float32)
    beta_vol = (mask * beta).astype(np.float32)

    delta_path = out / "wires_delta.tif"
    beta_path = out / "wires_beta.tif"
    tifffile.imwrite(str(delta_path), delta_vol)
    tifffile.imwrite(str(beta_path), beta_vol)
    print(f"wrote {delta_path}  shape={delta_vol.shape}  voxel={voxel_size_um} um")
    print(f"wrote {beta_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="examples")
    p.add_argument("--n", type=int, default=256, help="voxels per side (cube)")
    p.add_argument("--voxel-size-um", type=float, default=0.5)
    p.add_argument("--n-wires", type=int, default=25)
    p.add_argument("--diameter-um", type=float, default=5.0)
    p.add_argument("--bundle-radius-um", type=float, default=None,
                   help="radial confinement of wire start points (default: n*vox/4)")
    p.add_argument("--delta", type=float, default=DELTA_NYLON_30keV,
                   help="refractive index decrement (default: nylon @ 30 keV)")
    p.add_argument("--beta", type=float, default=BETA_NYLON_30keV,
                   help="absorption index (default: nylon @ 30 keV)")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    main(args.out_dir, args.n, args.voxel_size_um, args.n_wires,
         args.diameter_um, args.bundle_radius_um, args.delta, args.beta, args.seed)
