"""Generate a thin slab densely populated with myelinated axons.

Two modes, both producing 3D voxel volumes (z is the X-ray propagation axis):

* ``--mode in-plane`` (default) — axons lie **perpendicular to the beam**,
  i.e. their long axes are in the (x, y) plane. Each axon is a true 3D
  cylinder of given diameter and length, randomly oriented in xy. Projecting
  along z therefore produces the correct chord-length taper across the wire.
* ``--mode cross`` — axons run **parallel to the beam** (along z). The xy
  cross-section is a disk; useful for the canonical white-matter look.

Common parameters:

* Field of view: configurable, default 100 × 100 µm.
* Slab thickness: configurable. Cylinders in ``cross`` mode span the slab;
  in ``in-plane`` mode the slab need only contain the largest axon diameter.
* Axon diameters drawn from a **log-uniform** distribution between
  ``--diameter-min-nm`` and ``--diameter-max-nm`` (default 100 to 500 nm).
* Axon orientations in ``in-plane``: uniform in [0, π). Pass
  ``--orientation-deg X`` to align all axons at a fixed angle (like a
  white-matter tract).

Material defaults are CXRO-style values for myelin (lipid, ρ ≈ 1 g/cm³) at
23.1 keV — δ = 4.0e-7, β = 2.0e-10. Override with ``--delta`` / ``--beta``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

DELTA_MYELIN_23keV = 4.0e-7
BETA_MYELIN_23keV = 2.0e-10


def _paint_disk_through_z(mask: np.ndarray, cx: float, cy: float, r: float) -> None:
    """Mark a full-z column of voxels inside a disk (cross-section mode)."""
    nz, ny, nx = mask.shape
    margin = int(np.ceil(r)) + 1
    xmin = max(0, int(np.floor(cx)) - margin)
    xmax = min(nx, int(np.ceil(cx)) + margin)
    ymin = max(0, int(np.floor(cy)) - margin)
    ymax = min(ny, int(np.ceil(cy)) + margin)
    if xmin >= xmax or ymin >= ymax:
        return
    xs = np.arange(xmin, xmax, dtype=np.float32)
    ys = np.arange(ymin, ymax, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    disk = ((XX - cx) ** 2 + (YY - cy) ** 2) <= (r * r)
    if disk.any():
        # broadcast along z
        mask[:, ymin:ymax, xmin:xmax] |= disk[None, :, :]


def _paint_cylinder_in_xy(
    mask: np.ndarray,
    x0: float, y0: float, x1: float, y1: float,
    z_mid: float, r: float,
) -> None:
    """Mark a 3D cylinder whose axis is the in-plane segment
    (x0, y0, z_mid)-(x1, y1, z_mid), radius r voxels."""
    nz, ny, nx = mask.shape
    margin = int(np.ceil(r)) + 1
    xmin = max(0, int(np.floor(min(x0, x1))) - margin)
    xmax = min(nx, int(np.ceil(max(x0, x1))) + margin)
    ymin = max(0, int(np.floor(min(y0, y1))) - margin)
    ymax = min(ny, int(np.ceil(max(y0, y1))) + margin)
    zmin = max(0, int(np.floor(z_mid - r)) - 1)
    zmax = min(nz, int(np.ceil(z_mid + r)) + 2)
    if xmin >= xmax or ymin >= ymax or zmin >= zmax:
        return

    dx = x1 - x0
    dy = y1 - y0
    L2 = dx * dx + dy * dy

    xs = np.arange(xmin, xmax, dtype=np.float32)
    ys = np.arange(ymin, ymax, dtype=np.float32)
    zs = np.arange(zmin, zmax, dtype=np.float32) - z_mid

    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    if L2 < 1e-12:
        d_xy_2 = (XX - x0) ** 2 + (YY - y0) ** 2
    else:
        t = np.clip(((XX - x0) * dx + (YY - y0) * dy) / L2, 0.0, 1.0)
        px = x0 + t * dx
        py = y0 + t * dy
        d_xy_2 = (XX - px) ** 2 + (YY - py) ** 2

    # 3D distance² = d_xy² + (z - z_mid)²
    d3d_2 = d_xy_2[None, :, :] + (zs * zs)[:, None, None]
    mask[zmin:zmax, ymin:ymax, xmin:xmax] |= d3d_2 <= (r * r)


def make_axon_volume(args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    n = int(round(args.fov_um / args.voxel_size_um))
    # In-plane mode: slab must contain the largest cylinder's z-extent.
    min_thickness_um = (args.diameter_max_nm * 1e-3) + 2 * args.voxel_size_um
    eff_thickness_um = max(args.thickness_um, min_thickness_um) if args.mode == "in-plane" else args.thickness_um
    nz = max(1, int(round(eff_thickness_um / args.voxel_size_um)))
    if eff_thickness_um > args.thickness_um and args.mode == "in-plane":
        print(f"  (slab grown to {eff_thickness_um:.3f} µm to contain {args.diameter_max_nm} nm cylinders)")

    mask = np.zeros((nz, n, n), dtype=bool)
    z_mid = (nz - 1) / 2.0

    log_dmin = np.log(args.diameter_min_nm)
    log_dmax = np.log(args.diameter_max_nm)

    fixed_theta = (
        None if args.orientation_deg is None else np.deg2rad(float(args.orientation_deg))
    )

    for _ in range(args.n_axons):
        diam_nm = float(np.exp(rng.uniform(log_dmin, log_dmax)))
        radius_vox = 0.5 * (diam_nm * 1e-3 / args.voxel_size_um)
        cx = rng.uniform(0.0, n)
        cy = rng.uniform(0.0, n)
        if args.mode == "cross":
            _paint_disk_through_z(mask, cx, cy, radius_vox)
        elif args.mode == "in-plane":
            theta = fixed_theta if fixed_theta is not None else rng.uniform(0.0, np.pi)
            half_L_vox = 0.5 * args.segment_length_um / args.voxel_size_um
            ux = np.cos(theta)
            uy = np.sin(theta)
            x0 = cx - half_L_vox * ux
            y0 = cy - half_L_vox * uy
            x1 = cx + half_L_vox * ux
            y1 = cy + half_L_vox * uy
            _paint_cylinder_in_xy(mask, x0, y0, x1, y1, z_mid, radius_vox)
        else:
            raise ValueError(f"unknown mode {args.mode!r}")
    return mask


def main(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask = make_axon_volume(args)
    nz, ny, nx = mask.shape
    fill_3d = float(mask.mean())
    fill_xy = float(mask.any(axis=0).mean())
    print(
        f"3D mask fill: {fill_3d*100:.2f}%  "
        f"(xy footprint coverage: {fill_xy*100:.2f}%)  "
        f"shape (z,y,x) = {mask.shape}  mode={args.mode}"
    )

    delta_vol = (mask * args.delta).astype(np.float32)
    beta_vol = (mask * args.beta).astype(np.float32)
    delta_path = out / "axons_delta.tif"
    beta_path = out / "axons_beta.tif"
    tifffile.imwrite(str(delta_path), delta_vol)
    tifffile.imwrite(str(beta_path), beta_vol)
    print(f"wrote {delta_path}  shape={delta_vol.shape}  voxel={args.voxel_size_um} µm  "
          f"FOV={args.fov_um} µm  slab={nz * args.voxel_size_um:.3f} µm  axons={args.n_axons}")
    print(f"wrote {beta_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="examples")
    p.add_argument("--mode", choices=["in-plane", "cross"], default="in-plane",
                   help="axon orientation (default: in-plane = perpendicular to the beam)")
    p.add_argument("--fov-um", type=float, default=100.0)
    p.add_argument("--voxel-size-um", type=float, default=0.05)
    p.add_argument("--thickness-um", type=float, default=1.0,
                   help="slab thickness (grown automatically in in-plane mode if too small)")
    p.add_argument("--n-axons", type=int, default=200,
                   help="number of axons (200 default for in-plane; raise for cross)")
    p.add_argument("--diameter-min-nm", type=float, default=100.0)
    p.add_argument("--diameter-max-nm", type=float, default=500.0)
    p.add_argument("--segment-length-um", type=float, default=80.0,
                   help="axon segment length (in-plane mode)")
    p.add_argument("--orientation-deg", type=float, default=None,
                   help="if set, all axons share this angle (degrees from +x). "
                        "Default None = random uniform orientations.")
    p.add_argument("--delta", type=float, default=DELTA_MYELIN_23keV)
    p.add_argument("--beta", type=float, default=BETA_MYELIN_23keV)
    p.add_argument("--seed", type=int, default=11)
    main(p.parse_args())
