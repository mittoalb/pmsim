"""Command-line entry point: ``pmsim simulate --config foo.json [...]``"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .simulator import Simulator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pmsim",
        description="Propagation-based phase-contrast X-ray projection microscope simulator.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sim = sub.add_parser("simulate", help="Run a simulation from a JSON config.")
    sim.add_argument("--config", "-c", required=True, type=Path, help="Path to JSON config file.")
    sim.add_argument("--sample", type=Path, default=None, help="Override delta_path in config.")
    sim.add_argument("--beta", type=Path, default=None, help="Override beta_path in config.")
    sim.add_argument("--voxel-size", type=float, default=None, help="Override voxel_size_um.")
    sim.add_argument("--r1", type=float, default=None, help="Override source_to_sample_mm.")
    sim.add_argument("--r2", type=float, default=None, help="Override sample_to_detector_mm.")
    sim.add_argument("--output", "-o", type=Path, default=None, help="Output TIFF path.")
    sim.add_argument("--source-samples", type=int, default=None)
    sim.add_argument("--seed", type=int, default=None)
    sim.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None,
                     help="Compute backend (overrides simulation.device).")
    sim.add_argument(
        "--print-metadata",
        action="store_true",
        help="Print simulation metadata as JSON on stdout after writing the image.",
    )

    info = sub.add_parser(
        "info",
        help="Print field-of-view / resolution / geometry for a config (no simulation).",
    )
    info.add_argument("--config", "-c", required=True, type=Path)
    info.add_argument("--r1", type=float, default=None)
    info.add_argument("--r2", type=float, default=None)
    info.add_argument("--voxel-size", type=float, default=None)

    return parser


def _collect_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    sample: dict = {}
    # Resolve CLI paths to absolute paths so they aren't re-joined with the
    # config-file directory in load_config (which is the right behaviour for
    # paths typed inside the JSON, wrong for ones typed on the command line
    # relative to the shell's CWD).
    if args.sample is not None:
        sample["delta_path"] = str(Path(args.sample).resolve())
    if args.beta is not None:
        sample["beta_path"] = str(Path(args.beta).resolve())
    if args.voxel_size is not None:
        sample["voxel_size_um"] = args.voxel_size
    if sample:
        overrides["sample"] = sample

    geom: dict = {}
    if args.r1 is not None:
        geom["source_to_sample_mm"] = args.r1
    if args.r2 is not None:
        geom["sample_to_detector_mm"] = args.r2
    if geom:
        overrides["geometry"] = geom

    if args.output is not None:
        overrides["output"] = {"path": str(args.output)}

    sim: dict = {}
    if args.source_samples is not None:
        sim["source_samples"] = args.source_samples
    if args.seed is not None:
        sim["seed"] = args.seed
    if args.device is not None:
        sim["device"] = args.device
    if sim:
        overrides["simulation"] = sim
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "simulate":
        cfg = load_config(args.config, overrides=_collect_overrides(args))
        sim = Simulator(cfg)
        # Always show a one-paragraph geometry summary before running.
        geo = sim.geometry_summary()
        _print_geometry(geo)
        result = sim.run()
        out_path = sim.save(result)
        print(f"wrote {out_path}", file=sys.stderr)
        if args.print_metadata:
            print(json.dumps(result.metadata, indent=2))
        return 0
    if args.cmd == "info":
        overrides: dict = {}
        geom: dict = {}
        if args.r1 is not None:
            geom["source_to_sample_mm"] = args.r1
        if args.r2 is not None:
            geom["sample_to_detector_mm"] = args.r2
        if geom:
            overrides["geometry"] = geom
        if args.voxel_size is not None:
            overrides["sample"] = {"voxel_size_um": args.voxel_size}
        cfg = load_config(args.config, overrides=overrides)
        _print_geometry(Simulator(cfg).geometry_summary())
        return 0
    parser.print_help()
    return 1


def _print_geometry(geo: dict) -> None:
    fov_c = geo["fov_camera_mm"]
    fov_s = geo["fov_sample_um"]
    pix = geo["detector_pixels"]
    M_geom = geo["magnification_geometric"]
    M_obj = geo["magnification_objective"]
    M_tot = geo["magnification_total"]
    print(
        "pmsim geometry summary\n"
        f"  energy              : {geo['energy_keV']:.3f} keV   (lambda = {geo['wavelength_m']*1e12:.3f} pm)\n"
        f"  R1 (focus→sample)   : {geo['R1_mm']:.3f} mm\n"
        f"  R2 (sample→det.)    : {geo['R2_mm']:.3f} mm\n"
        f"  geometric mag       : {M_geom:.4f} ×   (cone-beam, (R1+R2)/R1)\n"
        f"  objective mag       : {M_obj:.4f} ×   (lens-coupled scintillator)\n"
        f"  TOTAL mag           : {M_tot:.4f} ×\n"
        "  --- pixel sizes ---\n"
        f"  camera pixel        : {geo['camera_pixel_um']:.4f} µm   (physical CCD pixel)\n"
        f"  detector pixel      : {geo['dx_det_um']:.5f} µm   (at scintillator = camera / objective)\n"
        f"  sample-plane pixel  : {geo['dx_eff_um']:.6f} µm   (= camera / total mag)\n"
        "  --- field of view ---\n"
        f"  camera array        : {pix[0]} × {pix[1]} pixels\n"
        f"  camera FOV          : {fov_c[0]:.3f} × {fov_c[1]:.3f} mm\n"
        f"  sample FOV          : {fov_s[0]:.4f} × {fov_s[1]:.4f} µm\n"
        "  --- input sampling ---\n"
        f"  phantom voxel       : {geo['voxel_um']:.4f} µm  "
        f"(should be ≤ sample-plane pixel for the projection to be well sampled)\n"
        "  --- partial-coherence blur (geometric, pre-deconvolution) ---\n"
        f"  source PSF FWHM     : {geo['source_blur_sample_um']:.6f} µm @ sample  /  "
        f"{geo['source_blur_detector_um']:.4f} µm @ detector\n"
        f"  detector-pixel limit: {geo['dx_eff_um']:.6f} µm @ sample\n"
        f"  geometric PSF FWHM  : {geo['effective_resolution_sample_um']:.6f} µm @ sample\n"
        "                        (pre-deconvolution PSF; deconvolution against a\n"
        "                         known source distribution is SNR-limited, not\n"
        "                         PSF-limited)\n"
        "  --- Fresnel sampling diagnostics ---\n"
        f"  sim grid (padded)   : {geo['sim_grid_padded'][0]} × {geo['sim_grid_padded'][1]}  "
        f"(fft_padding={geo['fft_padding']}; raise this if memory allows)\n"
        f"  Q = N·dx²/(λ·d_eff) : {geo['fresnel_sampling_Q']:.3f}    "
        f"({'OK — angular spectrum well sampled' if geo['fresnel_sampling_Q'] >= 1.0 else 'UNDERSAMPLED — chirp aliases; will auto-split into ' + str(geo['fresnel_auto_n_steps']) + ' substeps'})\n"
        "  --- beam cone vs. detector / sample ---\n"
        f"  divergence (full)   : {geo['divergence_mrad_full']:.3f} mrad   (taken as full cone-opening angle)\n"
        f"  beam ⌀ @ sample     : {geo['beam_diameter_sample_um']:.3f} µm   (sample FOV diag = {geo['sample_fov_diagonal_um']:.3f} µm)\n"
        f"  beam ⌀ @ detector   : {geo['beam_diameter_detector_mm']:.3f} mm   (camera diag      = {geo['camera_diagonal_mm']:.3f} mm)\n"
        f"  at sample           : {geo['beam_vs_sample']}\n"
        f"  at detector         : {geo['beam_vs_camera']}\n",
        file=sys.stderr,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
