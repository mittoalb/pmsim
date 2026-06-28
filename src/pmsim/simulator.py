"""Top-level simulator that ties config, sample, propagation, source and detector together."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


@contextmanager
def _stopwatch(label: str, verbose: bool):
    if not verbose:
        yield
        return
    t0 = time.perf_counter()
    yield
    print(f"  [pmsim] {label}: {time.perf_counter() - t0:6.2f} s", file=sys.stderr, flush=True)

from .backend import get_xp, to_device, to_host
from .config import Config
from .constants import wavelength_m
from .detector import apply_detector, lookup_efficiency
from .projector import ConeBeam, resample_centered, tilt_carrier
from .propagation import fresnel_propagate, fresnel_sampling_product
from .sample import VoxelSample, load_sample


@dataclass
class SimulationResult:
    image: np.ndarray            # final (ny_det, nx_det) image (photon counts, post-noise)
    flatfield: np.ndarray        # same shape, no-sample reference (post-noise)
    phase_map: np.ndarray | None  # demagnified-grid projected phase (rad), if save_intermediate
    absorption_map: np.ndarray | None  # demagnified-grid μ·t map, if save_intermediate
    metadata: dict[str, Any]


class Simulator:
    def __init__(self, config: Config):
        self.config = config
        self._sample: VoxelSample | None = None

    # ---- public API -----------------------------------------------------------
    def geometry_summary(self) -> dict[str, Any]:
        """Compute field-of-view, resolution and magnification without simulating.

        Useful as a sanity check before launching a long run.
        """
        cfg = self.config
        det = cfg.detector
        geom = cfg.geometry
        beam = cfg.beam
        R1 = geom["source_to_sample_mm"] * 1e-3
        R2 = geom["sample_to_detector_mm"] * 1e-3
        cone = ConeBeam(R1_m=R1, R2_m=R2)
        camera_pixel_m = det["pixel_size_um"] * 1e-6
        obj_mag = float(det.get("objective_magnification", 1.0))
        dx_det_m = camera_pixel_m / obj_mag
        nx_det, ny_det = int(det["pixels"][0]), int(det["pixels"][1])
        dx_eff_m = cone.dx_eff_m(dx_det_m)
        M_total = cone.M * obj_mag
        fs_size_um = float(beam["focal_spot"]["size_um"])
        source_blur_det_m = fs_size_um * 1e-6 * R2 / R1
        source_blur_sample_m = source_blur_det_m / cone.M
        voxel_um = float(cfg.sample["voxel_size_um"])
        voxel_proj_camera_m = voxel_um * 1e-6 * M_total
        effective_res_sample_m = max(source_blur_sample_m, dx_eff_m)
        # ---- Fresnel sampling diagnostics ----
        # use the padded sim grid (fft_padding × detector pixels) for Q
        pad = int(cfg.simulation.get("fft_padding", 2))
        N_pad = nx_det * pad
        Q = fresnel_sampling_product(N_pad, dx_eff_m, wavelength_m(float(beam["energy_keV"])), cone.d_eff_m)
        auto_steps = max(1, int(np.ceil(1.0 / Q))) if np.isfinite(Q) else 1

        # ---- Beam-cone geometric diameter ----
        # Treat divergence_mrad as the FULL cone-opening angle (most common
        # synchrotron convention). The cone has half-angle = div/2.
        div_full_rad = float(beam.get("divergence_mrad", 0.0)) * 1e-3
        beam_dia_sample_m = R1 * div_full_rad
        beam_dia_det_m = (R1 + R2) * div_full_rad
        camera_diag_m = np.hypot(nx_det * camera_pixel_m, ny_det * camera_pixel_m)
        sample_fov_diag_m = np.hypot(nx_det * dx_eff_m, ny_det * dx_eff_m)
        # Classify three relevant comparisons (only meaningful if div > 0).
        if div_full_rad > 0.0:
            beam_vs_camera = (
                "beam larger than camera — camera sees only the central cone (no clipping in image)"
                if beam_dia_det_m >= camera_diag_m else
                "beam SMALLER than camera — corners of the camera are in shadow (image is clipped by the cone)"
            )
            beam_vs_sample = (
                "beam larger than sample FOV — sample uniformly illuminated"
                if beam_dia_sample_m >= sample_fov_diag_m else
                "beam SMALLER than sample FOV — only the central region of the sample is illuminated"
            )
        else:
            beam_vs_camera = "divergence_mrad not set — beam cone not modelled (assumed infinite)"
            beam_vs_sample = beam_vs_camera

        return {
            "energy_keV": float(beam["energy_keV"]),
            "wavelength_m": wavelength_m(float(beam["energy_keV"])),
            "R1_mm": geom["source_to_sample_mm"],
            "R2_mm": geom["sample_to_detector_mm"],
            "magnification_geometric": cone.M,
            "magnification_objective": obj_mag,
            "magnification_total": M_total,
            "d_eff_m": cone.d_eff_m,
            "camera_pixel_um": camera_pixel_m * 1e6,
            "dx_det_um": dx_det_m * 1e6,
            "dx_eff_um": dx_eff_m * 1e6,
            "detector_pixels": [nx_det, ny_det],
            "fov_camera_mm": [nx_det * camera_pixel_m * 1e3, ny_det * camera_pixel_m * 1e3],
            "fov_detector_mm": [nx_det * dx_det_m * 1e3, ny_det * dx_det_m * 1e3],
            "fov_sample_um": [nx_det * dx_eff_m * 1e6, ny_det * dx_eff_m * 1e6],
            "voxel_um": voxel_um,
            "voxel_projected_on_camera_um": voxel_proj_camera_m * 1e6,
            "source_blur_detector_um": source_blur_det_m * 1e6,
            "source_blur_sample_um": source_blur_sample_m * 1e6,
            "effective_resolution_sample_um": effective_res_sample_m * 1e6,
            "fresnel_sampling_Q": Q,
            "fresnel_auto_n_steps": auto_steps,
            "fft_padding": pad,
            "sim_grid_padded": [N_pad, ny_det * pad],
            "divergence_mrad_full": float(beam.get("divergence_mrad", 0.0)),
            "beam_diameter_sample_um": beam_dia_sample_m * 1e6,
            "beam_diameter_detector_mm": beam_dia_det_m * 1e3,
            "camera_diagonal_mm": camera_diag_m * 1e3,
            "sample_fov_diagonal_um": sample_fov_diag_m * 1e6,
            "beam_vs_camera": beam_vs_camera,
            "beam_vs_sample": beam_vs_sample,
        }

    def load(self) -> VoxelSample:
        s = self.config.sample
        self._sample = load_sample(
            delta_path=s["delta_path"],
            beta_path=s.get("beta_path"),
            voxel_size_um=s["voxel_size_um"],
            axis_order=s.get("axis_order", "zyx"),
        )
        return self._sample

    def run(self) -> SimulationResult:
        if self._sample is None:
            self.load()
        assert self._sample is not None

        cfg = self.config
        sim = cfg.simulation
        geom = cfg.geometry
        det = cfg.detector
        beam = cfg.beam

        R1 = geom["source_to_sample_mm"] * 1e-3
        R2 = geom["sample_to_detector_mm"] * 1e-3
        cone = ConeBeam(R1_m=R1, R2_m=R2)

        # camera-pixel and effective-detector-pixel (after the optical objective)
        camera_pixel_m = det["pixel_size_um"] * 1e-6
        obj_mag = float(det.get("objective_magnification", 1.0))
        dx_det_m = camera_pixel_m / obj_mag           # footprint on scintillator
        nx_det, ny_det = int(det["pixels"][0]), int(det["pixels"][1])
        dx_eff_m = cone.dx_eff_m(dx_det_m)             # footprint at sample plane

        pad = int(sim.get("fft_padding", 2))
        ny_sim = ny_det * pad
        nx_sim = nx_det * pad
        cdtype = np.complex64 if sim.get("dtype", "complex64") == "complex64" else np.complex128
        sim_shape = (ny_sim, nx_sim)

        # ------ device backend (cpu / cuda / auto) -----------------------------
        device_req = str(sim.get("device", "cpu"))
        xp, device = get_xp(device_req)
        self._device = device  # surfaced via metadata
        cdtype_xp = xp.complex64 if cdtype is np.complex64 else xp.complex128
        verbose = bool(sim.get("verbose", True))
        if verbose:
            mem = ""
            if device == "cuda":
                try:
                    import cupy as cp  # type: ignore[import-not-found]
                    free, total = cp.cuda.runtime.memGetInfo()
                    mem = f"  (GPU mem free {free/1e9:.1f} / {total/1e9:.1f} GB)"
                except Exception:
                    pass
            print(
                f"[pmsim] device={device}  sim_grid={nx_sim}×{ny_sim}  "
                f"dtype={cdtype.__name__}{mem}",
                file=sys.stderr, flush=True,
            )

        # ------ spectrum --------------------------------------------------------
        spectrum = beam.get("spectrum")
        if spectrum:
            energies = np.asarray(spectrum, dtype=np.float64)
            E_list = energies[:, 0].tolist()
            w_list = energies[:, 1].tolist()
            total_w = sum(w_list)
            weights = [w / total_w for w in w_list]
        else:
            E_list = [float(beam["energy_keV"])]
            weights = [1.0]

        # ------ source samples --------------------------------------------------
        from .source import sample_source

        fs = beam["focal_spot"]
        source_pts = sample_source(
            shape=fs["shape"],
            size_um=fs["size_um"],
            n_samples=int(sim.get("source_samples", 64)),
            seed=int(sim.get("seed", 42)),
        )

        # ------ accumulators ----------------------------------------------------
        intensity_sum = np.zeros((ny_det, nx_det), dtype=np.float64)
        flatfield_sum = np.zeros_like(intensity_sum)

        phase_keep: np.ndarray | None = None
        absorb_keep: np.ndarray | None = None

        for E_keV, w_E in zip(E_list, weights):
            lam = wavelength_m(E_keV)
            eff = lookup_efficiency(det.get("efficiency", 1.0), E_keV)

            with _stopwatch(f"project + resample to {nx_sim}×{ny_sim} (CPU)", verbose):
                # axis-aligned projection of delta and beta
                delta_proj = self._sample.delta.sum(axis=0) * self._sample.voxel_size_m
                beta_proj = self._sample.beta.sum(axis=0) * self._sample.voxel_size_m
                k = 2.0 * np.pi / lam

                # resample onto simulation grid (centred, zero-padded by pad_factor)
                # — CPU only (see resample_centered docstring for rationale).
                phase = -k * resample_centered(
                    delta_proj.astype(np.float32),
                    self._sample.voxel_size_m,
                    sim_shape,
                    dx_eff_m,
                )
                absorb = 2.0 * k * resample_centered(
                    beta_proj.astype(np.float32),
                    self._sample.voxel_size_m,
                    sim_shape,
                    dx_eff_m,
                )

            with _stopwatch(f"build transmission + push to {device}", verbose):
                T_sample_np = np.exp(1j * phase - 0.5 * absorb).astype(cdtype)
                T_sample = to_device(T_sample_np, xp)

            # ------ Monte-Carlo over source points -----------------------------
            n_src = source_pts.shape[0]
            with _stopwatch(f"propagate × {n_src} source points (with + flat-field)", verbose):
                I_sample = xp.zeros(sim_shape, dtype=xp.float64)
                I_flat = xp.zeros(sim_shape, dtype=xp.float64)
                for x_s, y_s in source_pts:
                    alpha_x = x_s / R1
                    alpha_y = y_s / R1
                    carrier = tilt_carrier(
                        sim_shape, dx_eff_m, alpha_x, alpha_y, lam, dtype=cdtype_xp, xp=xp,
                    )
                    U = T_sample * carrier
                    U = fresnel_propagate(U, lam, cone.d_eff_m, dx_eff_m, pad_factor=1)
                    I_sample += (U.real ** 2 + U.imag ** 2)
                    U_f = carrier
                    U_f = fresnel_propagate(U_f, lam, cone.d_eff_m, dx_eff_m, pad_factor=1)
                    I_flat += (U_f.real ** 2 + U_f.imag ** 2)
                if device == "cuda":
                    import cupy as cp  # type: ignore[import-not-found]
                    cp.cuda.Stream.null.synchronize()
                I_sample /= n_src
                I_flat /= n_src

            # crop padded region back to detector size (centred), then host-copy
            y0 = (ny_sim - ny_det) // 2
            x0 = (nx_sim - nx_det) // 2
            I_sample_det = to_host(I_sample[y0:y0 + ny_det, x0:x0 + nx_det])
            I_flat_det = to_host(I_flat[y0:y0 + ny_det, x0:x0 + nx_det])

            intensity_sum += w_E * eff * I_sample_det
            flatfield_sum += w_E * eff * I_flat_det

            if cfg.output.get("save_intermediate") and phase_keep is None:
                phase_keep = phase[y0:y0 + ny_det, x0:x0 + nx_det].copy()
                absorb_keep = absorb[y0:y0 + ny_det, x0:x0 + nx_det].copy()

        # ------ detector noise --------------------------------------------------
        n_phot = float(sim.get("n_photons_per_pixel", 1.0e4))
        rn = float(det.get("read_noise_e", 0.0))
        poisson = bool(det.get("include_poisson", True))
        seed = int(sim.get("seed", 42))

        image = apply_detector(
            intensity_sum,
            n_photons_per_pixel=n_phot,
            efficiency=1.0,  # efficiency already folded into intensity_sum per-energy
            read_noise_e=rn,
            include_poisson=poisson,
            seed=seed,
        )
        flat = apply_detector(
            flatfield_sum,
            n_photons_per_pixel=n_phot,
            efficiency=1.0,
            read_noise_e=rn,
            include_poisson=poisson,
            seed=seed + 1,
        )

        # ------ FOV and resolution ---------------------------------------------
        # Detector FOV: camera-pixel × number of camera pixels (physical extent of the camera).
        fov_camera_m = (nx_det * camera_pixel_m, ny_det * camera_pixel_m)
        # Detector FOV at the scintillator (= X-ray detector plane) = camera FOV / objective.
        fov_det_m = (fov_camera_m[0] / obj_mag, fov_camera_m[1] / obj_mag)
        # Field of view at the sample plane = detector FOV / geometric magnification.
        fov_sample_m = (fov_det_m[0] / cone.M, fov_det_m[1] / cone.M)
        # Source geometric blur on the detector (FWHM) and as referred to the sample plane.
        fs_size_um = float(beam["focal_spot"]["size_um"])
        source_blur_det_m = fs_size_um * 1e-6 * R2 / R1
        source_blur_sample_m = source_blur_det_m / cone.M  # = fs * R2 / (R1+R2)
        # Sample voxel pitch as projected on the detector.
        voxel_proj_det_m = self._sample.voxel_size_m * cone.M
        # Effective resolution (FWHM) at the sample plane: max of source blur and pixel pitch.
        effective_res_sample_m = max(source_blur_sample_m, dx_eff_m)
        # Maximum sample thickness compatible with the FOV (so the geometric shadow stays
        # on the detector): ~ FOV / divergence-half-angle, useful for the user to know.
        max_sample_thickness_m = fov_sample_m[0] / 2.0 / (fs_size_um * 1e-6 / R1 + 1e-30)

        metadata = {
            "wavelength_m": [wavelength_m(E) for E in E_list],
            "energy_keV": E_list,
            "spectrum_weights": weights,
            "magnification_geometric": cone.M,
            "magnification_objective": obj_mag,
            "magnification_total": cone.M * obj_mag,
            "d_eff_m": cone.d_eff_m,
            "dx_eff_m": dx_eff_m,
            "dx_det_m": dx_det_m,
            "camera_pixel_m": camera_pixel_m,
            "source_samples": int(source_pts.shape[0]),
            "detector_pixels": [nx_det, ny_det],
            "fft_padding": pad,
            "fov_camera_m": list(fov_camera_m),
            "fov_detector_m": list(fov_det_m),
            "fov_sample_m": list(fov_sample_m),
            "source_blur_detector_m": source_blur_det_m,
            "source_blur_sample_m": source_blur_sample_m,
            "voxel_pitch_on_detector_m": voxel_proj_det_m,
            "effective_resolution_sample_m": effective_res_sample_m,
            "max_sample_thickness_in_fov_m": max_sample_thickness_m,
        }
        return SimulationResult(
            image=image,
            flatfield=flat,
            phase_map=phase_keep,
            absorption_map=absorb_keep,
            metadata=metadata,
        )

    def save(self, result: SimulationResult, path: str | Path | None = None) -> Path:
        out = Path(path or self.config.output.get("path", "projection.tif"))
        tifffile.imwrite(str(out), result.image.astype(np.float32))
        flat_path = out.with_name(out.stem + "_flatfield" + out.suffix)
        tifffile.imwrite(str(flat_path), result.flatfield.astype(np.float32))
        if self.config.output.get("save_intermediate"):
            if result.phase_map is not None:
                tifffile.imwrite(
                    str(out.with_name(out.stem + "_phase" + out.suffix)),
                    result.phase_map.astype(np.float32),
                )
            if result.absorption_map is not None:
                tifffile.imwrite(
                    str(out.with_name(out.stem + "_absorption" + out.suffix)),
                    result.absorption_map.astype(np.float32),
                )
        return out
