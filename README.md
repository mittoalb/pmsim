# pmsim

Propagation-based phase-contrast X-ray projection microscope simulator.

`pmsim` simulates a cone-beam X-ray imaging geometry (lab microfocus or
focused-beam nano-imaging) given a 3D voxel object specified by its complex
refractive index decrement (δ, β) and a JSON configuration describing the
source, detector and geometry. The partial coherence of the extended focal
spot is modelled honestly as an incoherent sum over many source points — not
as a post-hoc Gaussian blur applied to a coherent image.

The detector model supports indirect (scintillator + visible-light objective +
camera) detection, so the **geometric magnification from the focused beam** and
the **optical magnification from the objective** are tracked as two separate
multipliers and the report shows pixel sizes at every plane (camera,
scintillator, sample).

Optional **GPU backend** via CuPy: drop-in for the Fresnel inner loop, ~30–80×
speedup at large grid sizes; auto-selected when CuPy is importable.

## Install

```bash
# any modern Python ≥3.9
pip install -e .                # runtime only (CPU)
pip install -e .[dev]           # adds pytest, h5py, ruff
pip install -e .[cuda12]        # add CuPy for the GPU backend (CUDA 12)
pip install -e .[cuda11]        # CUDA 11 variant
```

## Quick start — three commands

```bash
# 1. build a phantom (here: bundle of randomly-oriented nylon wires)
python examples/make_wire_bundle.py --out-dir examples \
       --n 256 --voxel-size-um 0.5 --n-wires 25 --diameter-um 5.0

# 2. print the geometry without running the sim (cheap sanity check)
pmsim info -c examples/config_wires.json

# 3. run the simulation (uses the GPU if CuPy is available)
pmsim simulate -c examples/config_wires.json -o wires_projection.tif
```

`pmsim simulate` writes two TIFFs next to the output path:
`<out>.tif` (the image, with noise) and `<out>_flatfield.tif` (matching
empty-beam reference). Divide one by the other in any image viewer to expose
the propagation fringes. With `output.save_intermediate: true` you also get
`_phase.tif` and `_absorption.tif` (projected phase and μ·t maps on the
simulation grid).

## CLI

`pmsim info -c CONFIG.json [--r1 mm] [--r2 mm] [--voxel-size um]`
&nbsp;&nbsp; prints a one-screen geometry summary — magnifications, pixel
sizes at every plane, FOV, source PSF, **Fresnel-sampling diagnostic**, and
**beam-cone fit** to the detector. No simulation runs. Use this to size
phantoms and pick distances before committing to a long run.

`pmsim simulate -c CONFIG.json [overrides...]`
&nbsp;&nbsp; runs the simulation. The geometry summary is always shown first;
per-stage timings are printed as the run progresses.

Overrides (each takes precedence over the JSON; relative file paths on the
command line are resolved against the current working directory, not the
config-file directory):
`--sample PATH`, `--beta PATH`, `--voxel-size UM`,
`--r1 MM`, `--r2 MM`,
`--source-samples N`, `--seed N`,
`--device {cpu|cuda|auto}`,
`--output PATH`, `--print-metadata`.

### What `pmsim info` prints

```
pmsim geometry summary
  energy              : 23.100 keV   (lambda = 53.673 pm)
  R1 (focus→sample)   : 75.000 mm
  R2 (sample→det.)    : 850.000 mm
  geometric mag       : 12.3333 ×   (cone-beam, (R1+R2)/R1)
  objective mag       : 10.0000 ×   (lens-coupled scintillator)
  TOTAL mag           : 123.3333 ×
  --- pixel sizes ---
  camera pixel        : 4.6000 µm   (physical CCD pixel)
  detector pixel      : 0.46000 µm   (at scintillator = camera / objective)
  sample-plane pixel  : 0.037297 µm   (= camera / total mag)
  --- field of view ---
  camera array        : 4432 × 2368 pixels
  camera FOV          : 20.387 × 10.893 mm
  sample FOV          : 165.302 × 88.320 µm
  --- input sampling ---
  phantom voxel       : 0.0500 µm  (should be ≤ sample-plane pixel)
  --- partial-coherence blur (geometric, pre-deconvolution) ---
  source PSF FWHM     : 0.018378 µm @ sample  /  0.227 µm @ detector
  detector-pixel limit: 0.037297 µm @ sample
  geometric PSF FWHM  : 0.037297 µm @ sample
  --- Fresnel sampling diagnostics ---
  sim grid (padded)   : 8864 × 4736  (fft_padding=2)
  Q = N·dx²/(λ·d_eff) : 3.333    (OK — angular spectrum well sampled)
  --- beam cone vs. detector / sample ---
  divergence (full)   : 20.000 mrad
  beam ⌀ @ sample     : 1500.000 µm   (sample FOV diag = 187.417 µm)
  beam ⌀ @ detector   : 18.500 mm    (camera diag      = 23.115 mm)
  at sample           : beam larger than sample FOV — uniformly illuminated
  at detector         : beam SMALLER than camera — corners in shadow
```

## Python API

```python
from pmsim import Simulator, load_config

cfg = load_config("examples/config_wires.json", overrides={
    "geometry": {"source_to_sample_mm": 5.0},
    "simulation": {"device": "cuda"},
})
sim = Simulator(cfg)
print(sim.geometry_summary())          # dict — same numbers as `pmsim info`
result = sim.run()                     # SimulationResult
sim.save(result, "wires_projection.tif")
print(result.metadata["magnification_total"],
      result.metadata["fresnel_sampling_Q"],
      result.metadata["fresnel_auto_n_steps"])
```

## Configuration schema

Validated against [`src/pmsim/schema/config.schema.json`](src/pmsim/schema/config.schema.json).

| group        | required keys                                | optional keys |
|--------------|----------------------------------------------|---------------|
| `beam`       | `energy_keV`, `focal_spot {size_um, shape}` | `spectrum` (`[[E_keV, weight], …]`), `divergence_mrad` |
| `detector`   | `pixels` (Nx, Ny), `pixel_size_um`           | `objective_magnification` (default 1), `efficiency` (scalar or `[E, eff]` table), `read_noise_e`, `include_poisson` |
| `geometry`   | `source_to_sample_mm` (R1), `sample_to_detector_mm` (R2) | |
| `sample`     | `type: voxel`, `voxel_size_um`               | `delta_path`, `beta_path` (TIFF / NPY / HDF5), `axis_order` (`zyx` default) |
| `simulation` | —                                            | `source_samples`, `n_photons_per_pixel`, `seed`, `fft_padding`, `dtype` (`complex64`/`complex128`), `device` (`cpu`/`cuda`/`auto`, default `auto`) |
| `output`     | —                                            | `path`, `save_intermediate` |

`focal_spot.shape` ∈ `{gaussian, uniform_disk, uniform_square, point}`.
For `gaussian`, `size_um` is the FWHM; for the uniform shapes it's the
diameter / side. `point` ignores `size_um`.

`beam.divergence_mrad` is currently **informational only** — it drives the
beam-cone diagnostic in `pmsim info` (and the metadata) so you can spot
camera-corner vignetting, but no intensity falloff is applied to the
simulated image.

## Bundled examples

- [`examples/config_example.json`](examples/config_example.json) — 30 keV,
  5 µm focal spot, 100 + 500 mm, 512² @ 6.5 µm direct detector. Paired with
  [`make_sphere_phantom.py`](examples/make_sphere_phantom.py).
- [`examples/config_wires.json`](examples/config_wires.json) — 23.1 keV,
  20 nm focal spot, focused-beam geometry, **Hamamatsu Orca Fire**
  (4432×2368 @ 4.6 µm) with 10× objective. Paired with either of:
  - [`make_wire_bundle.py`](examples/make_wire_bundle.py) — randomly-oriented
    nylon wires inside a 3D cube. Knobs for diameter, count, bundle radius.
  - [`make_axon_field.py`](examples/make_axon_field.py) — myelinated brain
    axons. Default `--mode in-plane` lays cylinders flat in the (x, y)
    plane (perpendicular to the beam) so projection along z gives the
    correct chord-length taper across each wire. `--mode cross` puts them
    along z for the canonical white-matter cross-section view.

## Physics summary

1. **Cone-beam → parallel beam** via the Fresnel scaling theorem:
   `M_geom = (R1+R2) / R1`, `d_eff = R1·R2 / (R1+R2)`,
   `dx_eff = dx_det / M_geom`  with  `dx_det = camera_pixel / objective_mag`.
2. **Sample projection** (axis-aligned, integrated along z):
   `φ = -k ∫ δ dz`, `μt = 2k ∫ β dz`. The maps are cubic-resampled onto the
   simulation grid with a Gaussian voxel-Nyquist low-pass to suppress
   bilinear-seam Talbot artefacts when the phantom voxel is coarser than
   the sample-plane pixel.
3. **Transmission**: `T(x,y) = exp(iφ - μt/2)`.
4. **Partial coherence**: for each of `N_s` source points (Halton-sampled
   from the focal-spot intensity distribution, Box-Muller for Gaussian),
   the incident wave at the sample plane is a tilted plane wave
   `exp(-i k (α_x x + α_y y))` with `α = source_pos / R1`. The product
   `T · carrier` is Fresnel-propagated by `d_eff` and the intensity is
   accumulated. The detector image is `I = ⟨|U|²⟩` over the source points;
   the carrier tilt induces the correct geometric image shift of
   `−x_s · R2/R1` on the detector.
5. **Polychromatic (optional)**: outer loop over `[E_keV, weight]` entries
   in `beam.spectrum`, weighted by detector efficiency at each energy.
6. **Detector**: simulation runs on the sample-plane grid at `dx_eff` —
   already aligned with detector pixels after the geometric demagnification.
   Scintillator pixel = camera_pixel / `objective_magnification`. Efficiency
   scales mean counts, then Poisson shot noise and Gaussian read noise.

### Fresnel-sampling Q and auto multi-step propagation

The angular-spectrum FFT propagator only samples the chirp without aliasing
when `Q ≡ N·dx² / (λ·d_eff) ≥ 1`. At very high magnification or large
propagation distance this is easy to violate (a typical nano-imaging setup
can land at `Q ≈ 0.1`). `pmsim` automatically splits the propagation into
`K = ⌈1/Q⌉` substeps so each one is well-sampled in frequency space, and
they compose to the same total distance. This is a software construct, not
a physical limit — increasing `K` (or `fft_padding`) recovers accuracy to
machine precision. The result is reported as `fresnel_auto_n_steps` in
the metadata.

### What "geometric resolution" means here

`pmsim info` reports a `geometric PSF FWHM` at the sample plane, taken as
the larger of the sample-plane pixel and the source-blur FWHM
(`spot × R2/(R1+R2)`). This is the **pre-deconvolution** system PSF — what
one bright point will smear to on a noisy image. It is **not** a hard
resolution cutoff: with a known source distribution (the metadata records
enough to reconstruct it), deconvolution is SNR-limited, not PSF-limited.

## Compute backends and performance

`simulation.device` selects the backend (default `auto`):

| value | behaviour |
|---|---|
| `cpu` | NumPy + SciPy FFT. Always works. |
| `cuda` | CuPy on a CUDA device. Errors if CuPy/CUDA aren't available. |
| `auto` | `cuda` if CuPy importable and a device is visible, otherwise `cpu`. |

The Fresnel inner loop (FFT → chirp multiply → IFFT, repeated over source
points and substeps) is what scales with `source_samples × n_steps × N_energies`
and is fully on the chosen device. The one-shot per-energy projection
resample stays on CPU (scipy.ndimage) because some CUDA toolkits can't
JIT-compile `cupyx.scipy.ndimage.map_coordinates`.

Indicative wall times on a Hamamatsu-Orca-Fire grid (8864 × 4736 padded
sim grid, 64 source points):

| Backend | Time |
|---|---|
| CPU (16-core Xeon) | ~14 min |
| CuPy on GTX 1080 Ti | ~25 s |

## Approximations and limitations

- The in-sample ray shear `α · z` from an off-axis source is neglected —
  the axis-aligned projection is reused for every source point. For typical
  configurations (e.g. a 20 nm spot at R1 = 5 mm with a 1 µm sample), this
  is well under one voxel; the dominant partial-coherence effect, the
  propagation-side source blur, is captured exactly. For very thick samples
  or very large spots, plug a sheared projection into
  [`projector.py`](src/pmsim/projector.py).
- δ(E) and β(E) come from the user-supplied voxel stack. A polychromatic
  run reuses the same stack at every energy — supply per-energy stacks
  externally if you need spectral material dispersion.
- No detector MTF or pixel cross-talk in v1 — only objective demagnification
  + efficiency + Poisson shot noise + Gaussian read noise.
- `beam.divergence_mrad` does not vignette the simulated image (only flagged
  in the diagnostic). Add an explicit cone-aperture mask in `simulator.py`
  if you need to enforce it.

## Tests

```bash
pytest -q
```

Eight unit / integration tests cover the Fresnel propagator (round-trip,
plane-wave intensity preservation), the sample loader (axis ordering,
sphere chord-length sanity), partial coherence (fringe power decreases
monotonically with focal-spot size), and an end-to-end sphere phantom.
