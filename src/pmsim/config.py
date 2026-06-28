"""Load and validate the simulator JSON configuration."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import jsonschema

_DEFAULT_SIMULATION = {
    "source_samples": 64,
    "n_photons_per_pixel": 1.0e4,
    "seed": 42,
    "fft_padding": 2,
    "dtype": "complex64",
    # "auto" = use CUDA when CuPy is importable, else CPU. Set to "cpu" or
    # "cuda" in your config to pin the choice.
    "device": "auto",
}

_DEFAULT_DETECTOR = {
    "efficiency": 1.0,
    "read_noise_e": 0.0,
    "include_poisson": True,
    "objective_magnification": 1.0,
}

_DEFAULT_OUTPUT = {
    "path": "projection.tif",
    "save_intermediate": False,
}

_DEFAULT_SAMPLE = {
    "axis_order": "zyx",
}


@dataclass
class Config:
    raw: dict[str, Any]
    source_path: Path | None = None
    cli_overrides: dict[str, Any] = field(default_factory=dict)

    # ---- convenience accessors -------------------------------------------------
    @property
    def beam(self) -> dict[str, Any]:
        return self.raw["beam"]

    @property
    def detector(self) -> dict[str, Any]:
        return self.raw["detector"]

    @property
    def geometry(self) -> dict[str, Any]:
        return self.raw["geometry"]

    @property
    def sample(self) -> dict[str, Any]:
        return self.raw["sample"]

    @property
    def simulation(self) -> dict[str, Any]:
        return self.raw.get("simulation", dict(_DEFAULT_SIMULATION))

    @property
    def output(self) -> dict[str, Any]:
        return self.raw.get("output", dict(_DEFAULT_OUTPUT))


def _schema() -> dict[str, Any]:
    text = files("pmsim.schema").joinpath("config.schema.json").read_text()
    return json.loads(text)


def _apply_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg.setdefault("simulation", {})
    for k, v in _DEFAULT_SIMULATION.items():
        cfg["simulation"].setdefault(k, v)
    for k, v in _DEFAULT_DETECTOR.items():
        cfg["detector"].setdefault(k, v)
    for k, v in _DEFAULT_SAMPLE.items():
        cfg["sample"].setdefault(k, v)
    cfg.setdefault("output", {})
    for k, v in _DEFAULT_OUTPUT.items():
        cfg["output"].setdefault(k, v)
    cfg["beam"].setdefault("spectrum", None)
    cfg["beam"].setdefault("divergence_mrad", 0.0)
    return cfg


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load a JSON config from disk, apply overrides, validate against the schema."""
    src = Path(path)
    with src.open("r") as fh:
        cfg = json.load(fh)

    if overrides:
        _deep_update(cfg, overrides)

    cfg = _apply_defaults(cfg)
    jsonschema.validate(cfg, _schema())

    # resolve sample paths relative to the config file's directory if not absolute
    sample = cfg["sample"]
    for key in ("delta_path", "beta_path"):
        if key in sample and not Path(sample[key]).is_absolute():
            sample[key] = str((src.parent / sample[key]).resolve())

    return Config(raw=cfg, source_path=src, cli_overrides=dict(overrides or {}))


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = deepcopy(v)
