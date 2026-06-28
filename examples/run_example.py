"""Run the bundled example: build the sphere phantom, then simulate."""

from __future__ import annotations

from pathlib import Path

from pmsim import Simulator, load_config

from make_sphere_phantom import main as make_phantom


def main() -> None:
    here = Path(__file__).parent
    make_phantom(out_dir=str(here), n=128, radius_um=25.0, voxel_size_um=0.5)
    cfg = load_config(here / "config_example.json", overrides={
        "output": {"path": str(here / "projection.tif")},
    })
    sim = Simulator(cfg)
    result = sim.run()
    out = sim.save(result)
    print(f"wrote {out}")
    print("metadata:", result.metadata)


if __name__ == "__main__":
    main()
