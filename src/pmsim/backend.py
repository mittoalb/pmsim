"""Tiny array-module dispatcher: numpy on CPU, cupy on CUDA.

`get_xp(device)` returns the numpy-compatible module that owns arrays for the
chosen device:

* ``"cpu"`` → ``numpy``
* ``"cuda"`` → ``cupy`` (raises if cupy not installed or no CUDA device visible)
* ``"auto"`` → cupy if importable, otherwise numpy

The Fresnel propagator and simulator inner loop use only operations that exist
on both — FFT, ``exp``, element-wise multiply, ``abs**2``, ``zeros`` — so they
work unchanged on either backend.
"""

from __future__ import annotations

from typing import Any


def get_xp(device: str = "cpu") -> tuple[Any, str]:
    """Return ``(array_module, resolved_device)``."""
    dev = device.lower()
    if dev not in ("cpu", "cuda", "auto"):
        raise ValueError(f"device must be one of cpu / cuda / auto, got {device!r}")
    if dev == "cpu":
        import numpy as np
        return np, "cpu"
    if dev in ("cuda", "auto"):
        try:
            import cupy as cp  # type: ignore[import-not-found]
            if cp.cuda.runtime.getDeviceCount() == 0:
                raise RuntimeError("no CUDA device visible")
            return cp, "cuda"
        except Exception:
            if dev == "cuda":
                raise
            import numpy as np
            return np, "cpu"
    raise AssertionError("unreachable")


def to_device(arr, xp) -> Any:
    """Copy a NumPy array to the active backend (no-op if already on it)."""
    if xp.__name__ == "numpy":
        import numpy as np
        return np.asarray(arr)
    return xp.asarray(arr)


def to_host(arr) -> Any:
    """Copy a backend array to host (NumPy). No-op for NumPy input."""
    if hasattr(arr, "get"):
        return arr.get()
    import numpy as np
    return np.asarray(arr)
