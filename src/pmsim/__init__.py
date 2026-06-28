"""pmsim — propagation-based phase-contrast X-ray projection microscope simulator."""

from .config import load_config
from .simulator import Simulator

__all__ = ["Simulator", "load_config", "__version__"]
__version__ = "0.1.0"
