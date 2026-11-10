"""Stereo matching: rectified pair -> disparity map -> depth.

Each algorithm lives in its own module and implements StereoMatcher:
  sgbm        OpenCV StereoSGBM (default)
  census-sgm  census-transform cost with semi-global matching, in numpy
  elas        ELAS (Geiger et al. 2010), a numpy port of libelas's main steps

  matcher = depth.create("elas")          # or create() for GCS_DEPTH_ALGO
  disp = matcher.compute(left, right)     # float32 pixels, NaN = no match
  z = depth.depth_mm(disp, rect.focal_px, rect.baseline_mm)

Run `python -m modules.depth --help` for the live viewer.
"""
import importlib

import numpy as np

from .. import config
from .base import StereoMatcher

## name -> (module, class). Modules are imported on first use, so e.g. scipy
## only loads when ELAS is picked.
ALGORITHMS: dict[str, tuple[str, str]] = {
    "sgbm": ("sgbm", "SGBM"),
    "census-sgm": ("census_sgm", "CensusSGM"),
    "elas": ("elas", "ELAS"),
}

__all__ = ["ALGORITHMS", "StereoMatcher", "create", "depth_mm"]


def create(name: str = config.DEPTH_ALGO, **params) -> StereoMatcher:
    """Build the matcher called `name`. Extra keyword arguments go to its constructor."""
    try:
        module, cls = ALGORITHMS[name]
    except KeyError:
        raise ValueError(f"unknown depth algorithm {name!r}; choose from {', '.join(ALGORITHMS)}") from None
    return getattr(importlib.import_module(f".{module}", __name__), cls)(**params)


def depth_mm(disparity: np.ndarray, focal_px: float, baseline_mm: float) -> np.ndarray:
    """Distance along the optical axis, Z = f * B / d, in mm. NaN where d <= 0 or invalid."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (focal_px * baseline_mm) / disparity
    z[~(disparity > 0)] = np.nan
    return z
