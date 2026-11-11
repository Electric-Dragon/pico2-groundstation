"""Data contracts: the agreed shapes, types and units of data passed between stages."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera intrinsics, in pixels. Image size is (width, height)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self):
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"focal lengths must be positive, got fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"image size must be positive, got {self.width}x{self.height}")

    def matrix(self):
        """The 3x3 camera matrix K that maps camera-frame points to pixels."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )


@dataclass(frozen=True, eq=False)
class DepthFrame:
    """One depth image in metres (float32, NaN = invalid) plus its metadata."""

    depth: np.ndarray
    intrinsics: Intrinsics
    frame_id: int = 0
    timestamp: float = 0.0
    rig_id: str = "unknown"
    calibration_id: str = "unknown"

    def __post_init__(self):
        d = self.depth
        if not isinstance(d, np.ndarray) or d.ndim != 2:
            raise ValueError("depth must be a 2D numpy array (height, width)")
        if d.dtype != np.float32:
            raise ValueError(f"depth must be float32 metres, got dtype {d.dtype}")
        expected = (self.intrinsics.height, self.intrinsics.width)
        if d.shape != expected:
            raise ValueError(f"depth shape {d.shape} does not match intrinsics (h, w) {expected}")

    @property
    def valid_mask(self):
        """True where the depth is a usable measurement (finite and > 0)."""
        return np.isfinite(self.depth) & (self.depth > 0)
