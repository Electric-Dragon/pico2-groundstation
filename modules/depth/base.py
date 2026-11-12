"""The interface every stereo matcher implements."""
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from .. import config


class StereoMatcher(ABC):
    """Turns a rectified left/right pair into a disparity map for the left image.

    compute() returns float32 (H, W) disparities in pixels, with NaN where no
    reliable match was found. A left pixel at x matches the right pixel at x - d.
    Disparities searched are 0 .. num_disparities - 1.
    """
    name: ClassVar[str]

    def __init__(self, num_disparities: int = config.NUM_DISPARITIES):
        if num_disparities < 2:
            raise ValueError(f"num_disparities must be at least 2, got {num_disparities}")
        self.num_disparities = num_disparities

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if left.shape != right.shape or left.ndim != 2:
            raise ValueError(f"need two grayscale images of one size, got {left.shape} and {right.shape}")
        if left.dtype != np.uint8 or right.dtype != np.uint8:
            raise ValueError(f"need uint8 images, got {left.dtype} and {right.dtype}")
        return self._compute(left, right)

    @abstractmethod
    def _compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray: ...

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({params})"
