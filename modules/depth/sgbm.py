"""OpenCV StereoSGBM: block-matching cost with semi-global smoothing. Fast and
robust; the default."""
import logging

import cv2 as cv
import numpy as np

from .. import config
from .base import StereoMatcher

log = logging.getLogger(__name__)


class SGBM(StereoMatcher):
    """cv.StereoSGBM with settings suited to small grayscale frames.

    block_size: odd matching window (3-11). Bigger is smoother and loses detail.
    P1/P2 follow the OpenCV recommendation 8*bs^2 and 32*bs^2 for one channel.
    full_dp: all 8 paths (MODE_HH) instead of 5; slower, a bit better.
    num_disparities is rounded up to a multiple of 16, as OpenCV requires.
    """
    name = "sgbm"

    def __init__(self, num_disparities: int = config.NUM_DISPARITIES,
                 block_size: int = config.SGBM_BLOCK_SIZE, uniqueness: int = 10,
                 speckle_size: int = 100, speckle_range: int = 2, lr_max_diff: int = 1,
                 full_dp: bool = False):
        rounded = -(-num_disparities // 16) * 16
        if rounded != num_disparities:
            log.info("SGBM: num_disparities %d rounded up to %d", num_disparities, rounded)
        if block_size < 1 or block_size % 2 == 0:
            raise ValueError(f"block_size must be odd and positive, got {block_size}")
        super().__init__(rounded)
        self.block_size = block_size
        self.uniqueness = uniqueness
        self.full_dp = full_dp
        self._sgbm = cv.StereoSGBM_create(
            minDisparity=0, numDisparities=rounded, blockSize=block_size,
            P1=8 * block_size ** 2, P2=32 * block_size ** 2,
            disp12MaxDiff=lr_max_diff, preFilterCap=31, uniquenessRatio=uniqueness,
            speckleWindowSize=speckle_size, speckleRange=speckle_range,
            mode=cv.StereoSGBM_MODE_HH if full_dp else cv.StereoSGBM_MODE_SGBM)

    def _compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        disp = self._sgbm.compute(left, right).astype(np.float32) / 16  # fixed point, 4 fractional bits
        disp[disp < 0] = np.nan  # invalid pixels come back as (minDisparity - 1)
        return disp
