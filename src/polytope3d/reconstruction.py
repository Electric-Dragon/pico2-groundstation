"""Depth image -> metric 3D point cloud (pinhole model, camera frame X right, Y down, Z forward)."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def _ray_grid(intrinsics):
    """Per-pixel normalised coordinates (X/Z, Y/Z), computed once per camera and cached."""
    v, u = np.indices((intrinsics.height, intrinsics.width), dtype=np.float32)
    x_n = (u - intrinsics.cx) / intrinsics.fx
    y_n = (v - intrinsics.cy) / intrinsics.fy
    x_n.flags.writeable = False
    y_n.flags.writeable = False
    return x_n, y_n


def depth_to_points(frame, min_range=0.1, max_range=10.0):
    """Convert a DepthFrame to an (N, 3) float32 point cloud in metres.

    Only pixels that are valid and have min_range <= Z <= max_range are converted.
    Z is the depth along the optical axis, not the straight-line distance.
    """
    if not 0 < min_range < max_range:
        raise ValueError(f"need 0 < min_range < max_range, got {min_range}, {max_range}")
    x_n, y_n = _ray_grid(frame.intrinsics)
    z = frame.depth
    keep = frame.valid_mask & (z >= min_range) & (z <= max_range)
    z_keep = z[keep]
    return np.stack((x_n[keep] * z_keep, y_n[keep] * z_keep, z_keep), axis=1)


def project_points(points, intrinsics):
    """Project (N, 3) camera-frame points to (N, 2) pixel coordinates (u, v).

    The inverse of depth_to_points. All points must have Z > 0.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if np.any(points[:, 2] <= 0):
        raise ValueError("all points must be in front of the camera (Z > 0)")
    u = intrinsics.fx * points[:, 0] / points[:, 2] + intrinsics.cx
    v = intrinsics.fy * points[:, 1] / points[:, 2] + intrinsics.cy
    return np.stack((u, v), axis=1)
