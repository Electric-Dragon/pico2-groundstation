import numpy as np


def depth_to_points(depth, fx, fy, cx, cy):
    """Convert a depth image in metres to an (N, 3) float32 point cloud in metres.

    Invalid pixels (NaN, zero or negative depth) are dropped.
    Camera frame: X right, Y down, Z forward.
    """
    assert depth.ndim == 2, "depth must be a 2D array (height, width)"
    height, width = depth.shape
    v, u = np.indices((height, width))
    z = depth.astype(np.float32)
    valid = np.isfinite(z) & (z > 0)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1)[valid]
    return points.astype(np.float32)
