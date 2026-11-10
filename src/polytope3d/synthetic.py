import numpy as np


def make_box_cloud(center, size, n_points, seed=0):
    """Random points filling a box. Returns an (n_points, 3) float32 array in metres."""
    rng = np.random.default_rng(seed)
    half = np.asarray(size, dtype=np.float64) / 2
    points = rng.uniform(-half, half, size=(n_points, 3)) + np.asarray(center)
    return points.astype(np.float32)

def render_depth(height, width, fx, fy, cx, cy, cam_height, boxes=()):
    """Depth image (metres, float32) of a flat floor plus axis-aligned boxes.

    Camera frame: X right, Y down, Z forward. The floor is the plane Y = cam_height.
    Each box is (lo, hi): its min and max corners as (x, y, z) in metres.
    Pixels that hit nothing are NaN.
    """
    v, u = np.indices((height, width))
    dx = (u - cx) / fx
    dy = (v - cy) / fy
    depth = np.full((height, width), np.nan)

    below_horizon = dy > 1e-9
    depth[below_horizon] = cam_height / dy[below_horizon]

    rays = np.stack((dx, dy, np.ones_like(dx)), axis=-1)
    for lo, hi in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = np.asarray(lo, dtype=float) / rays
            t_hi = np.asarray(hi, dtype=float) / rays
        t_enter = np.minimum(t_lo, t_hi).max(axis=-1)
        t_exit = np.maximum(t_lo, t_hi).min(axis=-1)
        hit = (t_enter <= t_exit) & (t_enter > 0)
        closer = hit & (np.isnan(depth) | (t_enter < depth))
        depth[closer] = t_enter[closer]

    return depth.astype(np.float32)