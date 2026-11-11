"""Synthetic test scenes with known ground truth."""

from dataclasses import dataclass

import numpy as np

from polytope3d.contracts import DepthFrame


@dataclass(frozen=True)
class Box:
    """Axis-aligned box given by its min corner `lo` and max corner `hi`, (x, y, z) in metres."""

    lo: tuple
    hi: tuple

    def __post_init__(self):
        if len(self.lo) != 3 or len(self.hi) != 3:
            raise ValueError("lo and hi must each have 3 values")
        if not all(low < high for low, high in zip(self.lo, self.hi)):
            raise ValueError(f"need lo < hi on every axis, got lo={self.lo}, hi={self.hi}")


def make_box_surface_cloud(box, n_points, seed=0):
    """Random points on the surface of a box. Returns an (n_points, 3) float32 array in metres."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(box.lo, dtype=float), np.asarray(box.hi, dtype=float)
    size = hi - lo
    area_per_axis = np.array([size[1] * size[2], size[0] * size[2], size[0] * size[1]])
    face_prob = np.repeat(area_per_axis, 2) / (2 * area_per_axis.sum())
    points = rng.uniform(lo, hi, size=(n_points, 3))
    face = rng.choice(6, size=n_points, p=face_prob)
    axis, side = face // 2, face % 2
    points[np.arange(n_points), axis] = np.where(side == 0, lo[axis], hi[axis])
    return points.astype(np.float32)


def _ray_box_entry(rays, box):
    """Distance parameter t where each ray first enters the box (inf where it misses)."""
    t_enter = np.full(rays.shape[:2], -np.inf)
    t_exit = np.full(rays.shape[:2], np.inf)
    for axis in range(3):
        lo, hi = box.lo[axis], box.hi[axis]
        d = rays[..., axis]
        parallel = np.abs(d) < 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = lo / d, hi / d
        t_min, t_max = np.minimum(t1, t2), np.maximum(t1, t2)
        origin_in_slab = lo <= 0.0 <= hi
        t_min = np.where(parallel, -np.inf if origin_in_slab else np.inf, t_min)
        t_max = np.where(parallel, np.inf if origin_in_slab else -np.inf, t_max)
        t_enter = np.maximum(t_enter, t_min)
        t_exit = np.minimum(t_exit, t_max)
    hit = (t_enter <= t_exit) & (t_enter > 0)
    return np.where(hit, t_enter, np.inf)


def render_depth(intrinsics, cam_height, boxes=(), noise_k=0.0, dropout=0.0, seed=0):
    """Render a DepthFrame of a flat floor plus boxes, seen by a level camera.

    Camera frame: X right, Y down, Z forward; the floor is the plane Y = cam_height.
    noise_k: depth noise standard deviation is noise_k * Z**2 (metres), 0 = perfect.
    dropout: fraction of pixels randomly set to NaN (simulates stereo holes).
    Pixels that hit nothing are NaN.
    """
    rng = np.random.default_rng(seed)
    v, u = np.indices((intrinsics.height, intrinsics.width))
    dx = (u - intrinsics.cx) / intrinsics.fx
    dy = (v - intrinsics.cy) / intrinsics.fy
    depth = np.full(dx.shape, np.inf)

    below_horizon = dy > 1e-9
    depth[below_horizon] = cam_height / dy[below_horizon]

    rays = np.stack((dx, dy, np.ones_like(dx)), axis=-1)
    for box in boxes:
        depth = np.minimum(depth, _ray_box_entry(rays, box))

    depth[~np.isfinite(depth)] = np.nan
    if noise_k > 0:
        depth = depth + rng.normal(size=depth.shape) * noise_k * depth**2
    if dropout > 0:
        depth[rng.random(depth.shape) < dropout] = np.nan
    return DepthFrame(
        depth=depth.astype(np.float32),
        intrinsics=intrinsics,
        rig_id="synthetic",
        calibration_id="synthetic",
    )
