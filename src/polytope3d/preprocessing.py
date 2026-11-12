"""Point-cloud cleanup: workspace crop, voxel downsampling, outlier removal, floor removal."""

import numpy as np
from scipy.spatial import cKDTree

from polytope3d.contracts import Plane


def crop_to_workspace(points, lo, hi):
    """Keep points inside the axis-aligned box [lo, hi] (camera frame, metres)."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if lo.shape != (3,) or hi.shape != (3,) or np.any(lo >= hi):
        raise ValueError("lo and hi must be 3-vectors with lo < hi")
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    return points[keep]


def voxel_downsample(points, voxel_size):
    """Replace all points inside each cubic voxel by their centroid. Returns float32."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if len(points) == 0:
        return points
    cell = np.floor(points / voxel_size).astype(np.int64)
    cell -= cell.min(axis=0)
    dims = cell.max(axis=0) + 1
    if float(dims[0]) * float(dims[1]) * float(dims[2]) > 2.0**62:
        raise ValueError("voxel_size too small for the extent of the cloud")
    flat = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
    _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    sums = np.stack(
        [np.bincount(inverse, weights=points[:, k], minlength=len(counts)) for k in range(3)],
        axis=1,
    )
    return (sums / counts[:, None]).astype(np.float32)


def remove_statistical_outliers(points, k=16, std_ratio=2.0):
    """Drop points whose mean distance to their k nearest neighbours is unusually large.

    A point is removed if that mean distance exceeds (global mean + std_ratio * global std).
    """
    if k < 1 or std_ratio <= 0:
        raise ValueError("k must be >= 1 and std_ratio > 0")
    if len(points) <= k:
        return points
    dist, _ = cKDTree(points).query(points, k=k + 1)
    mean_dist = dist[:, 1:].mean(axis=1)
    limit = mean_dist.mean() + std_ratio * mean_dist.std()
    return points[mean_dist <= limit]


def _tolerance(points, threshold, threshold_per_z2):
    return threshold + threshold_per_z2 * points[:, 2].astype(np.float64) ** 2


def _least_squares_plane(points, up):
    """Best-fit plane through points: normal = direction of least variance."""
    centroid = points.mean(axis=0)
    centred = points - centroid
    _, vectors = np.linalg.eigh(centred.T @ centred)
    normal = vectors[:, 0]
    if normal @ up < 0:
        normal = -normal
    return Plane(normal=normal, d=float(-normal @ centroid))


def fit_floor_plane(
    points,
    threshold=0.02,
    threshold_per_z2=0.0,
    iterations=200,
    max_tilt_deg=30.0,
    up=(0.0, -1.0, 0.0),
    seed=0,
):
    """Fit the floor with RANSAC, then refine with least squares on its inliers.

    A point is on the plane if |distance| <= threshold + threshold_per_z2 * Z**2 (metres).
    Candidate planes whose normal is more than max_tilt_deg away from `up` are rejected,
    so a large wall is never mistaken for the floor. The returned normal points toward
    `up`; for a floor below the camera, plane.d is then the camera height.
    Raises ValueError if no acceptable plane exists.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if len(points) < 3:
        raise ValueError("need at least 3 points")
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    cos_limit = np.cos(np.radians(max_tilt_deg))
    rng = np.random.default_rng(seed)
    pts = points.astype(np.float64)
    tol = _tolerance(points, threshold, threshold_per_z2)

    best_mask, best_count = None, 2
    for _ in range(iterations):
        a, b, c = pts[rng.choice(len(pts), size=3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9 or abs(normal @ up) / length < cos_limit:
            continue
        normal /= length
        mask = np.abs(pts @ normal - normal @ a) <= tol
        count = int(mask.sum())
        if count > best_count:
            best_mask, best_count = mask, count
    if best_mask is None:
        raise ValueError("no acceptable floor plane found")

    return _least_squares_plane(pts[best_mask], up)


def split_by_plane(points, plane, threshold=0.02, threshold_per_z2=0.0):
    """Split points into (above, on_plane, below) using signed distance to the plane.

    `above` is on the side the plane normal points to (the obstacle candidates for a floor).
    """
    dist = plane.signed_distance(points.astype(np.float64))
    tol = _tolerance(points, threshold, threshold_per_z2)
    return points[dist > tol], points[np.abs(dist) <= tol], points[dist < -tol]
