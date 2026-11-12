"""Axis-aligned bounding box: min and max per axis, six planes. The speed baseline."""

import numpy as np

from polytope3d.polytope import Polytope, check_points


def fit_aabb(points, obstacle_id=0, min_extent=0.01):
    """Fit an axis-aligned box. Axes thinner than min_extent metres are widened symmetrically."""
    pts = check_points(points, min_count=1)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    thin = (hi - lo) < min_extent
    mid = (lo + hi) / 2
    lo = np.where(thin, mid - min_extent / 2, lo)
    hi = np.where(thin, mid + min_extent / 2, hi)
    A = np.vstack([np.eye(3), -np.eye(3)])
    b = np.concatenate([hi, -lo])
    return Polytope(A, b, method="aabb", obstacle_id=obstacle_id, n_points=len(pts))
