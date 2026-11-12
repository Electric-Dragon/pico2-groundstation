"""PCA-oriented bounding box. Not the minimum-volume box; can jitter for near-symmetric data."""

import numpy as np

from polytope3d.polytope import Polytope, check_points


def pca_axes(pts):
    """Right-handed orthonormal axes (as columns), longest-variance first, signs made canonical."""
    centred = pts - pts.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    axes = vectors[:, ::-1].copy()
    for k in range(2):
        if axes[np.argmax(np.abs(axes[:, k])), k] < 0:
            axes[:, k] = -axes[:, k]
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    return axes


def fit_obb(points, obstacle_id=0, min_extent=0.01):
    """Fit a box aligned with the principal axes of the points."""
    pts = check_points(points, min_count=4)
    axes = pca_axes(pts)
    s = pts @ axes
    lo, hi = s.min(axis=0), s.max(axis=0)
    thin = (hi - lo) < min_extent
    mid = (lo + hi) / 2
    lo = np.where(thin, mid - min_extent / 2, lo)
    hi = np.where(thin, mid + min_extent / 2, hi)
    A = np.vstack([axes.T, -axes.T])
    b = np.concatenate([hi, -lo])
    return Polytope(A, b, method="obb", obstacle_id=obstacle_id, n_points=len(pts))
