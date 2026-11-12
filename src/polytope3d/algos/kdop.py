"""Fixed-direction k-DOP: one supporting plane per direction from a published direction set."""

import numpy as np

from polytope3d.polytope import Polytope, check_points

AXES_7 = (
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
)
AXES_13 = AXES_7 + (
    (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
)


def make_directions(axes):
    """Unit normals for each axis and its opposite. Row 2j is +axis j, row 2j+1 is -axis j.

    The first three axes must be x, y and z, so rows 0..5 are +x, -x, +y, -y, +z, -z.
    """
    axes = np.asarray(axes, dtype=np.float64)
    if axes.ndim != 2 or axes.shape[1] != 3 or len(axes) < 3:
        raise ValueError("axes must be (K, 3) with K >= 3")
    if not np.array_equal(axes[:3], np.eye(3)):
        raise ValueError("the first three axes must be x, y and z")
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("axes must be non-zero")
    unit = axes / norms
    return np.stack([unit, -unit], axis=1).reshape(-1, 3)


def fit_kdop(points, axes=AXES_7, obstacle_id=0, min_extent=0.01, method="kdop"):
    """Fit a k-DOP, where k = 2 * len(axes). Each offset is the largest projection of any point
    onto that direction, so every point is inside by construction. Cost is N * k.
    """
    pts = check_points(points, min_count=1)
    directions = make_directions(axes)
    b = (pts @ directions.T).max(axis=0)
    for a in range(3):
        extent = b[2 * a] + b[2 * a + 1]
        if extent < min_extent:
            pad = (min_extent - extent) / 2
            b[2 * a] += pad
            b[2 * a + 1] += pad
    return Polytope(directions, b, method=method, obstacle_id=obstacle_id, n_points=len(pts))


def fit_kdop14(points, obstacle_id=0, min_extent=0.01):
    """14-DOP: 3 coordinate axes + 4 cube diagonals (7 axes, 14 planes)."""
    return fit_kdop(points, AXES_7, obstacle_id, min_extent, method="kdop14")


def fit_kdop26(points, obstacle_id=0, min_extent=0.01):
    """26-DOP: the 14-DOP axes + 6 edge directions (13 axes, 26 planes)."""
    return fit_kdop(points, AXES_13, obstacle_id, min_extent, method="kdop26")