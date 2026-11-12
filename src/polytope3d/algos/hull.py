"""Full convex hull via SciPy/Qhull: the tightest single convex enclosure of the points."""

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from polytope3d.polytope import DegenerateGeometry, Polytope, check_points


def fit_hull(points, obstacle_id=0, decimals=4):
    """Convex hull as a Polytope. Near-duplicate facet planes are merged, and every offset
    is recomputed from the points so containment holds exactly.

    Raises DegenerateGeometry for fewer than 4 points or points that are flat or collinear.
    """
    pts = check_points(points, min_count=4)
    try:
        hull = ConvexHull(pts)
    except QhullError as exc:
        raise DegenerateGeometry("points are flat, collinear or otherwise degenerate") from exc
    normals = hull.equations[:, :3]
    _, first = np.unique(np.round(normals, decimals), axis=0, return_index=True)
    A = normals[np.sort(first)]
    b = (pts @ A.T).max(axis=0)
    return Polytope(A, b, method="hull", obstacle_id=obstacle_id, n_points=len(pts))
