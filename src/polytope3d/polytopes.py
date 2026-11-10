import numpy as np
from scipy.spatial import ConvexHull


def aabb(points):
    """Axis-aligned bounding box as half-spaces A x <= b (6 planes, unit normals)."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    A = np.vstack([np.eye(3), -np.eye(3)])
    b = np.concatenate([hi, -lo])
    return A, b


def convex_hull(points):
    """Convex hull via SciPy/Qhull. Returns (A, b, vertices) with A x <= b."""
    hull = ConvexHull(points)
    A = hull.equations[:, :3]
    b = -hull.equations[:, 3]
    return A, b, points[hull.vertices]


def contains(A, b, points, tol=1e-5):
    """Boolean mask: True where a point satisfies every half-space (with tolerance)."""
    return np.all(points @ A.T <= b + tol, axis=1)


def volume_of_aabb(A, b):
    """Volume of an AABB given in the 6-plane form produced by `aabb`."""
    hi, lo = b[:3], -b[3:]
    return float(np.prod(hi - lo))